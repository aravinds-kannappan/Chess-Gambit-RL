"""Shannon's Gambit - Hugging Face Space backend (FastAPI).

This Space *trains and serves* the chess agents end to end:

* serves moves / predictions / watch-mode pairings by target Elo from the
  checkpoint ladder (`shannons_gambit.serve.ModelServer`),
* runs continuous self-play in a background thread (`ContinualTrainer`), pushing
  each new generation to the Hub so the ladder survives Space restarts,
* logs human games and fine-tunes a personal checkpoint on request (`/adapt`).

There is no heuristic fallback: every move comes from a trained network.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from shannons_gambit.serve import ModelServer

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")  # e.g. legacyaravind/shannons-gambit
TRAIN_ENABLED = os.environ.get("TRAIN_ENABLED", "1") == "1"
# Opt-in exact-endgame agents the phase router dispatches to (e.g. "KRvK,KQvK").
# Off by default to keep cold starts fast on free CPU.
ENDGAME_MDPS = tuple(n for n in os.environ.get("ENDGAME_MDP", "").split(",") if n.strip())

server = ModelServer(MODELS_DIR, endgame_mdps=ENDGAME_MDPS)
_train_state = {"running": False, "last_gen": None, "last_elo": None}
_calib_state = {"running": False, "result": None}


def _background_trainer() -> None:
    from shannons_gambit.agents.alphazero.continual import ContinualTrainer
    from shannons_gambit.config import ContinualConfig, NetConfig
    from shannons_gambit.export import push_ladder_to_hub

    cfg = ContinualConfig(
        run_dir=MODELS_DIR, init_from=str(Path(MODELS_DIR) / "gen-0000.pt"),
        net=NetConfig(channels=32, blocks=3), games_per_gen=8, simulations=24,
        max_moves=60, eval_games=12, eval_sims=16, device="cpu",
    )
    trainer = ContinualTrainer(cfg)
    _train_state["running"] = True
    while True:
        try:
            entry = trainer.step()
            _train_state.update(last_gen=entry.gen, last_elo=entry.elo)
            server.reload()
            if HF_MODEL_REPO and os.environ.get("HF_TOKEN"):
                try:
                    push_ladder_to_hub(HF_MODEL_REPO, MODELS_DIR)
                except Exception:  # noqa: BLE001 - Hub push is best-effort
                    pass
        except Exception as exc:  # noqa: BLE001 - keep the worker alive
            print("trainer error:", exc, flush=True)
            time.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if HF_MODEL_REPO:
        try:
            from shannons_gambit.export import pull_ladder_from_hub

            pull_ladder_from_hub(HF_MODEL_REPO, MODELS_DIR)
            server.reload()
        except Exception:  # noqa: BLE001
            pass
    server.ensure_seeded()
    if TRAIN_ENABLED:
        threading.Thread(target=_background_trainer, daemon=True).start()
    yield


app = FastAPI(title="Shannon's Gambit API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MoveReq(BaseModel):
    fen: str
    elo: float | None = None
    session: str | None = None


class PredictReq(BaseModel):
    fen: str


class WatchReq(BaseModel):
    fen: str
    white_elo: float
    black_elo: float


class LogReq(BaseModel):
    session_id: str
    fens: list[str]
    moves: list[str]
    result: float  # +1 / 0 / -1 for the agent


class AdaptReq(BaseModel):
    session_id: str


@app.get("/healthz")
def healthz() -> dict:
    info = server.ladder_info()
    return {"status": "ok", "generations": info["generations"],
            "best_elo": info["best_elo"], "calibrated_elo": info["calibrated_elo"],
            "training": _train_state, "calibration": _calib_state}


@app.get("/ladder")
def ladder() -> dict:
    return server.ladder_info()


def _run_calibration() -> None:
    _calib_state["running"] = True
    try:
        _calib_state["result"] = server.calibrate()
        server.ladder.save()
    except Exception as exc:  # noqa: BLE001 - surface, do not crash the worker
        _calib_state["result"] = {"error": str(exc)}
    finally:
        _calib_state["running"] = False


@app.post("/calibrate")
def calibrate() -> dict:
    """Grade the served agent against Stockfish and store its calibrated Elo.

    Runs in the background (it plays real games vs Stockfish); poll /healthz.
    """
    if _calib_state["running"]:
        return {"status": "already running", "calibration": _calib_state}
    threading.Thread(target=_run_calibration, daemon=True).start()
    return {"status": "started"}


@app.post("/move")
def move(req: MoveReq) -> dict:
    personal = Path(MODELS_DIR) / "personal" / f"{req.session}.pt" if req.session else None
    if personal and personal.exists():
        import chess

        from shannons_gambit.agents.alphazero.mcts import AlphaZeroAgent

        agent = AlphaZeroAgent.from_checkpoint(str(personal), simulations=server.base_sims)
        mv = agent.select_move(chess.Board(req.fen))
        return {"move": mv.uci(), "source": "personal", "gen": -1, "elo": None}
    return server.move(req.fen, elo=req.elo)


@app.post("/watch-move")
def watch_move(req: WatchReq) -> dict:
    return server.watch_move(req.fen, req.white_elo, req.black_elo)


@app.post("/predict")
def predict(req: PredictReq) -> dict:
    return server.predict(req.fen)


@app.post("/log_game")
def log_game(req: LogReq) -> dict:
    import json

    sess_dir = Path(MODELS_DIR) / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    with (sess_dir / f"{req.session_id}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"fens": req.fens, "moves": req.moves, "result": req.result}) + "\n")
    return {"status": "logged", "session": req.session_id}


@app.post("/adapt")
def adapt(req: AdaptReq) -> dict:
    import json

    from shannons_gambit.agents.adaptive import adapt_to_games

    path = Path(MODELS_DIR) / "sessions" / f"{req.session_id}.jsonl"
    if not path.exists():
        return {"error": "no logged games for this session"}
    games = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    best = server.ladder.best()
    out = Path(MODELS_DIR) / "personal" / f"{req.session_id}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = adapt_to_games(server._resolve(best), games, str(out))
    return {"adapted": True, "n_games": len(games), **result}
