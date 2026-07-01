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
# Serve-only by default: a CPU Space that also self-plays starves the web server
# (healthz times out -> the site shows "backend warming up"). Train on HF Jobs
# (deploy/hf_job) and set TRAIN_ENABLED=1 only on a Space you don't also serve from.
TRAIN_ENABLED = os.environ.get("TRAIN_ENABLED", "0") == "1"
# Seconds the background trainer sleeps between generations to yield CPU to
# request handling when training and serving do share a box.
TRAIN_SLEEP = float(os.environ.get("TRAIN_SLEEP", "5"))
# Opt-in exact-endgame agents the phase router dispatches to (e.g. "KRvK,KQvK").
# Off by default to keep cold starts fast on free CPU.
ENDGAME_MDPS = tuple(n for n in os.environ.get("ENDGAME_MDP", "").split(",") if n.strip())

server = ModelServer(MODELS_DIR, endgame_mdps=ENDGAME_MDPS)
_train_state = {"running": False, "last_gen": None, "last_elo": None}
_calib_state = {"running": False, "result": None}
# Adaptation state: cached personal agents (loading a 40MB checkpoint per move
# is what made adapted play crawl) and an in-flight guard per session.
_personal_agents: dict[str, object] = {}
_adapt_state: dict[str, dict] = {}


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
    # Drop the worker's scheduling priority so request handling stays responsive.
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    while True:
        try:
            entry = trainer.step()
            _train_state.update(last_gen=entry.gen, last_elo=entry.elo,
                                promoted=entry.metrics.get("promoted"))
            server.reload()
            if HF_MODEL_REPO and os.environ.get("HF_TOKEN"):
                try:
                    push_ladder_to_hub(HF_MODEL_REPO, MODELS_DIR)
                except Exception:  # noqa: BLE001 - Hub push is best-effort
                    pass
            time.sleep(TRAIN_SLEEP)  # yield CPU between generations
        except Exception as exc:  # noqa: BLE001 - keep the worker alive
            print("trainer error:", exc, flush=True)
            time.sleep(10)


def _warmup() -> None:
    """Pull Hub artifacts OFF the startup path so the app is healthy immediately.

    A blocking startup that downloads the ladder + every checkpoint can take many
    minutes (the ladder can hold thousands of generations); HF's health check then
    times out and the Space restart-loops. Doing it in a daemon thread lets uvicorn
    answer /healthz at once and the served net swaps in as each piece lands.
    """
    try:
        from shannons_gambit.export import pull_ladder_from_hub

        pull_ladder_from_hub(HF_MODEL_REPO, MODELS_DIR)
        server.reload()
    except Exception as exc:  # noqa: BLE001
        print("ladder pull skipped:", exc, flush=True)
    try:  # learned opening book (served for the first plies); optional
        from shannons_gambit.export import pull_opening_book

        pull_opening_book(HF_MODEL_REPO, MODELS_DIR)
        server.reload()
    except Exception as exc:  # noqa: BLE001
        print("book pull skipped:", exc, flush=True)
    # Serve from the strong pre-trained net (pretrain/model.pt), scaled to the
    # target Elo, instead of weak self-play generations.
    try:
        from shannons_gambit.export import pull_base_model

        base = pull_base_model(HF_MODEL_REPO, MODELS_DIR)
        if base:
            # The reference Elo must be the net's MEASURED strength, or every
            # scaled level is wrong (a 1600 reference on a ~980 net meant
            # "900" was served with 50% random blunders). Prefer the champion's
            # Stockfish-calibrated rating; BASE_ELO env only overrides explicitly.
            champ = server.ladder.champion()
            measured = champ.metrics.get("calibrated_elo") if champ else None
            elo = float(os.environ.get("BASE_ELO") or measured or (champ.elo if champ else 1000))
            server.set_base(base, elo=elo)
            print(f"serving base at reference elo {elo}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("base model load skipped:", exc, flush=True)
    server.ensure_seeded()
    if TRAIN_ENABLED:
        _background_trainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed something servable instantly (local, no network), then warm up from the
    # Hub in the background so the Space passes its health check right away.
    server.ensure_seeded()
    if HF_MODEL_REPO:
        threading.Thread(target=_warmup, daemon=True).start()
    elif TRAIN_ENABLED:
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
            "ceiling": info["ceiling"],
            "training": _train_state, "calibration": _calib_state,
            "adapting_sessions": sum(1 for s in _adapt_state.values() if s.get("running"))}


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


def _personal_agent(session: str):
    """Cached personal agent for a session: local file first, then the Hub.

    Returns ``None`` when the session has no adapted checkpoint anywhere. The
    Hub lookup is what makes adaptation survive Space restarts (free-tier disks
    are ephemeral, so a local-only personal net used to vanish every restart).
    """
    if session in _personal_agents:
        return _personal_agents[session]
    path = Path(MODELS_DIR) / "personal" / f"{session}.pt"
    if not path.exists() and HF_MODEL_REPO:
        try:
            from shannons_gambit.export import pull_personal

            pull_personal(HF_MODEL_REPO, session, MODELS_DIR)
        except Exception:  # noqa: BLE001 - fall through to the shared net
            pass
    agent = None
    if path.exists():
        from shannons_gambit.agents.alphazero.mcts import AlphaZeroAgent

        agent = AlphaZeroAgent.from_checkpoint(str(path), simulations=server.base_sims,
                                               temperature=0.3)
    _personal_agents[session] = agent
    return agent


@app.post("/move")
def move(req: MoveReq) -> dict:
    if req.session:
        agent = _personal_agent(req.session)
        if agent is not None:
            import chess

            mv = agent.select_move(chess.Board(req.fen))
            ceiling = server.ceiling()
            return {"move": mv.uci(), "source": "personal", "gen": -1,
                    "elo": round(ceiling, 0), "ceiling": round(ceiling, 0),
                    "route": "personal", "adapted": True}
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
    # Adaptation should not require a button: fine-tune the personal net in the
    # background after every finished game (guarded so one session adapts once
    # at a time). The next /move for this session serves the adapted net.
    started = _start_adapt(req.session_id)
    return {"status": "logged", "session": req.session_id, "adapting": started}


def _base_checkpoint() -> str | None:
    """Path to the checkpoint personal fine-tuning starts from (base net, else best)."""
    base = Path(MODELS_DIR) / "model.pt"
    if base.exists():
        return str(base)
    best = server.ladder.best() or server.ladder.latest()
    if best is None:
        return None
    resolved = server._resolve(best)
    return resolved if Path(resolved).exists() else None


def _do_adapt(session_id: str) -> dict:
    """Fine-tune the session's personal checkpoint on its logged games."""
    import json

    from shannons_gambit.agents.adaptive import adapt_to_games

    path = Path(MODELS_DIR) / "sessions" / f"{session_id}.jsonl"
    if not path.exists():
        return {"error": "no logged games for this session yet - play a full game first"}
    base = _base_checkpoint()
    if base is None:
        return {"error": "no base checkpoint available to fine-tune from"}
    games = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = Path(MODELS_DIR) / "personal" / f"{session_id}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = adapt_to_games(base, games, str(out))
    _personal_agents.pop(session_id, None)  # next /move loads the new weights
    if HF_MODEL_REPO and os.environ.get("HF_TOKEN"):
        try:  # persist across restarts (free-tier disk is ephemeral)
            from shannons_gambit.export import push_personal

            push_personal(HF_MODEL_REPO, session_id, str(out))
            result["persisted"] = True
        except Exception as exc:  # noqa: BLE001 - adaptation still works locally
            print("personal push skipped:", exc, flush=True)
    return {"adapted": True, "n_games": len(games), **result}


def _start_adapt(session_id: str) -> bool:
    """Kick off a background adapt for a session (no-op if one is running)."""
    state = _adapt_state.get(session_id)
    if state and state.get("running"):
        return False

    def worker() -> None:
        _adapt_state[session_id] = {"running": True, "result": None}
        try:
            _adapt_state[session_id]["result"] = _do_adapt(session_id)
        except Exception as exc:  # noqa: BLE001
            _adapt_state[session_id]["result"] = {"error": f"adapt failed: {exc}"}
        finally:
            _adapt_state[session_id]["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return True


@app.post("/adapt")
def adapt(req: AdaptReq) -> dict:
    """Manual adapt trigger; also reports a background adapt's latest result."""
    state = _adapt_state.get(req.session_id)
    if state and state.get("running"):
        return {"status": "adapting"}
    if state and state.get("result") is not None:
        return state["result"]
    try:
        return _do_adapt(req.session_id)
    except Exception as exc:  # noqa: BLE001 - report instead of a bare 500
        return {"error": f"adapt failed: {exc}"}
