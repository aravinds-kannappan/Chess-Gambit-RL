"""Model-serving core used by the Hugging Face Space.

Loads the checkpoint ladder, picks a snapshot by target Elo (with simulation /
temperature / blunder tuning for fine-grained strength), and answers move /
predict / watch / eval requests. Kept in the package (not the Space) so it is
unit-testable. The Space (``deploy/hf_space/app.py``) is a thin FastAPI + trainer
wrapper around this.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import chess

from .agents.alphazero.mcts import AlphaZeroAgent
from .agents.ladder import Ladder, LadderEntry
from .agents.router import MDPAgent, PhaseRouter
from .models.net import ChessNet, save_model
from .models.prediction import Predictor


class ModelServer:
    def __init__(self, models_dir: str = "runs/continual", *, device: str = "cpu",
                 base_sims: int = 48, endgame_mdps: tuple[str, ...] = ()) -> None:
        self.models_dir = models_dir
        self.device = device
        self.base_sims = base_sims
        # Exact-endgame specialists the phase router can dispatch to (lazy-built).
        self.endgame_mdps = tuple(endgame_mdps)
        self._endgame_agent: MDPAgent | None = None
        self._endgame_loaded = False
        self.ladder = Ladder.load(models_dir)
        self._models: dict[int, ChessNet] = {}
        self._predictors: dict[int, Predictor] = {}

    # --- seeding / state ---------------------------------------------------
    def ensure_seeded(self, supervised_path: str = "runs/supervised/model.pt") -> None:
        """Guarantee at least one checkpoint exists (supervised bootstrap or scratch)."""
        if self.ladder.entries:
            return
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        ckpt = Path(self.models_dir) / "gen-0000.pt"
        if Path(supervised_path).exists():
            from .models.net import load_model

            model, extra = load_model(supervised_path, map_location=self.device)
            save_model(model, str(ckpt), extra={"gen": 0, "elo": 1200.0, **extra})
            elo = 1200.0
        else:
            model = ChessNet()
            save_model(model, str(ckpt), extra={"gen": 0, "elo": 600.0})
            elo = 600.0
        self.ladder.add(0, str(ckpt), elo, {"seeded": True})
        self.ladder.save()

    def reload(self) -> None:
        self.ladder = Ladder.load(self.models_dir)
        self._models.clear()
        self._predictors.clear()

    def _resolve(self, entry: LadderEntry) -> str:
        """Return a usable checkpoint path (stored path, or by name in models_dir)."""
        if Path(entry.path).exists():
            return entry.path
        fallback = Path(self.models_dir) / f"{entry.name}.pt"
        return str(fallback)

    def _model(self, gen: int) -> ChessNet:
        if gen not in self._models:
            from .models.net import load_model

            entry = next(e for e in self.ladder.entries if e.gen == gen)
            model, _ = load_model(self._resolve(entry), map_location=self.device)
            self._models[gen] = model.to(self.device).eval()
        return self._models[gen]

    def _predictor(self, gen: int) -> Predictor:
        if gen not in self._predictors:
            entry = next(e for e in self.ladder.entries if e.gen == gen)
            self._predictors[gen] = Predictor.from_checkpoint(
                self._resolve(entry), device=self.device)
        return self._predictors[gen]

    # --- strength selection ------------------------------------------------
    def agent_for_elo(self, target_elo: float, *, seed: int = 0
                      ) -> tuple[AlphaZeroAgent, LadderEntry]:
        entry = self.ladder.nearest(target_elo) or self.ladder.latest()
        if entry is None:
            raise RuntimeError("no checkpoints available; call ensure_seeded()")
        model = self._model(entry.gen)
        gap = target_elo - entry.elo
        # A light temperature even at full strength so games are not identical
        # every time (MCTS with no root noise is otherwise deterministic).
        sims, temperature, blunder = self.base_sims, 0.35, 0.0
        if gap < -50:  # weaker than this snapshot
            blunder = min(0.6, -gap / 800.0)
            temperature = 0.7
            sims = max(8, self.base_sims // 2)
        elif gap > 50:  # stronger
            sims = self.base_sims * 2
        agent = AlphaZeroAgent(model, device=self.device, simulations=sims,
                               temperature=temperature, blunder_rate=blunder, seed=seed)
        return agent, entry

    # --- multi-agent routing ----------------------------------------------
    def endgame_agent(self) -> MDPAgent | None:
        """An exact-endgame ``MDPAgent`` (lazily solved), or ``None`` if disabled."""
        if not self._endgame_loaded:
            self._endgame_loaded = True
            if self.endgame_mdps:
                from .mdp.chess_mdp import load_endgame

                cache = str(Path(self.models_dir) / "mdp_cache")
                mdps = []
                for name in self.endgame_mdps:
                    try:
                        mdps.append(load_endgame(name, cache_dir=cache))
                    except Exception:  # noqa: BLE001 - skip an endgame we cannot build
                        pass
                self._endgame_agent = MDPAgent(mdps) if mdps else None
        return self._endgame_agent

    def router_for_elo(self, target_elo: float, *, seed: int = 0
                       ) -> tuple[PhaseRouter, LadderEntry]:
        """The served agent: a phase router (neural general + exact endgame)."""
        neural, entry = self.agent_for_elo(target_elo, seed=seed)
        router = PhaseRouter(general=neural, endgame=self.endgame_agent())
        return router, entry

    # --- request handlers --------------------------------------------------
    def move(self, fen: str, *, elo: float | None = None, seed: int | None = None) -> dict:
        # A fresh random seed per call so stochastic move selection actually
        # varies -- otherwise every game is a deterministic replay.
        if seed is None:
            seed = random.randrange(1, 2**31)
        board = chess.Board(fen)
        best = self.ladder.best()
        target = elo if elo is not None else (best.elo if best else 1000)
        router, entry = self.router_for_elo(target, seed=seed)
        mv = router.select_move(board)
        return {
            "move": mv.uci(),
            "gen": entry.gen,
            "elo": entry.elo,
            "calibrated_elo": entry.metrics.get("calibrated_elo"),
            "route": router.last_route,
            "source": "model",
        }

    def watch_move(self, fen: str, white_elo: float, black_elo: float) -> dict:
        board = chess.Board(fen)
        target = white_elo if board.turn == chess.WHITE else black_elo
        return self.move(fen, elo=target)

    def predict(self, fen: str) -> dict:
        best = self.ladder.best() or self.ladder.latest()
        pred = self._predictor(best.gen).predict(chess.Board(fen))
        return {**pred.to_dict(), "gen": best.gen, "source": "model"}

    def ladder_info(self) -> dict:
        best = self.ladder.best()
        return {
            "generations": len(self.ladder.entries),
            "best_elo": best.elo if best else None,
            "calibrated_elo": best.metrics.get("calibrated_elo") if best else None,
            "levels": self.ladder.levels(),
            "elo_curve": self.ladder.elo_curve(),
        }

    def calibrate(self, *, stockfish_path: str | None = None,
                  anchors: tuple[int, ...] = (1350, 1700, 2100),
                  elo_games: int = 4, n_positions: int = 60,
                  movetime_ms: int = 30, with_elo: bool = True) -> dict:
        """Score the served agent against Stockfish and store a calibrated Elo.

        This is the only place Stockfish touches serving: it grades the agent (it
        never plays for it). Writes ``calibrated_elo`` / ``acpl`` / ``top1`` into
        the best entry's metrics so subsequent moves report the benchmarked Elo.
        Heavy on CPU (real games vs Stockfish); call it on demand, not per move.
        """
        from .agents.stockfish import find_stockfish
        from .eval.benchmark import assess_elo, move_quality, random_positions

        if find_stockfish(stockfish_path) is None:
            raise RuntimeError("no Stockfish binary; set $STOCKFISH_PATH or install it")
        best = self.ladder.best() or self.ladder.latest()
        if best is None:
            raise RuntimeError("no checkpoints to calibrate; call ensure_seeded()")

        router, _ = self.router_for_elo(best.elo)
        positions = random_positions(n_positions)
        quality = move_quality(router, positions, movetime_ms=movetime_ms, path=stockfish_path)
        best.metrics["acpl"] = quality["avg_centipawn_loss"]
        best.metrics["top1"] = quality["top1_agreement"]
        result = {"gen": best.gen, **quality}
        if with_elo:
            router2, _ = self.router_for_elo(best.elo)
            elo = assess_elo(router2, anchors=anchors, games=elo_games,
                             movetime_ms=movetime_ms, path=stockfish_path)
            best.metrics["calibrated_elo"] = elo["elo"]
            result["calibrated_elo"] = elo["elo"]
            result["score_vs_anchor"] = elo["score_vs_anchor"]
        self.ladder.save()
        return result


@lru_cache(maxsize=1)
def get_server(models_dir: str = "runs/continual") -> ModelServer:
    server = ModelServer(models_dir)
    server.ensure_seeded()
    return server
