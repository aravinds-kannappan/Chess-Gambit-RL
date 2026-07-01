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
        # Learned opening book (lazy-loaded from <models_dir>/opening_book.json).
        self._opening_agent = None
        self._opening_loaded = False
        self.ladder = Ladder.load(models_dir)
        self._models: dict[int, ChessNet] = {}
        self._predictors: dict[int, Predictor] = {}
        # Optional pre-trained base net (the supervised model.pt). When set, every
        # served move comes from this strong net scaled to the requested Elo,
        # rather than from a noisy self-play generation.
        self._base_model: ChessNet | None = None
        self._base_elo: float = 1500.0

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

    def set_base(self, path: str, *, elo: float = 1500.0) -> bool:
        """Use a pre-trained checkpoint as the served base net (scaled to target Elo)."""
        if not Path(path).exists():
            return False
        from .models.net import load_model

        model, _ = load_model(path, map_location=self.device)
        self._base_model = model.to(self.device).eval()
        self._base_elo = elo
        return True

    def reload(self) -> None:
        self.ladder = Ladder.load(self.models_dir)
        self._models.clear()
        self._predictors.clear()
        self._opening_loaded = False  # re-read a freshly pulled opening_book.json

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
    def ceiling(self) -> float:
        """The engine's honest maximum strength: the reference Elo it serves from.

        The base net's measured (Stockfish-calibrated when available) rating.
        Requests above this are served at full strength and *reported* at the
        ceiling instead of echoing an Elo the engine cannot actually play at.
        """
        if self._base_model is not None:
            return float(self._base_elo)
        entry = self.ladder.champion() or self.ladder.latest()
        if entry is None:
            return 600.0
        return float(entry.metrics.get("calibrated_elo") or entry.elo)

    def agent_for_elo(self, target_elo: float, *, seed: int = 0
                      ) -> tuple[AlphaZeroAgent, LadderEntry]:
        # Serve from the strongest net available (pre-trained base if set, else
        # the gated CHAMPION -- a checkpoint that actually beat its predecessor,
        # never a lucky-noise generation), then scale it to the requested Elo.
        # ``reference`` is that net's own Elo.
        entry = self.ladder.champion() or self.ladder.latest()
        if self._base_model is not None:
            model = self._base_model
            reference = self._base_elo
        else:
            if entry is None:
                raise RuntimeError("no checkpoints available; call ensure_seeded()")
            model = self._model(entry.gen)
            reference = float(entry.metrics.get("calibrated_elo") or entry.elo)
        gap = target_elo - reference
        # Light temperature even at full strength so games are not identical
        # (MCTS with no root noise is otherwise deterministic).
        sims, temperature, blunder = self.base_sims, 0.3, 0.0
        if gap < -50:  # play down to a weaker target via blunders + lower search
            blunder = min(0.8, (reference - target_elo) / 1400.0)
            temperature = 0.85
            sims = max(6, int(self.base_sims * max(0.25, 1 + gap / 1600.0)))
        elif gap > 50:  # play up: more search, sharper
            sims = self.base_sims * 2
            temperature = 0.15
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

    def opening_agent(self):
        """Opening-book agent from ``<models_dir>/opening_book.json``, or ``None``."""
        if not self._opening_loaded:
            self._opening_loaded = True
            path = Path(self.models_dir) / "opening_book.json"
            if path.exists():
                try:
                    from .agents.opening_book import OpeningBook, OpeningBookAgent

                    self._opening_agent = OpeningBookAgent(OpeningBook.load(str(path)))
                except Exception:  # noqa: BLE001 - a bad book must not break serving
                    self._opening_agent = None
        return self._opening_agent

    def router_for_elo(self, target_elo: float, *, seed: int = 0
                       ) -> tuple[PhaseRouter, LadderEntry]:
        """The served agent: a phase router (opening book + neural + exact endgame)."""
        neural, entry = self.agent_for_elo(target_elo, seed=seed)
        router = PhaseRouter(general=neural, endgame=self.endgame_agent(),
                             opening=self.opening_agent())
        return router, entry

    # --- request handlers --------------------------------------------------
    def move(self, fen: str, *, elo: float | None = None, seed: int | None = None) -> dict:
        # A fresh random seed per call so stochastic move selection actually
        # varies -- otherwise every game is a deterministic replay.
        if seed is None:
            seed = random.randrange(1, 2**31)
        board = chess.Board(fen)
        ceiling = self.ceiling()
        target = elo if elo is not None else ceiling
        router, entry = self.router_for_elo(target, seed=seed)
        mv = router.select_move(board)
        # Honesty rule: never echo an Elo above what the engine can actually
        # play. Above the ceiling it serves full strength and says so.
        served = min(float(target), ceiling)
        return {
            "move": mv.uci(),
            "gen": entry.gen if entry else -1,
            "elo": round(served, 0),
            "requested_elo": round(float(target), 0),
            "ceiling": round(ceiling, 0),
            "at_ceiling": float(target) >= ceiling,
            "calibrated_elo": entry.metrics.get("calibrated_elo") if entry else None,
            "route": router.last_route,
            "source": "model",
        }

    def watch_move(self, fen: str, white_elo: float, black_elo: float) -> dict:
        board = chess.Board(fen)
        target = white_elo if board.turn == chess.WHITE else black_elo
        return self.move(fen, elo=target)

    def predict(self, fen: str) -> dict:
        champ = self.ladder.champion() or self.ladder.latest()
        pred = self._predictor(champ.gen).predict(chess.Board(fen))
        return {**pred.to_dict(), "gen": champ.gen, "source": "model"}

    def ladder_info(self) -> dict:
        champ = self.ladder.champion()
        return {
            "generations": len(self.ladder.entries),
            # The served strength is the champion's, not the noisiest-highest gen.
            "best_elo": champ.elo if champ else None,
            "champion_gen": champ.gen if champ else None,
            "calibrated_elo": champ.metrics.get("calibrated_elo") if champ else None,
            # Honest playable range for UIs: sliders should cap here.
            "ceiling": round(self.ceiling(), 0),
            "levels": self.ladder.levels(),
            "elo_curve": self.ladder.elo_curve(),
        }

    def calibrate(self, *, stockfish_path: str | None = None,
                  anchors: tuple[int, ...] = (800, 1100, 1400, 1700),
                  elo_games: int = 6, n_positions: int = 60,
                  movetime_ms: int = 30, with_elo: bool = True,
                  with_phase_acpl: bool = True) -> dict:
        """Score the served agent against Stockfish and store a calibrated Elo.

        This is the only place Stockfish touches serving: it grades the agent (it
        never plays for it). Writes ``calibrated_elo`` / ``acpl`` / ``top1`` into
        the best entry's metrics so subsequent moves report the benchmarked Elo.
        Heavy on CPU (real games vs Stockfish); call it on demand, not per move.
        """
        from .agents.stockfish import find_stockfish
        from .eval.benchmark import (
            assess_elo,
            move_quality,
            move_quality_by_phase,
            random_positions,
        )

        if find_stockfish(stockfish_path) is None:
            raise RuntimeError("no Stockfish binary; set $STOCKFISH_PATH or install it")
        best = self.ladder.champion() or self.ladder.latest()
        if best is None:
            raise RuntimeError("no checkpoints to calibrate; call ensure_seeded()")

        router, _ = self.router_for_elo(best.elo)
        positions = random_positions(n_positions)
        quality = move_quality(router, positions, movetime_ms=movetime_ms, path=stockfish_path)
        best.metrics["acpl"] = quality["avg_centipawn_loss"]
        best.metrics["top1"] = quality["top1_agreement"]
        result = {"gen": best.gen, **quality}
        if with_phase_acpl:
            router_p, _ = self.router_for_elo(best.elo)
            by_phase = move_quality_by_phase(router_p, movetime_ms=movetime_ms,
                                             path=stockfish_path)
            acpl_by_phase = {p: q["avg_centipawn_loss"] for p, q in by_phase.items()}
            best.metrics["acpl_by_phase"] = acpl_by_phase
            result["acpl_by_phase"] = acpl_by_phase
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
