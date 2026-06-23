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
from .models.net import ChessNet, save_model
from .models.prediction import Predictor


class ModelServer:
    def __init__(self, models_dir: str = "runs/continual", *, device: str = "cpu",
                 base_sims: int = 48) -> None:
        self.models_dir = models_dir
        self.device = device
        self.base_sims = base_sims
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

    # --- request handlers --------------------------------------------------
    def move(self, fen: str, *, elo: float | None = None, seed: int | None = None) -> dict:
        # A fresh random seed per call so stochastic move selection actually
        # varies -- otherwise every game is a deterministic replay.
        if seed is None:
            seed = random.randrange(1, 2**31)
        board = chess.Board(fen)
        best = self.ladder.best()
        target = elo if elo is not None else (best.elo if best else 1000)
        agent, entry = self.agent_for_elo(target, seed=seed)
        mv = agent.select_move(board)
        return {"move": mv.uci(), "gen": entry.gen, "elo": entry.elo, "source": "model"}

    def watch_move(self, fen: str, white_elo: float, black_elo: float) -> dict:
        board = chess.Board(fen)
        target = white_elo if board.turn == chess.WHITE else black_elo
        return self.move(fen, elo=target)

    def predict(self, fen: str) -> dict:
        best = self.ladder.best() or self.ladder.latest()
        pred = self._predictor(best.gen).predict(chess.Board(fen))
        return {**pred.to_dict(), "gen": best.gen, "source": "model"}

    def ladder_info(self) -> dict:
        return {
            "generations": len(self.ladder.entries),
            "best_elo": self.ladder.best().elo if self.ladder.best() else None,
            "levels": self.ladder.levels(),
            "elo_curve": self.ladder.elo_curve(),
        }


@lru_cache(maxsize=1)
def get_server(models_dir: str = "runs/continual") -> ModelServer:
    server = ModelServer(models_dir)
    server.ensure_seeded()
    return server
