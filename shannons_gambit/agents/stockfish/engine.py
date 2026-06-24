"""Stockfish discovery + an Elo-throttled :class:`Agent`.

The Elo-to-options mapping is a pure function (:func:`elo_to_uci_options`) so it
can be unit-tested without a binary on the machine. The agent itself opens a UCI
engine lazily and clamps requested options to the engine's reported ranges.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import replace

import chess
import chess.engine

from ...config import StockfishConfig
from ..base import Agent

# Common install locations checked after $STOCKFISH_PATH and $PATH.
_CANDIDATE_PATHS = (
    "/opt/homebrew/bin/stockfish",  # Apple-silicon Homebrew
    "/usr/local/bin/stockfish",     # Intel Homebrew / manual
    "/usr/games/stockfish",         # Debian/Ubuntu apt
    "/usr/bin/stockfish",
)


class StockfishUnavailable(RuntimeError):
    """Raised when no Stockfish binary can be located."""


def find_stockfish(explicit: str | None = None) -> str | None:
    """Locate a Stockfish binary, or return ``None`` if none is found.

    Resolution order: explicit argument, ``$STOCKFISH_PATH``, ``$PATH``, then a
    few well-known install locations.
    """
    for cand in (explicit, os.environ.get("STOCKFISH_PATH")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    for cand in _CANDIDATE_PATHS:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def skill_for_elo(elo: int, *, floor: int = 1320, min_elo: int = 600) -> int:
    """Map a sub-floor Elo to a Stockfish ``Skill Level`` in ``[0, 19]``.

    ``UCI_Elo`` bottoms out around ``floor``; below that we lean on Skill Level
    (linear from level 0 at ``min_elo`` to level 19 just under ``floor``).
    """
    if elo >= floor:
        return 20
    span = max(1, floor - min_elo)
    frac = (elo - min_elo) / span
    return max(0, min(19, round(frac * 19)))


def elo_to_uci_options(
    elo: int, *, floor: int = 1320, ceiling: int = 3190
) -> dict:
    """Return UCI options that throttle Stockfish toward ``elo``.

    At or above ``floor`` we use the engine's native strength limiter
    (``UCI_LimitStrength`` + ``UCI_Elo``). Below it we disable that limiter and
    use ``Skill Level`` instead (the limiter clamps up to ``floor`` otherwise).
    """
    if elo >= floor:
        return {
            "UCI_LimitStrength": True,
            "UCI_Elo": int(max(floor, min(ceiling, elo))),
        }
    return {
        "UCI_LimitStrength": False,
        "Skill Level": skill_for_elo(elo, floor=floor),
    }


class StockfishAgent(Agent):
    """A Stockfish opponent calibrated to :class:`StockfishConfig.elo`.

    Opens the UCI engine lazily on first move and reuses it for the agent's
    lifetime. Use as a context manager (or call :meth:`close`) to release the
    subprocess; the destructor closes it as a backstop.
    """

    def __init__(self, cfg: StockfishConfig | None = None, *, path: str | None = None) -> None:
        self.cfg = cfg or StockfishConfig()
        self.path = find_stockfish(path)
        if self.path is None:
            raise StockfishUnavailable(
                "no Stockfish binary found; set $STOCKFISH_PATH or install stockfish"
            )
        self.name = f"stockfish-{self.cfg.elo}"
        self._engine: chess.engine.SimpleEngine | None = None

    # --- engine lifecycle --------------------------------------------------
    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            engine = chess.engine.SimpleEngine.popen_uci(self.path)
            # Clamp the Elo window to what *this* binary actually reports.
            floor, ceiling = self.cfg.uci_elo_floor, self.cfg.uci_elo_ceiling
            opt = engine.options.get("UCI_Elo")
            if opt is not None and opt.min is not None and opt.max is not None:
                floor, ceiling = int(opt.min), int(opt.max)
            options = {"Threads": self.cfg.threads, "Hash": self.cfg.hash_mb}
            options.update(elo_to_uci_options(self.cfg.elo, floor=floor, ceiling=ceiling))
            # Drop options this build does not expose (keeps older binaries happy).
            options = {k: v for k, v in options.items() if k in engine.options}
            engine.configure(options)
            self._engine = engine
        return self._engine

    def _limit(self) -> chess.engine.Limit:
        kw: dict = {}
        if self.cfg.movetime_ms > 0:
            kw["time"] = self.cfg.movetime_ms / 1000.0
        if self.cfg.depth > 0:
            kw["depth"] = self.cfg.depth
        if not kw:  # never let Stockfish think unboundedly
            kw["time"] = 0.05
        return chess.engine.Limit(**kw)

    # --- Agent API ---------------------------------------------------------
    def select_move(self, board: chess.Board) -> chess.Move:
        engine = self._ensure_engine()
        result = engine.play(board, self._limit())
        if result.move is None:  # pragma: no cover - only on terminal boards
            raise ValueError("Stockfish returned no move for a non-terminal board")
        return result.move

    def evaluate(self, board: chess.Board, *, depth: int | None = None) -> chess.engine.PovScore:
        """Return Stockfish's score for ``board`` from the side-to-move's view."""
        engine = self._ensure_engine()
        limit = chess.engine.Limit(depth=depth) if depth else self._limit()
        info = engine.analyse(board, limit)
        return info["score"]

    def best_move_and_value(
        self, board: chess.Board, *, cp_scale: float = 400.0
    ) -> tuple[chess.Move, float]:
        """Return (best move, value in [-1, 1]) for distillation targets.

        The value is ``tanh(centipawns / cp_scale)`` from the side-to-move's
        perspective; forced mates clamp to ``+/-1``.
        """
        import math

        engine = self._ensure_engine()
        info = engine.analyse(board, self._limit())
        pv = info.get("pv") or []
        move = pv[0] if pv else engine.play(board, self._limit()).move
        if move is None:  # pragma: no cover - terminal board
            raise ValueError("Stockfish returned no move for a non-terminal board")
        score = info["score"].pov(board.turn)
        mate = score.mate()
        if mate is not None:
            value = 1.0 if mate > 0 else -1.0
        else:
            cp = score.score() or 0
            value = math.tanh(cp / cp_scale)
        return move, value

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    def at_elo(self, elo: int) -> StockfishAgent:
        """A sibling agent at a different Elo (shares the same binary path)."""
        return StockfishAgent(replace(self.cfg, elo=elo), path=self.path)

    def __enter__(self) -> StockfishAgent:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
