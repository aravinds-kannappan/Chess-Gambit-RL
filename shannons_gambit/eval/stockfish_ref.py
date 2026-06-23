"""Stockfish-anchored Elo (feature-detected).

Without a calibrated reference, a "2000 Elo" claim is meaningless. When a
Stockfish binary is available, this places an agent on a real scale by playing
it against Stockfish at fixed skill levels (whose approximate Elos are known) and
solving for the agent's rating. No binary -> the functions raise a clear error so
callers can skip rather than fabricate a number.
"""

from __future__ import annotations

import shutil

import chess

from ..agents.base import Agent
from .elo import estimate_rating

# Approximate Lichess-scale Elo of Stockfish at each "Skill Level" (0-20),
# from community calibration; coarse but a real anchor.
SKILL_ELO = {0: 1100, 1: 1200, 3: 1400, 5: 1600, 8: 1900, 11: 2200, 14: 2500, 20: 3200}


def stockfish_path() -> str | None:
    return shutil.which("stockfish")


class StockfishAgent(Agent):
    """A fixed-skill Stockfish opponent used purely as an Elo anchor."""

    def __init__(self, skill: int = 5, *, movetime: float = 0.05, path: str | None = None) -> None:
        import chess.engine

        binary = path or stockfish_path()
        if binary is None:
            raise RuntimeError("no stockfish binary found (install Stockfish to anchor Elo)")
        self.name = f"stockfish-skill{skill}"
        self.skill = skill
        self.movetime = movetime
        self._engine = chess.engine.SimpleEngine.popen_uci(binary)
        self._engine.configure({"Skill Level": skill})

    def select_move(self, board: chess.Board) -> chess.Move:
        import chess.engine

        result = self._engine.play(board, chess.engine.Limit(time=self.movetime))
        assert result.move is not None
        return result.move

    def close(self) -> None:
        self._engine.quit()


def estimate_elo_vs_stockfish(agent: Agent, *, skills=(1, 3, 5, 8), games: int = 10,
                              max_moves: int = 120) -> float:
    """Place ``agent`` on the Stockfish-anchored Elo scale."""
    if stockfish_path() is None:
        raise RuntimeError("no stockfish binary; cannot anchor Elo")
    from .arena import play_game

    results = []
    for skill in skills:
        opp = StockfishAgent(skill)
        pts = 0.0
        try:
            for i in range(games):
                white, black = (agent, opp) if i % 2 == 0 else (opp, agent)
                rec = play_game(white, black, max_moves=max_moves)
                if rec.result == "1-0":
                    pts += 1.0 if white is agent else 0.0
                elif rec.result == "0-1":
                    pts += 1.0 if black is agent else 0.0
                else:
                    pts += 0.5
        finally:
            opp.close()
        results.append((float(SKILL_ELO[skill]), pts, games))
    return estimate_rating(results, init=1800.0)
