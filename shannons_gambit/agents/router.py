"""Phase-aware multi-agent play: the right method for the position.

The project owns several move-selection policies, each strongest in a different
regime:

* :class:`MDPAgent` -- exact Bellman dynamic programming, optimal wherever the
  position falls inside a fully-solved endgame (e.g. KRvK / KQvK).
* a **reward** agent (:class:`~shannons_gambit.agents.dqn.DQNAgent`) and a
  **PPO** agent (:class:`~shannons_gambit.agents.ppo.PPOAgent`) -- learned RL
  policies for the low-material regime.
* a general full-board agent (the AlphaZero-lite net or supervised policy) for
  the opening / middlegame.

:class:`PhaseRouter` dispatches each move to the appropriate sub-agent based on
the position, and records which one acted (so the Stockfish backend can score
each agent's contribution separately). None of these agents consults Stockfish;
Stockfish is only ever the external benchmark.
"""

from __future__ import annotations

from collections import Counter

import chess

from ..mdp.chess_mdp import EndgameMDP, _state_key
from .base import Agent


def piece_count(board: chess.Board) -> int:
    """Total men on the board (the simplest phase signal)."""
    return chess.popcount(board.occupied)


class MDPAgent(Agent):
    """Plays the exact optimal move from one or more solved endgame MDPs."""

    name = "mdp"

    def __init__(self, mdps: EndgameMDP | list[EndgameMDP]) -> None:
        self.mdps = [mdps] if isinstance(mdps, EndgameMDP) else list(mdps)
        if not self.mdps:
            raise ValueError("MDPAgent needs at least one solved EndgameMDP")

    def _mdp_for(self, board: chess.Board) -> EndgameMDP | None:
        """The solved MDP whose material exactly matches ``board`` (or None)."""
        for mdp in self.mdps:
            spec = mdp.spec
            strong = board.pieces(spec.strong_piece, chess.WHITE)
            if len(strong) != 1:
                continue
            # Exactly: white king, one white strong piece, black king, nothing else.
            if piece_count(board) != 3:
                continue
            if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                continue
            key = _state_key(board, spec.strong_piece)
            if key is not None and key in mdp.index:
                return mdp
        return None

    def applies(self, board: chess.Board) -> bool:
        return self._mdp_for(board) is not None

    def select_move(self, board: chess.Board) -> chess.Move:
        mdp = self._mdp_for(board)
        if mdp is None:
            raise ValueError("MDPAgent does not cover this position; check applies() first")
        move = mdp.optimal_move(board)
        if move is None:  # pragma: no cover - terminal board
            raise ValueError("no legal move from a terminal position")
        return move


class PhaseRouter(Agent):
    """Route each move to the sub-agent that owns the current position.

    Priority: ``opening`` (a learned opening book) while it has a move for the
    position, then ``endgame`` (MDP / PPO / reward) where ``endgame_applies`` is
    true, otherwise ``general``. The most recent and cumulative routing decisions
    are recorded for the benchmark to attribute move quality to the agent that
    actually played.
    """

    name = "router"

    def __init__(self, general: Agent, *, endgame: Agent | None = None,
                 opening: Agent | None = None, endgame_max_men: int = 7) -> None:
        self.general = general
        self.endgame = endgame
        self.opening = opening
        self.endgame_max_men = endgame_max_men
        self.last_route: str | None = None
        self.routes: Counter[str] = Counter()

    def opening_applies(self, board: chess.Board) -> bool:
        if self.opening is None:
            return False
        applies = getattr(self.opening, "applies", None)
        return bool(applies(board)) if callable(applies) else False

    def endgame_applies(self, board: chess.Board) -> bool:
        if self.endgame is None:
            return False
        applies = getattr(self.endgame, "applies", None)
        if callable(applies):  # exact-domain agents (e.g. MDPAgent) know their scope
            return bool(applies(board))
        return piece_count(board) <= self.endgame_max_men

    def _route(self, board: chess.Board) -> tuple[Agent, str]:
        if self.opening is not None and self.opening_applies(board):
            return self.opening, getattr(self.opening, "name", "opening")
        if self.endgame is not None and self.endgame_applies(board):
            return self.endgame, getattr(self.endgame, "name", "endgame")
        return self.general, getattr(self.general, "name", "general")

    def select_move(self, board: chess.Board) -> chess.Move:
        agent, label = self._route(board)
        self.last_route = label
        self.routes[label] += 1
        return agent.select_move(board)
