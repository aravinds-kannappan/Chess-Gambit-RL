"""Agents driven by a trained network (policy prior or value lookahead)."""

from __future__ import annotations

import chess

from ..models.prediction import Predictor
from .base import Agent


class NeuralAgent(Agent):
    """Greedy over the network's legal-masked policy head."""

    name = "supervised"

    def __init__(self, predictor: Predictor, *, name: str | None = None) -> None:
        self.predictor = predictor
        if name:
            self.name = name

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu", **kw) -> NeuralAgent:
        return cls(Predictor.from_checkpoint(path, device=device), **kw)

    def select_move(self, board: chess.Board) -> chess.Move:
        dist = self.predictor.policy_distribution(board)
        return max(dist.items(), key=lambda kv: kv[1])[0]


class ValueAgent(Agent):
    """One-ply lookahead maximising the network's value head (negamax depth 1)."""

    name = "value"

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def select_move(self, board: chess.Board) -> chess.Move:
        best_move = None
        best_score = -2.0
        for move in board.legal_moves:
            board.push(move)
            # value is from the side-to-move's view; after our move it's the
            # opponent's, so negate to score from our perspective.
            score = -self.predictor.predict(board, top_k=1).value
            board.pop()
            if score > best_score:
                best_score, best_move = score, move
        assert best_move is not None
        return best_move
