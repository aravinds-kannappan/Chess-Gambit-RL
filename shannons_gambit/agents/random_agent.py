"""Uniform-random baseline agent (the arena's zero point)."""

from __future__ import annotations

import random

import chess

from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))
