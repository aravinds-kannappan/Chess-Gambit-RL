"""Common agent interface so every policy is interchangeable in the arena."""

from __future__ import annotations

import abc

import chess


class Agent(abc.ABC):
    """A chess move-selection policy."""

    name: str = "agent"

    @abc.abstractmethod
    def select_move(self, board: chess.Board) -> chess.Move:
        """Return a legal move for ``board`` (which is not modified)."""

    def __str__(self) -> str:
        return self.name
