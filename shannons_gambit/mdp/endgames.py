"""Specifications for the tractable endgames we solve exactly.

The strong side (White) owns one heavy piece; Black is a lone king. These
spaces are small enough to enumerate fully, so Bellman value iteration yields
the provably optimal policy (which must force mate from won positions).

Pawn endgames are intentionally excluded from the exact solver because a
promotion leaves the state space; KRvK and KQvK are closed under legal moves
apart from the strong piece being captured (which collapses to bare kings = a
draw, handled via a single absorbing state in ``chess_mdp``).
"""

from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class EndgameSpec:
    name: str
    strong_piece: int  # chess.ROOK or chess.QUEEN
    description: str


SPECS: dict[str, EndgameSpec] = {
    "KRvK": EndgameSpec("KRvK", chess.ROOK, "King + rook vs. king"),
    "KQvK": EndgameSpec("KQvK", chess.QUEEN, "King + queen vs. king"),
}


def get_spec(name: str) -> EndgameSpec:
    if name not in SPECS:
        raise KeyError(f"unsupported endgame {name!r}; choose from {sorted(SPECS)}")
    return SPECS[name]
