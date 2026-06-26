"""Game-phase classification shared by the router, data tagging, and evaluation.

A deterministic split by men-on-board and move number so the project can finally
treat the **opening** and **middlegame** as first-class phases (previously every
non-3-piece position fell to a single "general" net):

* ``endgame``    -- ``<= endgame_men`` pieces (exact MDP / careful net),
* ``opening``    -- first ``opening_plies`` plies (opening book / net),
* ``middlegame`` -- everything else (the net, with deeper search).
"""

from __future__ import annotations

import chess

OPENING_PLIES = 24   # ~12 full moves
ENDGAME_MEN = 10
PHASES = ("opening", "middlegame", "endgame")


def piece_count(board: chess.Board) -> int:
    return chess.popcount(board.occupied)


def game_phase(board: chess.Board, *, ply: int | None = None,
               opening_plies: int = OPENING_PLIES, endgame_men: int = ENDGAME_MEN) -> str:
    """Classify ``board`` into one of :data:`PHASES`.

    ``ply`` defaults to ``board.ply()`` (derivable from a FEN), so this works on
    standalone positions as well as live games.
    """
    if piece_count(board) <= endgame_men:
        return "endgame"
    p = board.ply() if ply is None else ply
    if p < opening_plies:
        return "opening"
    return "middlegame"
