"""A weighted opening book learned from real games (not hand-authored).

The book maps a position (python-chess Zobrist hash) to the moves strong players
chose there, weighted by how often. At serve time the first ``max_ply`` plies are
played from the book whenever an entry exists, giving the agent sound, varied,
*named* openings for almost no compute -- the cheapest real strength gain in the
opening phase, which the project previously left entirely to a shallow net.

Built by :func:`build_book` from any PGN stream (the same Lichess source the
network trains on) and consulted by :class:`OpeningBookAgent` via the
:class:`~shannons_gambit.agents.router.PhaseRouter`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import chess
import chess.polyglot
import numpy as np

from .base import Agent


def _key(board: chess.Board) -> str:
    """Stable, position-only key (side to move, castling, ep all folded in)."""
    return f"{chess.polyglot.zobrist_hash(board):016x}"


def _safe_int(value) -> int | None:
    try:
        return int(value) if value not in (None, "?") else None
    except (TypeError, ValueError):
        return None


class OpeningBook:
    def __init__(self, table: dict[str, list[list]], *, max_ply: int = 24) -> None:
        # table: zobrist-hex -> [[uci, weight], ...]
        self.table = table
        self.max_ply = max_ply

    def __len__(self) -> int:
        return len(self.table)

    def lookup(self, board: chess.Board) -> list[tuple[chess.Move, int]]:
        """Legal book moves for ``board`` with weights (empty if out of book)."""
        if board.ply() >= self.max_ply:
            return []
        entry = self.table.get(_key(board))
        if not entry:
            return []
        out: list[tuple[chess.Move, int]] = []
        for uci, weight in entry:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move in board.legal_moves:
                out.append((move, int(weight)))
        return out

    def sample(self, board: chess.Board,
               rng: np.random.Generator | None = None) -> chess.Move | None:
        moves = self.lookup(board)
        if not moves:
            return None
        rng = rng or np.random.default_rng()
        weights = np.array([w for _, w in moves], dtype=np.float64)
        probs = weights / weights.sum()
        return moves[int(rng.choice(len(moves), p=probs))][0]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"max_ply": self.max_ply, "positions": self.table}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str, *, max_ply: int = 24) -> OpeningBook:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "positions" in data:
            return cls(data["positions"], max_ply=int(data.get("max_ply", max_ply)))
        return cls(data, max_ply=max_ply)  # tolerate a bare mapping


def build_book(source: str, *, max_games: int = 20_000, min_elo: int = 2000,
               max_ply: int = 24, min_count: int = 5, top_k: int = 4) -> OpeningBook:
    """Build an :class:`OpeningBook` from a PGN stream (``.zst`` / remote / local).

    Counts the moves players rated ``>= min_elo`` made in the first ``max_ply``
    plies, then keeps, per position, the ``top_k`` moves seen at least
    ``min_count`` times (so noise and one-off blunders are filtered out).
    """
    from ..data.lichess import SAMPLE_PGN, iter_games

    src = (source if source and (source.startswith("http") or Path(source).exists())
           else str(SAMPLE_PGN))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for game in iter_games(src, max_games=max_games):
        white_elo = _safe_int(game.headers.get("WhiteElo"))
        black_elo = _safe_int(game.headers.get("BlackElo"))
        if min_elo and min(white_elo or 0, black_elo or 0) < min_elo:
            continue
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= max_ply:
                break
            counts[_key(board)][move.uci()] += 1
            board.push(move)

    table: dict[str, list[list]] = {}
    for key, moves in counts.items():
        ranked = sorted(
            ([uci, cnt] for uci, cnt in moves.items() if cnt >= min_count),
            key=lambda kv: kv[1], reverse=True,
        )[:top_k]
        if ranked:
            table[key] = ranked
    return OpeningBook(table, max_ply=max_ply)


class OpeningBookAgent(Agent):
    """Plays a weighted book move; the router checks :meth:`applies` first."""

    name = "opening"

    def __init__(self, book: OpeningBook, *, seed: int = 0) -> None:
        self.book = book
        self._rng = np.random.default_rng(seed)

    def applies(self, board: chess.Board) -> bool:
        return bool(self.book.lookup(board))

    def select_move(self, board: chess.Board) -> chess.Move:
        move = self.book.sample(board, self._rng)
        if move is None:
            raise ValueError("position is out of book; check applies() first")
        return move
