"""Board and move encoding shared by every model in the project.

* ``encode_board`` -> ``(NUM_PLANES, 8, 8)`` float32 planes.
* ``move_to_index`` / ``index_to_move`` implement the AlphaZero 4672-move
  scheme (56 queen moves + 8 knight moves + 9 underpromotions per square),
  which is a reversible, position-independent action map. ``index_to_move``
  needs the board only to resolve queen-promotions and side-to-move.

The TypeScript ``web/app/lib/encode.ts`` mirrors this exactly so the deployed
site produces identical model inputs.
"""

from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 18
BOARD_SHAPE = (NUM_PLANES, 8, 8)
POLICY_SIZE = 64 * 73  # 4672

# 8 compass directions as (file_delta_sign, rank_delta_sign): N, NE, E, SE, S, SW, W, NW.
_QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_KNIGHT_OFFSETS = [
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING,
]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position as ``(18, 8, 8)`` float32 planes (white's orientation)."""
    planes = np.zeros(BOARD_SHAPE, dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        offset = 0 if piece.color == chess.WHITE else 6
        plane = offset + _PIECE_ORDER.index(piece.piece_type)
        planes[plane, rank, file] = 1.0
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
    planes[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        planes[17, ep_rank, ep_file] = 1.0
    return planes


def move_to_index(move: chess.Move) -> int:
    """Map a chess move to its policy index in ``[0, POLICY_SIZE)``."""
    from_sq = move.from_square
    to_sq = move.to_square
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        promo_index = _UNDERPROMO_PIECES.index(move.promotion)
        dir_index = df + 1  # -1, 0, 1 -> 0, 1, 2
        plane = 64 + promo_index * 3 + dir_index
        return from_sq * 73 + plane

    # Queen-like (also queen promotions, king steps, pawn pushes) or knight moves.
    if (abs(df), abs(dr)) in {(1, 2), (2, 1)}:
        plane = 56 + _KNIGHT_OFFSETS.index((df, dr))
        return from_sq * 73 + plane

    direction = (_sign(df), _sign(dr))
    distance = max(abs(df), abs(dr))
    plane = _QUEEN_DIRS.index(direction) * 7 + (distance - 1)
    return from_sq * 73 + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """Inverse of :func:`move_to_index`; returns ``None`` if off-board."""
    from_sq = index // 73
    plane = index % 73
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    promotion: int | None = None

    if plane < 56:
        dir_index, dist = divmod(plane, 7)
        dist += 1
        sdf, sdr = _QUEEN_DIRS[dir_index]
        to_file = from_file + sdf * dist
        to_rank = from_rank + sdr * dist
    elif plane < 64:
        df, dr = _KNIGHT_OFFSETS[plane - 56]
        to_file = from_file + df
        to_rank = from_rank + dr
    else:
        u = plane - 64
        promo_index, dir_index = divmod(u, 3)
        promotion = _UNDERPROMO_PIECES[promo_index]
        to_file = from_file + (dir_index - 1)
        to_rank = from_rank + (1 if board.turn == chess.WHITE else -1)

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        return None
    to_sq = chess.square(to_file, to_rank)

    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and to_rank in (0, 7):
            promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_policy_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape ``(POLICY_SIZE,)`` marking legal-move indices."""
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask
