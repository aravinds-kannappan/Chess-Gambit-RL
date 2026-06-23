"""Self-play game generation producing (state, MCTS-policy, value) examples."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from ...data.encode import POLICY_SIZE, encode_board, move_to_index
from .mcts import MCTS


@dataclass
class Example:
    state: np.ndarray  # (18, 8, 8) uint8
    policy: np.ndarray  # (POLICY_SIZE,) float32 visit distribution
    value: float  # from the side-to-move perspective, filled in after the game


def _visit_policy(visits: dict[chess.Move, int]) -> np.ndarray:
    target = np.zeros(POLICY_SIZE, dtype=np.float32)
    total = sum(visits.values())
    if total == 0:
        return target
    for move, n in visits.items():
        target[move_to_index(move)] = n / total
    return target


def play_game(
    mcts: MCTS,
    *,
    simulations: int,
    temperature_moves: int = 15,
    max_moves: int = 80,
    rng: np.random.Generator | None = None,
) -> list[Example]:
    rng = rng or np.random.default_rng()
    board = chess.Board()
    examples: list[Example] = []
    movers: list[bool] = []  # True if White moved at that ply

    for ply in range(max_moves):
        if board.is_game_over():
            break
        visits = mcts.run(board, simulations=simulations, add_noise=True)
        policy = _visit_policy(visits)
        examples.append(Example(encode_board(board).astype(np.uint8), policy, 0.0))
        movers.append(board.turn == chess.WHITE)

        moves = list(visits)
        counts = np.array([visits[m] for m in moves], dtype=np.float64)
        if ply < temperature_moves and counts.sum() > 0:
            probs = counts / counts.sum()
            move = moves[int(rng.choice(len(moves), p=probs))]
        else:
            move = moves[int(counts.argmax())]
        board.push(move)

    result = _white_result(board)
    for ex, white_moved in zip(examples, movers, strict=False):
        ex.value = result if white_moved else -result
    return examples


def _white_result(board: chess.Board) -> float:
    if board.is_checkmate():
        # side to move is mated; the other side won.
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0  # draw / adjudicated at move cap
