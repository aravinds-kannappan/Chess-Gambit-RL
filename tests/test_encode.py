"""Board/move encoding round-trips and shapes."""

from __future__ import annotations

import random
import unittest

import chess

from shannons_gambit.data.encode import (
    BOARD_SHAPE,
    POLICY_SIZE,
    encode_board,
    index_to_move,
    legal_policy_mask,
    move_to_index,
)


class TestEncode(unittest.TestCase):
    def test_board_shape(self):
        self.assertEqual(encode_board(chess.Board()).shape, BOARD_SHAPE)

    def test_legal_mask_matches_movecount(self):
        board = chess.Board()
        self.assertEqual(int(legal_policy_mask(board).sum()), board.legal_moves.count())

    def test_move_index_roundtrip_random_games(self):
        rng = random.Random(0)
        checked = 0
        for _ in range(120):
            board = chess.Board()
            for _ in range(rng.randint(0, 40)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            for move in board.legal_moves:
                idx = move_to_index(move)
                self.assertTrue(0 <= idx < POLICY_SIZE)
                self.assertEqual(index_to_move(idx, board), move)
                checked += 1
        self.assertGreater(checked, 1000)

    def test_underpromotion_roundtrip(self):
        board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        for promo in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            move = chess.Move(chess.A7, chess.A8, promotion=promo)
            self.assertIn(move, board.legal_moves)
            self.assertEqual(index_to_move(move_to_index(move), board), move)


if __name__ == "__main__":
    unittest.main()
