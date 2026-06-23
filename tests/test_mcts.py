"""MCTS search returns legal, sensible visit distributions."""

from __future__ import annotations

import unittest

import chess

from shannons_gambit.agents.alphazero.mcts import MCTS, AlphaZeroAgent
from shannons_gambit.models.net import ChessNet


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.model = ChessNet(channels=16, blocks=1)

    def test_visits_are_legal(self):
        board = chess.Board()
        mcts = MCTS(self.model, device="cpu")
        visits = mcts.run(board, simulations=16, add_noise=True)
        legal = set(board.legal_moves)
        self.assertTrue(set(visits).issubset(legal))
        self.assertEqual(sum(visits.values()), 16)

    def test_agent_returns_legal_move(self):
        agent = AlphaZeroAgent(self.model, device="cpu", simulations=8)
        board = chess.Board()
        move = agent.select_move(board)
        self.assertIn(move, board.legal_moves)


if __name__ == "__main__":
    unittest.main()
