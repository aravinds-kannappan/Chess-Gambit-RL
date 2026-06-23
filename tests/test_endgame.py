"""Exact endgame solve: value iteration converges and forces mate.

This builds the full KRvK state space (~0.4M states); it is the integration
test for the MDP/Bellman pillar and takes ~20s on first run (cached after).
"""

from __future__ import annotations

import tempfile
import unittest

import chess
import numpy as np

from shannons_gambit.mdp.chess_mdp import EndgameMDP
from shannons_gambit.mdp.endgames import get_spec


class TestEndgame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.mdp = EndgameMDP(get_spec("KRvK")).build(cache_dir=cls.tmp)
        cls.V, cls.history = cls.mdp.solve(gamma=0.99, theta=1e-9)

    def test_value_iteration_converges(self):
        self.assertLess(self.history[-1], 1e-8)
        self.assertGreater(len(self.mdp.states), 100_000)

    def test_optimal_policy_forces_mate(self):
        rng = np.random.default_rng(0)
        mated = 0
        trials = 60
        for _ in range(trials):
            board = self.mdp.board_from_state(self.mdp.sample_won_state(rng))
            plies = 0
            while not board.is_game_over() and plies < 80:
                board.push(self.mdp.optimal_move(board))
                plies += 1
            mated += int(board.is_checkmate())
        self.assertEqual(mated, trials)

    def test_known_position_is_won(self):
        board = chess.Board("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1")
        self.assertGreater(self.mdp.value_of(board), 0.0)
        self.assertIsNotNone(self.mdp.mate_distance(board))


if __name__ == "__main__":
    unittest.main()
