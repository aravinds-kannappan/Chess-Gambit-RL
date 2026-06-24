"""PPO trains/plays on the solved endgame, and PhaseRouter dispatches by phase."""

from __future__ import annotations

import tempfile
import unittest

import chess
import numpy as np

from shannons_gambit.agents.ppo import PPOAgent
from shannons_gambit.agents.random_agent import RandomAgent
from shannons_gambit.agents.router import MDPAgent, PhaseRouter, piece_count
from shannons_gambit.config import PPOConfig
from shannons_gambit.mdp.chess_mdp import load_endgame


class TestPPO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mdp = load_endgame("KRvK")

    def test_train_and_play(self):
        cfg = PPOConfig(run_dir=tempfile.mkdtemp(), rollout_steps=64, minibatch_size=32,
                        update_epochs=1, total_updates=2, device="cpu")
        agent = PPOAgent(self.mdp, cfg)
        history = agent.train()
        self.assertTrue(history)
        board = self.mdp.board_from_state(self.mdp.sample_won_state(np.random.default_rng(0)))
        self.assertIn(agent.select_move(board), board.legal_moves)


class TestRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mdp = load_endgame("KRvK")

    def test_mdp_agent_scope(self):
        agent = MDPAgent(self.mdp)
        endgame = self.mdp.board_from_state(self.mdp.sample_won_state(np.random.default_rng(1)))
        self.assertTrue(agent.applies(endgame))
        self.assertEqual(piece_count(endgame), 3)
        self.assertIn(agent.select_move(endgame), endgame.legal_moves)
        self.assertFalse(agent.applies(chess.Board()))  # full board out of scope

    def test_router_dispatches_by_phase(self):
        router = PhaseRouter(general=RandomAgent(), endgame=MDPAgent(self.mdp))
        endgame = self.mdp.board_from_state(self.mdp.sample_won_state(np.random.default_rng(2)))
        router.select_move(endgame)
        self.assertEqual(router.last_route, "mdp")
        router.select_move(chess.Board())
        self.assertEqual(router.last_route, "random")
        self.assertEqual(router.routes["mdp"], 1)


if __name__ == "__main__":
    unittest.main()
