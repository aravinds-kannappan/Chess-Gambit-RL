"""Elo fitting and arena bookkeeping."""

from __future__ import annotations

import unittest

import numpy as np

from shannons_gambit.agents.random_agent import RandomAgent
from shannons_gambit.eval.arena import play_game, round_robin
from shannons_gambit.eval.elo import expected_score, fit_elo


class TestElo(unittest.TestCase):
    def test_stronger_player_gets_higher_rating(self):
        names = ["A", "B", "C"]
        # A beats B and C; B beats C. 10 games each pair.
        score = np.array([[0, 9, 8], [1, 0, 7], [2, 3, 0]], dtype=float)
        games = np.full((3, 3), 10.0)
        np.fill_diagonal(games, 0.0)
        elo = fit_elo(names, score, games)
        self.assertGreater(elo["A"], elo["B"])
        self.assertGreater(elo["B"], elo["C"])

    def test_expected_score_symmetry(self):
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5, places=9)
        self.assertGreater(expected_score(1600, 1400), 0.5)


class TestArena(unittest.TestCase):
    def test_play_game_returns_valid_result(self):
        rec = play_game(RandomAgent(1), RandomAgent(2), max_moves=40)
        self.assertIn(rec.result, {"1-0", "0-1", "1/2-1/2"})

    def test_round_robin_leaderboard(self):
        agents = [RandomAgent(1), RandomAgent(2)]
        out = round_robin(agents, games_per_pair=2, max_moves=30)
        self.assertEqual(len(out["leaderboard"]), 2)
        self.assertIn("elo", out)


if __name__ == "__main__":
    unittest.main()
