"""Stockfish layer: pure Elo mapping always runs; engine tests skip if no binary."""

from __future__ import annotations

import unittest

import chess

from shannons_gambit.agents.random_agent import RandomAgent
from shannons_gambit.agents.stockfish import (
    StockfishAgent,
    elo_to_uci_options,
    find_stockfish,
    skill_for_elo,
)
from shannons_gambit.config import StockfishConfig

_HAS_SF = find_stockfish() is not None
_SKIP = "no Stockfish binary on this machine"


class TestEloMapping(unittest.TestCase):
    """Pure functions -- no binary needed."""

    def test_strong_uses_uci_elo(self):
        opt = elo_to_uci_options(2000)
        self.assertTrue(opt["UCI_LimitStrength"])
        self.assertEqual(opt["UCI_Elo"], 2000)

    def test_weak_uses_skill_level(self):
        opt = elo_to_uci_options(900)
        self.assertFalse(opt["UCI_LimitStrength"])
        self.assertIn("Skill Level", opt)

    def test_uci_elo_clamped_to_window(self):
        self.assertEqual(elo_to_uci_options(5000, ceiling=3190)["UCI_Elo"], 3190)

    def test_skill_monotonic_and_bounded(self):
        self.assertLessEqual(skill_for_elo(600), skill_for_elo(1200))
        self.assertEqual(skill_for_elo(2000), 20)
        self.assertTrue(0 <= skill_for_elo(800) <= 19)


@unittest.skipUnless(_HAS_SF, _SKIP)
class TestStockfishEngine(unittest.TestCase):
    def test_throttled_agent_plays_legal_move(self):
        with StockfishAgent(StockfishConfig(elo=1500, movetime_ms=20)) as sf:
            board = chess.Board()
            self.assertIn(sf.select_move(board), board.legal_moves)

    def test_best_move_and_value_range(self):
        with StockfishAgent(StockfishConfig(elo=2500, movetime_ms=20)) as sf:
            move, value = sf.best_move_and_value(chess.Board())
            self.assertIn(move, chess.Board().legal_moves)
            self.assertTrue(-1.0 <= value <= 1.0)

    def test_benchmark_rates_random_below_stockfish(self):
        from shannons_gambit.eval.benchmark import benchmark_suite

        rep = benchmark_suite({"random": RandomAgent()}, n_positions=6,
                              elo_games=2, movetime_ms=15)
        self.assertIn("elo", rep["random"])
        self.assertGreaterEqual(rep["random"]["avg_centipawn_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
