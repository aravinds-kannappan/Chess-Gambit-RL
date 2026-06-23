"""Supervised training really fits and the predictor round-trips."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import chess

from shannons_gambit.config import DataConfig, NetConfig, SupervisedConfig
from shannons_gambit.data.dataset import build_dataset
from shannons_gambit.data.lichess import SAMPLE_PGN
from shannons_gambit.models.prediction import Predictor
from shannons_gambit.models.supervised import train_supervised


class TestSupervised(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        build_dataset(replace(DataConfig(), url=str(SAMPLE_PGN), out_dir=self.tmp,
                              max_games=10, shard_size=100))

    def test_train_and_predict(self):
        cfg = replace(
            SupervisedConfig(), data_dir=self.tmp, run_dir=str(Path(self.tmp, "run")),
            epochs=4, batch_size=64, net=NetConfig(channels=16, blocks=1), device="cpu",
        )
        res = train_supervised(cfg)
        hist = res["history"]
        self.assertEqual(len(hist), 4)
        # the policy loss decreases over training (genuine fitting).
        self.assertLess(hist[-1]["loss_policy"], hist[0]["loss_policy"])

        predictor = Predictor.from_checkpoint(str(Path(self.tmp, "run", "model.pt")))
        pred = predictor.predict(chess.Board())
        self.assertIn(chess.Move.from_uci(pred.best_move), chess.Board().legal_moves)
        self.assertAlmostEqual(sum(pred.wdl.values()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
