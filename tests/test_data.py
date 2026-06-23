"""Data pipeline on the bundled real games (offline)."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from shannons_gambit.config import DataConfig
from shannons_gambit.data.dataset import PositionDataset, build_dataset, load_records
from shannons_gambit.data.lichess import SAMPLE_PGN


class TestData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_build_and_load(self):
        cfg = replace(DataConfig(), url=str(SAMPLE_PGN), out_dir=self.tmp,
                      max_games=10, shard_size=100)
        summary = build_dataset(cfg)
        self.assertGreater(summary["positions"], 100)
        self.assertTrue(Path(self.tmp, "positions").exists())

        records = load_records(self.tmp)
        self.assertEqual(len(records["fen"]), summary["positions"])
        # move indices are valid and FENs parse.
        self.assertTrue((records["move_index"] >= 0).all())

        ds = PositionDataset(records)
        self.assertEqual(ds.x.shape[1:], (18, 8, 8))
        self.assertEqual(len(ds), summary["positions"])
        self.assertGreater(int(ds.rating_mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
