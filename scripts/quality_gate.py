#!/usr/bin/env python3
"""Local quality gate: lint, unit tests, and a tiny end-to-end smoke train.

Mirrors the CI checks so they can be run with one command:
    python scripts/quality_gate.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def smoke_train() -> None:
    """Build a tiny dataset from the bundled games and train one epoch."""
    from shannons_gambit.config import DataConfig, NetConfig, SupervisedConfig
    from shannons_gambit.data.dataset import build_dataset
    from shannons_gambit.data.lichess import SAMPLE_PGN
    from shannons_gambit.models.supervised import train_supervised

    with tempfile.TemporaryDirectory() as tmp:
        build_dataset(replace(DataConfig(), url=str(SAMPLE_PGN), out_dir=tmp,
                              max_games=10, shard_size=500))
        cfg = replace(SupervisedConfig(), data_dir=tmp, run_dir=str(Path(tmp, "run")),
                      epochs=1, batch_size=64, net=NetConfig(channels=16, blocks=1),
                      device="cpu")
        res = train_supervised(cfg)
        assert res["history"], "smoke train produced no metrics"
        print("smoke train OK:", res["history"][-1])


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    run([sys.executable, "-m", "ruff", "check", str(repo / "shannons_gambit")])
    run([sys.executable, "-m", "unittest", "discover", "-s", str(repo / "tests")])
    smoke_train()
    print("\nquality gate passed ✅")


if __name__ == "__main__":
    main()
