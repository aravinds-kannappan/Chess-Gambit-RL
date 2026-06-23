#!/usr/bin/env python3
"""Download a real Lichess open-database dump and build position shards.

Usage:
    python scripts/download_data.py --games 20000
    python scripts/download_data.py --source https://database.lichess.org/standard/<file>.pgn.zst

The default source is a monthly standard-rated dump; see
https://database.lichess.org for the full list (many include Stockfish evals).
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from shannons_gambit.config import DataConfig
from shannons_gambit.data.dataset import build_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=None, help="PGN(.zst) url or local path")
    p.add_argument("--games", type=int, default=20_000)
    p.add_argument("--out", default="data")
    p.add_argument("--min-elo", type=int, default=0, dest="min_elo")
    p.add_argument("--require-eval", action="store_true", dest="require_eval")
    args = p.parse_args()

    cfg = DataConfig()
    if args.source:
        cfg = replace(cfg, url=args.source)
    cfg = replace(cfg, max_games=args.games, out_dir=args.out,
                  min_elo=args.min_elo, require_eval=args.require_eval)
    summary = build_dataset(cfg)
    print(f"built {summary['positions']} positions in {summary['shards']} shard(s) "
          f"-> {summary['out_dir']}")


if __name__ == "__main__":
    main()
