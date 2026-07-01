"""Nightly continual-training job, sized for a free CI runner (GitHub Actions).

This is what makes "the agents are actually training" true without paid GPU
infrastructure: every night a 2-core runner

1. pulls the current champion, ladder, and opening book from the Hub,
2. builds (or restores from cache) a slice of real 2000+ Lichess games,
3. runs a few **gated** self-play generations with human-data (SFT) replay,
   so the net refines toward strong play instead of drifting,
4. re-grades the champion against Stockfish (calibrated Elo),
5. pushes the updated ladder + checkpoints back to the Hub and pokes the
   serving Space to reload.

A generation is only promoted if it beats the current champion head to head,
so a bad night can never make the served engine weaker. Run manually with
``python -m shannons_gambit.nightly_job --gens 1 --games-per-gen 4`` for a smoke.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

RUN_DIR = "runs/continual"


def _init_checkpoint(run_dir: str) -> str:
    """The checkpoint to resume from: the gated champion, else the base net."""
    from shannons_gambit.agents.ladder import Ladder

    ladder = Ladder.load(run_dir)
    champ = ladder.champion()
    if champ is not None:
        for cand in (champ.path, str(Path(run_dir) / f"{champ.name}.pt")):
            if Path(cand).exists():
                return cand
    base = Path(run_dir) / "model.pt"
    return str(base) if base.exists() else ""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gens", type=int, default=6)
    p.add_argument("--games-per-gen", type=int, default=16, dest="games_per_gen")
    p.add_argument("--simulations", type=int, default=48)
    p.add_argument("--eval-games", type=int, default=8, dest="eval_games")
    p.add_argument("--sft-ratio", type=float, default=0.5, dest="sft_ratio")
    p.add_argument("--min-elo", type=int, default=2000, dest="min_elo")
    p.add_argument("--games", type=int, default=30_000, help="games to scan for the dataset")
    p.add_argument("--url", default="https://database.lichess.org/standard/"
                   "lichess_db_standard_rated_2014-07.pgn.zst")
    p.add_argument("--hf-repo", default="legacyaravind/shannons-gambit", dest="hf_repo")
    p.add_argument("--space-id", default="legacyaravind/shannons-gambit", dest="space_id",
                   help="Space to restart after pushing ('' skips)")
    p.add_argument("--skip-calibrate", action="store_true", dest="skip_calibrate")
    args = p.parse_args(argv)

    from shannons_gambit.agents.alphazero.continual import ContinualTrainer
    from shannons_gambit.config import ContinualConfig, DataConfig
    from shannons_gambit.data.dataset import build_dataset
    from shannons_gambit.export import (
        pull_base_model,
        pull_ladder_from_hub,
        pull_opening_book,
        push_ladder_to_hub,
    )

    # 1) current state from the Hub
    print("== pulling ladder + base + book from the Hub ==", flush=True)
    pull_ladder_from_hub(args.hf_repo, RUN_DIR)
    pull_base_model(args.hf_repo, RUN_DIR)
    pull_opening_book(args.hf_repo, RUN_DIR)

    # 2) real strong-player data (restored from CI cache when available)
    shards = list(Path("data/positions").glob("shard_*.parquet"))
    if shards:
        print(f"== dataset cache hit ({len(shards)} shards) ==", flush=True)
    else:
        print("== building dataset ==", flush=True)
        summary = build_dataset(replace(DataConfig(), url=args.url, out_dir="data",
                                        min_elo=args.min_elo, max_games=args.games))
        print(summary, flush=True)

    # 3) gated self-play with SFT replay, resuming the champion
    init_from = _init_checkpoint(RUN_DIR)
    print(f"== training from {init_from or 'scratch'} ==", flush=True)
    cfg = replace(ContinualConfig(), run_dir=RUN_DIR, init_from=init_from,
                  games_per_gen=args.games_per_gen, simulations=args.simulations,
                  eval_games=args.eval_games, sft_data_dir="data",
                  sft_ratio=args.sft_ratio, device="cpu")
    trainer = ContinualTrainer(cfg)
    entries = trainer.run(args.gens)
    for entry in entries:
        m = entry.metrics
        print(f"gen {entry.gen}: elo={entry.elo} promoted={m.get('promoted')} "
              f"score_vs_champ={m.get('score_vs_champion')} sft={m.get('sft_batches')}",
              flush=True)
    promoted = any(e.metrics.get("promoted") for e in entries)

    # 4) honest rating. Re-grade ONLY when the champion actually changed (or has
    #    no rating yet). Re-grading the same net every night just churns the
    #    number with sampling noise; a stable champion keeps its stable rating.
    from shannons_gambit.agents.ladder import Ladder as _L

    champ = _L.load(RUN_DIR).champion()
    needs_rating = champ is not None and champ.metrics.get("calibrated_elo") is None
    if args.skip_calibrate:
        print("== calibration skipped by flag ==", flush=True)
    elif not (promoted or needs_rating):
        print("== champion unchanged; keeping its calibrated rating ==", flush=True)
    else:
        from shannons_gambit.agents.stockfish import find_stockfish
        from shannons_gambit.serve import ModelServer

        if find_stockfish(None) is None:
            print("== no stockfish binary; skipping calibration ==", flush=True)
        else:
            print("== calibrating new champion vs Stockfish ==", flush=True)
            server = ModelServer(RUN_DIR)
            try:
                print(server.calibrate(movetime_ms=30, with_phase_acpl=False), flush=True)
            except Exception as exc:  # noqa: BLE001 - rating is best-effort
                print("calibration failed:", exc, flush=True)

    # 5) publish + poke the serving Space
    print("== pushing ladder to the Hub ==", flush=True)
    print(push_ladder_to_hub(args.hf_repo, RUN_DIR), flush=True)
    if args.space_id and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi

            HfApi().restart_space(args.space_id)
            print(f"== restarted {args.space_id} to serve the new champion ==", flush=True)
        except Exception as exc:  # noqa: BLE001
            print("space restart skipped:", exc, flush=True)
    print("== done ==", flush=True)


if __name__ == "__main__":
    main()
