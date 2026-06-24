"""Heavy training job for Hugging Face Jobs (GPU) - the path toward 2000 Elo.

Run on a GPU (pay-per-run, funded by the HF credit) to do what a laptop CPU
cannot: train on a large slice of real strong-player Lichess games, then refine
by self-play, calibrate Elo, and push the ladder to the Hub.

    hf jobs run --flavor a10g-small \
        --secrets HF_TOKEN \
        python deploy/hf_job/run.py --min-elo 2000 --games 400000 --epochs 12

On an A10G this trains on millions of real positions in well under an hour; the
behavioural-cloning policy imitates 2000+ rated play, and MCTS adds search on top.
Strength is reported on whatever anchor is configured (see eval/stockfish_ref.py).
"""

from __future__ import annotations

import argparse
from dataclasses import replace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="https://database.lichess.org/standard/"
                   "lichess_db_standard_rated_2014-07.pgn.zst")
    p.add_argument("--min-elo", type=int, default=2000, dest="min_elo")
    p.add_argument("--games", type=int, default=400_000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--selfplay-gens", type=int, default=20, dest="selfplay_gens")
    p.add_argument("--hf-repo", default="legacyaravind/shannons-gambit", dest="hf_repo")
    args = p.parse_args()

    from shannons_gambit.agents.alphazero.continual import run_generations
    from shannons_gambit.config import (
        ContinualConfig,
        DataConfig,
        NetConfig,
        SupervisedConfig,
    )
    from shannons_gambit.data.dataset import build_dataset
    from shannons_gambit.export import push_ladder_to_hub, push_model_to_hf
    from shannons_gambit.models.supervised import train_supervised

    # 1) real strong-player data
    print("== building dataset ==", flush=True)
    summary = build_dataset(replace(DataConfig(), url=args.url, out_dir="data",
                                    min_elo=args.min_elo, max_games=args.games))
    print(summary, flush=True)

    # 2) behavioural-cloning pre-training (the bulk of real strength)
    print("== supervised pretraining ==", flush=True)
    net = NetConfig(channels=args.channels, blocks=args.blocks)
    train_supervised(replace(SupervisedConfig(), data_dir="data",
                             run_dir="runs/supervised", net=net, epochs=args.epochs,
                             max_positions=10_000_000, device="auto"))

    # 3) self-play refinement on top, versioned to the ladder
    print("== self-play refinement ==", flush=True)
    res = run_generations(replace(ContinualConfig(), run_dir="runs/continual",
                                  init_from="runs/supervised/model.pt", net=net,
                                  games_per_gen=64, simulations=200, eval_games=40,
                                  device="auto"), args.selfplay_gens)
    print("elo curve:", res["elo_curve"], flush=True)

    # 4) publish: the improved pre-trained net is the served base (model.pt),
    #    plus the self-play ladder + checkpoints (under checkpoints/).
    print("== pushing base model + ladder to the Hub ==", flush=True)
    print(push_model_to_hf(args.hf_repo, "runs/supervised/model.pt", with_handler=False),
          flush=True)
    print(push_ladder_to_hub(args.hf_repo, "runs/continual"), flush=True)


if __name__ == "__main__":
    main()
