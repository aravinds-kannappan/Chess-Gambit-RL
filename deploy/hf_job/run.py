"""Heavy training job for Hugging Face Jobs (GPU) - the path toward 2000 Elo.

Thin wrapper around :mod:`shannons_gambit.train_job` (the importable entry point
so HF Jobs can run it after a plain ``pip install`` with no repo checkout):

    hf jobs run --flavor a10g-small --secrets HF_TOKEN \
        python -m shannons_gambit.train_job --min-elo 2000 --games 120000 --epochs 8

It builds a dataset of real 2000+ rated Lichess games, behavioural-clone
pre-trains, refines by self-play, and publishes ``pretrain/model.pt`` (served
base) + ``posttrain/`` (RL generations) to the Hub.
"""

from __future__ import annotations

from shannons_gambit.train_job import main

if __name__ == "__main__":
    main()
