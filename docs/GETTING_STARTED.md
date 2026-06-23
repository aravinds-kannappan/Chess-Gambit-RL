# Getting started

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev]"
```

Python 3.10+; PyTorch uses MPS on Apple Silicon, CUDA on NVIDIA, else CPU.

## The five-minute tour (offline)

Everything below runs on the bundled real games with no network:

```bash
# Solve KRvK exactly with Bellman value iteration and verify forced mate
sgambit mdp --endgame KRvK

# Build a tiny dataset from the bundled games and train the model
python -c "from shannons_gambit.config import *; from shannons_gambit.data.dataset import build_dataset; from shannons_gambit.data.lichess import SAMPLE_PGN; from dataclasses import replace; build_dataset(replace(DataConfig(), url=str(SAMPLE_PGN), out_dir='data', max_games=10))"
sgambit train supervised --preset local_smoke

# Information-theory report and a live prediction
sgambit analyze
sgambit predict --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Arena leaderboard
sgambit arena
```

## Real data

```bash
python scripts/download_data.py --games 50000      # a real Lichess monthly dump
sgambit train supervised --preset local_full
```

See https://database.lichess.org for available dumps (many carry Stockfish evals;
add `--require-eval`).

## Presets

`local_smoke` (fast smoke), `local_full` (a real laptop run), `cloud` (bigger
nets, more self-play). Pass with `--preset` on `sgambit train`.

## Reproduce everything

```bash
python scripts/quality_gate.py        # lint + tests + smoke train
jupyter notebook notebooks/shannons_gambit_reproduce.ipynb
```

## Web app

```bash
cd web && npm install && npm run dev   # http://localhost:3000
```

Copy `web/.env.example` to `web/.env.local` and set the `HF_*` variables to
connect the trained model on Hugging Face; otherwise the built-in agent is used.
