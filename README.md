<div align="center">

# ♟ Shannon's Gambit

### A self-improving chess intelligence

**Continuous self-play reinforcement learning · trained + served on Hugging Face · real Lichess data · adaptive play · a deployable web app**

[![CI](https://github.com/aravinds-kannappan/Chess-Gambit-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/aravinds-kannappan/Chess-Gambit-RL/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

</div>

---

## What it is

An end-to-end chess RL system that **owns the whole pipeline**. It is a
**multi-agent** engine: each position is routed to the method that owns it - exact
**MDP/Bellman** dynamic programming in solved endgames, **PPO** and **reward (DQN)**
RL in the low-material regime, and an AlphaZero-lite **neural net** for the opening
and middlegame (`agents/router.py`). It pre-trains on real Lichess games, improves
by **continuous self-play**, **adapts to how you play**, and serves it all from a
**Hugging Face Space that both trains and serves** - with **no heuristic fallback**.

**Stockfish is the referee, never a player.** The agents never call Stockfish to
choose a move. A separate backend evaluator (`eval/benchmark.py`) uses Stockfish
purely as a calibrated yardstick: it scores each agent's **centipawn loss** and
top-1 agreement, and assigns a **calibrated Elo** via gauntlets against Elo-throttled
Stockfish. That rating is the level each agent plays at - and climbs as it learns.

| Surface | What it does |
| --- | --- |
| **Play** | Play the routed agent; it **adapts to your level**, your games fine-tune a *personal* checkpoint, and **Competitive Mode** cranks it to ~2300 tournament strength. |
| **Tiers** | Many agent-vs-agent games at once across **Elo tiers**; each finished game is fresh training data for the ladder. |
| **Watch** | Pair two agents at chosen **Elo** levels and watch a single game. |
| **Dashboard** | Live training graphs: Elo per generation, Stockfish-assessed ratings, falling loss curves, and the information-theoretic analysis of real games. |
| **Predict** | FEN/PGN → win/draw/loss, best move, value, from the current best network. |
| **Ladder** | Every self-play generation as a rated checkpoint. |

## How strength works (read this before judging the Elo)

Genuine playing strength comes from two things this project is honest about:

1. **Real data.** The network is pre-trained by behavioural cloning on **real
   Lichess games filtered to strong players** (the bundled run uses 2000+ rated
   games from the [Lichess open database](https://database.lichess.org/)). This is
   what teaches it real chess; a network trained on a few toy games cannot play.
2. **Search + scale.** MCTS adds lookahead at serve time, and self-play refines the
   network over generations. AlphaZero reached superhuman with *thousands of
   TPU-hours* - strength scales with compute and data.

**On the Elo number:** a meaningful "2000 Elo" must be **anchored to a calibrated
reference**. Stockfish is that reference and *only* that - the agents never use it to
play. `eval/benchmark.py` throttles Stockfish to known Elo bands (`UCI_LimitStrength`
+ `UCI_Elo`, with a `Skill Level` fallback below the floor), plays each agent a
gauntlet, and fits its rating (Bradley-Terry MLE in `eval/elo.py`); it also reports
centipawn loss and top-1 agreement per agent. Run it with `sgambit benchmark`.
Without that anchor, any Elo is only relative. The repo ships:

- the pre-training pipeline on real strong-player Lichess data,
- a **Stockfish-anchored evaluator** to place the agent on a real scale,
- a **GPU training job** (`deploy/hf_job/run.py`) that trains on hundreds of
  thousands of 2000+ games and self-plays - the realistic path to and past 2000.

Reaching a *certified* 2000+ requires running that job on a GPU (HF Jobs, funded by
HF credit, or your own machine) with a Stockfish binary present. A laptop CPU
trains a network that genuinely plays, but cannot certify or reach 2000 on its own.

## Architecture

```
  ┌──────────────────────── Hugging Face Space (Docker, FastAPI) ────────────────────────┐
  │  TRAINS: continuous self-play (ContinualTrainer) → new versioned generation           │
  │  SERVES: /move /watch-move /predict by target Elo from the checkpoint ladder           │
  │  ADAPTS: /log_game, /adapt → per-session fine-tuned personal checkpoint                 │
  │     │ push checkpoints + ladder.json            │ (GPU bursts via HF Jobs, optional)    │
  │     ▼                                            ▼                                       │
  │  model repo  legacyaravind/shannons-gambit (gen-*.pt, ladder.json)                      │
  └───────────────────────────────────▲───────────────────────────────────────────────────┘
                                       │ HTTPS (server-side, no fallback)
                Next.js on Vercel:  /play  /watch  /research  /predict  /ladder
```

- **Pre-train** - behavioural cloning on real Lichess games (`models/supervised.py`).
- **Self-play RL** - MCTS self-play improves the net each generation; each checkpoint
  is rated on a stable anchored Elo ladder (`agents/alphazero/continual.py`, `agents/ladder.py`).
- **Adapt** - live opponent-modeling + genuine per-session fine-tuning (`agents/adaptive.py`).
- **Serve** - the Space (`deploy/hf_space/app.py`) over the package's `serve.py`; the site
  has **no heuristic fallback** (`web/`).
- **Foundations** - exact MDP/Bellman endgames, tabular Q, DQN, and the information-theory
  analysis from the project's first phase remain in `mdp/`, `agents/`, `infotheory/`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev]"

# Real strong-player data from the Lichess open database
python scripts/download_data.py --source \
  https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst \
  --games 120000 --min-elo 2000

sgambit train supervised --preset local_full     # pre-train (behavioural cloning)
sgambit train-continual --gens 5                 # self-play generations (versioned ladder)
sgambit ladder                                   # Elo per generation

# Serve locally (the Space app)
cd deploy/hf_space && MODELS_DIR=../../runs/continual TRAIN_ENABLED=0 \
  uvicorn app:app --port 7860
# then run the site against it:
cd ../../web && HF_SPACE_URL=http://localhost:7860 npm install && npm run dev
```

To chase 2000 on GPU:

```bash
hf jobs run --flavor a10g-small --secrets HF_TOKEN \
  python deploy/hf_job/run.py --min-elo 2000 --games 400000 --epochs 12
```

## Repository structure

```
Chess-Gambit-RL/
├── shannons_gambit/
│   ├── agents/
│   │   ├── alphazero/{mcts,selfplay,train,continual}.py   # MCTS self-play + continual loop
│   │   ├── ladder.py            # versioned Elo ladder
│   │   ├── adaptive.py          # opponent model + per-session fine-tune
│   │   └── tabular_q.py, dqn.py, random_agent.py
│   ├── serve.py                 # ModelServer: pick checkpoint by Elo, tune strength
│   ├── models/  data/  mdp/  infotheory/  eval/ (+ stockfish_ref.py)
│   ├── export.py                # Hub persistence (push/pull ladder)
│   └── cli.py                   # sgambit: data, train, train-continual, ladder, analyze, ...
├── deploy/
│   ├── hf_space/                # Docker FastAPI Space (trains + serves)
│   ├── hf_job/                  # GPU training job (path to 2000)
│   └── hf_endpoint/             # legacy single-checkpoint inference handler
├── web/                         # Next.js app: play / watch / research / predict / ladder (no fallback)
├── notebooks/                   # reproducible notebook with graphs
└── tests/                       # unittest suite
```

## Deploying

1. **HF Space** - create a Docker Space, point it at `deploy/hf_space/`, set
   `HF_MODEL_REPO`, `HF_TOKEN`, `TRAIN_ENABLED=1`. It trains and serves; checkpoints
   persist to the model repo. (See `deploy/hf_space/README.md`.)
2. **Vercel** - root `web/`, set `HF_SPACE_URL` (and `HF_SPACE_TOKEN` if private) to the
   Space. The site has no fallback, so the Space must be reachable.

## Limitations (stated honestly)

- **Certified 2000+ Elo is not produced on a laptop CPU.** It needs GPU-scale training
  on real strong-player data plus a Stockfish anchor; the job and evaluator are provided.
- Self-play loss falls cleanly, but *objective* strength gains per generation are small and
  noisy on a few CPU minutes (a gauntlet found ~zero gen↔Elo correlation at that budget).
- Without a Stockfish binary, the ladder Elo is **relative** (anchored to a random baseline),
  not a FIDE/Lichess-calibrated number.
- Personal fine-tuning is light and KL-regularized - it nudges style, not a rebuild.

## License

[Apache-2.0](LICENSE). Trained weights are released on the Hugging Face Hub under the same licence.
