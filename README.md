<div align="center">

# ♟ Shannon's Gambit

### A chess engine that learns, at an honest rating

**Real Lichess data · gated self-play that trains itself nightly · adapts to how you play · rated only as high as Stockfish says it can actually play**

[![CI](https://github.com/aravinds-kannappan/Chess-Gambit-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/aravinds-kannappan/Chess-Gambit-RL/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

</div>

---

## What it is

An end-to-end chess RL system that **owns the whole pipeline**. It is a
**multi-agent** engine: each position is routed to the method that owns it - a
**learned opening book** for the first moves, an AlphaZero-lite **neural net** for the
middlegame, exact **MDP/Bellman** dynamic programming in solved endgames, with
**PPO** and **reward (DQN)** RL in the low-material regime (`agents/router.py`). It
pre-trains on real Lichess games (sampled evenly across opening/middlegame/endgame),
improves by **nightly gated self-play with champion gating** (a new generation is served
only if it actually beats the current one, so strength never regresses), and **adapts
to how you play** (finished games auto fine-tune a personal checkpoint that persists
across restarts). Play and prediction have **no heuristic fallback** (every move is a
trained net); the spectate views fall back to a local engine so they never go dark
when the backend is asleep.

**Stockfish is the referee, never a player.** The agents never call Stockfish to
choose a move. A separate backend evaluator (`eval/benchmark.py`) uses Stockfish
purely as a calibrated yardstick: it scores each agent's **centipawn loss** and
top-1 agreement, and assigns a **calibrated Elo** via gauntlets against Elo-throttled
Stockfish. That calibrated rating is the engine's **ceiling**: the serving layer
scales strength to any target *up to* that ceiling and, above it, plays full strength
and reports the ceiling rather than echoing a number it cannot play. Nightly training
is what raises the ceiling.

| Surface | What it does |
| --- | --- |
| **Play** | Play the routed engine. **Match me** estimates your level from your move quality and retargets it per move; **Set level** caps at the honest ceiling; **Full** plays its Stockfish-rated best. Finished games auto fine-tune a *personal* checkpoint for your browser (persisted to the Hub). |
| **Watch** | Pair two engines at chosen ratings (capped at the ceiling) for a single exhibition game. |
| **Scorebook** | The training scorebook: rating per generation, Stockfish-calibrated rating, loss curves, and information-theory over real games. |
| **Analysis** | FEN/PGN → win/draw/loss, best move, value from the champion network, with the network's move and engine lines drawn on the board. |
| **Tables / Ladder** | Every club table (tiers) and every rated generation, all spaced under the honest ceiling. |

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
- a **nightly training job** (`nightly_job.py` + `.github/workflows/train.yml`) that
  runs gated self-play on a free runner and re-grades vs Stockfish every night, plus a
  one-shot `train_job.py` for a bigger run on CPU/MPS or a GPU.

The engine you can play today grades around **1000 Elo**, and the site is careful never
to claim a level it cannot play. Reaching a stronger *certified* rating means more
training on real strong-player data: many nights of the free job, or a single large run
(GPU scale gets there fastest). The point of the pipeline is that the number stays
honest at every size.

## Architecture

```
  ┌──── GitHub Actions (nightly, free runner) ────┐      ┌──── HF Space (Docker, FastAPI) ────┐
  │  nightly_job: pull champion → gated self-play  │      │  SERVES: /move /watch-move /predict │
  │  + human-data (SFT) replay → Stockfish grade   │      │  by target Elo, capped at the       │
  │  → push ladder → poke the Space to reload      │      │  calibrated ceiling                 │
  └───────────────────────┬────────────────────────┘      │  ADAPTS: /log_game auto fine-tunes  │
                          │ push checkpoints + ladder.json │  a personal checkpoint (→ Hub)      │
                          ▼                                 └──────────────┬─────────────────────┘
              model repo  legacyaravind/shannons-gambit                    │ HTTPS (no fallback
              (pretrain/, posttrain/, personal/, ladder.json, book)  ◀─────┘  for play/predict)
                                       ▲
                Next.js on Vercel:  /play  /watch  /research (scorebook)  /predict  /arena
```

- **Pre-train** - behavioural cloning on real Lichess games, phase-balanced so the net
  learns the middlegame and not just opening moves (`models/supervised.py`).
- **Opening book** - a weighted book learned from strong-player games by position hash,
  routed for the first plies (`agents/opening_book.py`, `sgambit build-book`).
- **Train nightly** - GitHub Actions runs `nightly_job.py`: gated MCTS self-play with
  **human-data (SFT) replay** so it refines toward strong play instead of drifting. A
  contender is promoted only if it beats the champion head-to-head, else its weights are
  rolled back (`agents/alphazero/continual.py`). Free runner, no GPU required.
- **Adapt** - finished games auto fine-tune a per-session checkpoint (REINFORCE + KL to
  base), cached in memory and persisted to the Hub so it survives Space restarts
  (`agents/adaptive.py`, `deploy/hf_space/app.py`).
- **Serve** - the Space over the package's `serve.py`; serves the gated champion + book,
  scaled only up to the calibrated **ceiling** (`serve.ceiling()`), no fallback for
  play/predict (`web/`).
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
sgambit build-book --min-elo 2000                # learn an opening book from real games
sgambit train-continual --gens 5                 # self-play generations (gated ladder)
sgambit ladder                                   # Elo per generation

# Serve locally (the Space app)
cd deploy/hf_space && MODELS_DIR=../../runs/continual TRAIN_ENABLED=0 \
  uvicorn app:app --port 7860
# then run the site against it:
cd ../../web && HF_SPACE_URL=http://localhost:7860 npm install && npm run dev
```

### Training it (three paths, no paid GPU required)

```bash
# 1) Nightly, hands-off: GitHub Actions runs this on a free runner every night.
#    Trigger a run by hand (small smoke or full):
gh workflow run nightly-training -f gens=6 -f games_per_gen=16
#    or run the same job locally:
python -m shannons_gambit.nightly_job --gens 6 --games-per-gen 16

# 2) One big local run (Apple Silicon MPS / any CPU), pushes to the Hub:
python -m shannons_gambit.train_job --min-elo 2000 --games 120000 --epochs 8

# 3) GPU burst (needs an HF Pro/Team plan for HF Jobs):
hf jobs run --flavor a10g-small --secrets HF_TOKEN \
  python -m shannons_gambit.train_job --min-elo 2000 --games 400000 --epochs 12
```

All three push `pretrain/model.pt`, the opening book, and the gated `posttrain/`
ladder to the model repo; the serving Space reloads the new champion. Every path is
gated, so a bad run can never make the served engine weaker.

## Repository structure

```
Chess-Gambit-RL/
├── shannons_gambit/
│   ├── agents/
│   │   ├── alphazero/{mcts,selfplay,train,continual}.py   # MCTS self-play + continual loop
│   │   ├── ladder.py            # versioned Elo ladder
│   │   ├── adaptive.py          # opponent model + per-session fine-tune
│   │   └── tabular_q.py, dqn.py, random_agent.py
│   ├── serve.py                 # ModelServer: champion + book, scaled up to the ceiling
│   ├── nightly_job.py           # nightly gated self-play + SFT replay + calibrate + push
│   ├── train_job.py             # one-shot full training run (local / GPU), pushes to Hub
│   ├── models/  data/  mdp/  infotheory/  eval/ (+ stockfish_ref.py)
│   ├── export.py                # Hub persistence (ladder, base, book, personal checkpoints)
│   └── cli.py                   # sgambit: data, train, build-book, train-continual, ...
├── .github/workflows/train.yml  # nightly training on a free GitHub Actions runner
├── deploy/
│   ├── hf_space/                # Docker FastAPI Space (serves; optional background self-play)
│   ├── hf_job/                  # optional GPU training job (HF Jobs, needs a paid plan)
│   └── hf_endpoint/             # legacy single-checkpoint inference handler
├── web/                         # Next.js app: play / watch / scorebook / analysis (no fallback)
├── notebooks/                   # reproducible notebook with graphs
└── tests/                       # unittest suite
```

## Deploying

1. **HF Space (serving)** - create a Docker Space, point it at `deploy/hf_space/`, set
   `HF_MODEL_REPO` (and `HF_TOKEN` if private). It serves the published champion +
   opening book; checkpoints persist to the model repo. Leave `TRAIN_ENABLED=0`: a CPU
   Space that also self-plays starves the web server (`/healthz` times out and the site
   shows "backend warming up"). (See `deploy/hf_space/README.md`.)
2. **Nightly training (free)** - add an `HF_TOKEN` repo secret (write access to the model
   repo + Space); `.github/workflows/train.yml` then trains every night on a GitHub
   runner, re-grades vs Stockfish, pushes the ladder, and restarts the Space. Run it on
   demand with `gh workflow run nightly-training`. A GPU is optional (HF Jobs, paid);
   the nightly job and `train_job.py` run on free CPU/MPS.
3. **Vercel (frontend only)** - root `web/`, set `HF_SPACE_URL` (and `HF_SPACE_TOKEN` if
   private) to the Space. **Vercel only runs the Next.js site; it does not train anything**
   - API keys there do not start training. Play/predict need the Space reachable.

## Limitations (stated honestly)

- **The current ceiling is modest** (the deployed champion grades around 1000 Elo, not
  2000). The site never labels a level above that ceiling; slider tops and "Full"
  strength report the Stockfish-calibrated number. Reaching a stronger, *certified* Elo
  needs more training on real strong-player data (the nightly job, or a GPU run for scale).
- **Gating trades climbing speed for safety.** On a short run, self-play rarely beats the
  champion, so the ladder holds rather than climbing. By design it can never *regress*
  into weak play either. Consistent climbing needs many nights (or a big run).
- Without a Stockfish binary the ladder Elo is **relative** (anchored to a random
  baseline), not a FIDE/Lichess-calibrated number; the nightly job installs Stockfish so
  its runs are graded.
- Personal fine-tuning is light and KL-regularized: it nudges style toward beating *you*,
  it is not a full rebuild.

## License

[Apache-2.0](LICENSE). Trained weights are released on the Hugging Face Hub under the same licence.
