<div align="center">

# ♟ Shannon's Gambit

### Information-Theoretic Reinforcement Learning for Chess

**Markov Decision Processes · Bellman optimality · Deep RL · real Lichess data · a deployable web app**

[![CI](https://github.com/aravinds-kannappan/Chess-Gambit-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/aravinds-kannappan/Chess-Gambit-RL/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/web-Next.js%20on%20Vercel-black.svg)](web/)

</div>

---

## Why "Shannon's Gambit"?

Claude Shannon wrote both *"A Mathematical Theory of Communication"* (1948), which
founded **information theory**, and *"Programming a Computer for Playing Chess"*
(1950), which founded **computer chess**. This project deliberately sits at the
intersection of those two papers: it treats chess as a problem of *uncertainty
and information*, formalises play as a **Markov Decision Process**, solves it with
**Bellman optimality** where tractable, and learns it with **deep reinforcement
learning** where it is not - all on **real human game data**.

---

## What this project is

A single, coherent system that unifies three bodies of theory and ships the
result as an interactive, deployable web app:

| Pillar | What it does | Where |
| --- | --- | --- |
| **Information theory** | Shannon entropy, KL/JS divergence, mutual information; quantifies policies, positions, and "where games are decided" | `shannons_gambit/infotheory/` |
| **MDP + Bellman** | Enumerates KRvK/KQvK endgames and solves them *exactly* with value/policy iteration; tabular Q-learning recovers the optimum from experience | `shannons_gambit/mdp/`, `agents/tabular_q.py` |
| **Deep RL** | A DQN validated against the exact value table, and a small **AlphaZero-lite** (policy/value net + PUCT MCTS self-play) bootstrapped from supervised play | `shannons_gambit/agents/dqn.py`, `agents/alphazero/` |
| **Prediction model** | Multi-head residual net trained on real games: next-move, win/draw/loss, value, and player-rating heads | `shannons_gambit/models/` |
| **Web app** | Play the agent, explore the information dashboard, run live predictions, browse the Elo arena | `web/` (Next.js → Vercel) |

Everything is **real**: the data is real Lichess games, the models are actually
trained (losses decrease, checkpoints are written), and the theory is verified
against analytic identities and exact ground truth.

---

## Quickstart

```bash
# 1. Install (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev]"

# 2. Get real data (bundled sample works offline; or pull a real Lichess dump)
python scripts/download_data.py --games 20000          # real Lichess open DB
#   ... or just use the bundled historical games for an offline smoke run.

# 3. Solve an endgame exactly with Bellman value iteration
sgambit mdp --endgame KRvK
#   -> value iteration converges; the optimal policy forces mate from every won position

# 4. Train the prediction / behavioural-cloning model on real games
sgambit train supervised --preset local_full

# 5. Information-theory report (entropy, mutual information, info-gain-per-ply)
sgambit analyze

# 6. Round-robin Elo arena between the agents
sgambit arena

# 7. Live prediction from any position
sgambit predict --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

# 8. Run the web app locally
cd web && npm install && npm run dev      # http://localhost:3000
```

Everything also runs as a tiny, fast smoke via `--preset local_smoke`, and scales
to a serious run via `--preset cloud` - a config change, not a rewrite.

---

## System design

```
                         real Lichess PGN (zstd)
                                  │  stream + parse (python-chess)
                                  ▼
                    ┌──────────────────────────┐
                    │  data/  encode.py         │  board → 18×8×8 planes
                    │         dataset.py        │  parquet position shards
                    └────────────┬─────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────────┐
        ▼                        ▼                              ▼
 ┌──────────────┐        ┌───────────────┐             ┌──────────────────┐
 │  mdp/        │        │  models/      │             │  infotheory/     │
 │  Bellman VI  │        │  ChessNet     │  supervised │  entropy, KL,    │
 │  exact KRvK  │        │  4 heads      │◀── pretrain │  mutual info     │
 └──────┬───────┘        └──────┬────────┘             └────────┬─────────┘
        │ V* (potential)        │ bootstrap                     │
        ▼                       ▼                               │
 ┌──────────────┐   ┌────────────────────────┐                 │
 │ agents/      │   │ agents/alphazero/      │                 │
 │ tabular_q    │   │ MCTS + self-play       │                 │
 │ dqn (shaped) │   │ (deep RL)              │                 │
 └──────┬───────┘   └───────────┬────────────┘                 │
        └──────────────┬────────┘                              │
                       ▼                                        ▼
                 eval/ arena + Elo  ───────────►  export →  web/ (Next.js / Vercel)
                                                  HF hub  →  Hugging Face model
```

**Key design decisions**

- **One shared encoding** (`data/encode.py`): an 18-plane board tensor and the
  reversible AlphaZero **4672-move index**. Every model - supervised, DQN,
  AlphaZero, prediction - speaks the same representation, tested by an exact
  round-trip over thousands of legal moves.
- **One shared network** (`models/net.py`): a residual tower with four heads
  (policy / value / WDL / rating). Supervised pretraining produces both the
  released prediction model **and** the bootstrap weights for self-play, so a
  laptop run refines a sensible network instead of starting from noise.
- **Exact where possible, learned where necessary.** Full chess (~10⁴⁴ states)
  is intractable for tabular methods, so Bellman is made *provably correct* on
  enumerable endgames; deep RL takes over for the full game.
- **The Bellman solution feeds the deep RL.** The exact value table V\* is used
  as a **potential-based shaping reward** for DQN - theoretically policy-preserving,
  and a concrete bridge from the MDP pillar to the deep-RL pillar.
- **The heavy model never runs on Vercel.** Weights live on Hugging Face; the
  web app proxies an Inference Endpoint and falls back to a built-in TypeScript
  agent so the site is always responsive.

---

## Repository structure

```
Chess-Gambit-RL/
├── shannons_gambit/          # Python package (import name)
│   ├── cli.py                # `sgambit` CLI: data, mdp, train, analyze, arena, predict, export
│   ├── config.py             # frozen dataclass configs + local_smoke / local_full / cloud presets
│   ├── data/                 # encode.py (planes + 4672 moves), lichess.py, dataset.py, sample_games.pgn
│   ├── mdp/                  # bellman.py (VI/PI/QVI), endgames.py, chess_mdp.py (env + exact solver)
│   ├── infotheory/           # entropy.py, divergence.py, analysis.py
│   ├── agents/               # base, random, tabular_q, dqn, alphazero/{mcts,selfplay,train}, neural
│   ├── models/               # net.py (ChessNet), supervised.py, prediction.py
│   ├── eval/                 # arena.py, elo.py
│   ├── reports.py            # ties data + models to the information-theory measures
│   └── export.py             # web JSON + Hugging Face upload + model card
├── tests/                    # unittest suite (encoding, Bellman, info theory, data, MCTS, arena, supervised, endgame)
├── scripts/                  # download_data.py, quality_gate.py
├── notebooks/                # reproducible notebook (tests + graphs)
├── web/                      # Next.js app (Vercel): play / analysis / predict / arena + API routes
├── docs/                     # ARCHITECTURE.md, GETTING_STARTED.md, THEORY.md
└── pyproject.toml            # distribution "shannons-gambit"; CLI entry `sgambit`
```

> **Naming:** the GitHub repo is **Chess-Gambit-RL**, the distribution is
> **shannons-gambit**, and the import package is **shannons_gambit** - the same
> three-name pattern as `scikit-learn` → `sklearn`. They are intentionally distinct.

---

## Results (reproducible)

> Numbers below come from the local pipeline; the notebook regenerates every plot.

- **MDP / Bellman (exact).** KRvK enumerates **399,112** states; value iteration
  converges in ~33 sweeps to `‖ΔV‖∞ = 0`. The derived optimal policy **forces
  mate from 100% of won positions**, and the implied mate-distance matches actual
  play (e.g., a central position mates in 23 plies = DTM 23).
- **Tabular Q-learning (learned).** With a mate-distance curriculum, it converts
  **~71%** of positions within 4-8 plies of mate against a random defender,
  degrading gracefully with distance - genuine learning from experience, never
  shown the value table.
- **Information theory.** Over real positions, **material difference carries the
  most information** about the result (highest mutual information), ahead of
  mobility; outcome entropy collapses sharply at the decisive moment of a game.
- **Prediction model.** Multi-head training drives policy, WDL, value, and rating
  losses down together; the WDL head and rating head give the live-prediction page.
- **Deep RL (DQN) - see Limitations.** A flat 4672-action DQN is a poor structural
  fit for forced-mate search; potential-based shaping with V\* helps, but tabular
  memorisation and AlphaZero-style policy/value+MCTS remain the effective methods.

---

## Tradeoffs & limitations

- **DQN vs. the problem structure.** A Q-network over 4672 move indices must learn
  a position-specific *mapping* to the mating move, which generalises poorly from
  sparse reward; tabular Q (memorisation) and AlphaZero-lite (policy/value + search)
  both do better. We keep DQN as an honest, instructive baseline and use the exact
  Bellman value as a shaping potential to show how much the MDP solution helps.
- **Compute.** Designed to run on a laptop (Apple M4 / MPS); the default nets and
  self-play volumes are modest. The `cloud` preset scales depth/width, simulations,
  and data without code changes - AlphaZero-style strength needs that budget.
- **Offline data.** A small set of real historical games is bundled so the whole
  pipeline (and CI) runs with no network; serious training pulls a real monthly
  Lichess dump via `scripts/download_data.py`.
- **Exact solving is endgame-only.** Bellman value iteration is applied to KRvK/KQvK
  because they are enumerable; the full game is intractable for tabular methods by
  design, which is precisely why deep RL exists in the stack.

---

## Why open source?

- **Reproducible research over claims.** Information theory + RL is often discussed
  abstractly; here every measure is computed on real data, every model is trained
  from a single command, and a notebook regenerates the figures. Anyone can audit
  the gap between theory and practice (including the DQN limitation above).
- **A teaching artifact.** The codebase is a guided tour from the Bellman equation
  on a solvable endgame to MCTS self-play on the full game, with the information-
  theoretic lens connecting them. It is meant to be read.
- **Permissive licence.** Apache-2.0 (with an explicit patent grant) so the code,
  the trained weights (Hugging Face), and the analysis can be freely reused.

---

## Reproducibility

```bash
# Full local quality gate: lint + unit tests + a tiny end-to-end smoke train
python scripts/quality_gate.py

# Unit tests only
python -m unittest discover -s tests

# The notebook re-runs the checks and regenerates every graph
jupyter notebook notebooks/shannons_gambit_reproduce.ipynb
```

The test suite covers the move-encoding round-trip, Bellman convergence and the
forced-mate guarantee, the information-theoretic identities, the data pipeline on
real games, MCTS legality, Elo/arena bookkeeping, and a real supervised train step.

---

## Deploying the web app

The app under `web/` is a standard Next.js (App Router) project; point Vercel at
the `web/` directory and deploy. It works immediately with the built-in agent and
upgrades to the trained model when a Hugging Face endpoint is configured (below).

---

## Configuration - API keys & environment variables

Everything runs **with no keys at all** (offline data + the built-in web agent).
Keys only unlock the *hosted* paths:

| Variable | Where | Needed for | How to get it |
| --- | --- | --- | --- |
| `HF_TOKEN` | shell env (Python) | `sgambit export --hf` to upload weights to the Hugging Face Hub | https://huggingface.co/settings/tokens (a *write* token) |
| `HF_ENDPOINT_URL` | `web/.env.local` & Vercel env | The web app proxying the trained model for play/predict | URL of your deployed [HF Inference Endpoint](https://ui.endpoints.huggingface.co/) serving the model |
| `HF_API_TOKEN` | `web/.env.local` & Vercel env | Auth for the above endpoint (if private) | https://huggingface.co/settings/tokens (a *read* token) |
| `HF_MODEL_ID` | `web/.env.local` & Vercel env | Display / model card link | e.g. `legacyaravind/shannons-gambit` |

```bash
# Python side (uploading the trained model to Hugging Face)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
sgambit export --hf legacyaravind/shannons-gambit --model runs/supervised/model.pt

# Web side: web/.env.local  (copy from web/.env.example)
HF_ENDPOINT_URL=https://<your-endpoint>.endpoints.huggingface.cloud
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_MODEL_ID=legacyaravind/shannons-gambit
```

On Vercel, set the three `HF_*` web variables under **Project → Settings →
Environment Variables**. No GitHub Actions secrets are required for CI (tests run
fully offline). **Never commit real tokens** - `.env.local` is gitignored.

---

## License

[Apache-2.0](LICENSE). Trained weights are released under the same licence with a
model card on the Hugging Face Hub.

<div align="center">
<sub>Built with python-chess, PyTorch, Next.js, and a lot of Bellman backups.</sub>
</div>
