---
title: Shannon's Gambit API
emoji: ♟️
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Shannon's Gambit - serving Space

This Hugging Face Space **serves** the chess agents:

- **serves** moves/predictions/watch pairings by target Elo from the gated
  champion + opening book (no heuristic fallback - every move is a trained net),
- **adapts** to a player by fine-tuning a personal checkpoint on their games,
- can **optionally** run background self-play (`ContinualTrainer`) when
  `TRAIN_ENABLED=1`, but this is **off by default**: a single CPU Space that both
  trains and serves starves the web server (so `/healthz` times out and the site
  shows "backend warming up"). Do real training on **HF Jobs** (`deploy/hf_job`,
  GPU) and let this Space just serve the published checkpoints + book.

## Endpoints
`GET /healthz`, `GET /ladder`, `POST /move`, `POST /watch-move`, `POST /predict`,
`POST /log_game`, `POST /adapt`.

## Configuration (Space secrets / variables)
| Var | Purpose |
| --- | --- |
| `HF_MODEL_REPO` | model repo for checkpoints/ladder/book, e.g. `legacyaravind/shannons-gambit` |
| `HF_TOKEN` | write token (enables pushing new generations) |
| `TRAIN_ENABLED` | `1` to run background self-play; **default `0`** (serve only) |
| `TRAIN_SLEEP` | seconds the trainer sleeps between generations (default `5`) |
| `BASE_ELO` | Elo the served base net is scaled to (default `1600`) |
| `MODELS_DIR` | checkpoint dir (default `/data/models`) |

Build is defined by `Dockerfile`; the package is installed from GitHub via
`requirements.txt`.
