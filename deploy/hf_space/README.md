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

# Shannon's Gambit - training + serving Space

This Hugging Face Space **trains and serves** the chess agents:

- **serves** moves/predictions/watch pairings by target Elo from the checkpoint
  ladder (no heuristic fallback - every move is from a trained network),
- **trains** continuously via background self-play (`ContinualTrainer`), pushing
  each new generation to the model repo so the ladder survives restarts,
- **adapts** to a player by fine-tuning a personal checkpoint on their games.

## Endpoints
`GET /healthz`, `GET /ladder`, `POST /move`, `POST /watch-move`, `POST /predict`,
`POST /log_game`, `POST /adapt`.

## Configuration (Space secrets / variables)
| Var | Purpose |
| --- | --- |
| `HF_MODEL_REPO` | model repo for ladder persistence, e.g. `legacyaravind/shannons-gambit` |
| `HF_TOKEN` | write token (enables pushing new generations) |
| `TRAIN_ENABLED` | `1` to run the background self-play trainer (default on) |
| `MODELS_DIR` | checkpoint dir (default `/data/models`) |

Build is defined by `Dockerfile`; the package is installed from GitHub via
`requirements.txt`.
