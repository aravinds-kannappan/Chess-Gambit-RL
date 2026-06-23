# Architecture

This document describes how the pieces fit together and why.

## Data flow

1. **Ingest** (`data/lichess.py`): stream a zstd-compressed Lichess PGN, parse
   with python-chess, emit one record per position with the move played, the
   eventual result (side-to-move perspective), any Stockfish eval, and the mover
   Elo. A small bundled PGN of real historical games makes the whole thing run
   offline.
2. **Shard** (`data/dataset.py`): write records to parquet (FENs stay tiny on
   disk); `PositionDataset` encodes FENs to planes once at load time and exposes
   policy / value / WDL / rating targets as tensors.
3. **Encode** (`data/encode.py`): the single source of truth for representation -
   an 18-plane board tensor and the reversible 4672-move index. Tested by an
   exact round-trip over thousands of legal moves and underpromotions.

## Models

`models/net.py` defines `ChessNet`: a residual tower with four heads.

| Head | Output | Trained on | Used by |
| --- | --- | --- | --- |
| policy | 4672 logits | move played (behavioural cloning) / MCTS visits | play, MCTS prior |
| value | scalar tanh | game result | AlphaZero value, eval |
| wdl | 3 logits | game result | prediction page |
| rating | scalar | mover Elo | prediction page |

Supervised pretraining (`models/supervised.py`) produces both the released
prediction model and the bootstrap weights for self-play.

## MDP + Bellman

`mdp/bellman.py` implements value/policy/Q iteration on a vectorised CSR
transition model (`GameMDP`) that supports per-state min/max - i.e. the
turn-based (Markov game) Bellman operator. `mdp/chess_mdp.py` enumerates the full
KRvK/KQvK state space, compiles it to a `GameMDP`, solves it exactly, and exposes
an `EndgameEnv` (gym-style) for the learning agents. Losing the strong piece
collapses to a single absorbing draw state.

## Agents

- `agents/tabular_q.py`: tabular Q-learning on `EndgameEnv` with a mate-distance
  curriculum; converges toward the value-iteration optimum from experience.
- `agents/dqn.py`: a double-DQN over the plane encoding, optionally shaped by the
  exact value table V\* (potential-based reward).
- `agents/alphazero/`: PUCT MCTS (`mcts.py`), self-play example generation
  (`selfplay.py`), and the iterate-train loop (`train.py`), bootstrapped from
  supervised weights.
- `agents/neural.py`: `NeuralAgent` (greedy policy) and `ValueAgent` (one-ply
  value lookahead) for the arena.

## Evaluation & export

`eval/arena.py` runs round-robins and fits Elo (`eval/elo.py`, a Bradley-Terry
MLE). `export.py` writes the web JSON artifacts and uploads weights + a model
card to the Hugging Face Hub.

## Web app

`web/` is a Next.js (App Router) project. API routes proxy a Hugging Face
Inference Endpoint (if configured) and fall back to a TypeScript heuristic agent
so the site always works. The analysis and arena pages render exported JSON.

## Configuration

`config.py` holds frozen dataclasses and three presets (`local_smoke`,
`local_full`, `cloud`) so scaling is a flag, not a rewrite.
