"""Stockfish reference engine -- used **only as a benchmark**, never as a player.

The project's own agents (MDP / PPO / reward, routed by game phase) choose every
move themselves. Stockfish lives here purely so the backend evaluator
(:mod:`shannons_gambit.eval.benchmark`) can score those agents against a strong,
Elo-calibrated yardstick.

* :class:`StockfishAgent` throttles Stockfish to a named Elo band (a reliable
  reference opponent for gauntlets) and exposes per-position evaluations
  (centipawn / mate scores) for measuring an agent's move quality.

The Stockfish binary is optional at import time; helpers degrade gracefully and
tests skip when no engine is on the machine.
"""

from __future__ import annotations

from .engine import (
    StockfishAgent,
    StockfishUnavailable,
    elo_to_uci_options,
    find_stockfish,
    skill_for_elo,
)

__all__ = [
    "StockfishAgent",
    "StockfishUnavailable",
    "find_stockfish",
    "elo_to_uci_options",
    "skill_for_elo",
]
