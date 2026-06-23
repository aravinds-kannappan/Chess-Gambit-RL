"""Play agents against each other and produce an Elo leaderboard + replays."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from ..agents.base import Agent
from .elo import fit_elo


@dataclass
class GameRecord:
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: list[str] = field(default_factory=list)


def play_game(white: Agent, black: Agent, *, max_moves: int = 120) -> GameRecord:
    board = chess.Board()
    moves: list[str] = []
    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        agent = white if board.turn == chess.WHITE else black
        move = agent.select_move(board)
        moves.append(move.uci())
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"  # adjudicate the move-capped game as a draw
    return GameRecord(white.name, black.name, result, moves)


def round_robin(
    agents: list[Agent], *, games_per_pair: int = 2, max_moves: int = 120
) -> dict:
    """Every ordered pair plays ``games_per_pair`` games; returns scores + Elo."""
    names = [a.name for a in agents]
    idx = {a.name: i for i, a in enumerate(agents)}
    n = len(agents)
    score = np.zeros((n, n), dtype=np.float64)
    games = np.zeros((n, n), dtype=np.float64)
    records: list[GameRecord] = []

    for a in agents:
        for b in agents:
            if a is b:
                continue
            for _ in range(games_per_pair):
                rec = play_game(a, b, max_moves=max_moves)
                records.append(rec)
                i, j = idx[a.name], idx[b.name]
                games[i, j] += 1
                games[j, i] += 1
                if rec.result == "1-0":
                    score[i, j] += 1.0
                elif rec.result == "0-1":
                    score[j, i] += 1.0
                else:
                    score[i, j] += 0.5
                    score[j, i] += 0.5

    elo = fit_elo(names, score, games)
    table = sorted(
        (
            {
                "agent": name,
                "elo": elo[name],
                "points": float(score[idx[name]].sum()),
                "games": float(games[idx[name]].sum()),
            }
            for name in names
        ),
        key=lambda r: r["elo"],
        reverse=True,
    )
    return {
        "leaderboard": table,
        "elo": elo,
        "n_games": len(records),
        "replays": [rec.__dict__ for rec in records[:50]],
    }
