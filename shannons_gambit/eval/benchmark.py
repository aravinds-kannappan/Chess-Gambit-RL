"""The Stockfish benchmark backend: scores our agents, never plays for them.

Two complementary measurements, both using Stockfish purely as an external
yardstick:

* :func:`move_quality` -- average centipawn loss and top-1 agreement of an
  agent's moves vs. Stockfish over a spread of positions.
* :func:`assess_elo` -- the agent's calibrated Elo, found by gauntlets against
  Stockfish throttled to known Elo bands (Bradley-Terry fit via
  :func:`~shannons_gambit.eval.elo.estimate_rating`).

:func:`benchmark_suite` runs both for several agents so each is rated
*separately*. Everything no-ops with a clear error when no binary is present.
"""

from __future__ import annotations

import random

import chess

from ..agents.base import Agent
from ..agents.stockfish import StockfishAgent, find_stockfish
from ..config import StockfishConfig
from .elo import estimate_rating

# Stockfish UCI_Elo bands used as fixed anchors when assessing an agent's Elo.
DEFAULT_ANCHORS = (1350, 1600, 1900, 2200, 2500)
_MATE_CP = 100_000


def random_positions(n: int, *, seed: int = 0, max_plies: int = 16) -> list[chess.Board]:
    """Non-terminal boards from short random walks (a diverse benchmark set)."""
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, max_plies)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over() and any(board.legal_moves):
            out.append(board)
    return out


def _cp(score: chess.engine.PovScore, color: chess.Color) -> int:
    return score.pov(color).score(mate_score=_MATE_CP)


def move_quality(
    agent: Agent,
    positions: list[chess.Board],
    *,
    reference_elo: int = 2800,
    movetime_ms: int = 50,
    path: str | None = None,
    blunder_cp: int = 150,
) -> dict:
    """Centipawn loss / top-1 agreement of ``agent`` vs. a strong Stockfish.

    For each position we compare the side-to-move evaluation of the agent's move
    against Stockfish's best move (both scored by the same reference engine).
    """
    ref = StockfishAgent(StockfishConfig(elo=reference_elo, movetime_ms=movetime_ms), path=path)
    losses: list[int] = []
    matches = 0
    blunders = 0
    try:
        engine = ref._ensure_engine()
        for board in positions:
            color = board.turn
            info = engine.analyse(board, ref._limit())
            best_move = (info.get("pv") or [None])[0]
            best_cp = _cp(info["score"], color)

            agent_move = agent.select_move(board)
            after = board.copy()
            after.push(agent_move)
            if after.is_game_over():
                played_cp = _MATE_CP if after.is_checkmate() else 0
            else:
                played_cp = -_cp(engine.analyse(after, ref._limit())["score"], not color)

            loss = max(0, best_cp - played_cp)
            losses.append(loss)
            matches += int(best_move is not None and agent_move == best_move)
            blunders += int(loss >= blunder_cp)
    finally:
        ref.close()

    n = max(len(losses), 1)
    return {
        "agent": getattr(agent, "name", "agent"),
        "n_positions": len(losses),
        "reference_elo": reference_elo,
        "avg_centipawn_loss": round(sum(losses) / n, 1),
        "top1_agreement": round(matches / n, 3),
        "blunder_rate": round(blunders / n, 3),
    }


def phase_positions(n_per_phase: int = 20, *, seed: int = 0,
                    max_attempts: int = 5000) -> dict[str, list[chess.Board]]:
    """Sample non-terminal positions bucketed by game phase (opening/mid/end)."""
    from ..phases import PHASES, game_phase

    rng = random.Random(seed)
    buckets: dict[str, list[chess.Board]] = {p: [] for p in PHASES}
    for _ in range(max_attempts):
        if all(len(v) >= n_per_phase for v in buckets.values()):
            break
        board = chess.Board()
        for _ in range(rng.randint(0, 80)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over() or not any(board.legal_moves):
            continue
        bucket = buckets[game_phase(board)]
        if len(bucket) < n_per_phase:
            bucket.append(board)
    return buckets


def move_quality_by_phase(
    agent: Agent,
    *,
    n_per_phase: int = 20,
    reference_elo: int = 2800,
    movetime_ms: int = 50,
    path: str | None = None,
    seed: int = 0,
) -> dict:
    """Average centipawn loss of ``agent`` broken down by game phase.

    Surfaces *where* an agent is weak -- the opening/middlegame blind spot the
    project had no visibility into before. One Stockfish reference scores all
    phases identically; out-of-the-book opening play is graded here too.
    """
    buckets = phase_positions(n_per_phase, seed=seed)
    out: dict[str, dict] = {}
    for phase, positions in buckets.items():
        if positions:
            out[phase] = move_quality(agent, positions, reference_elo=reference_elo,
                                      movetime_ms=movetime_ms, path=path)
    return out


def assess_elo(
    agent: Agent,
    *,
    anchors: tuple[int, ...] = DEFAULT_ANCHORS,
    games: int = 10,
    max_moves: int = 120,
    movetime_ms: int = 50,
    path: str | None = None,
    init: float = 1600.0,
) -> dict:
    """Assign ``agent`` a calibrated Elo via gauntlets vs. Elo-throttled Stockfish."""
    from .arena import play_game

    results: list[tuple[float, float, int]] = []
    per_anchor: dict[int, float] = {}
    for anchor in anchors:
        opp = StockfishAgent(StockfishConfig(elo=anchor, movetime_ms=movetime_ms), path=path)
        pts = 0.0
        try:
            for i in range(games):
                white, black = (agent, opp) if i % 2 == 0 else (opp, agent)
                rec = play_game(white, black, max_moves=max_moves)
                if rec.result == "1-0":
                    pts += 1.0 if white is agent else 0.0
                elif rec.result == "0-1":
                    pts += 1.0 if black is agent else 0.0
                else:
                    pts += 0.5
        finally:
            opp.close()
        results.append((float(anchor), pts, games))
        per_anchor[anchor] = round(pts / max(games, 1), 3)
    return {
        "agent": getattr(agent, "name", "agent"),
        "elo": estimate_rating(results, init=init),
        "score_vs_anchor": per_anchor,
        "games_per_anchor": games,
    }


def benchmark_suite(
    agents: dict[str, Agent],
    positions: list[chess.Board] | None = None,
    *,
    n_positions: int = 200,
    elo_games: int = 8,
    movetime_ms: int = 50,
    path: str | None = None,
) -> dict:
    """Benchmark several agents separately (move quality + calibrated Elo)."""
    if find_stockfish(path) is None:
        raise RuntimeError("no Stockfish binary; set $STOCKFISH_PATH or install stockfish")
    positions = positions or random_positions(n_positions)
    report: dict[str, dict] = {}
    for name, agent in agents.items():
        quality = move_quality(agent, positions, movetime_ms=movetime_ms, path=path)
        elo = assess_elo(agent, games=elo_games, movetime_ms=movetime_ms, path=path)
        report[name] = {**quality, **{k: elo[k] for k in ("elo", "score_vs_anchor")}}
    return report
