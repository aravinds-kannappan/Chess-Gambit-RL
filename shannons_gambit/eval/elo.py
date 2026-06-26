"""Fit Elo ratings from round-robin results via a Bradley-Terry MLE.

Draws count as half a win for each side. Ratings are fit by gradient ascent on
the logistic likelihood and anchored to a configurable mean for readability.
"""

from __future__ import annotations

import math

import numpy as np

SCALE = 400.0
BASE = 10.0


def expected_score(r_i: float, r_j: float) -> float:
    return 1.0 / (1.0 + BASE ** ((r_j - r_i) / SCALE))


def elo_delta_from_score(score: float, *, clamp: float = 200.0) -> float:
    """Elo difference implied by a head-to-head score fraction in ``[0, 1]``.

    The inverse of :func:`expected_score`: a 50% score is +0, 75% is ~+191, etc.
    Clamped to ``±clamp`` so a near-sweep over a handful of games cannot imply a
    multi-thousand-point swing (a small sample simply cannot measure that far).
    """
    s = min(max(score, 1e-3), 1.0 - 1e-3)
    delta = -SCALE * math.log10(1.0 / s - 1.0)
    return max(-clamp, min(clamp, delta))


def estimate_rating(results: list[tuple[float, float, int]], *,
                    init: float = 1000.0, iters: int = 400, lr: float = 100.0,
                    max_step: float = 64.0) -> float:
    """Estimate one player's Elo against fixed-rating anchors (1-D logistic MLE).

    ``results`` is a list of ``(anchor_elo, score, games)`` where ``score`` is the
    player's total points (win=1, draw=0.5) over ``games`` games vs. that anchor.
    Because the anchors are held fixed (e.g. a random agent pinned to a known Elo
    plus prior generations), the returned rating is on a *stable absolute scale*
    comparable across generations. Gradient ascent on the Bradley-Terry
    log-likelihood, with a per-step cap (``max_step``) so a near-sweep over a few
    games converges smoothly instead of overshooting by thousands of points.
    """
    total_games = sum(g for _, _, g in results)
    if total_games == 0:
        return init
    r = init
    for _ in range(iters):
        grad = 0.0
        for anchor_elo, score, games in results:
            if games == 0:
                continue
            grad += score - games * expected_score(r, anchor_elo)
        step = lr * grad / total_games
        r += max(-max_step, min(max_step, step))
        r = max(0.0, min(4000.0, r))
    return float(round(r, 1))


def fit_elo(
    names: list[str],
    score_matrix: np.ndarray,
    games_matrix: np.ndarray,
    *,
    anchor_mean: float = 1000.0,
    iters: int = 2000,
    lr: float = 8.0,
) -> dict[str, float]:
    """Fit ratings from aggregated pairwise scores.

    ``score_matrix[i, j]`` is the total points player ``i`` scored against
    ``j`` (win=1, draw=0.5); ``games_matrix[i, j]`` the number of games.
    """
    n = len(names)
    ratings = np.zeros(n, dtype=np.float64)
    for _ in range(iters):
        grad = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j or games_matrix[i, j] == 0:
                    continue
                exp = expected_score(ratings[i], ratings[j])
                grad[i] += score_matrix[i, j] - games_matrix[i, j] * exp
        ratings += lr * grad / max(games_matrix.sum(), 1)
        ratings -= ratings.mean()
    ratings += anchor_mean
    return {name: float(round(r, 1)) for name, r in zip(names, ratings, strict=False)}
