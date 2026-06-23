"""Fit Elo ratings from round-robin results via a Bradley-Terry MLE.

Draws count as half a win for each side. Ratings are fit by gradient ascent on
the logistic likelihood and anchored to a configurable mean for readability.
"""

from __future__ import annotations

import numpy as np

SCALE = 400.0
BASE = 10.0


def expected_score(r_i: float, r_j: float) -> float:
    return 1.0 / (1.0 + BASE ** ((r_j - r_i) / SCALE))


def estimate_rating(results: list[tuple[float, float, int]], *,
                    init: float = 1000.0, iters: int = 200, lr: float = 400.0) -> float:
    """Estimate one player's Elo against fixed-rating anchors (1-D logistic MLE).

    ``results`` is a list of ``(anchor_elo, score, games)`` where ``score`` is the
    player's total points (win=1, draw=0.5) over ``games`` games vs. that anchor.
    Because the anchors are held fixed (e.g. a random agent pinned to a known Elo
    plus prior generations), the returned rating is on a *stable absolute scale*
    comparable across generations. Gradient ascent on the Bradley-Terry log-likelihood.
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
        r += lr * grad / total_games
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
