"""Bellman optimality for finite turn-based zero-sum MDPs (Markov games).

The chess endgames here are two-player, so a state is solved with the *game*
Bellman operator: maximiser states take ``max`` over successors, minimiser
states take ``min``. A single-agent MDP is the special case where every state
is a maximiser (``is_max`` all True), which is what the textbook value/policy
iteration tests exercise.

States are stored in a CSR-style layout so each value-iteration sweep is a
couple of vectorised NumPy reductions (``np.maximum.reduceat`` /
``np.minimum.reduceat``) rather than a Python loop over states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GameMDP:
    """Vectorised transition model for a finite turn-based MDP.

    Attributes
    ----------
    is_max: per-state flag; True for maximiser (the mating side) to move.
    terminal: per-state terminal flag.
    terminal_value: value of terminal states (0 for non-terminals); the
        non-terminal entries double as the value-iteration initialisation.
    nt: indices of non-terminal states, in CSR segment order.
    succ_flat: concatenated successor state-indices for every ``nt`` state.
    seg_starts: offset into ``succ_flat`` where each ``nt`` state's block begins.
    """

    is_max: np.ndarray
    terminal: np.ndarray
    terminal_value: np.ndarray
    nt: np.ndarray
    succ_flat: np.ndarray
    seg_starts: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.is_max.shape[0])


def value_iteration(
    mdp: GameMDP,
    *,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iters: int = 1000,
) -> tuple[np.ndarray, list[float]]:
    """Run game value iteration; return ``(V, delta_history)``.

    ``delta_history`` is the per-sweep ``||V_{k+1} - V_k||_inf`` trace, which is
    the empirical proof of contraction/convergence.
    """
    V = mdp.terminal_value.astype(np.float64).copy()
    ismax_nt = mdp.is_max[mdp.nt]
    history: list[float] = []
    for _ in range(max_iters):
        contrib = gamma * V[mdp.succ_flat]
        seg_max = np.maximum.reduceat(contrib, mdp.seg_starts)
        seg_min = np.minimum.reduceat(contrib, mdp.seg_starts)
        new_v = np.where(ismax_nt, seg_max, seg_min)
        delta = float(np.max(np.abs(new_v - V[mdp.nt]))) if mdp.nt.size else 0.0
        V[mdp.nt] = new_v
        history.append(delta)
        if delta < theta:
            break
    return V, history


def greedy_offsets(mdp: GameMDP, V: np.ndarray, *, gamma: float) -> np.ndarray:
    """For each non-terminal state, the offset into ``succ_flat`` it would pick."""
    contrib = gamma * V[mdp.succ_flat]
    seg_ends = np.append(mdp.seg_starts[1:], mdp.succ_flat.shape[0])
    ismax_nt = mdp.is_max[mdp.nt]
    out = np.empty(mdp.nt.shape[0], dtype=np.int64)
    for i, (lo, hi, is_max) in enumerate(zip(mdp.seg_starts, seg_ends, ismax_nt, strict=False)):
        block = contrib[lo:hi]
        rel = int(np.argmax(block) if is_max else np.argmin(block))
        out[i] = lo + rel
    return out


def policy_iteration(
    mdp: GameMDP,
    *,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iters: int = 1000,
    eval_iters: int = 1000,
) -> tuple[np.ndarray, int]:
    """Policy iteration for the turn-based game; return ``(V, n_sweeps)``."""
    chosen = mdp.seg_starts.copy()  # start with each state's first successor
    V = mdp.terminal_value.astype(np.float64).copy()
    for sweep in range(1, max_iters + 1):
        # --- policy evaluation (iterative) ---
        succ_states = mdp.succ_flat[chosen]
        for _ in range(eval_iters):
            new_v_nt = gamma * V[succ_states]
            if np.max(np.abs(new_v_nt - V[mdp.nt])) < theta:
                V[mdp.nt] = new_v_nt
                break
            V[mdp.nt] = new_v_nt
        # --- policy improvement ---
        new_chosen = greedy_offsets(mdp, V, gamma=gamma)
        if np.array_equal(new_chosen, chosen):
            return V, sweep
        chosen = new_chosen
    return V, max_iters


def q_value_iteration(
    mdp: GameMDP,
    *,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iters: int = 1000,
) -> np.ndarray:
    """Q-value iteration for the single-agent case; returns Q over ``succ_flat``.

    ``Q[e]`` is the action-value of taking the transition stored at offset ``e``.
    Requires every state to be a maximiser (a single-agent MDP).
    """
    if not bool(mdp.is_max[mdp.nt].all()):
        raise ValueError("q_value_iteration expects a single-agent (all-max) MDP")
    V, _ = value_iteration(mdp, gamma=gamma, theta=theta, max_iters=max_iters)
    return gamma * V[mdp.succ_flat]
