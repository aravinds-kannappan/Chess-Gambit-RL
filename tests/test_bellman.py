"""Bellman value/policy/Q iteration on small hand-built MDPs."""

from __future__ import annotations

import unittest

import numpy as np

from shannons_gambit.mdp.bellman import (
    GameMDP,
    policy_iteration,
    q_value_iteration,
    value_iteration,
)


def _chain_mdp() -> GameMDP:
    """s0 -> s1 -> terminal(value=1); single agent, gamma test."""
    return GameMDP(
        is_max=np.array([True, True, True]),
        terminal=np.array([False, False, True]),
        terminal_value=np.array([0.0, 0.0, 1.0]),
        nt=np.array([0, 1]),
        succ_flat=np.array([1, 2]),
        seg_starts=np.array([0, 1]),
    )


def _choice_mdp(is_max: bool) -> GameMDP:
    """s0 picks between terminal(0) and terminal(1)."""
    return GameMDP(
        is_max=np.array([is_max, True, True]),
        terminal=np.array([False, True, True]),
        terminal_value=np.array([0.0, 0.0, 1.0]),
        nt=np.array([0]),
        succ_flat=np.array([1, 2]),
        seg_starts=np.array([0]),
    )


class TestBellman(unittest.TestCase):
    def test_value_iteration_discounting(self):
        V, hist = value_iteration(_chain_mdp(), gamma=0.9, theta=1e-12)
        self.assertAlmostEqual(V[2], 1.0, places=6)
        self.assertAlmostEqual(V[1], 0.9, places=5)
        self.assertAlmostEqual(V[0], 0.81, places=5)
        # contraction: deltas are monotone non-increasing toward 0.
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(hist, hist[1:])))

    def test_max_vs_min(self):
        Vmax, _ = value_iteration(_choice_mdp(True), gamma=1.0, theta=1e-12)
        Vmin, _ = value_iteration(_choice_mdp(False), gamma=1.0, theta=1e-12)
        self.assertAlmostEqual(Vmax[0], 1.0, places=6)  # maximiser grabs the win
        self.assertAlmostEqual(Vmin[0], 0.0, places=6)  # minimiser avoids it

    def test_policy_iteration_matches_value_iteration(self):
        mdp = _chain_mdp()
        V_vi, _ = value_iteration(mdp, gamma=0.9, theta=1e-12)
        V_pi, sweeps = policy_iteration(mdp, gamma=0.9)
        self.assertTrue(np.allclose(V_vi, V_pi, atol=1e-5))
        self.assertGreaterEqual(sweeps, 1)

    def test_q_value_iteration_single_agent(self):
        q = q_value_iteration(_chain_mdp(), gamma=0.9, theta=1e-12)
        # Q of the only action at s0 equals gamma * V(s1) = 0.9 * 0.9.
        self.assertAlmostEqual(q[0], 0.81, places=5)


if __name__ == "__main__":
    unittest.main()
