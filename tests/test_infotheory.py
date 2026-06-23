"""Information-theoretic identities."""

from __future__ import annotations

import unittest

import numpy as np

from shannons_gambit.infotheory.divergence import (
    js_divergence,
    kl_divergence,
    mutual_information,
    mutual_information_from_samples,
)
from shannons_gambit.infotheory.entropy import (
    conditional_entropy,
    cross_entropy,
    normalize,
    perplexity,
    shannon_entropy,
)


class TestInfoTheory(unittest.TestCase):
    def test_uniform_entropy_is_log2_n(self):
        for n in (2, 4, 8, 16):
            self.assertAlmostEqual(shannon_entropy(np.ones(n) / n), np.log2(n), places=9)

    def test_kl_nonneg_and_zero_on_equal(self):
        p = normalize([3, 1, 1, 5])
        q = normalize([1, 1, 1, 1])
        self.assertAlmostEqual(kl_divergence(p, p), 0.0, places=12)
        self.assertGreater(kl_divergence(p, q), 0.0)

    def test_js_symmetric_bounded(self):
        p = normalize([3, 1, 1, 5])
        q = normalize([1, 4, 1, 1])
        self.assertAlmostEqual(js_divergence(p, q), js_divergence(q, p), places=12)
        self.assertGreaterEqual(js_divergence(p, q), 0.0)
        self.assertLessEqual(js_divergence(p, q), 1.0)

    def test_mutual_information_bounds(self):
        indep = np.outer([0.5, 0.5], [0.5, 0.5])
        self.assertAlmostEqual(mutual_information(indep), 0.0, places=9)
        dep = np.array([[0.5, 0.0], [0.0, 0.5]])
        self.assertAlmostEqual(mutual_information(dep), 1.0, places=9)

    def test_conditional_entropy_le_marginal(self):
        joint = np.array([[5.0, 1.0], [1.0, 3.0]])
        h_y = shannon_entropy(normalize(joint.sum(axis=0)))
        self.assertLessEqual(conditional_entropy(joint), h_y + 1e-9)

    def test_perplexity_uniform(self):
        u = np.ones(4) / 4
        self.assertAlmostEqual(perplexity(u, u), 4.0, places=6)
        self.assertAlmostEqual(cross_entropy(u, u), 2.0, places=6)

    def test_sample_mi_detects_dependence(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=4000)
        dep = mutual_information_from_samples(x, (x > 0).astype(int))
        indep = mutual_information_from_samples(rng.normal(size=4000),
                                                rng.integers(0, 2, 4000))
        self.assertGreater(dep, 0.5)
        self.assertLess(indep, 0.05)


if __name__ == "__main__":
    unittest.main()
