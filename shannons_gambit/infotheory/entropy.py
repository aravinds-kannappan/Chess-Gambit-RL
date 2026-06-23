"""Core Shannon information measures (bits by default).

Everything operates on NumPy probability vectors/distributions and is unit
tested against analytic identities (uniform entropy = log2 n, etc.).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def normalize(counts: np.ndarray) -> np.ndarray:
    """Turn non-negative counts/weights into a probability distribution."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot normalize a zero/negative total")
    return counts / total


def shannon_entropy(p: np.ndarray, *, base: float = 2.0) -> float:
    """H(p) = -sum p log p, ignoring zero-probability outcomes."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * (np.log(p) / np.log(base))).sum())


def cross_entropy(p: np.ndarray, q: np.ndarray, *, base: float = 2.0) -> float:
    """H(p, q) = -sum p log q; the score a model q pays under truth p."""
    p = np.asarray(p, dtype=np.float64)
    q = np.clip(np.asarray(q, dtype=np.float64), _EPS, None)
    return float(-(p * (np.log(q) / np.log(base))).sum())


def perplexity(p: np.ndarray, q: np.ndarray) -> float:
    """2 ** cross-entropy(p, q) in bits; the model's effective branching."""
    return float(2.0 ** cross_entropy(p, q, base=2.0))


def conditional_entropy(joint: np.ndarray, *, base: float = 2.0) -> float:
    """H(Y | X) from a joint count/probability matrix ``joint[x, y]``."""
    joint = np.asarray(joint, dtype=np.float64)
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(joint > 0, joint * (np.log(px / joint) / np.log(base)), 0.0)
    return float(cond.sum())
