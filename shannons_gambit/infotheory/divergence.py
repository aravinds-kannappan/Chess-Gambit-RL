"""Divergences and mutual information used to compare policies and features."""

from __future__ import annotations

import numpy as np

from .entropy import shannon_entropy

_EPS = 1e-12


def kl_divergence(p: np.ndarray, q: np.ndarray, *, base: float = 2.0) -> float:
    """D_KL(p || q); how many extra bits using q to code samples from p."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    p = p[mask]
    q = np.clip(q[mask], _EPS, None)
    return float((p * (np.log(p / q) / np.log(base))).sum())


def js_divergence(p: np.ndarray, q: np.ndarray, *, base: float = 2.0) -> float:
    """Jensen-Shannon divergence: symmetric, bounded in ``[0, 1]`` bits."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m, base=base) + 0.5 * kl_divergence(q, m, base=base))


def mutual_information(joint: np.ndarray, *, base: float = 2.0) -> float:
    """I(X; Y) = H(X) + H(Y) - H(X, Y) from a joint count/probability matrix."""
    joint = np.asarray(joint, dtype=np.float64)
    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    h_x = shannon_entropy(px, base=base)
    h_y = shannon_entropy(py, base=base)
    h_xy = shannon_entropy(joint.ravel(), base=base)
    return float(max(h_x + h_y - h_xy, 0.0))


def mutual_information_from_samples(
    x: np.ndarray, y: np.ndarray, *, n_bins: int = 8, base: float = 2.0
) -> float:
    """Estimate I(X; Y) from paired samples via histogram binning.

    ``x`` is binned into ``n_bins`` quantiles; ``y`` is treated as a discrete
    label (e.g., game outcome in {-1, 0, 1}).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    x_binned = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    labels = np.unique(y)
    joint = np.zeros((n_bins, labels.size), dtype=np.float64)
    for j, lab in enumerate(labels):
        sel = y == lab
        if sel.any():
            joint[:, j] = np.bincount(x_binned[sel], minlength=n_bins)
    return mutual_information(joint, base=base)
