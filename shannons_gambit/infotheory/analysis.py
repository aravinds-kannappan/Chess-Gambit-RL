"""Higher-level information-theoretic analyses over games and datasets.

These build on :mod:`entropy` / :mod:`divergence` and produce the series the
notebook and the web dashboard visualise: per-position move entropy, the drop
in outcome-uncertainty per ply (where games are decided), and the mutual
information between board features and the eventual result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .divergence import mutual_information_from_samples
from .entropy import normalize, shannon_entropy


def move_entropy(move_probs: np.ndarray, *, base: float = 2.0) -> float:
    """Entropy (bits) of the move distribution at a single position."""
    return shannon_entropy(normalize(move_probs), base=base)


def outcome_entropy(wdl: np.ndarray, *, base: float = 2.0) -> float:
    """Entropy of a win/draw/loss probability vector."""
    return shannon_entropy(normalize(wdl), base=base)


@dataclass(frozen=True)
class GameInfoProfile:
    """Per-ply outcome entropy and information gain for one game."""

    entropy: np.ndarray  # H(result | position) per ply, bits
    info_gain: np.ndarray  # entropy[t-1] - entropy[t], bits
    total_info: float  # entropy[0] - entropy[-1]
    decisive_ply: int  # ply with the largest single-step info gain


def game_info_profile(wdl_per_ply: np.ndarray, *, base: float = 2.0) -> GameInfoProfile:
    """Information profile of a game from per-ply W/D/L distributions.

    ``wdl_per_ply`` has shape ``(n_plies, 3)``. The total information gained is
    the collapse of outcome entropy from the opening to the result.
    """
    wdl_per_ply = np.asarray(wdl_per_ply, dtype=np.float64)
    ent = np.array([outcome_entropy(row, base=base) for row in wdl_per_ply])
    gain = np.zeros_like(ent)
    gain[1:] = ent[:-1] - ent[1:]
    decisive = int(np.argmax(gain)) if gain.size else 0
    total = float(ent[0] - ent[-1]) if ent.size else 0.0
    return GameInfoProfile(entropy=ent, info_gain=gain, total_info=total, decisive_ply=decisive)


def feature_outcome_mi(
    features: dict[str, np.ndarray], outcomes: np.ndarray, *, n_bins: int = 8
) -> dict[str, float]:
    """Mutual information (bits) between each scalar feature and the result.

    ``outcomes`` are discrete labels (e.g., -1/0/1). Returns a name -> I(X; Y)
    map, the ranking of which features actually carry winning information.
    """
    return {
        name: mutual_information_from_samples(values, outcomes, n_bins=n_bins)
        for name, values in features.items()
    }
