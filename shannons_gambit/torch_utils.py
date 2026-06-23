"""Small PyTorch helpers (device resolution) shared by the trainers."""

from __future__ import annotations


def resolve_device(name: str = "auto") -> str:
    """Resolve ``"auto"`` to mps -> cuda -> cpu; otherwise return ``name``."""
    if name != "auto":
        return name
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
