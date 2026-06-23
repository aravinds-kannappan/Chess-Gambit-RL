"""Supervised pretraining on real games: behavioural cloning + prediction heads.

Trains :class:`ChessNet` jointly on the move (policy), result (value + WDL),
and mover-Elo (rating) targets. The resulting checkpoint is both the
HuggingFace prediction model and the bootstrap for AlphaZero-lite self-play.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import SupervisedConfig
from ..data.dataset import PositionDataset, load_records
from ..logging_utils import JsonlLogger
from ..seeding import seed_everything
from ..torch_utils import resolve_device
from .net import ChessNet, save_model


def train_supervised(cfg: SupervisedConfig) -> dict:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    records = load_records(cfg.data_dir, max_positions=cfg.max_positions)
    dataset = PositionDataset(records)
    tensors = dataset.to_torch()
    loader = torch.utils.data.DataLoader(
        tensors, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )

    model = ChessNet(channels=cfg.net.channels, blocks=cfg.net.blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    policy_loss = nn.CrossEntropyLoss()
    wdl_loss = nn.CrossEntropyLoss()

    run_dir = Path(cfg.run_dir)
    logger = JsonlLogger(run_dir, config=cfg)
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        agg = _Accum()
        for x, pol, val, wdl, rating, rmask in loader:
            x, pol, val = x.to(device), pol.to(device), val.to(device)
            wdl, rating, rmask = wdl.to(device), rating.to(device), rmask.to(device)
            out = model(x)
            l_pol = policy_loss(out["policy"], pol)
            l_val = ((out["value"] - val) ** 2).mean()
            l_wdl = wdl_loss(out["wdl"], wdl)
            denom = rmask.sum().clamp(min=1.0)
            l_rat = (((out["rating"] - rating) ** 2) * rmask).sum() / denom
            loss = l_pol + cfg.value_weight * l_val + l_wdl + cfg.rating_weight * l_rat

            opt.zero_grad()
            loss.backward()
            opt.step()
            agg.update(out, pol, val, wdl, rating, rmask, l_pol, l_val, l_wdl, l_rat, x.size(0))

        metrics = agg.summary(dataset.rating_std)
        metrics["epoch"] = epoch
        logger.log(**metrics)
        history.append(metrics)

    run_dir.mkdir(parents=True, exist_ok=True)
    save_model(
        model,
        str(run_dir / "model.pt"),
        extra={
            "rating_mean": dataset.rating_mean,
            "rating_std": dataset.rating_std,
            "n_positions": len(dataset),
            "final_metrics": history[-1] if history else {},
        },
    )
    logger.close()
    return {"run_dir": str(run_dir), "history": history, "n_positions": len(dataset)}


class _Accum:
    def __init__(self) -> None:
        self.n = 0
        self.sums = {k: 0.0 for k in ("pol", "val", "wdl", "rat", "p_correct", "w_correct")}
        self.rating_abs = 0.0
        self.rating_n = 0.0

    def update(self, out, pol, val, wdl, rating, rmask, l_pol, l_val, l_wdl, l_rat, bs) -> None:
        self.n += bs
        self.sums["pol"] += float(l_pol.detach()) * bs
        self.sums["val"] += float(l_val.detach()) * bs
        self.sums["wdl"] += float(l_wdl.detach()) * bs
        self.sums["rat"] += float(l_rat.detach()) * bs
        self.sums["p_correct"] += float((out["policy"].argmax(1) == pol).sum())
        self.sums["w_correct"] += float((out["wdl"].argmax(1) == wdl).sum())
        self.rating_abs += float((torch.abs(out["rating"].detach() - rating) * rmask).sum())
        self.rating_n += float(rmask.sum())

    def summary(self, rating_std: float) -> dict:
        n = max(self.n, 1)
        rn = max(self.rating_n, 1.0)
        return {
            "loss_policy": round(self.sums["pol"] / n, 4),
            "loss_value": round(self.sums["val"] / n, 4),
            "loss_wdl": round(self.sums["wdl"] / n, 4),
            "loss_rating": round(self.sums["rat"] / n, 4),
            "policy_acc": round(self.sums["p_correct"] / n, 4),
            "wdl_acc": round(self.sums["w_correct"] / n, 4),
            "rating_mae_elo": round((self.rating_abs / rn) * rating_std, 1),
        }


def _load_value_baseline(records: dict[str, np.ndarray]) -> float:
    """Majority-class WDL accuracy baseline (for context in reports)."""
    counts = np.bincount(records["stm_result"].astype(int), minlength=3)
    return float(counts.max() / counts.sum())
