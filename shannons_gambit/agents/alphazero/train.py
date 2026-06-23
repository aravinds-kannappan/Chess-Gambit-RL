"""AlphaZero-lite training loop: self-play -> learn -> repeat.

Small but real: bootstraps from the supervised checkpoint (``init_from``) so a
laptop run refines a sensible network rather than starting from noise. Each
iteration generates self-play games with the current net, then trains the
policy head toward the MCTS visit distribution and the value head toward the
game outcome. The ``cloud`` preset scales sims/games/net without code changes.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...config import AlphaZeroConfig
from ...logging_utils import JsonlLogger
from ...models.net import ChessNet, load_model, save_model
from ...seeding import seed_everything
from ...torch_utils import resolve_device
from .mcts import MCTS
from .selfplay import Example, play_game


def _init_model(cfg: AlphaZeroConfig, device: str) -> ChessNet:
    if cfg.init_from and Path(cfg.init_from).exists():
        model, _ = load_model(cfg.init_from, map_location=device)
        return model.to(device)
    return ChessNet(channels=cfg.net.channels, blocks=cfg.net.blocks).to(device)


def train_alphazero(cfg: AlphaZeroConfig) -> dict:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    model = _init_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    logger = JsonlLogger(Path(cfg.run_dir), config=cfg)
    replay: deque[Example] = deque(maxlen=cfg.buffer_games * cfg.max_moves)
    history: list[dict] = []

    for it in range(1, cfg.iters + 1):
        model.eval()
        mcts = MCTS(model, device=device, c_puct=cfg.c_puct,
                    dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_eps=cfg.dirichlet_eps)
        new_examples = 0
        outcomes = []
        for _ in range(cfg.games_per_iter):
            game = play_game(
                mcts, simulations=cfg.simulations,
                temperature_moves=cfg.temperature_moves, max_moves=cfg.max_moves, rng=rng,
            )
            replay.extend(game)
            new_examples += len(game)
            outcomes.append(game[0].value if game else 0.0)

        loss_stats = _train_on_replay(model, opt, replay, cfg, device)
        metrics = {
            "iter": it,
            "new_examples": new_examples,
            "replay": len(replay),
            "avg_game_len": round(new_examples / max(cfg.games_per_iter, 1), 1),
            **loss_stats,
        }
        logger.log(**metrics)
        history.append(metrics)
        save_model(model, str(Path(cfg.run_dir) / "model.pt"),
                   extra={"iter": it, "metrics": metrics})

    logger.close()
    return {"run_dir": cfg.run_dir, "history": history}


def _train_on_replay(model: ChessNet, opt, replay, cfg, device) -> dict:
    if len(replay) < cfg.batch_size:
        return {"loss": None, "loss_policy": None, "loss_value": None}
    model.train()
    data = list(replay)
    rng = np.random.default_rng(0)
    n_batches = max(1, (len(data) // cfg.batch_size)) * cfg.epochs_per_iter
    tot_p = tot_v = tot = 0.0
    for _ in range(n_batches):
        idx = rng.integers(0, len(data), size=cfg.batch_size)
        batch = [data[i] for i in idx]
        x = torch.from_numpy(np.stack([b.state for b in batch])).float().to(device)
        pi = torch.from_numpy(np.stack([b.policy for b in batch])).float().to(device)
        z = torch.tensor([b.value for b in batch], dtype=torch.float32, device=device)
        out = model(x)
        logp = torch.log_softmax(out["policy"], dim=1)
        loss_policy = -(pi * logp).sum(dim=1).mean()
        loss_value = ((out["value"] - z) ** 2).mean()
        loss = loss_policy + loss_value
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot_p += float(loss_policy.detach())
        tot_v += float(loss_value.detach())
        tot += float(loss.detach())
    return {
        "loss": round(tot / n_batches, 4),
        "loss_policy": round(tot_p / n_batches, 4),
        "loss_value": round(tot_v / n_batches, 4),
    }
