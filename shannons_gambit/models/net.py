"""The shared residual network: a chess backbone with four heads.

* policy: distribution over the 4672 move indices (behavioural cloning / MCTS prior)
* value: scalar tanh evaluation in ``[-1, 1]`` (AlphaZero-style)
* wdl: win/draw/loss logits (the outcome-prediction head)
* rating: scalar Elo estimate (standardised target)

The same class serves supervised pretraining, the prediction model, and the
AlphaZero-lite agent; ``init_from`` lets self-play resume from the trained
supervised weights.
"""

from __future__ import annotations

import torch
from torch import nn

from ..data.encode import NUM_PLANES, POLICY_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class ChessNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, POLICY_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 64, 128),
            nn.ReLU(inplace=True),
        )
        self.value_out = nn.Linear(128, 1)
        self.wdl_out = nn.Linear(128, 3)
        self.rating_out = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.tower(self.stem(x))
        z = self.value_head(h)
        return {
            "policy": self.policy_head(h),
            "value": torch.tanh(self.value_out(z)).squeeze(-1),
            "wdl": self.wdl_out(z),
            "rating": self.rating_out(z).squeeze(-1),
        }

    @property
    def config(self) -> dict:
        return {"channels": self.channels, "blocks": self.blocks}


def save_model(model: ChessNet, path: str, *, extra: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "config": model.config}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_model(path: str, *, map_location: str = "cpu") -> tuple[ChessNet, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = ChessNet(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("extra", {})
