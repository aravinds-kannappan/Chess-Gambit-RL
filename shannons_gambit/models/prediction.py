"""Inference wrapper around a trained :class:`ChessNet` checkpoint.

Powers the CLI ``predict`` command, the neural arena agent, and the JSON the
web app expects (best move, win/draw/loss, value, estimated rating).
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
import torch

from ..data.encode import encode_board, index_to_move, legal_policy_mask, move_to_index
from .net import ChessNet, load_model


@dataclass
class Prediction:
    best_move: str
    top_moves: list[tuple[str, float]]
    value: float
    wdl: dict[str, float]
    rating: float

    def to_dict(self) -> dict:
        return {
            "best_move": self.best_move,
            "top_moves": [{"uci": u, "prob": round(p, 4)} for u, p in self.top_moves],
            "value": round(self.value, 4),
            "wdl": {k: round(v, 4) for k, v in self.wdl.items()},
            "rating": round(self.rating, 1),
        }


class Predictor:
    def __init__(self, model: ChessNet, extra: dict, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.rating_mean = float(extra.get("rating_mean", 1500.0))
        self.rating_std = float(extra.get("rating_std", 1.0))

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> Predictor:
        model, extra = load_model(path, map_location=device)
        return cls(model, extra, device=device)

    @torch.no_grad()
    def _forward(self, board: chess.Board) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(encode_board(board)).float().unsqueeze(0).to(self.device)
        return self.model(x)

    def policy_distribution(self, board: chess.Board) -> dict[chess.Move, float]:
        """Legal-masked softmax over the policy head."""
        out = self._forward(board)
        logits = out["policy"][0].cpu().numpy()
        mask = legal_policy_mask(board)
        masked = np.where(mask, logits, -1e9)
        masked = masked - masked.max()
        probs = np.exp(masked)
        probs[~mask] = 0.0
        probs /= probs.sum()
        dist: dict[chess.Move, float] = {}
        for move in board.legal_moves:
            dist[move] = float(probs[move_to_index(move)])
        return dist

    def predict(self, board: chess.Board, *, top_k: int = 5) -> Prediction:
        out = self._forward(board)
        dist = self.policy_distribution(board)
        ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        best = ordered[0][0] if ordered else None
        wdl_logits = out["wdl"][0].cpu().numpy()
        wdl = np.exp(wdl_logits - wdl_logits.max())
        wdl /= wdl.sum()
        rating = float(out["rating"][0].cpu()) * self.rating_std + self.rating_mean
        return Prediction(
            best_move=best.uci() if best else "",
            top_moves=[(m.uci(), p) for m, p in ordered[:top_k]],
            value=float(out["value"][0].cpu()),
            wdl={"loss": float(wdl[0]), "draw": float(wdl[1]), "win": float(wdl[2])},
            rating=rating,
        )


def index_to_uci(index: int, board: chess.Board) -> str | None:
    move = index_to_move(index, board)
    return move.uci() if move is not None else None
