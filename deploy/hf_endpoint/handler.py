"""Hugging Face Inference Endpoint handler for the Shannon's Gambit model.

Deploy this alongside ``model.pt`` and ``requirements.txt`` in a Hugging Face
model repo, then create an Inference Endpoint pointing at that repo. The endpoint
receives ``{"inputs": {"fen": "<FEN>"}}`` and returns the best move, win/draw/loss,
value, estimated rating, and policy entropy - exactly the shape the web app's
``web/app/lib/hf.ts`` expects.

The encoding/model code is reused from the published package (installed via
``requirements.txt``), so there is no risk of representation drift.
"""

from __future__ import annotations

import os
from typing import Any

import chess
import numpy as np

from shannons_gambit.infotheory.analysis import move_entropy
from shannons_gambit.models.prediction import Predictor


class EndpointHandler:
    def __init__(self, path: str = "") -> None:
        # The served base net lives under pretrain/ (supervised behavioural
        # cloning); fall back to the legacy repo-root location.
        candidates = [os.path.join(path, "pretrain", "model.pt"), os.path.join(path, "model.pt")]
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])
        self.predictor = Predictor.from_checkpoint(model_path, device="cpu")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        inputs = data.get("inputs", data)
        fen = inputs.get("fen") if isinstance(inputs, dict) else inputs
        if not fen:
            return {"error": "provide inputs.fen (a FEN string)"}
        try:
            board = chess.Board(fen)
        except (ValueError, AttributeError) as exc:
            return {"error": f"invalid FEN: {exc}"}

        pred = self.predictor.predict(board)
        dist = self.predictor.policy_distribution(board) if board.legal_moves else {}
        entropy = move_entropy(np.array(list(dist.values()))) if dist else 0.0
        return {
            "best_move": pred.best_move,
            "wdl": pred.wdl,
            "value": pred.value,
            "rating": pred.rating,
            "policy_entropy_bits": float(entropy),
            "top_moves": [{"uci": u, "prob": p} for u, p in pred.top_moves],
        }
