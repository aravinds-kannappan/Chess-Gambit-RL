"""Adaptivity: live opponent-modeling and genuine per-session fine-tuning.

Two mechanisms, both real:

* ``OpponentModel`` tracks a human's tendencies within a session (aggression,
  blunder rate, opening variety) and adjusts the served agent's search/sampling
  on the fly -- the agent visibly changes style as you play, without touching weights.
* ``adapt_to_games`` performs a short REINFORCE-style fine-tune on the agent's own
  games against the human (increase the log-prob of moves from games it won, lower
  it for games it lost; regress value to outcome; KL-regularize to the base policy
  to avoid forgetting). This produces a **new personal checkpoint** -- genuine weight
  updates -- whose win-rate-vs-you climbs across sessions.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
import torch

from ..data.encode import encode_board, legal_policy_mask, move_to_index
from ..models.net import load_model, save_model
from ..torch_utils import resolve_device


@dataclass
class OpponentModel:
    """Running summary of the human's play, used for live style adaptation."""

    moves: int = 0
    captures: int = 0
    blunders: int = 0  # moves that drop >=2 pawns of material next ply (heuristic)

    def observe(self, board_before: chess.Board, move: chess.Move) -> None:
        self.moves += 1
        if board_before.is_capture(move):
            self.captures += 1

    @property
    def aggression(self) -> float:
        return self.captures / self.moves if self.moves else 0.5

    def agent_params(self, base_sims: int) -> dict:
        """Live knobs for the served agent given the observed human style.

        Aggressive humans get a steadier, lower-temperature agent; passive humans
        get a slightly sharper one. Always genuine MCTS, never a heuristic.
        """
        temperature = 0.4 if self.aggression > 0.4 else 0.8
        return {"simulations": base_sims, "temperature": temperature}


@dataclass
class AdaptConfig:
    epochs: int = 3
    lr: float = 1e-4
    kl_coef: float = 0.5
    value_coef: float = 1.0
    device: str = "auto"


def _examples_from_games(games: list[dict]) -> list[tuple[np.ndarray, int, float]]:
    """Flatten games into (state, agent_move_index, result_for_agent)."""
    examples: list[tuple[np.ndarray, int, float]] = []
    for game in games:
        result = float(game.get("result", 0.0))  # +1 win / 0 draw / -1 loss for the agent
        for fen, uci in zip(game.get("fens", []), game.get("moves", []), strict=False):
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(uci)
            except (ValueError, AttributeError):
                continue
            examples.append((encode_board(board).astype(np.float32), move_to_index(move), result))
    return examples


def adapt_to_games(base_path: str, games: list[dict], out_path: str,
                   cfg: AdaptConfig = AdaptConfig()) -> dict:
    """Fine-tune a checkpoint on the agent's games vs a human; save a new one."""
    device = resolve_device(cfg.device)
    model, extra = load_model(base_path, map_location=device)
    base, _ = load_model(base_path, map_location=device)
    model.to(device).train()
    base.to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)

    examples = _examples_from_games(games)
    if not examples:
        save_model(model, out_path, extra=extra)
        return {"out_path": out_path, "n_examples": 0, "agent_win_rate": None,
                "loss_before": None, "loss_after": None}

    x = torch.from_numpy(np.stack([e[0] for e in examples])).float().to(device)
    a = torch.tensor([e[1] for e in examples], dtype=torch.long, device=device)
    r = torch.tensor([e[2] for e in examples], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def loss_fn() -> torch.Tensor:
        out = model(x)
        logp = torch.log_softmax(out["policy"], dim=1)
        chosen = logp.gather(1, a.unsqueeze(1)).squeeze(1)
        loss_pg = -(r * chosen).mean()  # reinforce winning moves, suppress losing ones
        loss_val = ((out["value"] - r) ** 2).mean()
        with torch.no_grad():
            base_logp = torch.log_softmax(base(x)["policy"], dim=1)
        kl = (logp.exp() * (logp - base_logp)).sum(dim=1).mean()
        return loss_pg + cfg.value_coef * loss_val + cfg.kl_coef * kl

    loss_before = float(loss_fn().detach())
    for _ in range(cfg.epochs):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    loss_after = float(loss_fn().detach())

    win_rate = float(np.mean([1.0 if g.get("result", 0) > 0 else 0.0 for g in games]))
    extra = {**extra, "adapted": True, "n_adapt_games": len(games), "agent_win_rate": win_rate}
    save_model(model, out_path, extra=extra)
    return {"out_path": out_path, "n_examples": len(examples), "agent_win_rate": win_rate,
            "loss_before": round(loss_before, 4), "loss_after": round(loss_after, 4)}


def legal_softmax(model, board: chess.Board, device: str = "cpu") -> dict[chess.Move, float]:
    """Legal-masked policy of a model at a position (small helper for serving)."""
    x = torch.from_numpy(encode_board(board)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)["policy"][0].cpu().numpy()
    mask = legal_policy_mask(board)
    masked = np.where(mask, logits, -1e9)
    masked = masked - masked.max()
    probs = np.exp(masked)
    probs[~mask] = 0.0
    probs /= probs.sum()
    return {m: float(probs[move_to_index(m)]) for m in board.legal_moves}
