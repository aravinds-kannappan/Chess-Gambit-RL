"""PUCT Monte-Carlo Tree Search guided by :class:`ChessNet`.

Standard AlphaZero search: priors and a leaf value come from the network, the
selection rule is ``Q + c_puct * P * sqrt(N_parent) / (1 + N_child)``, and the
backup alternates sign each ply (values are stored from each node's
side-to-move perspective). Returns visit-count move probabilities.
"""

from __future__ import annotations

import math

import chess
import numpy as np
import torch

from ...data.encode import encode_board, legal_policy_mask, move_to_index
from ..base import Agent


class _Node:
    __slots__ = ("prior", "n", "w", "children", "expanded")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.n = 0
        self.w = 0.0
        self.children: dict[chess.Move, _Node] = {}
        self.expanded = False

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


def _terminal_value(board: chess.Board) -> float:
    """Value from the side-to-move perspective for a finished game."""
    if board.is_checkmate():
        return -1.0
    return 0.0


class MCTS:
    def __init__(self, model, *, device: str = "cpu", c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        x = torch.from_numpy(encode_board(board)).float().unsqueeze(0).to(self.device)
        out = self.model(x)
        logits = out["policy"][0].cpu().numpy()
        value = float(out["value"][0].cpu())
        mask = legal_policy_mask(board)
        masked = np.where(mask, logits, -1e9)
        masked = masked - masked.max()
        probs = np.exp(masked)
        probs[~mask] = 0.0
        probs /= probs.sum()
        priors = {m: float(probs[move_to_index(m)]) for m in board.legal_moves}
        return priors, value

    def _expand(self, node: _Node, board: chess.Board) -> float:
        priors, value = self._evaluate(board)
        for move, p in priors.items():
            node.children[move] = _Node(p)
        node.expanded = True
        return value

    def _select(self, node: _Node) -> tuple[chess.Move, _Node]:
        sqrt_n = math.sqrt(node.n)
        best_score = -1e30
        best: tuple[chess.Move, _Node] | None = None
        for move, child in node.children.items():
            q = -child.q  # child Q is from the opponent's perspective
            u = self.c_puct * child.prior * sqrt_n / (1 + child.n)
            score = q + u
            if score > best_score:
                best_score, best = score, (move, child)
        assert best is not None
        return best

    def run(self, board: chess.Board, *, simulations: int,
            add_noise: bool = False) -> dict[chess.Move, int]:
        root = _Node(0.0)
        self._expand(root, board)
        if add_noise and root.children:
            self._add_dirichlet(root)
        for _ in range(simulations):
            sim_board = board.copy()
            node = root
            path = [node]
            while node.expanded and node.children and not sim_board.is_game_over():
                move, node = self._select(node)
                sim_board.push(move)
                path.append(node)
            if sim_board.is_game_over():
                value = _terminal_value(sim_board)
            else:
                value = self._expand(node, sim_board)
            for n in reversed(path):
                n.n += 1
                n.w += value
                value = -value
        return {move: child.n for move, child in root.children.items()}

    def _add_dirichlet(self, root: _Node) -> None:
        moves = list(root.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        for move, eta in zip(moves, noise, strict=False):
            child = root.children[move]
            child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * float(eta)


class AlphaZeroAgent(Agent):
    """MCTS agent with strength knobs for Elo calibration.

    Strength is lowered by reducing ``simulations``, raising ``temperature``
    (sample softer from visit counts), or injecting a ``blunder_rate`` of uniform
    random moves -- so one checkpoint can be tuned to a target Elo between ladder
    snapshots.
    """

    name = "alphazero"

    def __init__(self, model, *, device: str = "cpu", simulations: int = 64,
                 c_puct: float = 1.5, temperature: float = 0.0,
                 blunder_rate: float = 0.0, name: str | None = None,
                 seed: int = 0) -> None:
        self.mcts = MCTS(model, device=device, c_puct=c_puct)
        self.simulations = simulations
        self.temperature = temperature
        self.blunder_rate = blunder_rate
        self._rng = np.random.default_rng(seed)
        if name:
            self.name = name

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu", **kw) -> AlphaZeroAgent:
        from ...models.net import load_model

        model, _ = load_model(path, map_location=device)
        return cls(model, device=device, **kw)

    def select_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if self.blunder_rate > 0 and self._rng.random() < self.blunder_rate:
            return legal[int(self._rng.integers(len(legal)))]
        visits = self.mcts.run(board, simulations=self.simulations, add_noise=False)
        moves = list(visits)
        counts = np.array([visits[m] for m in moves], dtype=np.float64)
        if self.temperature <= 0 or counts.sum() == 0:
            return moves[int(counts.argmax())]
        weights = counts ** (1.0 / self.temperature)
        probs = weights / weights.sum()
        return moves[int(self._rng.choice(len(moves), p=probs))]
