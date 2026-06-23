"""Continuous self-play: one resumable generation = self-play -> train -> rate.

A generation generates self-play games with the current network, trains on the
replay buffer, then measures the new network's strength on a **stable absolute
Elo scale** via an anchored gauntlet (a random agent pinned to a fixed Elo, plus
the previous best generation). The result is a versioned checkpoint registered on
the :class:`Ladder`, so strength can be tracked over time and served by target Elo.

``ContinualTrainer`` holds the state (model, optimizer, replay) so the HF Space can
call :meth:`step` repeatedly in a background worker; the Hub is the durable store.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

from ...config import ContinualConfig
from ...logging_utils import JsonlLogger
from ...models.net import ChessNet, load_model, save_model
from ...seeding import seed_everything
from ...torch_utils import resolve_device
from ..ladder import Ladder, LadderEntry
from ..random_agent import RandomAgent
from .mcts import AlphaZeroAgent
from .selfplay import Example, play_game


def _init_model(cfg: ContinualConfig, device: str) -> ChessNet:
    if cfg.init_from and Path(cfg.init_from).exists():
        model, _ = load_model(cfg.init_from, map_location=device)
        return model.to(device)
    return ChessNet(channels=cfg.net.channels, blocks=cfg.net.blocks).to(device)


_PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _material(board: chess.Board) -> int:
    return sum((_PIECE_VAL[p.piece_type] if p.color == chess.WHITE else -_PIECE_VAL[p.piece_type])
               for p in board.piece_map().values())


def _play_white_score(white, black, max_moves: int) -> float:
    """Play one game; adjudicate unfinished games by material (sensitive signal)."""
    board = chess.Board()
    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        agent = white if board.turn == chess.WHITE else black
        board.push(agent.select_move(board))
    if board.is_checkmate():
        return 0.0 if board.turn == chess.WHITE else 1.0
    m = _material(board)
    return 1.0 if m > 0.5 else (0.0 if m < -0.5 else 0.5)


def _match_points(a, b, n: int, max_moves: int) -> float:
    """Points scored by ``a`` over ``n`` games vs ``b`` (alternating colors).

    Unfinished games are adjudicated by material, so a learning agent that wins
    material against a weak opponent registers progress even before it can force
    checkmate -- the signal that drives a visible Elo climb.
    """
    pts = 0.0
    for i in range(n):
        if i % 2 == 0:
            pts += _play_white_score(a, b, max_moves)
        else:
            pts += 1.0 - _play_white_score(b, a, max_moves)
    return pts


class ContinualTrainer:
    def __init__(self, cfg: ContinualConfig) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.model = _init_model(cfg, self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)
        self.replay: deque[Example] = deque(maxlen=cfg.buffer_games * cfg.max_moves)
        self.ladder = Ladder.load(cfg.run_dir, random_anchor_elo=cfg.random_anchor_elo)
        self.logger = JsonlLogger(Path(cfg.run_dir), config=cfg)

    # --- one generation ----------------------------------------------------
    def step(self) -> LadderEntry:
        cfg = self.cfg
        gen = self.ladder.next_gen()

        # 1) self-play with the current network
        from .mcts import MCTS

        mcts = MCTS(self.model, device=self.device, c_puct=cfg.c_puct,
                    dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_eps=cfg.dirichlet_eps)
        self.model.eval()
        new_examples = 0
        for _ in range(cfg.games_per_gen):
            game = play_game(mcts, simulations=cfg.simulations,
                             temperature_moves=cfg.temperature_moves,
                             max_moves=cfg.max_moves, rng=self.rng)
            self.replay.extend(game)
            new_examples += len(game)

        # 2) train on the replay buffer
        loss_stats = self._train()

        # 3) anchored-Elo evaluation
        elo, win_rate_random = self._evaluate()

        # 4) version the checkpoint and register it on the ladder
        ckpt = Path(cfg.run_dir) / f"gen-{gen:04d}.pt"
        metrics = {
            "loss_policy": loss_stats.get("loss_policy"),
            "loss_value": loss_stats.get("loss_value"),
            "win_rate_vs_random": round(win_rate_random, 3),
            "new_examples": new_examples,
        }
        save_model(self.model, str(ckpt), extra={"gen": gen, "elo": elo, "metrics": metrics})
        entry = self.ladder.add(gen, str(ckpt), elo, metrics)
        self.ladder.save()
        # keep a "latest.pt" pointer for convenience
        save_model(self.model, str(Path(cfg.run_dir) / "latest.pt"),
                   extra={"gen": gen, "elo": elo, "metrics": metrics})
        self.logger.log(gen=gen, elo=elo, **metrics)
        return entry

    def run(self, n_gens: int) -> list[LadderEntry]:
        return [self.step() for _ in range(n_gens)]

    # --- internals ---------------------------------------------------------
    def _train(self) -> dict:
        cfg = self.cfg
        if len(self.replay) < cfg.batch_size:
            return {"loss_policy": None, "loss_value": None}
        self.model.train()
        data = list(self.replay)
        n_batches = max(1, len(data) // cfg.batch_size) * cfg.epochs_per_gen
        tot_p = tot_v = 0.0
        for _ in range(n_batches):
            idx = self.rng.integers(0, len(data), size=cfg.batch_size)
            batch = [data[i] for i in idx]
            x = torch.from_numpy(np.stack([b.state for b in batch])).float().to(self.device)
            pi = torch.from_numpy(np.stack([b.policy for b in batch])).float().to(self.device)
            z = torch.tensor([b.value for b in batch], dtype=torch.float32, device=self.device)
            out = self.model(x)
            logp = torch.log_softmax(out["policy"], dim=1)
            loss_policy = -(pi * logp).sum(dim=1).mean()
            loss_value = ((out["value"] - z) ** 2).mean()
            loss = loss_policy + loss_value
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            tot_p += float(loss_policy.detach())
            tot_v += float(loss_value.detach())
        return {"loss_policy": round(tot_p / n_batches, 4),
                "loss_value": round(tot_v / n_batches, 4)}

    def _evaluate(self) -> tuple[float, float]:
        """Return (anchored Elo, win-rate vs random)."""
        cfg = self.cfg
        self.model.eval()
        new_agent = AlphaZeroAgent(self.model, device=self.device, simulations=cfg.eval_sims)
        random_agent = RandomAgent(seed=123)

        anchors: list[tuple[object, float]] = [(random_agent, cfg.random_anchor_elo)]
        best = self.ladder.best()
        if best is not None and Path(best.path).exists():
            prev_model, _ = load_model(best.path, map_location=self.device)
            anchors.append(
                (AlphaZeroAgent(prev_model.to(self.device), device=self.device,
                                simulations=cfg.eval_sims), best.elo)
            )

        from ...eval.elo import estimate_rating

        results = []
        win_rate_random = 0.5
        for agent, anchor_elo in anchors:
            pts = _match_points(new_agent, agent, cfg.eval_games, cfg.max_moves)
            results.append((anchor_elo, pts, cfg.eval_games))
            if agent is random_agent:
                win_rate_random = pts / cfg.eval_games
        init = best.elo if best is not None else (cfg.random_anchor_elo + 200)
        elo = estimate_rating(results, init=init)
        return elo, win_rate_random


def run_generations(cfg: ContinualConfig, n_gens: int) -> dict:
    """Convenience: run ``n_gens`` generations and return the Elo curve."""
    trainer = ContinualTrainer(cfg)
    trainer.run(n_gens)
    trainer.logger.close()
    return {"run_dir": cfg.run_dir, "elo_curve": trainer.ladder.elo_curve(),
            "levels": trainer.ladder.levels()}
