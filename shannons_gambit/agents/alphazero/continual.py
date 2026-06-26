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

        # 3) gate the contender against the champion (+ vs-random sanity check)
        result = self._evaluate_and_gate()
        elo = result["elo"]

        # 4) version the checkpoint and register it on the ladder
        ckpt = Path(cfg.run_dir) / f"gen-{gen:04d}.pt"
        metrics = {
            "loss_policy": loss_stats.get("loss_policy"),
            "loss_value": loss_stats.get("loss_value"),
            "score_vs_champion": round(result["score_vs_champion"], 3),
            "win_rate_vs_random": round(result["win_rate_vs_random"], 3),
            "promoted": result["promoted"],
            "new_examples": new_examples,
        }
        save_model(self.model, str(ckpt), extra={"gen": gen, "elo": elo, "metrics": metrics})
        entry = self.ladder.add(gen, str(ckpt), elo, metrics)

        if result["promoted"]:
            # the contender earned the crown: it becomes the served champion.
            self.ladder.set_champion(gen)
            for name in ("champion.pt", "latest.pt"):
                save_model(self.model, str(Path(cfg.run_dir) / name),
                           extra={"gen": gen, "elo": elo, "metrics": metrics})
        else:
            # it failed to beat the champion -- roll the weights back so the next
            # generation self-plays from the champion again. This is what stops
            # the network from spiralling down into random play.
            self._restore_champion()
        self.ladder.save()
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

    def _resolve_ckpt(self, entry) -> str:
        """Checkpoint path: stored path, else by name in the run dir (Hub-pulled)."""
        if Path(entry.path).exists():
            return entry.path
        return str(Path(self.cfg.run_dir) / f"{entry.name}.pt")

    def _evaluate_and_gate(self) -> dict:
        """Gauntlet the contender, decide promotion, and place it on the Elo scale.

        The contender plays a cheap sanity gauntlet vs a fixed random anchor (an
        honest "can it still beat random?" signal that previously masqueraded as
        the rating). If a champion exists, the contender's Elo is the champion's
        plus the *clamped* Elo implied by their head-to-head, and it is promoted
        only if it scores at least ``gate_threshold`` -- so a noisy or degraded
        generation can never lower the served strength. Returns a dict with
        ``elo``, ``score_vs_champion``, ``win_rate_vs_random`` and ``promoted``.
        """
        cfg = self.cfg
        self.model.eval()
        contender = AlphaZeroAgent(self.model, device=self.device, simulations=cfg.eval_sims)

        from ...eval.elo import elo_delta_from_score, estimate_rating

        random_agent = RandomAgent(seed=123)
        rnd_pts = _match_points(contender, random_agent, cfg.eval_games, cfg.max_moves)
        win_rate_random = round(rnd_pts / cfg.eval_games, 3)

        champ = self.ladder.champion()
        champ_ckpt = self._resolve_ckpt(champ) if champ is not None else None
        if champ is None or champ_ckpt is None or not Path(champ_ckpt).exists():
            # No champion yet: anchor to the random scale, capped (beating random
            # proves you are above it, not that you are 2000 -- only a Stockfish
            # anchor measures that, via serve.calibrate()).
            elo = estimate_rating([(cfg.random_anchor_elo, rnd_pts, cfg.eval_games)],
                                  init=cfg.random_anchor_elo + 200)
            elo = min(elo, cfg.first_gen_elo_cap)
            return {"elo": elo, "score_vs_champion": 1.0,
                    "win_rate_vs_random": win_rate_random, "promoted": True}

        prev_model, _ = load_model(champ_ckpt, map_location=self.device)
        champ_agent = AlphaZeroAgent(prev_model.to(self.device), device=self.device,
                                     simulations=cfg.eval_sims)
        pts = _match_points(contender, champ_agent, cfg.eval_games, cfg.max_moves)
        score = pts / cfg.eval_games
        delta = elo_delta_from_score(score, clamp=cfg.elo_step_clamp)
        elo = round(champ.elo + delta, 1)
        return {"elo": elo, "score_vs_champion": score,
                "win_rate_vs_random": win_rate_random,
                "promoted": score >= cfg.gate_threshold}

    def _restore_champion(self) -> None:
        """Roll the in-place model back to the champion's weights (anti-collapse)."""
        champ = self.ladder.champion()
        if champ is None:
            return
        path = self._resolve_ckpt(champ)
        if not Path(path).exists():
            return
        model, _ = load_model(path, map_location=self.device)
        self.model.load_state_dict(model.state_dict())
        # fresh optimizer so collapse momentum does not carry into the next gen.
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)


def run_generations(cfg: ContinualConfig, n_gens: int) -> dict:
    """Convenience: run ``n_gens`` generations and return the Elo curve."""
    trainer = ContinualTrainer(cfg)
    trainer.run(n_gens)
    trainer.logger.close()
    return {"run_dir": cfg.run_dir, "elo_curve": trainer.ladder.elo_curve(),
            "levels": trainer.ladder.levels()}
