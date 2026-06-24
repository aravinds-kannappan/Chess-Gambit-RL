"""Proximal Policy Optimization on the endgame MDP (on-policy actor-critic).

PPO is the on-policy sibling of the off-policy :mod:`~shannons_gambit.agents.dqn`
reward agent: same exact-ground-truth arena (the solved endgame), same shared
18-plane encoding, validated the same way (conversion rate vs. optimal). The
actor outputs logits over the 4672 move indices (masked to legal moves) and the
critic estimates the state value; training uses clipped surrogate objective with
GAE advantages.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

from ..config import PPOConfig
from ..data.encode import NUM_PLANES, POLICY_SIZE, encode_board, legal_policy_mask, move_to_index
from ..logging_utils import JsonlLogger
from ..mdp.chess_mdp import EndgameEnv, EndgameMDP
from ..seeding import seed_everything
from ..torch_utils import resolve_device
from .base import Agent

_NEG_INF = -1e9


class ActorCritic(nn.Module):
    """Shared conv trunk with a policy head (logits) and a value head (scalar)."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(channels * 64, 512), nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(512, POLICY_SIZE)
        self.value = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, _NEG_INF)


class PPOAgent(Agent):
    """An endgame policy trained with PPO; plays the argmax legal move."""

    name = "ppo"

    def __init__(self, mdp: EndgameMDP, cfg: PPOConfig | None = None) -> None:
        self.mdp = mdp
        self.cfg = cfg or PPOConfig()
        self.device = resolve_device(self.cfg.device)
        self.net = ActorCritic(self.cfg.channels).to(self.device)
        self.rng = np.random.default_rng(self.cfg.seed)

    # --- play --------------------------------------------------------------
    @torch.no_grad()
    def _policy_value(self, board: chess.Board) -> tuple[np.ndarray, float]:
        x = torch.from_numpy(encode_board(board)).float().unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        mask = torch.from_numpy(legal_policy_mask(board)).to(self.device).unsqueeze(0)
        probs = torch.softmax(_masked_logits(logits, mask), dim=1)[0].cpu().numpy()
        return probs, float(value.item())

    def select_move(self, board: chess.Board, *, sample: bool = False) -> chess.Move:
        moves = list(board.legal_moves)
        probs, _ = self._policy_value(board)
        if sample:
            idx = [move_to_index(m) for m in moves]
            p = np.array([probs[i] for i in idx], dtype=np.float64)
            p = p / p.sum() if p.sum() > 0 else None
            return moves[int(self.rng.choice(len(moves), p=p))]
        return max(moves, key=lambda m: probs[move_to_index(m)])

    @classmethod
    def from_checkpoint(cls, mdp: EndgameMDP, path: str, cfg: PPOConfig | None = None
                        ) -> PPOAgent:
        agent = cls(mdp, cfg)
        state = torch.load(path, map_location=agent.device, weights_only=True)
        agent.net.load_state_dict(state)
        agent.net.eval()
        return agent

    # --- training ----------------------------------------------------------
    def train(self) -> list[dict]:
        cfg = self.cfg
        seed_everything(cfg.seed)
        env = EndgameEnv(self.mdp, opponent="random", max_plies=cfg.max_plies,
                         shaping=cfg.shaping, gamma=cfg.gamma, seed=cfg.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        logger = JsonlLogger(Path(cfg.run_dir), config=cfg)
        history: list[dict] = []

        board = self._curriculum_reset(env, 0)
        for update in range(1, cfg.total_updates + 1):
            roll, board, stats = self._rollout(env, board, update)
            metrics = self._update(roll, opt)
            metrics.update(stats)
            metrics["update"] = update
            if update % max(1, cfg.total_updates // 20) == 0 or update == cfg.total_updates:
                logger.log(**metrics)
                history.append(metrics)

        Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), Path(cfg.run_dir) / "ppo.pt")
        logger.close()
        return history

    def _rollout(self, env: EndgameEnv, board: chess.Board, update: int):
        cfg = self.cfg
        S, A, LP, V, R, D, M = [], [], [], [], [], [], []
        ep_wins = ep_count = 0
        for _ in range(cfg.rollout_steps):
            planes = encode_board(board)
            mask = legal_policy_mask(board)
            x = torch.from_numpy(planes).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, value = self.net(x)
                m = torch.from_numpy(mask).to(self.device).unsqueeze(0)
                dist = torch.distributions.Categorical(logits=_masked_logits(logits, m))
                action = dist.sample()
                logp = dist.log_prob(action)
            a_idx = int(action.item())
            move = self._index_to_legal_move(board, a_idx)
            next_board, reward, done, info = env.step(move)

            S.append(planes)
            A.append(a_idx)
            LP.append(float(logp.item()))
            V.append(float(value.item()))
            R.append(reward)
            D.append(float(done))
            M.append(mask)

            if done:
                ep_count += 1
                ep_wins += int(info.get("result") == "win")
                board = self._curriculum_reset(env, update)
            else:
                board = next_board

        with torch.no_grad():
            last_x = torch.from_numpy(encode_board(board)).float().unsqueeze(0).to(self.device)
            _, last_v = self.net(last_x)
        roll = self._finish(S, A, LP, V, R, D, M, float(last_v.item()))
        stats = {"rollout_win_rate": round(ep_wins / max(ep_count, 1), 3),
                 "episodes": ep_count}
        return roll, board, stats

    def _finish(self, S, A, LP, V, R, D, M, last_v: float) -> dict:
        cfg = self.cfg
        rewards = np.array(R, dtype=np.float32)
        values = np.array(V + [last_v], dtype=np.float32)
        dones = np.array(D, dtype=np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + values[:-1]
        return {
            "s": torch.from_numpy(np.asarray(S, dtype=np.float32)),
            "a": torch.tensor(A, dtype=torch.long),
            "logp": torch.tensor(LP, dtype=torch.float32),
            "mask": torch.from_numpy(np.asarray(M)),
            "adv": torch.from_numpy(adv),
            "ret": torch.from_numpy(returns),
        }

    def _update(self, roll: dict, opt) -> dict:
        cfg = self.cfg
        n = roll["a"].shape[0]
        s = roll["s"].to(self.device)
        a = roll["a"].to(self.device)
        old_logp = roll["logp"].to(self.device)
        mask = roll["mask"].to(self.device)
        adv = roll["adv"].to(self.device)
        ret = roll["ret"].to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        last = {"pol": 0.0, "val": 0.0, "ent": 0.0}
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                logits, value = self.net(s[idx])
                dist = torch.distributions.Categorical(logits=_masked_logits(logits, mask[idx]))
                logp = dist.log_prob(a[idx])
                ratio = torch.exp(logp - old_logp[idx])
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
                pol_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
                val_loss = ((value - ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pol_loss + cfg.value_coef * val_loss - cfg.entropy_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                opt.step()
                last = {"pol": float(pol_loss.detach()), "val": float(val_loss.detach()),
                        "ent": float(ent.detach())}
        return {"loss_policy": round(last["pol"], 4), "loss_value": round(last["val"], 4),
                "entropy": round(last["ent"], 4)}

    # --- helpers -----------------------------------------------------------
    def _index_to_legal_move(self, board: chess.Board, idx: int) -> chess.Move:
        for m in board.legal_moves:  # mask guarantees a legal idx, but stay safe
            if move_to_index(m) == idx:
                return m
        return next(iter(board.legal_moves))

    def _curriculum_reset(self, env: EndgameEnv, update: int) -> chess.Board:
        frac = update / max(self.cfg.total_updates, 1)
        ceiling = min(self.mdp.max_dtm, self.cfg.max_train_dtm)
        cap = max(2, min(ceiling, int(frac * ceiling) + 2))
        state = self.mdp.sample_won_state(self.rng, max_dtm=cap)
        env.board = self.mdp.board_from_state(state)
        env.plies = 0
        return env.board

    def evaluate(self, *, n: int = 500, max_dtm: int | None = None, seed: int = 7) -> dict:
        env = EndgameEnv(self.mdp, opponent="random", max_plies=self.cfg.max_plies, seed=seed)
        rng = np.random.default_rng(seed)
        wins = 0
        for _ in range(n):
            state = self.mdp.sample_won_state(rng, max_dtm=max_dtm)
            env.board = self.mdp.board_from_state(state)
            env.plies = 0
            board, done, info = env.board, False, {}
            while not done:
                _, _, done, info = env.step(self.select_move(board))
            wins += int(info.get("result") == "win")
        return {"n": n, "max_dtm": max_dtm, "win_rate": wins / n}
