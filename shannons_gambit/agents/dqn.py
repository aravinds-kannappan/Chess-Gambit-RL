"""Deep Q-Network on the endgame MDP (double-DQN, target net, replay buffer).

DQN is trained where we hold exact ground truth (the solved value table), so
its learned conversion rate can be validated against optimal play. Inputs are
the shared 18-plane board encoding; outputs are Q over the 4672 move indices,
masked to legal moves at action-selection time.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

from ..config import DQNConfig
from ..data.encode import NUM_PLANES, POLICY_SIZE, encode_board, legal_policy_mask, move_to_index
from ..logging_utils import JsonlLogger
from ..mdp.chess_mdp import EndgameEnv, EndgameMDP
from ..seeding import seed_everything
from ..torch_utils import resolve_device
from .base import Agent


class QNet(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(channels * 64, 512), nn.ReLU(inplace=True),
            nn.Linear(512, POLICY_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _Replay:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.s = np.zeros((capacity, NUM_PLANES, 8, 8), dtype=np.uint8)
        self.ns = np.zeros((capacity, NUM_PLANES, 8, 8), dtype=np.uint8)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.nmask = np.zeros((capacity, POLICY_SIZE), dtype=bool)
        self.size = 0
        self.ptr = 0

    def add(self, s, a, r, ns, done, nmask) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.ns[i] = s, a, r, ns
        self.done[i], self.nmask[i] = done, nmask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch)
        return (self.s[idx], self.a[idx], self.r[idx], self.ns[idx],
                self.done[idx], self.nmask[idx])


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, mdp: EndgameMDP, cfg: DQNConfig) -> None:
        self.mdp = mdp
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.online = QNet(cfg.channels).to(self.device)
        self.target = QNet(cfg.channels).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.rng = np.random.default_rng(cfg.seed)

    def _q(self, net: QNet, planes: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(planes).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return net(x)[0].cpu().numpy()

    def select_move(self, board: chess.Board, *, epsilon: float = 0.0) -> chess.Move:
        moves = list(board.legal_moves)
        if epsilon and self.rng.random() < epsilon:
            return moves[int(self.rng.integers(len(moves)))]
        q = self._q(self.online, encode_board(board))
        return max(moves, key=lambda m: q[move_to_index(m)])

    def train(self, history_every: int = 5000) -> list[dict]:
        cfg = self.cfg
        seed_everything(cfg.seed)
        env = EndgameEnv(self.mdp, opponent="random", max_plies=cfg.max_plies,
                         shaping=cfg.shaping, gamma=cfg.gamma, seed=cfg.seed)
        buffer = _Replay(cfg.buffer_size)
        opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        logger = JsonlLogger(Path(cfg.run_dir), config=cfg)
        history: list[dict] = []
        max_dtm = self.mdp.max_dtm

        board = self._curriculum_reset(env, 0, max_dtm)
        ep_wins = ep_count = 0
        running_loss = 0.0
        loss_n = 0
        for step in range(1, cfg.train_steps + 1):
            eps = self._epsilon(step)
            move = self.select_move(board, epsilon=eps)
            a = move_to_index(move)
            s = encode_board(board).astype(np.uint8)
            next_board, reward, done, info = env.step(move)
            ns = encode_board(next_board).astype(np.uint8)
            nmask = legal_policy_mask(next_board)
            buffer.add(s, a, reward, ns, float(done), nmask)

            if done:
                ep_count += 1
                ep_wins += int(info.get("result") == "win")
                board = self._curriculum_reset(env, step, max_dtm)
            else:
                board = next_board

            if buffer.size >= cfg.warmup:
                running_loss += self._learn(buffer, opt, cfg)
                loss_n += 1
            if step % cfg.target_sync == 0:
                self.target.load_state_dict(self.online.state_dict())
            if step % history_every == 0:
                rec = {
                    "step": step,
                    "epsilon": round(eps, 3),
                    "train_win_rate": round(ep_wins / max(ep_count, 1), 3),
                    "loss": round(running_loss / max(loss_n, 1), 4),
                }
                logger.log(**rec)
                history.append(rec)
                ep_wins = ep_count = 0
                running_loss = 0.0
                loss_n = 0
        Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.online.state_dict(), Path(cfg.run_dir) / "dqn.pt")
        logger.close()
        return history

    def _curriculum_reset(self, env: EndgameEnv, step: int, max_dtm: int) -> chess.Board:
        frac = step / max(self.cfg.train_steps, 1)
        ceiling = min(max_dtm, self.cfg.max_train_dtm)
        cap = max(2, min(ceiling, int(frac * ceiling) + 2))
        state = self.mdp.sample_won_state(self.rng, max_dtm=cap)
        env.board = self.mdp.board_from_state(state)
        env.plies = 0
        return env.board

    def _epsilon(self, step: int) -> float:
        cfg = self.cfg
        frac = min(1.0, step / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def _learn(self, buffer: _Replay, opt, cfg: DQNConfig) -> float:
        s, a, r, ns, done, nmask = buffer.sample(cfg.batch_size, self.rng)
        s_t = torch.from_numpy(s).float().to(self.device)
        ns_t = torch.from_numpy(ns).float().to(self.device)
        a_t = torch.from_numpy(a).long().to(self.device)
        r_t = torch.from_numpy(r).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)
        mask_t = torch.from_numpy(nmask).to(self.device)

        q = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = self.online(ns_t).masked_fill(~mask_t, -1e9)
            next_actions = next_online.argmax(1, keepdim=True)
            next_q = self.target(ns_t).gather(1, next_actions).squeeze(1)
            target = r_t + (1.0 - done_t) * cfg.gamma * next_q
        loss = nn.functional.smooth_l1_loss(q, target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        opt.step()
        return float(loss.detach())

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
