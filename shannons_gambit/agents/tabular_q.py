"""Tabular Q-learning on the endgame MDP.

This is the bridge between theory and learning: the agent never sees the solved
value table, it learns purely from sampled experience via Bellman backups
``Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]``, and is then
checked to converge toward the value-iteration optimum.
"""

from __future__ import annotations

from collections import defaultdict

import chess
import numpy as np

from ..config import TabularQConfig
from ..data.encode import move_to_index
from ..mdp.chess_mdp import EndgameEnv, EndgameMDP, _state_key
from .base import Agent


class TabularQAgent(Agent):
    name = "tabular_q"

    def __init__(self, mdp: EndgameMDP, cfg: TabularQConfig) -> None:
        self.mdp = mdp
        self.cfg = cfg
        self.q: dict[tuple, dict[int, float]] = defaultdict(dict)
        self._rng = np.random.default_rng(cfg.seed)

    def _state(self, board: chess.Board) -> tuple:
        key = _state_key(board, self.mdp.spec.strong_piece)
        return key if key is not None else ("draw",)

    def _best_value(self, board: chess.Board) -> float:
        row = self.q.get(self._state(board))
        if not row:
            return 0.0
        return max(row.values())

    def select_move(self, board: chess.Board, *, epsilon: float = 0.0) -> chess.Move:
        moves = list(board.legal_moves)
        if epsilon and self._rng.random() < epsilon:
            return moves[int(self._rng.integers(len(moves)))]
        row = self.q.get(self._state(board), {})
        scored = [(row.get(move_to_index(m), 0.0), i, m) for i, m in enumerate(moves)]
        return max(scored, key=lambda t: (t[0], -t[1]))[2]

    def train(self, history_every: int = 1000, *, curriculum: bool = True) -> list[dict]:
        cfg = self.cfg
        env = EndgameEnv(
            self.mdp, opponent=cfg.opponent, max_plies=cfg.max_plies, seed=cfg.seed
        )
        history: list[dict] = []
        wins = 0
        max_dtm = self.mdp.max_dtm
        for ep in range(1, cfg.episodes + 1):
            if curriculum:
                # Grow the start horizon from near-mate outward over training.
                frac = ep / cfg.episodes
                cap = max(2, int(frac * max_dtm) + 2)
                state = self.mdp.sample_won_state(self._rng, max_dtm=cap)
                env.board = self.mdp.board_from_state(state)
                env.plies = 0
                board = env.board
            else:
                board = env.reset()
            done = False
            while not done:
                state = self._state(board)
                move = self.select_move(board, epsilon=cfg.epsilon)
                a = move_to_index(move)
                next_board, reward, done, info = env.step(move)
                target = reward + (0.0 if done else cfg.gamma * self._best_value(next_board))
                old = self.q[state].get(a, 0.0)
                self.q[state][a] = old + cfg.alpha * (target - old)
                board = next_board
            wins += int(info.get("result") == "win")
            if ep % history_every == 0:
                history.append(
                    {"episode": ep, "win_rate": wins / history_every, "n_states": len(self.q)}
                )
                wins = 0
        return history

    def evaluate(self, *, n: int = 1000, max_dtm: int | None = None,
                 opponent: str = "random", seed: int = 123) -> dict:
        """Greedy roll-outs from sampled won positions; report outcome rates."""
        env = EndgameEnv(self.mdp, opponent=opponent, max_plies=self.cfg.max_plies, seed=seed)
        rng = np.random.default_rng(seed)
        wins = draws = plies_to_mate = 0
        for _ in range(n):
            state = self.mdp.sample_won_state(rng, max_dtm=max_dtm)
            env.board = self.mdp.board_from_state(state)
            env.plies = 0
            board = env.board
            done = False
            info: dict = {}
            while not done:
                _, _, done, info = env.step(self.select_move(board))
            if info.get("result") == "win":
                wins += 1
                plies_to_mate += env.plies
            elif info.get("result") in ("draw",):
                draws += 1
        return {
            "n": n,
            "max_dtm": max_dtm,
            "win_rate": wins / n,
            "draw_rate": draws / n,
            "avg_plies_to_mate": (plies_to_mate / wins) if wins else None,
        }

    def greedy_value_agreement(self, sample: int = 2000) -> float:
        """Fraction of sampled won states where the greedy move is optimal."""
        rng = np.random.default_rng(0)
        agree = 0
        total = 0
        for _ in range(sample):
            state = self.mdp.sample_won_state(rng)
            board = self.mdp.board_from_state(state)
            if self._state(board) not in self.q:
                continue
            total += 1
            mine = self.select_move(board)
            opt_val = self.mdp.value_of(board)
            board.push(mine)
            reached = self.mdp.value_of(board) * self.cfg.gamma
            board.pop()
            # optimal if my move preserves the optimal value (allowing float slack)
            agree += int(reached >= opt_val - 1e-6)
        return agree / max(total, 1)
