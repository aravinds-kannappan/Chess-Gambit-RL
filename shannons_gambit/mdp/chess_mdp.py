"""Enumerate a chess endgame as a finite Markov game and solve it with Bellman.

``EndgameMDP`` builds the full state space for KRvK/KQvK, compiles it to a
:class:`~shannons_gambit.mdp.bellman.GameMDP`, and caches the result. White is
the maximiser trying to mate; Black is the minimiser (lone king). Losing the
strong piece collapses to bare kings, represented by one absorbing draw state.

``EndgameEnv`` wraps the same dynamics as a small gym-style environment for the
RL agents (DQN, tabular Q), with the optional perfect-defence opponent driven
by the solved value table.
"""

from __future__ import annotations

import math
from pathlib import Path

import chess
import numpy as np

from .bellman import GameMDP, value_iteration
from .endgames import EndgameSpec, get_spec

State = tuple[int, int, int, bool]  # (white_king, strong_piece, black_king, white_to_move)


def _state_key(board: chess.Board, piece_type: int) -> State | None:
    """Return the state tuple, or ``None`` if the strong piece was captured."""
    strong = board.pieces(piece_type, chess.WHITE)
    if not strong:
        return None
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    assert wk is not None and bk is not None
    return (wk, next(iter(strong)), bk, board.turn)


class EndgameMDP:
    def __init__(self, spec: EndgameSpec) -> None:
        self.spec = spec
        self.states: list[State] = []
        self.index: dict[State, int] = {}
        self.draw_index: int = -1
        self.game: GameMDP | None = None
        self._V: np.ndarray | None = None

    # --- board <-> state ---------------------------------------------------
    def board_from_state(self, state: State) -> chess.Board:
        wk, ws, bk, turn = state
        board = chess.Board(None)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(ws, chess.Piece(self.spec.strong_piece, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.turn = turn
        return board

    # --- construction ------------------------------------------------------
    def build(self, cache_dir: str | Path = "data/mdp_cache") -> EndgameMDP:
        cache = Path(cache_dir) / f"{self.spec.name}.npz"
        if cache.exists():
            self._load(cache)
            return self
        self._enumerate()
        self._compile()
        self._save(cache)
        return self

    def _enumerate(self) -> None:
        squares = range(64)
        states: list[State] = []
        for wk in squares:
            for ws in squares:
                if ws == wk:
                    continue
                for bk in squares:
                    if bk in (wk, ws):
                        continue
                    for turn in (True, False):
                        board = self.board_from_state((wk, ws, bk, turn))
                        if board.is_valid():
                            states.append((wk, ws, bk, turn))
        self.states = states
        self.index = {s: i for i, s in enumerate(states)}
        self.draw_index = len(states)

    def _compile(self) -> None:
        n = len(self.states) + 1  # + absorbing draw state
        is_max = np.zeros(n, dtype=bool)
        terminal = np.zeros(n, dtype=bool)
        terminal_value = np.zeros(n, dtype=np.float64)
        nt_list: list[int] = []
        succ_flat: list[int] = []
        seg_starts: list[int] = []

        for i, state in enumerate(self.states):
            board = self.board_from_state(state)
            is_max[i] = board.turn == chess.WHITE
            if board.is_checkmate():
                terminal[i] = True
                # side to move is mated; +1 for White if Black is mated.
                terminal_value[i] = 1.0 if board.turn == chess.BLACK else -1.0
                continue
            if board.is_stalemate() or board.is_insufficient_material():
                terminal[i] = True
                terminal_value[i] = 0.0
                continue
            seg_starts.append(len(succ_flat))
            nt_list.append(i)
            for move in board.legal_moves:
                board.push(move)
                key = _state_key(board, self.spec.strong_piece)
                board.pop()
                succ_flat.append(self.draw_index if key is None else self.index[key])

        terminal[self.draw_index] = True  # absorbing draw
        self.game = GameMDP(
            is_max=is_max,
            terminal=terminal,
            terminal_value=terminal_value,
            nt=np.asarray(nt_list, dtype=np.int64),
            succ_flat=np.asarray(succ_flat, dtype=np.int64),
            seg_starts=np.asarray(seg_starts, dtype=np.int64),
        )

    # --- persistence -------------------------------------------------------
    def _save(self, cache: Path) -> None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        g = self.game
        assert g is not None
        np.savez_compressed(
            cache,
            states=np.asarray(self.states, dtype=np.int16),
            is_max=g.is_max,
            terminal=g.terminal,
            terminal_value=g.terminal_value,
            nt=g.nt,
            succ_flat=g.succ_flat,
            seg_starts=g.seg_starts,
        )

    def _load(self, cache: Path) -> None:
        data = np.load(cache)
        self.states = [tuple(row) for row in data["states"].astype(int).tolist()]
        self.states = [(s[0], s[1], s[2], bool(s[3])) for s in self.states]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.draw_index = len(self.states)
        self.game = GameMDP(
            is_max=data["is_max"],
            terminal=data["terminal"],
            terminal_value=data["terminal_value"],
            nt=data["nt"],
            succ_flat=data["succ_flat"],
            seg_starts=data["seg_starts"],
        )

    # --- solving & policy --------------------------------------------------
    def solve(self, *, gamma: float = 0.99, theta: float = 1e-6,
              max_iters: int = 1000) -> tuple[np.ndarray, list[float]]:
        assert self.game is not None, "call build() first"
        V, history = value_iteration(
            self.game, gamma=gamma, theta=theta, max_iters=max_iters
        )
        self._V = V
        self._gamma = gamma
        return V, history

    def value_of(self, board: chess.Board) -> float:
        assert self._V is not None
        key = _state_key(board, self.spec.strong_piece)
        if key is None or key not in self.index:
            return 0.0
        return float(self._V[self.index[key]])

    def optimal_move(self, board: chess.Board) -> chess.Move | None:
        """Greedy move from the solved value table (White max / Black min)."""
        assert self._V is not None
        moves = list(board.legal_moves)
        if not moves:
            return None
        is_max = board.turn == chess.WHITE
        scored: list[tuple[float, chess.Move]] = []
        for move in moves:
            board.push(move)
            key = _state_key(board, self.spec.strong_piece)
            val = 0.0 if key is None else float(self._V[self.index[key]])
            board.pop()
            scored.append((self._gamma * val, move))
        best = max(scored, key=lambda t: t[0]) if is_max else min(scored, key=lambda t: t[0])
        return best[1]

    def mate_distance(self, board: chess.Board) -> int | None:
        """Plies-to-mate implied by ``V = gamma**dtm`` for a won position."""
        v = self.value_of(board)
        if v <= 0:
            return None
        return round(math.log(v) / math.log(self._gamma))

    def _won_states(self) -> np.ndarray:
        assert self._V is not None and self.game is not None
        if not hasattr(self, "_won_cache"):
            nt = self.game.nt
            self._won_cache = nt[self.game.is_max[nt] & (self._V[nt] > 0.0)]
            with np.errstate(divide="ignore"):
                self._dtm_cache = np.round(
                    np.log(np.clip(self._V[self._won_cache], 1e-12, None))
                    / math.log(self._gamma)
                ).astype(int)
        return self._won_cache

    def sample_won_state(self, rng: np.random.Generator, *, max_dtm: int | None = None) -> State:
        """A random White-to-move winning position, optionally within ``max_dtm`` plies.

        ``max_dtm`` enables a mate-distance curriculum for learning agents.
        """
        won = self._won_states()
        if max_dtm is not None:
            won = won[self._dtm_cache <= max_dtm]
            if won.size == 0:
                won = self._won_states()
        return self.states[int(rng.choice(won))]

    @property
    def max_dtm(self) -> int:
        self._won_states()
        return int(self._dtm_cache.max())


class EndgameEnv:
    """Gym-style environment: White (agent) tries to mate the lone king."""

    def __init__(self, mdp: EndgameMDP, *, opponent: str = "optimal",
                 max_plies: int = 60, step_penalty: float = 0.0,
                 shaping: bool = False, gamma: float = 0.99,
                 seed: int = 0) -> None:
        self.mdp = mdp
        self.opponent = opponent
        self.max_plies = max_plies
        self.step_penalty = step_penalty
        # Potential-based reward shaping using the exact Bellman value V* as the
        # potential Phi(s). By the shaping theorem this leaves the optimal policy
        # unchanged but turns the sparse mate reward into a dense signal -- the
        # information in the solved MDP bootstrapping the deep-RL agent.
        self.shaping = shaping
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.board = chess.Board(None)
        self.plies = 0

    def reset(self) -> chess.Board:
        state = self.mdp.sample_won_state(self.rng)
        self.board = self.mdp.board_from_state(state)
        self.plies = 0
        return self.board

    def _opponent_move(self) -> None:
        moves = list(self.board.legal_moves)
        if not moves:
            return
        if self.opponent == "optimal" and self.mdp._V is not None:
            move = self.mdp.optimal_move(self.board)
        else:
            move = moves[int(self.rng.integers(len(moves)))]
        self.board.push(move)

    def _phi(self) -> float:
        return self.mdp.value_of(self.board) if self.mdp._V is not None else 0.0

    def _shaped(self, base: float, phi_before: float, terminal: bool) -> float:
        if not self.shaping:
            return base
        phi_after = 0.0 if terminal else self._phi()
        return base + self.gamma * phi_after - phi_before

    def step(self, move: chess.Move) -> tuple[chess.Board, float, bool, dict]:
        """Apply the agent's (White) move, then the opponent replies."""
        phi_before = self._phi() if self.shaping else 0.0
        self.board.push(move)
        self.plies += 1
        if self.board.is_checkmate():
            return self.board, self._shaped(1.0, phi_before, True), True, {"result": "win"}
        if self.board.is_game_over() or self._strong_captured():
            return self.board, self._shaped(0.0, phi_before, True), True, {"result": "draw"}

        self._opponent_move()
        self.plies += 1
        base = -self.step_penalty
        if self.board.is_checkmate():  # opponent mated itself? impossible, but guard
            return self.board, self._shaped(base, phi_before, True), True, {"result": "win"}
        if self.board.is_game_over():
            return self.board, self._shaped(base, phi_before, True), True, {"result": "draw"}
        if self.plies >= self.max_plies:
            return self.board, self._shaped(base, phi_before, True), True, {"result": "timeout"}
        return self.board, self._shaped(base, phi_before, False), False, {}

    def _strong_captured(self) -> bool:
        return not self.board.pieces(self.mdp.spec.strong_piece, chess.WHITE)


def load_endgame(name: str, *, gamma: float = 0.99,
                 cache_dir: str | Path = "data/mdp_cache") -> EndgameMDP:
    """Build (or load) and solve an endgame in one call."""
    mdp = EndgameMDP(get_spec(name)).build(cache_dir=cache_dir)
    mdp.solve(gamma=gamma)
    return mdp
