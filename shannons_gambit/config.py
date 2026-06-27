"""Frozen dataclass configs passed to data pipelines, trainers, and agents.

Presets (``local_smoke``, ``local_full``, ``cloud``) live in :func:`get_preset`
so a bigger run is a config change rather than a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class DataConfig:
    """Lichess open-database ingestion settings."""

    # A real monthly standard-rated dump from https://database.lichess.org
    url: str = (
        "https://database.lichess.org/standard/"
        "lichess_db_standard_rated_2014-09.pgn.zst"
    )
    out_dir: str = "data"
    max_games: int = 5_000
    min_elo: int = 0
    require_eval: bool = False
    shard_size: int = 50_000  # positions per parquet shard
    seed: int = 0


@dataclass(frozen=True)
class MDPConfig:
    """Tabular MDP / Bellman settings for tractable endgames."""

    endgame: str = "KRvK"  # one of KRvK, KQvK, KPvK
    gamma: float = 0.99
    theta: float = 1e-6  # value-iteration convergence threshold
    max_iters: int = 1_000
    max_plies: int = 60  # episode cap for the endgame environment


@dataclass(frozen=True)
class TabularQConfig:
    """Tabular Q-learning on the endgame MDP."""

    endgame: str = "KRvK"
    gamma: float = 0.99
    alpha: float = 0.3
    epsilon: float = 0.25
    episodes: int = 20_000
    max_plies: int = 60
    opponent: str = "random"  # "random" (learnable) or "optimal" (perfect defence)
    seed: int = 0


@dataclass(frozen=True)
class NetConfig:
    """Residual policy+value / prediction network shape."""

    channels: int = 64
    blocks: int = 4


@dataclass(frozen=True)
class SupervisedConfig:
    """Behavioural-cloning + prediction-head training on real games."""

    data_dir: str = "data"
    run_dir: str = "runs/supervised"
    net: NetConfig = field(default_factory=NetConfig)
    batch_size: int = 256
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    rating_weight: float = 0.5
    device: str = "auto"  # auto -> mps/cuda/cpu
    max_positions: int = 200_000
    # Sample opening/middlegame/endgame roughly equally. Early positions vastly
    # outnumber the rest, so without this the net overfits openings and barely
    # learns the middlegame -- exactly the gap this project was missing.
    balance_phases: bool = True
    seed: int = 0


@dataclass(frozen=True)
class DQNConfig:
    """Deep Q-network trained on an endgame MDP with ground-truth validation."""

    endgame: str = "KRvK"
    run_dir: str = "runs/dqn"
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 50_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    target_sync: int = 500
    train_steps: int = 30_000
    warmup: int = 1_000
    max_plies: int = 60
    channels: int = 32
    # Cap the curriculum at a fixed mate distance so deep RL masters a
    # well-defined sub-task (conversion within K plies) that can be validated
    # exactly against the solved value table.
    max_train_dtm: int = 6
    # Use the exact Bellman value V* as a potential-based shaping reward.
    shaping: bool = True
    device: str = "auto"
    seed: int = 0


@dataclass(frozen=True)
class PPOConfig:
    """Proximal Policy Optimization (actor-critic) on an endgame MDP.

    PPO is trained where we hold exact ground truth (the solved value table), so
    its conversion rate can be validated against optimal play -- the on-policy
    counterpart to the off-policy DQN ``reward`` agent.
    """

    endgame: str = "KRvK"
    run_dir: str = "runs/ppo"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    rollout_steps: int = 1_024     # transitions collected per update
    minibatch_size: int = 256
    update_epochs: int = 4         # PPO epochs over each rollout
    total_updates: int = 200
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    channels: int = 32
    max_plies: int = 60
    # Curriculum: cap the mate distance of start states (matches the DQN setup so
    # the two RL agents are validated on the same well-defined sub-task).
    max_train_dtm: int = 6
    shaping: bool = True           # potential-based shaping from the exact V*
    device: str = "auto"
    seed: int = 0


@dataclass(frozen=True)
class AlphaZeroConfig:
    """Small-but-real AlphaZero-lite self-play training."""

    run_dir: str = "runs/alphazero"
    init_from: str = "runs/supervised/model.pt"  # bootstrap weights ("" = scratch)
    net: NetConfig = field(default_factory=NetConfig)
    simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature_moves: int = 15
    games_per_iter: int = 24
    iters: int = 5
    max_moves: int = 80
    batch_size: int = 128
    lr: float = 5e-4
    epochs_per_iter: int = 2
    buffer_games: int = 200
    device: str = "auto"
    seed: int = 0


@dataclass(frozen=True)
class StockfishConfig:
    """The Stockfish **benchmark** reference (never a player for our own agents).

    Used by the backend evaluator as a calibrated yardstick: an Elo-throttled
    reference opponent for gauntlets and a per-position scorer for centipawn-loss.
    Above ``uci_elo_floor`` we use the engine's own ``UCI_LimitStrength`` +
    ``UCI_Elo`` calibration; below it (where ``UCI_Elo`` bottoms out) we drop to
    ``Skill Level`` with a shallow search so weak bands are still believable.
    """

    elo: int = 1500
    # Per-move search budget. ``movetime_ms`` (if > 0) caps wall-clock per move;
    # ``depth`` caps search depth. At least one keeps weak levels fast and cheap.
    depth: int = 0          # 0 -> no explicit depth cap (rely on Elo throttle)
    movetime_ms: int = 50
    threads: int = 1
    hash_mb: int = 16
    # Engine's supported UCI_Elo window (Stockfish reports min 1320 historically;
    # queried/clamped at runtime, these are conservative defaults).
    uci_elo_floor: int = 1320
    uci_elo_ceiling: int = 3190
    seed: int = 0


@dataclass(frozen=True)
class ContinualConfig:
    """Continuous self-play: one resumable generation = self-play -> train -> rate."""

    run_dir: str = "runs/continual"
    init_from: str = "runs/supervised/model.pt"  # bootstrap ("" = scratch)
    net: NetConfig = field(default_factory=NetConfig)
    # self-play per generation
    simulations: int = 48
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature_moves: int = 12
    games_per_gen: int = 8
    max_moves: int = 80
    buffer_games: int = 80
    # training per generation
    batch_size: int = 128
    lr: float = 5e-4
    epochs_per_gen: int = 2
    # anchored Elo evaluation + champion gating
    eval_games: int = 6      # games vs each anchor agent
    eval_sims: int = 24      # MCTS sims during evaluation (cheap)
    random_anchor_elo: float = 600.0  # fixes the absolute Elo scale
    # A new generation only replaces the served champion if it beats it by at
    # least this score (0.5 = even). Below it, the net is reverted to the
    # champion -- so self-play can never *degrade* the served strength.
    gate_threshold: float = 0.55
    # Cap on how far a single gauntlet can move a checkpoint's Elo (a handful of
    # games cannot credibly measure a multi-hundred-point gap).
    elo_step_clamp: float = 200.0
    # The first generation has no champion to gate against; rate it vs the random
    # anchor but cap the result here (beating a random mover proves you are
    # *above* it, not that you are 2000 -- only a Stockfish anchor measures that).
    first_gen_elo_cap: float = 1000.0
    # Hybrid post-training: replay human (SFT) data alongside self-play so the net
    # refines toward strong human play instead of drifting. ``sft_data_dir`` is a
    # built dataset dir (with a positions/ shard folder); empty = pure self-play
    # (the original behaviour). ``sft_ratio`` is the fraction of train batches
    # drawn from the human data. ``kl_coef`` optionally anchors the policy to the
    # init/pretrain net (KL penalty) -- 0 disables it.
    sft_data_dir: str = ""
    sft_ratio: float = 0.5
    sft_max_positions: int = 200_000
    kl_coef: float = 0.0
    device: str = "auto"
    seed: int = 0


_PRESETS: dict[str, dict] = {
    "local_smoke": {
        "data": replace(DataConfig(), max_games=200),
        "supervised": replace(
            SupervisedConfig(), epochs=1, max_positions=4_000, batch_size=128
        ),
        "dqn": replace(DQNConfig(), train_steps=2_000, warmup=200, buffer_size=5_000),
        "alphazero": replace(
            AlphaZeroConfig(), iters=1, games_per_iter=4, simulations=16, init_from=""
        ),
    },
    "local_full": {
        "data": replace(DataConfig(), max_games=5_000),
        "supervised": SupervisedConfig(),
        "dqn": DQNConfig(),
        "alphazero": AlphaZeroConfig(),
    },
    "cloud": {
        "data": replace(DataConfig(), max_games=200_000, require_eval=True),
        "supervised": replace(
            SupervisedConfig(),
            epochs=10,
            max_positions=5_000_000,
            net=NetConfig(channels=128, blocks=10),
        ),
        "dqn": replace(DQNConfig(), train_steps=200_000),
        "alphazero": replace(
            AlphaZeroConfig(),
            iters=40,
            games_per_iter=200,
            simulations=200,
            net=NetConfig(channels=128, blocks=10),
        ),
    },
}


def get_preset(name: str) -> dict:
    """Return the config bundle for a named preset."""
    if name not in _PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(_PRESETS)}")
    return _PRESETS[name]
