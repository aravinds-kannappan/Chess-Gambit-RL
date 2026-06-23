"""Tie real data + trained models to the information-theoretic measures.

Produces the JSON series the notebook and the web dashboard render: which board
features carry the most information about the result (mutual information), how
peaked the learned policy is (entropy), and where a single game's outcome
uncertainty collapses (information gain per ply).
"""

from __future__ import annotations

import chess
import numpy as np

from .infotheory.analysis import feature_outcome_mi, game_info_profile, move_entropy
from .infotheory.entropy import shannon_entropy

_PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def board_features(fen: str) -> dict[str, float]:
    """Cheap scalar features computed from a position (side-to-move view)."""
    board = chess.Board(fen)
    sign = 1 if board.turn == chess.WHITE else -1
    material = 0
    count = 0
    for _, piece in board.piece_map().items():
        v = _PIECE_VALUE[piece.piece_type]
        material += v if piece.color == chess.WHITE else -v
        count += 1
    return {
        "material_diff": sign * material,
        "total_material": float(count),
        "mobility": float(board.legal_moves.count()),
        "in_check": float(board.is_check()),
        "ply": float(board.fullmove_number * 2),
    }


def feature_mi_report(records: dict[str, np.ndarray], *, max_n: int = 20000) -> dict[str, float]:
    """Mutual information (bits) between each feature and the game result."""
    fens = records["fen"][:max_n]
    outcomes = records["stm_value"][:max_n].astype(int)
    feats: dict[str, list[float]] = {}
    for fen in fens:
        for k, v in board_features(fen).items():
            feats.setdefault(k, []).append(v)
    feature_arrays = {k: np.asarray(v) for k, v in feats.items()}
    return feature_outcome_mi(feature_arrays, outcomes)


def policy_entropy_report(predictor, fens: np.ndarray, *, sample: int = 200,
                          seed: int = 0) -> dict[str, float]:
    """Mean/percentiles of the learned policy's per-position entropy (bits)."""
    rng = np.random.default_rng(seed)
    chosen = fens[rng.choice(len(fens), size=min(sample, len(fens)), replace=False)]
    ents = []
    for fen in chosen:
        board = chess.Board(fen)
        dist = predictor.policy_distribution(board)
        ents.append(move_entropy(np.array(list(dist.values()))))
    ents = np.asarray(ents)
    return {
        "mean_bits": float(ents.mean()),
        "median_bits": float(np.median(ents)),
        "p10_bits": float(np.percentile(ents, 10)),
        "p90_bits": float(np.percentile(ents, 90)),
        "n": int(ents.size),
    }


def game_info_curve(predictor, moves_uci: list[str]) -> dict:
    """Per-ply outcome-entropy collapse for a single game (info gain profile)."""
    board = chess.Board()
    wdl_series = []
    for uci in moves_uci:
        pred = predictor.predict(board, top_k=1)
        wdl_series.append([pred.wdl["loss"], pred.wdl["draw"], pred.wdl["win"]])
        board.push(chess.Move.from_uci(uci))
    pred = predictor.predict(board, top_k=1)
    wdl_series.append([pred.wdl["loss"], pred.wdl["draw"], pred.wdl["win"]])
    profile = game_info_profile(np.asarray(wdl_series))
    return {
        "entropy": [round(float(e), 4) for e in profile.entropy],
        "info_gain": [round(float(g), 4) for g in profile.info_gain],
        "total_info_bits": round(profile.total_info, 4),
        "decisive_ply": profile.decisive_ply,
    }


def first_game_curve(predictor, source: str | None = None) -> dict | None:
    """Information-gain curve for the first game in a PGN source (real game)."""
    import chess.pgn

    from .data.lichess import SAMPLE_PGN, _open_pgn_stream

    src = source or str(SAMPLE_PGN)
    try:
        stream = _open_pgn_stream(src)
    except Exception:
        return None
    try:
        game = chess.pgn.read_game(stream)
    finally:
        stream.close()
    if game is None:
        return None
    moves = [node.move.uci() for node in game.mainline()]
    if not moves:
        return None
    return game_info_curve(predictor, moves)


def outcome_prior_entropy(records: dict[str, np.ndarray]) -> float:
    """Baseline entropy of the result distribution (bits) over the dataset."""
    counts = np.bincount((records["stm_result"]).astype(int), minlength=3).astype(float)
    return shannon_entropy(counts / counts.sum())
