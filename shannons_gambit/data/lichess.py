"""Ingest real games from the Lichess open database into position records.

The monthly dumps at https://database.lichess.org are zstd-compressed PGN; we
stream-decompress so we never hold the whole file in memory, parse with
python-chess, and emit one record per position. A small bundled PGN of real
historical games (``sample_games.pgn``) lets the pipeline and tests run offline;
full training points ``DataConfig.url`` at an actual Lichess dump.
"""

from __future__ import annotations

import io
import ssl
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn
import zstandard

from ..config import DataConfig

SAMPLE_PGN = Path(__file__).with_name("sample_games.pgn")


def _open_pgn_stream(source: str) -> io.TextIOBase:
    """Open a (possibly .zst, possibly remote) PGN as a text stream."""
    if source.startswith("http://") or source.startswith("https://"):
        try:
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except ModuleNotFoundError:
            ctx = ssl.create_default_context()
        raw: io.BufferedIOBase = urllib.request.urlopen(source, context=ctx)  # noqa: S310
    else:
        raw = open(source, "rb")
    if source.endswith(".zst"):
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")


def iter_games(source: str, *, max_games: int) -> Iterator[chess.pgn.Game]:
    stream = _open_pgn_stream(source)
    try:
        count = 0
        while count < max_games:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            count += 1
            yield game
    finally:
        stream.close()


_RESULT_TO_WHITE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def iter_positions(cfg: DataConfig) -> Iterator[dict]:
    """Yield one record per position from the configured source.

    Record fields: ``fen, move_uci, move_index, stm_value, stm_result,
    eval_cp, mover_elo, ply, phase``. ``stm_value`` is the eventual game result
    from the side-to-move's perspective (+1 win / 0 draw / -1 loss); ``phase`` is
    ``opening`` / ``middlegame`` / ``endgame`` for phase-balanced training.
    """
    from ..phases import game_phase
    from .encode import move_to_index  # local import avoids a cycle at module load

    source = cfg.url if (cfg.url.startswith("http") or Path(cfg.url).exists()) else str(SAMPLE_PGN)
    for game in iter_games(source, max_games=cfg.max_games):
        white_elo = _safe_int(game.headers.get("WhiteElo"))
        black_elo = _safe_int(game.headers.get("BlackElo"))
        result_white = _RESULT_TO_WHITE.get(game.headers.get("Result", "*"))
        if result_white is None:
            continue
        if cfg.min_elo and min(white_elo or 0, black_elo or 0) < cfg.min_elo:
            continue
        board = game.board()
        ply = 0
        for node in game.mainline():
            move = node.move
            eval_cp = _node_eval_cp(node)
            if cfg.require_eval and eval_cp is None:
                board.push(move)
                ply += 1
                continue
            white_to_move = board.turn == chess.WHITE
            stm_value = result_white if white_to_move else -result_white
            mover_elo = (white_elo if white_to_move else black_elo)
            yield {
                "fen": board.fen(),
                "move_uci": move.uci(),
                "move_index": move_to_index(move),
                "stm_value": stm_value,
                "stm_result": stm_value + 1,  # 0=loss,1=draw,2=win for WDL head
                "eval_cp": eval_cp,
                "mover_elo": mover_elo,
                "ply": ply,
                "phase": game_phase(board, ply=ply),
            }
            board.push(move)
            ply += 1


def _safe_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "?") else None
    except ValueError:
        return None


def _node_eval_cp(node: chess.pgn.ChildNode) -> float | None:
    """Stockfish eval (centipawns, white perspective) if the PGN carries it."""
    try:
        score = node.eval()
    except (ValueError, KeyError):
        return None
    if score is None:
        return None
    return score.white().score(mate_score=10000)
