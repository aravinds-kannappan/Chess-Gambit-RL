"""Build parquet position shards and serve them as a torch dataset.

FENs are stored (tiny on disk) and encoded to planes once at load time, so a
run keeps the rich targets — policy (move), value/WDL (result), and rating
(mover Elo) — without persisting hundreds of MB of dense tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import DataConfig
from .lichess import iter_positions

_SCHEMA_FIELDS = [
    "fen", "move_uci", "move_index", "stm_value",
    "stm_result", "eval_cp", "mover_elo", "ply",
]


def build_dataset(cfg: DataConfig) -> dict:
    """Stream positions into parquet shards under ``<out_dir>/positions``.

    Returns a small summary dict (counts + Elo stats) for logging.
    """
    out = Path(cfg.out_dir) / "positions"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard_*.parquet"):
        old.unlink()

    buf: dict[str, list] = {k: [] for k in _SCHEMA_FIELDS}
    n_total = 0
    shard = 0

    def flush() -> None:
        nonlocal shard
        if not buf["fen"]:
            return
        table = pa.table({k: buf[k] for k in _SCHEMA_FIELDS})
        pq.write_table(table, out / f"shard_{shard:05d}.parquet")
        for k in buf:
            buf[k].clear()
        shard += 1

    for rec in iter_positions(cfg):
        for k in _SCHEMA_FIELDS:
            buf[k].append(rec.get(k))
        n_total += 1
        if len(buf["fen"]) >= cfg.shard_size:
            flush()
    flush()
    return {"positions": n_total, "shards": shard, "out_dir": str(out)}


def load_records(data_dir: str, *, max_positions: int | None = None) -> dict[str, np.ndarray]:
    """Load position records from parquet shards into NumPy arrays."""
    pos = Path(data_dir) / "positions"
    files = sorted(pos.glob("shard_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards in {pos}; run the data pipeline first")
    table = pa.concat_tables([pq.read_table(f) for f in files])
    cols = {name: table.column(name).to_numpy(zero_copy_only=False) for name in _SCHEMA_FIELDS}
    if max_positions is not None and len(cols["fen"]) > max_positions:
        cols = {k: v[:max_positions] for k, v in cols.items()}
    return cols


class PositionDataset:
    """In-memory dataset: pre-encodes FENs to uint8 planes and exposes targets."""

    def __init__(self, records: dict[str, np.ndarray]) -> None:
        import chess

        from .encode import encode_board

        fens = records["fen"]
        self.x = np.zeros((len(fens), 18, 8, 8), dtype=np.uint8)
        for i, fen in enumerate(fens):
            self.x[i] = encode_board(chess.Board(fen)).astype(np.uint8)
        self.policy = records["move_index"].astype(np.int64)
        self.value = records["stm_value"].astype(np.float32)
        self.wdl = records["stm_result"].astype(np.int64)

        elo = records["mover_elo"].astype(np.float64)
        self.rating_mask = np.isfinite(elo) & (elo > 0)
        valid = elo[self.rating_mask]
        self.rating_mean = float(valid.mean()) if valid.size else 1500.0
        self.rating_std = float(valid.std()) if valid.size and valid.std() > 0 else 1.0
        self.rating = np.where(
            self.rating_mask, (np.nan_to_num(elo) - self.rating_mean) / self.rating_std, 0.0
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.value)

    def to_torch(self):
        """Return a ``torch.utils.data.TensorDataset`` of all fields."""
        import torch

        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.x).float(),
            torch.from_numpy(self.policy),
            torch.from_numpy(self.value),
            torch.from_numpy(self.wdl),
            torch.from_numpy(self.rating),
            torch.from_numpy(self.rating_mask.astype(np.float32)),
        )
