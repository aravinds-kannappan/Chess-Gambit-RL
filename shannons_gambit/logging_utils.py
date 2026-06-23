"""Lightweight JSONL metric logging (no wandb/tensorboard dependency).

Mirrors the append-only JSONL convention: one JSON object per step in
``metrics.jsonl`` and a single ``config.json`` snapshot per run.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


class JsonlLogger:
    """Append-only JSONL logger writing to ``<run_dir>/metrics.jsonl``."""

    def __init__(self, run_dir: str | Path, *, config: Any | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._fh = self.metrics_path.open("a", encoding="utf-8")
        self._t0 = time.time()
        if config is not None:
            self.log_config(config)

    def log_config(self, config: Any) -> None:
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(_to_jsonable(config), indent=2), encoding="utf-8")

    def log(self, **metrics: Any) -> None:
        record = {"t": round(time.time() - self._t0, 4), **_to_jsonable(metrics)}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
