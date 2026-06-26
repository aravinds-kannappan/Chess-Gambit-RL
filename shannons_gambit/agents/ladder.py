"""Versioned checkpoint registry for continuous self-play.

Each self-play generation produces a checkpoint with a measured (anchored) Elo.
The ``Ladder`` tracks them in ``ladder.json`` so the server can pick a snapshot
by target Elo (watch mode / graded play) and the research page can plot Elo vs.
generation. The Hub is the source of truth; this is the local mirror.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LadderEntry:
    gen: int
    name: str          # e.g. "gen-0007"
    path: str          # local checkpoint path
    elo: float
    created: str
    metrics: dict = field(default_factory=dict)


@dataclass
class Ladder:
    run_dir: str
    random_anchor_elo: float = 600.0
    entries: list[LadderEntry] = field(default_factory=list)
    # The generation currently served: the strongest *gated* checkpoint (one that
    # actually beat its predecessor head-to-head), not merely the highest Elo. The
    # served net is promoted only on a verified win, so it never regresses.
    champion_gen: int | None = None

    @property
    def json_path(self) -> Path:
        return Path(self.run_dir) / "ladder.json"

    # --- mutation ----------------------------------------------------------
    def add(self, gen: int, ckpt_path: str, elo: float, metrics: dict) -> LadderEntry:
        entry = LadderEntry(
            gen=gen,
            name=f"gen-{gen:04d}",
            path=str(ckpt_path),
            elo=round(float(elo), 1),
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            metrics=metrics,
        )
        self.entries = [e for e in self.entries if e.gen != gen] + [entry]
        self.entries.sort(key=lambda e: e.gen)
        return entry

    # --- queries -----------------------------------------------------------
    def latest(self) -> LadderEntry | None:
        return self.entries[-1] if self.entries else None

    def best(self) -> LadderEntry | None:
        return max(self.entries, key=lambda e: e.elo) if self.entries else None

    def champion(self) -> LadderEntry | None:
        """The served checkpoint: the gated champion, or the best as a fallback."""
        if self.champion_gen is not None:
            entry = next((e for e in self.entries if e.gen == self.champion_gen), None)
            if entry is not None:
                return entry
        return self.best()

    def set_champion(self, gen: int) -> None:
        self.champion_gen = gen

    def next_gen(self) -> int:
        return (self.entries[-1].gen + 1) if self.entries else 0

    def nearest(self, target_elo: float) -> LadderEntry | None:
        if not self.entries:
            return None
        return min(self.entries, key=lambda e: abs(e.elo - target_elo))

    def levels(self) -> list[dict]:
        return [{"gen": e.gen, "name": e.name, "elo": e.elo} for e in self.entries]

    def elo_curve(self) -> list[dict]:
        return [{"gen": e.gen, "elo": e.elo, **e.metrics} for e in self.entries]

    # --- persistence -------------------------------------------------------
    def save(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "random_anchor_elo": self.random_anchor_elo,
            "champion_gen": self.champion_gen,
            "generations": [asdict(e) for e in self.entries],
        }
        self.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: str, *, random_anchor_elo: float = 600.0) -> Ladder:
        path = Path(run_dir) / "ladder.json"
        if not path.exists():
            return cls(run_dir=run_dir, random_anchor_elo=random_anchor_elo)
        data = json.loads(path.read_text())
        entries = [LadderEntry(**e) for e in data.get("generations", [])]
        return cls(
            run_dir=run_dir,
            random_anchor_elo=data.get("random_anchor_elo", random_anchor_elo),
            entries=entries,
            champion_gen=data.get("champion_gen"),
        )
