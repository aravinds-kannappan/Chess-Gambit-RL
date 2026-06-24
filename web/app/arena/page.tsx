"use client";

import { useEffect, useState } from "react";

interface Level { gen: number; name: string; elo: number }
interface Ladder { generations: number; best_elo?: number; calibrated_elo?: number; levels?: Level[] }

export default function LadderPage() {
  const [data, setData] = useState<Ladder | null>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch("/api/ladder");
        const d = await r.json();
        if (r.ok && d.levels) { setData(d); setLive(true); return; }
      } catch { /* fall through */ }
      try { const s = await fetch("/data/ladder.json"); if (s.ok) setData(await s.json()); } catch { /* none */ }
    };
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);

  // Showing every generation is useless and laggy. Collapse to a handful of
  // distinct Elo tiers: keep the strongest checkpoint per ~150-Elo band, cap 8.
  const all = data?.levels ?? [];
  const byBand = new Map<number, Level>();
  for (const lvl of all) {
    const band = Math.round(lvl.elo / 150);
    const cur = byBand.get(band);
    if (!cur || lvl.elo > cur.elo) byBand.set(band, lvl);
  }
  const rows = [...byBand.values()].sort((a, b) => a.elo - b.elo).slice(-8);
  const max = rows.length ? Math.max(...rows.map((r) => r.elo)) : 1;
  const min = rows.length ? Math.min(...rows.map((r) => r.elo)) : 0;
  const bestElo = rows.length ? Math.max(...rows.map((r) => r.elo)) : 0;

  return (
    <main className="container">
      <h1 className="title" style={{ fontSize: "2rem" }}>The <span>ladder</span></h1>
      <p className="subtitle" style={{ textAlign: "left", margin: "0.3rem 0 1.4rem" }}>
        The strongest checkpoint in each Elo band, as distinct tiers (not every
        generation). {live ? "Live from the training backend." : "Latest published run."}
        {" "}Graded play scales the best net to your chosen Elo.
      </p>

      {!data ? (
        <p className="muted">Loading…</p>
      ) : rows.length === 0 ? (
        <div className="card"><p className="muted">No generations yet - the backend is warming up its first checkpoints.</p></div>
      ) : (
        <div className="card">
          <div className="rungs">
            {rows.map((row) => {
              const frac = max === min ? 1 : (row.elo - min) / (max - min);
              const isBest = row.elo === bestElo;
              return (
                <div key={row.gen} className={`rung ${isBest ? "best" : ""}`}>
                  <span className="fill" style={{ width: `${30 + frac * 70}%` }} />
                  <span className="gen">{row.name}</span>
                  <span className="elo" style={{ color: isBest ? "var(--accent)" : "var(--text)" }}>{Math.round(row.elo)}</span>
                  <span className="pill" style={{ marginLeft: "auto" }}>{isBest ? "★ current best" : `gen ${row.gen}`}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </main>
  );
}
