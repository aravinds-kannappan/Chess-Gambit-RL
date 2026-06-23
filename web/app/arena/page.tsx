"use client";

import { useEffect, useState } from "react";

interface Level { gen: number; name: string; elo: number }
interface Ladder { generations: number; best_elo?: number; levels?: Level[] }

export default function ArenaPage() {
  const [data, setData] = useState<Ladder | null>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch("/api/ladder");
        const d = await r.json();
        if (r.ok && d.levels) { setData(d); setLive(true); return; }
      } catch {/* fall through */}
      try {
        const s = await fetch("/data/ladder.json");
        if (s.ok) setData(await s.json());
      } catch {/* none */}
    };
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);

  if (!data) return <main className="container"><p className="muted">Loading…</p></main>;
  const rows = [...(data.levels ?? [])].sort((a, b) => b.elo - a.elo);

  return (
    <main className="container">
      <h1 className="title">Ladder</h1>
      <p className="subtitle">
        Every self-play generation is a rated checkpoint. {live ? "Live from the training backend." : "Latest published run."}
        {" "}Watch mode and graded play pick the snapshot nearest your chosen Elo.
      </p>
      <div className="card">
        <table className="lb">
          <thead>
            <tr><th>#</th><th>Generation</th><th>Elo</th></tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={row.gen}>
                <td>{i + 1}</td>
                <td><span className="badge">{row.name}</span></td>
                <td className="mono">{Math.round(row.elo)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
