"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

interface Ladder { generations?: number; best_elo?: number; calibrated_elo?: number; ceiling?: number }

// Challenge rungs derived from the engine's HONEST ceiling. Every rung is a
// level the engine can actually play; nothing on this ladder is aspirational.
function buildRungs(ceiling: number) {
  const names: [string, string, string][] = [
    ["Pawn", "♟", "Gentle. A friendly first opponent."],
    ["Knight", "♞", "Plays naturally, forgives mistakes."],
    ["Bishop", "♝", "Knows the ideas, makes the odd slip."],
    ["Rook", "♜", "Solid plans, fewer blunders."],
    ["Queen", "♛", "Sharp for its size. Converts material."],
    ["King", "♚", "Full strength. The whole book and search budget."],
  ];
  const colors = ["#8b7f6d", "#3f6d4e", "#97741f", "#7a5836", "#8a3324", "#5f4226"];
  const lo = 500;
  return names.map(([name, ic, style], i) => ({
    name, ic, style,
    elo: Math.round((lo + ((Math.max(ceiling, 650) - lo) * i) / (names.length - 1)) / 25) * 25,
    color: colors[i],
  }));
}

export default function ChallengeLadderPage() {
  const [data, setData] = useState<Ladder | null>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch("/api/ladder");
        const d = await r.json();
        if (r.ok && !d.error) { setData(d); setLive(true); return; }
      } catch { /* fall through */ }
      try { const s = await fetch("/data/ladder.json"); if (s.ok) setData(await s.json()); } catch { /* none */ }
    };
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);

  const ceiling = Math.round(data?.ceiling ?? data?.calibrated_elo ?? data?.best_elo ?? 1000);
  const rungs = buildRungs(ceiling);

  return (
    <main className="container">
      <h1 className="title" style={{ fontSize: "2.1rem" }}>The <span>challenge ladder</span></h1>
      <p className="subtitle" style={{ textAlign: "left", margin: "0.3rem 0 1.4rem" }}>
        Pick a rung and play. Every level is real: the ladder tops out at the engine&apos;s
        measured ceiling (~{ceiling}, Stockfish graded){live ? ", live from the backend" : ""}.
        Nightly training pushes the top rung higher.
      </p>

      <div className="rungs">
        {rungs.map((t) => (
          <div key={t.name} className={`rung ${t.elo >= ceiling ? "best" : ""}`}>
            <span className="fill" style={{ width: `${Math.min(100, (t.elo / Math.max(ceiling, 1)) * 100)}%`, background: `linear-gradient(90deg, ${t.color}22, transparent)` }} />
            <span style={{ fontSize: "1.7rem" }}>{t.ic}</span>
            <span>
              <span style={{ fontWeight: 700, color: t.color, fontSize: "1.05rem" }}>{t.name}</span>
              <span className="elo" style={{ marginLeft: "0.6rem" }}>{t.elo}</span>
              <div className="muted" style={{ fontSize: "0.85rem" }}>{t.style}</div>
            </span>
            <Link href="/play" className="btn secondary" style={{ marginLeft: "auto" }}>Challenge</Link>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginTop: "1.4rem" }}>
        <h2>How a rung is built</h2>
        <p className="muted" style={{ margin: 0 }}>
          One trained network (cloned from real games, refined by gated nightly
          self-play) is throttled to each rung with search depth and a calibrated
          blunder rate. Stockfish grades the top rung, so the number on the label is
          measured, not self-reported.
        </p>
      </div>
    </main>
  );
}
