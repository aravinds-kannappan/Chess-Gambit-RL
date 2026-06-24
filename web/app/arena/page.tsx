"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

interface Ladder { generations?: number; best_elo?: number; calibrated_elo?: number }

// Playable strength tiers (not a list of training runs). Serving scales the base
// net to any of these Elos, so each is a level you can actually challenge.
const TIERS = [
  { name: "Elite", elo: 2300, ic: "♚", style: "Tournament prep. Punishes every loose move.", color: "#ff8fd0" },
  { name: "Master", elo: 2000, ic: "♛", style: "Sharp tactics and clean conversion.", color: "#f85149" },
  { name: "Expert", elo: 1700, ic: "♜", style: "Solid plans, rarely blunders.", color: "#b06dff" },
  { name: "Club", elo: 1400, ic: "♝", style: "Knows the ideas, makes the odd slip.", color: "#f5a623" },
  { name: "Casual", elo: 1100, ic: "♞", style: "Plays naturally, forgives mistakes.", color: "#3fb950" },
  { name: "Novice", elo: 800, ic: "♟", style: "Gentle. A friendly first opponent.", color: "#6db0ff" },
];

export default function TiersLadderPage() {
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

  const reach = Math.round(data?.calibrated_elo ?? data?.best_elo ?? 0);

  return (
    <main className="container">
      <h1 className="title" style={{ fontSize: "2.1rem" }}>Challenge <span>tiers</span></h1>
      <p className="subtitle" style={{ textAlign: "left", margin: "0.3rem 0 1.4rem" }}>
        Pick a level and play. The agent is the strong pre-trained net scaled to that
        Elo - not a list of training runs. {live && reach ? `Right now it benchmarks around ${reach} Elo.` : ""}
      </p>

      <div className="rungs">
        {TIERS.map((t) => {
          const reachable = reach > 0 && reach >= t.elo;
          return (
            <div key={t.name} className={`rung ${reachable ? "best" : ""}`}>
              <span className="fill" style={{ width: `${Math.min(100, (t.elo / 2300) * 100)}%`, background: `linear-gradient(90deg, ${t.color}22, transparent)` }} />
              <span style={{ fontSize: "1.7rem" }}>{t.ic}</span>
              <span>
                <span style={{ fontWeight: 700, color: t.color, fontSize: "1.05rem" }}>{t.name}</span>
                <span className="elo" style={{ marginLeft: "0.6rem" }}>{t.elo}</span>
                <div className="muted" style={{ fontSize: "0.85rem" }}>{t.style}</div>
              </span>
              <Link href="/play" className="btn secondary" style={{ marginLeft: "auto" }}>Challenge</Link>
            </div>
          );
        })}
      </div>

      <div className="card" style={{ marginTop: "1.4rem" }}>
        <h2>How a tier is built</h2>
        <p className="muted" style={{ margin: 0 }}>
          One strong network (pre-trained on real games, refined by self-play) is throttled
          to each tier with search depth and a calibrated blunder rate, then graded by
          Stockfish so the Elo on the label is real - not a self-reported number.
        </p>
      </div>
    </main>
  );
}
