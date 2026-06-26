"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Ladder = { generations?: number; best_elo?: number | null; calibrated_elo?: number | null; levels?: number[] };

const TILES = [
  { href: "/play", ic: "♟", h: "Play", p: "An opponent that meets your level and adapts to how you play.", c: "rgba(245,166,35,0.5)" },
  { href: "/watch", ic: "♛", h: "Watch", p: "Pit two agents at the Elos you choose - or watch every tier at once.", c: "rgba(176,109,255,0.5)" },
  { href: "/research", ic: "♚", h: "Dashboard", p: "The Elo ladder, Stockfish-calibrated ratings, and per-phase accuracy.", c: "rgba(109,176,255,0.5)" },
];

export default function Home() {
  const [ladder, setLadder] = useState<Ladder | null>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let on = true;
    const tick = async () => {
      try {
        const res = await fetch("/api/ladder", { cache: "no-store" });
        const data = await res.json();
        if (on && res.ok && !data.error) { setLadder(data); setLive(true); }
        else if (on) setLive(false);
      } catch { if (on) setLive(false); }
    };
    tick();
    const id = setInterval(tick, 15000);
    return () => { on = false; clearInterval(id); };
  }, []);

  const gens = ladder?.generations ?? 0;
  const rating = ladder?.calibrated_elo ?? ladder?.best_elo;
  const best = rating ? Math.round(rating) : null;
  const tiers = ladder?.levels?.length ?? 0;

  return (
    <main className="container">
      <section className="hero">
        <span className="eyebrow">A chess AI that adapts to you</span>
        <h1 className="title">Play chess against<br /><span>an engine that learns</span></h1>
        <p className="subtitle">
          Pick your level and play. An opening book and a trained network handle the
          opening and middlegame; exact endgame solvers finish the job - and Stockfish
          grades the whole thing on a real Elo scale.
        </p>
        <div className="cta-row">
          <Link href="/play" className="btn">Play now</Link>
          <Link href="/watch" className="btn accent2">Watch a duel</Link>
        </div>
      </section>

      <div className="grid cols-4">
        <div className="card stat"><div className="num">{gens || "-"}</div><div className="label">Generations</div></div>
        <div className="card stat"><div className="num">{best ?? "-"}</div><div className="label">{ladder?.calibrated_elo ? "Calibrated Elo" : "Best Elo"}</div></div>
        <div className="card stat"><div className="num">{tiers || "-"}</div><div className="label">Elo tiers</div></div>
        <div className="card stat"><div className="num"><span className={`badge ${live ? "live" : ""}`}>{live ? "live" : "warming"}</span></div><div className="label">Backend</div></div>
      </div>

      <div className="deck" style={{ gridTemplateColumns: "repeat(3, 1fr)", marginTop: "2rem" }}>
        {TILES.map((t) => (
          <Link key={t.href} href={t.href} className="tile">
            <span className="glowdot" style={{ background: t.c }} />
            <span className="ic">{t.ic}</span>
            <h3>{t.h}</h3>
            <p>{t.p}</p>
          </Link>
        ))}
      </div>

      <div className="card" style={{ marginTop: "2rem" }}>
        <h2>How it works</h2>
        <ul className="muted" style={{ margin: 0 }}>
          <li>A <b>phase router</b> hands each position to the right method: an
            <b> opening book</b> for the first moves, a <b>trained network</b> for the
            middlegame, and <b>exact Bellman solvers</b> in the endgame.</li>
          <li>The network is pre-trained on real Lichess games, then refined by
            self-play with <b>champion gating</b> - a new generation is only served if
            it actually beats the current one, so strength never regresses.</li>
          <li><b>Stockfish is only the referee</b>: the agents never use it to choose a
            move; it grades them (centipawn loss, per-phase accuracy, calibrated Elo).</li>
        </ul>
      </div>

      <p className="muted" style={{ marginTop: "1.4rem", fontSize: "0.9rem" }}>
        More: <Link href="/predict">Predict a position</Link> ·{" "}
        <Link href="/tiers">Live agent tiers</Link> ·{" "}
        <Link href="/arena">Challenge ladder</Link>
      </p>
    </main>
  );
}
