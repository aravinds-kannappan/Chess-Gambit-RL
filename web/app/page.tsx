"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Ladder = { generations?: number; best_elo?: number | null; calibrated_elo?: number | null; levels?: number[] };

const TILES = [
  { href: "/play", ic: "♟", h: "Play & grow", p: "An agent that meets your level and adapts. Competitive Mode goes tournament-strong.", c: "rgba(245,166,35,0.5)" },
  { href: "/tiers", ic: "⚔", h: "Arena tiers", p: "Many agent-vs-agent games at once across Elo tiers, streaming data back to train.", c: "rgba(176,109,255,0.5)" },
  { href: "/research", ic: "📈", h: "Live dashboard", p: "Elo per generation, Stockfish-calibrated ratings, loss curves, info-theory.", c: "rgba(109,176,255,0.5)" },
  { href: "/predict", ic: "🔮", h: "Predict", p: "Streaming move recommendations for your live game, plus any custom position.", c: "rgba(63,185,80,0.5)" },
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
        <span className="eyebrow">MDP · PPO · Reward · Stockfish-rated</span>
        <h1 className="title">Chess agents that<br /><span>learn as they play</span></h1>
        <p className="subtitle">
          A multi-agent engine that routes each position to the right method - exact
          dynamic programming in the endgame, learned RL in the middlegame - while
          Stockfish grades every agent on a real Elo scale.
        </p>
        <div className="cta-row">
          <Link href="/play" className="btn">Play an agent</Link>
          <Link href="/tiers" className="btn accent2">Watch the tiers</Link>
          <Link href="/research" className="btn secondary">Live dashboard</Link>
        </div>
      </section>

      <div className="grid cols-4">
        <div className="card stat"><div className="num">{gens || "-"}</div><div className="label">Generations</div></div>
        <div className="card stat"><div className="num">{best ?? "-"}</div><div className="label">{ladder?.calibrated_elo ? "Calibrated Elo" : "Best Elo"}</div></div>
        <div className="card stat"><div className="num">{tiers || "-"}</div><div className="label">Elo tiers</div></div>
        <div className="card stat"><div className="num"><span className={`badge ${live ? "live" : ""}`}>{live ? "live" : "warming"}</span></div><div className="label">Backend</div></div>
      </div>

      <h2 className="section-title"><span className="no">01</span>Step onto the board</h2>
      <div className="deck">
        {TILES.map((t) => (
          <Link key={t.href} href={t.href} className="tile">
            <span className="glowdot" style={{ background: t.c }} />
            <span className="ic">{t.ic}</span>
            <h3>{t.h}</h3>
            <p>{t.p}</p>
          </Link>
        ))}
      </div>

      <h2 className="section-title"><span className="no">02</span>How it works</h2>
      <div className="grid cols-2">
        <div className="card">
          <h2>One board, many minds</h2>
          <ul className="muted">
            <li><b>MDP agent</b> - exact Bellman optimal play in solved endgames (KRvK, KQvK).</li>
            <li><b>PPO &amp; reward agents</b> - on-policy and off-policy RL for the low-material regime.</li>
            <li><b>Neural net</b> - AlphaZero-lite self-play for the opening and middlegame.</li>
            <li>A <b>phase router</b> hands each position to the agent that owns it.</li>
          </ul>
        </div>
        <div className="card">
          <h2>Stockfish is the referee</h2>
          <ul className="muted">
            <li>The agents <b>never</b> use Stockfish to choose a move.</li>
            <li>A backend evaluator scores each agent separately: <b>centipawn loss</b>, top-1 agreement, and a <b>calibrated Elo</b> from gauntlets vs throttled Stockfish.</li>
            <li>That rating is what each agent plays at - and climbs as it learns.</li>
          </ul>
        </div>
      </div>
    </main>
  );
}
