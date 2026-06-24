"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Ladder = {
  generations?: number;
  best_elo?: number | null;
  levels?: number[];
  elo_curve?: { gen: number; elo: number }[];
};

export default function Home() {
  const [ladder, setLadder] = useState<Ladder | null>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let on = true;
    const tick = async () => {
      try {
        const res = await fetch("/api/ladder", { cache: "no-store" });
        const data = await res.json();
        if (on && res.ok && !data.error) {
          setLadder(data);
          setLive(true);
        } else if (on) {
          setLive(false);
        }
      } catch {
        if (on) setLive(false);
      }
    };
    tick();
    const id = setInterval(tick, 15000);
    return () => {
      on = false;
      clearInterval(id);
    };
  }, []);

  const gens = ladder?.generations ?? 0;
  const best = ladder?.best_elo ? Math.round(ladder.best_elo) : null;
  const tiers = ladder?.levels?.length ?? 0;

  return (
    <main className="container">
      <section className="hero">
        <span className="eyebrow">MDP · PPO · Reward · Stockfish-rated</span>
        <h1 className="title">
          Chess agents that<br />
          <span>learn as they play</span>
        </h1>
        <p className="subtitle">
          A multi-agent engine that routes each position to the right method - exact
          dynamic programming in the endgame, learned RL in the middlegame - while
          Stockfish grades every agent on a real Elo scale. Play it, or watch tiers
          of agents train against each other in real time.
        </p>
        <div className="cta-row">
          <Link href="/play" className="btn">Play an agent</Link>
          <Link href="/tiers" className="btn accent2">Watch the tiers</Link>
          <Link href="/research" className="btn secondary">Live dashboard</Link>
        </div>
      </section>

      <div className="grid cols-4">
        <div className="card stat">
          <div className="num">{gens || "-"}</div>
          <div className="label">Generations</div>
        </div>
        <div className="card stat">
          <div className="num">{best ?? "-"}</div>
          <div className="label">Best Elo</div>
        </div>
        <div className="card stat">
          <div className="num">{tiers || "-"}</div>
          <div className="label">Elo tiers</div>
        </div>
        <div className="card stat">
          <div className="num">
            <span className={`badge ${live ? "live" : ""}`}>{live ? "live" : "warming"}</span>
          </div>
          <div className="label">Backend</div>
        </div>
      </div>

      <h2 className="section-title">What you can do</h2>
      <div className="grid cols-2">
        <Link href="/play" className="card hover" style={{ display: "block" }}>
          <h2>♟ Play &amp; grow</h2>
          <p className="muted">
            Face an agent that reads your level and adapts to it. Your games fine-tune
            a personal checkpoint - and a Competitive Mode cranks it to tournament
            strength when you want a real fight.
          </p>
        </Link>
        <Link href="/tiers" className="card hover" style={{ display: "block" }}>
          <h2>⚔ Agent arena tiers</h2>
          <p className="muted">
            Many games at once across Elo tiers, agent vs agent. Every game streams
            fresh training data back to the agents so the whole ladder keeps improving.
          </p>
        </Link>
        <Link href="/research" className="card hover" style={{ display: "block" }}>
          <h2>📈 Live dashboard</h2>
          <p className="muted">
            Elo per generation, falling loss curves, Stockfish-assessed ratings, and
            the information-theoretic analysis of real games - updating as it trains.
          </p>
        </Link>
        <Link href="/predict" className="card hover" style={{ display: "block" }}>
          <h2>🔮 Predict</h2>
          <p className="muted">
            Paste a FEN to get win/draw/loss, the best move, and a value estimate from
            the current best network.
          </p>
        </Link>
      </div>

      <h2 className="section-title">How it works</h2>
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
            <li>That rating is what each agent plays at - and what it climbs as it learns.</li>
          </ul>
        </div>
      </div>
    </main>
  );
}
