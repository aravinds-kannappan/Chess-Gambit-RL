"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { eloMove } from "@/app/lib/engine";

type GenRow = { gen: number; elo: number; promoted?: boolean; created?: string };
type Ladder = {
  generations?: number;
  best_elo?: number | null;
  calibrated_elo?: number | null;
  ceiling?: number | null;
  elo_curve?: GenRow[];
};

export default function Home() {
  const [ladder, setLadder] = useState<Ladder | null>(null);
  const [live, setLive] = useState(false);
  const [bw, setBw] = useState(420);
  const gameRef = useRef(new Chess());
  const [fen, setFen] = useState(gameRef.current.fen());
  const stageRef = useRef<HTMLDivElement>(null);
  const visibleRef = useRef(true);

  // live stats from the backend (best-effort)
  useEffect(() => {
    let on = true;
    (async () => {
      try {
        const r = await fetch("/api/ladder", { cache: "no-store" });
        const d = await r.json();
        if (on && r.ok && !d.error) { setLadder(d); setLive(true); }
        else if (on) setLive(false);
      } catch { if (on) setLive(false); }
    })();
    return () => { on = false; };
  }, []);

  // responsive board size
  useEffect(() => {
    const f = () => setBw(Math.max(260, Math.min(420, window.innerWidth - 110)));
    f();
    window.addEventListener("resize", f);
    return () => window.removeEventListener("resize", f);
  }, []);

  // pause the demo game while it is scrolled out of view (keeps scroll smooth)
  useEffect(() => {
    const el = stageRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => { visibleRef.current = e.isIntersecting; }, { threshold: 0.08 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // a board that quietly plays itself (offline-safe ambience for the hero)
  useEffect(() => {
    let alive = true;
    const step = () => {
      if (!alive) return;
      if (!visibleRef.current) { setTimeout(step, 700); return; }
      const g = gameRef.current;
      if (g.isGameOver() || g.history().length > 120) {
        setTimeout(() => {
          if (!alive) return;
          gameRef.current = new Chess();
          setFen(gameRef.current.fen());
          setTimeout(step, 800);
        }, 1600);
        return;
      }
      const uci = eloMove(g.fen(), 1800);
      if (uci) {
        try {
          g.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.slice(4) || undefined });
          setFen(g.fen());
        } catch { /* skip a beat on a rare illegal */ }
      }
      setTimeout(step, 700);
    };
    const t = setTimeout(step, 500);
    return () => { alive = false; clearTimeout(t); };
  }, []);

  const rating = ladder?.calibrated_elo ?? ladder?.best_elo;
  const elo = rating ? Math.round(rating) : null;
  const gens = ladder?.generations ?? null;
  // scorebook rows: the most recent ladder entries, newest first
  const rows = (ladder?.elo_curve ?? []).slice(-6).reverse();
  const bestElo = ladder?.best_elo ?? -Infinity;

  return (
    <main>
      <section className="wrap hero2">
        <div>
          <div className="kicker">{live ? "the engine is at the table" : "a chess club with a learning engine"}</div>
          <h1 className="display" style={{ marginTop: "1.2rem" }}>
            A chess engine<br />that <em>learns</em>.
          </h1>
          <p className="lead" style={{ marginTop: "1.4rem" }}>
            Take a seat. The house engine opens from a book of master games, plays the
            middlegame with a network trained on real chess, and converts endgames with
            exact solvers. Every night it trains; Stockfish grades it, so the rating on
            the plate is honest.
          </p>
          <div style={{ display: "flex", gap: "0.8rem", marginTop: "2rem", flexWrap: "wrap" }}>
            <Link href="/play" className="btn">Challenge the engine</Link>
            <Link href="/watch" className="btn ghost">Watch a game</Link>
          </div>
          <div className="statline" style={{ marginTop: "2.4rem" }}>
            <span>rated <b>{elo ?? "..."}</b></span>
            <span className="dot">·</span>
            <span>stockfish graded</span>
            <span className="dot">·</span>
            <span>generation <b>{gens ?? "..."}</b></span>
            <span className="dot">·</span>
            <span>trains nightly</span>
          </div>
        </div>

        <div className="board-stage" ref={stageRef}>
          <div className="board-wrap" style={{ width: bw + 26 }}>
            <Chessboard
              position={fen}
              arePiecesDraggable={false}
              boardWidth={bw}
              animationDuration={250}
              customBoardStyle={{ borderRadius: "3px" }}
              customDarkSquareStyle={{ backgroundColor: "#9a6b44" }}
              customLightSquareStyle={{ backgroundColor: "#e8d2a8" }}
            />
          </div>
          <div className="plate">house engine · playing itself</div>
        </div>
      </section>

      <section className="band tint">
        <div className="band-inner">
          <div className="kicker">how it plays</div>
          <div className="how">
            <div>
              <span className="hn">I. OPENING</span>
              <span className="pc">♗</span>
              <h3>From the book</h3>
              <p>The first moves come from an opening book learned from thousands of 2000+ rated games. Sound, varied, principled.</p>
            </div>
            <div>
              <span className="hn">II. MIDDLEGAME</span>
              <span className="pc">♘</span>
              <h3>From the network</h3>
              <p>A residual net cloned from real games, then refined by nightly self-play. A new generation is kept only if it beats the champion.</p>
            </div>
            <div>
              <span className="hn">III. ENDGAME</span>
              <span className="pc">♖</span>
              <h3>From the tables</h3>
              <p>Solved endgames are played perfectly with exact dynamic programming. Won positions get converted, not fumbled.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="band">
        <div className="band-inner ref">
          <div>
            <div className="kicker">the arbiter</div>
            <h2 className="display" style={{ fontSize: "clamp(1.9rem, 4.2vw, 3rem)", marginTop: "0.9rem" }}>
              Stockfish grades it.<br />It never plays for it.
            </h2>
          </div>
          <p className="lead">
            Every move belongs to the trained agents. Stockfish sits outside the game as
            the arbiter: throttled to known ratings, it plays grading matches and signs
            off the calibrated Elo you see. No self-reported numbers.
          </p>
        </div>
      </section>

      <section className="band tint">
        <div className="band-inner">
          <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
            <div className="kicker">the scorebook</div>
            <Link href="/research" className="mono" style={{ fontSize: "0.8rem", color: "var(--accent)" }}>
              full scorebook ↗
            </Link>
          </div>
          <p className="lead" style={{ margin: "0.9rem 0 1.3rem" }}>
            Every training generation is an entry: its rating, and whether it earned the
            table. A generation is promoted only if it beats the sitting champion.
          </p>
          {rows.length > 0 ? (
            <table className="score">
              <thead>
                <tr><th>Gen</th><th>Rating</th><th>Verdict</th></tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.gen} className={r.elo >= bestElo ? "champ" : ""}>
                    <td>№ {r.gen}</td>
                    <td>{Math.round(r.elo)}</td>
                    <td>{r.elo >= bestElo ? "champion" : r.promoted ? "promoted" : "held off"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="pill">The scorebook opens when the backend wakes.</p>
          )}
        </div>
      </section>

      <section className="band">
        <div className="band-inner cta2">
          <h2 className="display" style={{ fontSize: "clamp(2.2rem, 5.5vw, 4rem)" }}>
            <em>Your move.</em>
          </h2>
          <Link href="/play" className="btn">Take a seat</Link>
        </div>
      </section>
    </main>
  );
}
