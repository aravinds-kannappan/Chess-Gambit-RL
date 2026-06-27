"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { eloMove } from "@/app/lib/engine";

type Ladder = { generations?: number; best_elo?: number | null; calibrated_elo?: number | null };

export default function Home() {
  const [ladder, setLadder] = useState<Ladder | null>(null);
  const [live, setLive] = useState(false);
  const [bw, setBw] = useState(440);
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
    const f = () => setBw(Math.max(260, Math.min(440, window.innerWidth - 80)));
    f();
    window.addEventListener("resize", f);
    return () => window.removeEventListener("resize", f);
  }, []);

  // pause the hero animation while it's scrolled out of view (keeps scroll smooth)
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
      if (!visibleRef.current) { setTimeout(step, 700); return; } // idle when offscreen
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
      setTimeout(step, 680);
    };
    const t = setTimeout(step, 500);
    return () => { alive = false; clearTimeout(t); };
  }, []);

  const rating = ladder?.calibrated_elo ?? ladder?.best_elo;
  const elo = rating ? Math.round(rating) : null;
  const gens = ladder?.generations ?? null;

  return (
    <main>
      <section className="wrap hero2">
        <div>
          <div className="kicker">{live ? "● serving live" : "chess · reinforcement learning"}</div>
          <h1 className="display" style={{ marginTop: "1.3rem" }}>
            A chess engine<br />that <em>learns</em>.
          </h1>
          <p className="lead" style={{ marginTop: "1.5rem" }}>
            Pick your level and play. An opening book and a network trained on real
            grandmaster games handle the opening and middlegame; exact solvers finish
            the endgame — and Stockfish grades the whole thing on a real Elo scale.
          </p>
          <div style={{ display: "flex", gap: "0.8rem", marginTop: "2.1rem", flexWrap: "wrap" }}>
            <Link href="/play" className="btn">Play the engine →</Link>
            <Link href="/watch" className="btn ghost">Watch it play</Link>
          </div>
          <div className="statline" style={{ marginTop: "2.6rem" }}>
            <span>elo <b>{elo ?? "—"}</b></span><span className="dot">·</span>
            <span>stockfish&nbsp;calibrated</span><span className="dot">·</span>
            <span>book <b>231</b></span><span className="dot">·</span>
            <span>generations <b>{gens ?? "—"}</b></span>
          </div>
        </div>

        <div className="board-stage" ref={stageRef}>
          <div className="ring" />
          <div className="board-wrap" style={{ width: bw + 18 }}>
            <Chessboard
              position={fen}
              arePiecesDraggable={false}
              boardWidth={bw}
              animationDuration={250}
              customBoardStyle={{ borderRadius: "8px" }}
              customDarkSquareStyle={{ backgroundColor: "#26241f" }}
              customLightSquareStyle={{ backgroundColor: "#cbb78f" }}
            />
          </div>
        </div>
      </section>

      <section className="band">
        <div className="band-inner">
          <div className="kicker">how it works</div>
          <div className="how">
            <div>
              <span className="hn">01</span>
              <h3>Opening book</h3>
              <p>The first moves come from a book learned from thousands of 2000+ rated games — sound, varied, and principled instead of offbeat.</p>
            </div>
            <div>
              <span className="hn">02</span>
              <h3>Trained network</h3>
              <p>A residual net pre-trained by behavioural cloning on real games, then refined by gated self-play that keeps a generation only if it actually wins.</p>
            </div>
            <div>
              <span className="hn">03</span>
              <h3>Exact endgames</h3>
              <p>Solved-endgame tablebases — Bellman dynamic programming — convert won positions perfectly once the board simplifies.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="band">
        <div className="band-inner ref">
          <div>
            <div className="kicker">the referee</div>
            <h2 className="display" style={{ fontSize: "clamp(2rem, 4.5vw, 3.2rem)", marginTop: "1rem" }}>
              Stockfish grades it.<br />It never plays for it.
            </h2>
          </div>
          <p className="lead">
            Every move comes from the trained agents — never from Stockfish. A separate
            evaluator throttles Stockfish to known Elo bands and plays gauntlets, turning
            the scores into one honest, calibrated rating: the number you see.
          </p>
        </div>
      </section>

      <section className="band">
        <div className="band-inner cta2">
          <h2 className="display" style={{ fontSize: "clamp(2.4rem, 6vw, 4.4rem)" }}>Your move.</h2>
          <Link href="/play" className="btn">Play now →</Link>
        </div>
      </section>
    </main>
  );
}
