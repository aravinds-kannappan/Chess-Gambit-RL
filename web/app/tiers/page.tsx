"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { eloMove } from "@/app/lib/engine";

// Six tables spaced from beginner up to the engine's HONEST ceiling (fetched
// live). No table is labelled above what the engine can actually play.
function buildTiers(ceiling: number) {
  const names = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
  const glyphs = ["♙", "♘", "♗", "♖", "♕", "♔"];
  const colors = ["#8b7f6d", "#3f6d4e", "#97741f", "#7a5836", "#8a3324", "#5f4226"];
  const lo = 500;
  return names.map((name, i) => ({
    name,
    glyph: glyphs[i],
    elo: Math.round((lo + ((Math.max(ceiling, 650) - lo) * i) / (names.length - 1)) / 25) * 25,
    color: colors[i],
  }));
}
type Tier = ReturnType<typeof buildTiers>[number];

type Counters = { games: number; moves: number; live: number; demo: number };

function TierBoard({
  tier,
  index,
  onMove,
  onGame,
}: {
  tier: Tier;
  index: number;
  onMove: (live: boolean) => void;
  onGame: () => void;
}) {
  const gameRef = useRef(new Chess());
  const aliveRef = useRef(true);
  const [fen, setFen] = useState(gameRef.current.fen());
  const [moveCount, setMoveCount] = useState(0);
  const [last, setLast] = useState<string>("opening");

  const step = useCallback(async () => {
    if (!aliveRef.current) return;
    const game = gameRef.current;
    if (game.isGameOver() || game.history().length > 160) {
      onGame();
      setLast(game.isCheckmate() ? "checkmate" : "draw, new game");
      setTimeout(() => {
        if (!aliveRef.current) return;
        gameRef.current = new Chess();
        setFen(gameRef.current.fen());
        setMoveCount(0);
        setTimeout(step, 400);
      }, 1400);
      return;
    }

    let uci: string | null = null;
    let live = false;
    try {
      const res = await fetch("/api/watch-move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: game.fen(), whiteElo: tier.elo, blackElo: tier.elo }),
      });
      const data = await res.json();
      if (res.ok && !data.error && data.move) {
        uci = data.move;
        live = true;
      }
    } catch {
      /* fall through to local self-play */
    }
    if (!uci) uci = eloMove(game.fen(), tier.elo); // local engine keeps the club open offline

    if (uci) {
      try {
        game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.slice(4) || undefined });
        setFen(game.fen());
        setMoveCount(game.history().length);
        setLast(live ? "trained agent" : "demo engine");
        onMove(live);
      } catch {
        /* illegal from backend: skip a beat */
      }
    }
    const delay = 650 + Math.random() * 500;
    setTimeout(step, delay);
  }, [tier.elo, onGame, onMove]);

  useEffect(() => {
    aliveRef.current = true;
    const start = setTimeout(step, 300 + index * 220); // stagger boards
    return () => {
      aliveRef.current = false;
      clearTimeout(start);
    };
  }, [step, index]);

  return (
    <div className="card tier-board">
      <div className="tier-head">
        <span style={{ color: tier.color, fontWeight: 700 }}>{tier.glyph} {tier.name}&apos;s table</span>
        <span className="elo-chip">~{tier.elo}</span>
      </div>
      <Chessboard
        position={fen}
        arePiecesDraggable={false}
        boardWidth={240}
        customBoardStyle={{ borderRadius: "3px" }}
        customDarkSquareStyle={{ backgroundColor: "#9a6b44" }}
        customLightSquareStyle={{ backgroundColor: "#e8d2a8" }}
      />
      <div className="row" style={{ justifyContent: "space-between", marginTop: "0.5rem" }}>
        <span className="pill mono">move {moveCount}</span>
        <span className="pill">{last}</span>
      </div>
    </div>
  );
}

export default function TiersPage() {
  const [c, setC] = useState<Counters>({ games: 0, moves: 0, live: 0, demo: 0 });
  const [ceiling, setCeiling] = useState(1000);

  useEffect(() => {
    let on = true;
    fetch("/api/ladder").then((r) => r.json()).then((d) => {
      const cl = Math.round(d?.ceiling ?? d?.calibrated_elo ?? d?.best_elo ?? 1000);
      if (on && cl > 0) setCeiling(cl);
    }).catch(() => {});
    return () => { on = false; };
  }, []);

  const onMove = useCallback((live: boolean) => {
    setC((p) => ({ ...p, moves: p.moves + 1, live: p.live + (live ? 1 : 0), demo: p.demo + (live ? 0 : 1) }));
  }, []);
  const onGame = useCallback(() => setC((p) => ({ ...p, games: p.games + 1 })), []);

  const sourced = c.live + c.demo;
  const livePct = sourced ? Math.round((c.live / sourced) * 100) : 0;
  const tiers = buildTiers(ceiling);

  return (
    <main className="container">
      <h1 className="title" style={{ fontSize: "2.2rem", textAlign: "left" }}>
        Every <span>table in the club</span>
      </h1>
      <p className="subtitle" style={{ margin: "0 0 1.5rem", textAlign: "left" }}>
        Six boards, six ratings, all playing at once, from beginner up to the engine&apos;s
        honest ceiling (~{ceiling}). Live trained-agent moves when the backend is awake,
        a labelled demo engine otherwise.
      </p>

      <div className="grid cols-4" style={{ marginBottom: "1.5rem" }}>
        <div className="card stat"><div className="num">{c.games}</div><div className="label">Games finished</div></div>
        <div className="card stat"><div className="num">{c.moves}</div><div className="label">Moves played</div></div>
        <div className="card stat"><div className="num">{tiers.length}</div><div className="label">Tables</div></div>
        <div className="card stat">
          <div className="num"><span className={`badge ${livePct > 0 ? "live" : ""}`}>{livePct}%</span></div>
          <div className="label">From backend</div>
        </div>
      </div>

      <div className="grid cols-3">
        {tiers.map((t, i) => (
          <TierBoard key={t.name} tier={t} index={i} onMove={onMove} onGame={onGame} />
        ))}
      </div>
    </main>
  );
}
