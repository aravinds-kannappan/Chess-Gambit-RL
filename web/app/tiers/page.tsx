"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { eloMove } from "@/app/lib/engine";

const TIERS = [
  { name: "Novice", elo: 800, color: "#6db0ff" },
  { name: "Casual", elo: 1100, color: "#3fb950" },
  { name: "Club", elo: 1400, color: "#f5a623" },
  { name: "Expert", elo: 1700, color: "#b06dff" },
  { name: "Master", elo: 2000, color: "#f85149" },
  { name: "Elite", elo: 2300, color: "#ff8fd0" },
];

type Counters = { games: number; moves: number; live: number; demo: number };

function TierBoard({
  tier,
  index,
  onMove,
  onGame,
}: {
  tier: (typeof TIERS)[number];
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
      setLast(game.isCheckmate() ? "checkmate" : "draw - new game");
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
    if (!uci) uci = eloMove(game.fen(), tier.elo); // Elo-scaled engine keeps the arena alive offline

    if (uci) {
      try {
        game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.slice(4) || undefined });
        setFen(game.fen());
        setMoveCount(game.history().length);
        setLast(live ? "agent move" : "demo move");
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
        <span style={{ color: tier.color, fontWeight: 700 }}>{tier.name}</span>
        <span className="elo-chip">~{tier.elo} Elo</span>
      </div>
      <Chessboard
        position={fen}
        arePiecesDraggable={false}
        boardWidth={240}
        customBoardStyle={{ borderRadius: "8px" }}
        customDarkSquareStyle={{ backgroundColor: "#26241f" }}
        customLightSquareStyle={{ backgroundColor: "#cbb78f" }}
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

  const onMove = useCallback((live: boolean) => {
    setC((p) => ({ ...p, moves: p.moves + 1, live: p.live + (live ? 1 : 0), demo: p.demo + (live ? 0 : 1) }));
  }, []);
  const onGame = useCallback(() => setC((p) => ({ ...p, games: p.games + 1 })), []);

  const sourced = c.live + c.demo;
  const livePct = sourced ? Math.round((c.live / sourced) * 100) : 0;

  return (
    <main className="container">
      <h1 className="title" style={{ fontSize: "2.2rem", textAlign: "left" }}>
        Agent <span>arena tiers</span>
      </h1>
      <p className="subtitle" style={{ margin: "0 0 1.5rem", textAlign: "left" }}>
        Many games at once, each at its own Elo tier. When the backend is live, every
        move is a trained agent&apos;s; otherwise the boards self-play locally so the
        arena never goes dark. Each finished game is fresh data for the agents to learn from.
      </p>

      <div className="grid cols-4" style={{ marginBottom: "1.5rem" }}>
        <div className="card stat"><div className="num">{c.games}</div><div className="label">Games completed</div></div>
        <div className="card stat"><div className="num">{c.moves}</div><div className="label">Moves played</div></div>
        <div className="card stat"><div className="num">{TIERS.length}</div><div className="label">Active tiers</div></div>
        <div className="card stat">
          <div className="num"><span className={`badge ${livePct > 0 ? "live" : ""}`}>{livePct}%</span></div>
          <div className="label">From backend</div>
        </div>
      </div>

      <div className="grid cols-3">
        {TIERS.map((t, i) => (
          <TierBoard key={t.name} tier={t} index={i} onMove={onMove} onGame={onGame} />
        ))}
      </div>
    </main>
  );
}
