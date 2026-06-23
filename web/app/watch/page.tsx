"use client";

import { useCallback, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";

export default function WatchPage() {
  const gameRef = useRef(new Chess());
  const playingRef = useRef(false);
  const [fen, setFen] = useState(gameRef.current.fen());
  const [whiteElo, setWhiteElo] = useState(1200);
  const [blackElo, setBlackElo] = useState(800);
  const [status, setStatus] = useState("Set each side's Elo and press Play.");
  const [playing, setPlaying] = useState(false);
  const [served, setServed] = useState<{ white?: number; black?: number }>({});

  const stop = useCallback(() => {
    playingRef.current = false;
    setPlaying(false);
  }, []);

  const reset = useCallback(() => {
    stop();
    gameRef.current = new Chess();
    setFen(gameRef.current.fen());
    setServed({});
    setStatus("Set each side's Elo and press Play.");
  }, [stop]);

  const step = useCallback(async () => {
    const game = gameRef.current;
    if (!playingRef.current) return;
    if (game.isGameOver()) {
      stop();
      setStatus(`Game over: ${game.isCheckmate() ? "checkmate" : "draw"}.`);
      return;
    }
    const turn = game.turn();
    try {
      const res = await fetch("/api/watch-move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: game.fen(), whiteElo, blackElo }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        stop();
        setStatus(`Backend warming up - try again shortly. (${data.reason ?? res.status})`);
        return;
      }
      game.move({ from: data.move.slice(0, 2), to: data.move.slice(2, 4), promotion: data.move.slice(4) || undefined });
      setServed((s) => ({ ...s, [turn === "w" ? "white" : "black"]: data.elo }));
      setFen(game.fen());
      setStatus(`${turn === "w" ? "White" : "Black"} (gen ${data.gen}, Elo ${Math.round(data.elo)}) played ${data.move}.`);
      setTimeout(step, 500);
    } catch {
      stop();
      setStatus("Backend unreachable.");
    }
  }, [whiteElo, blackElo, stop]);

  const play = useCallback(() => {
    if (gameRef.current.isGameOver()) reset();
    playingRef.current = true;
    setPlaying(true);
    void step();
  }, [step, reset]);

  return (
    <main className="container">
      <h1 className="title">Watch mode</h1>
      <p className="subtitle">
        Pair two trained agents at chosen Elo levels and watch them play. The
        backend serves the nearest ladder snapshot, tuned to the target strength.
      </p>
      <div className="grid cols-2">
        <div className="card">
          <Chessboard position={fen} arePiecesDraggable={false} boardWidth={420}
            customBoardStyle={{ borderRadius: "8px" }} />
        </div>
        <div className="card">
          <h2>Match setup</h2>
          <label className="muted">White Elo: <b>{whiteElo}</b></label>
          <input type="range" min={500} max={1800} step={50} value={whiteElo}
            onChange={(e) => setWhiteElo(Number(e.target.value))} style={{ width: "100%" }} />
          <label className="muted">Black Elo: <b>{blackElo}</b></label>
          <input type="range" min={500} max={1800} step={50} value={blackElo}
            onChange={(e) => setBlackElo(Number(e.target.value))} style={{ width: "100%" }} />
          <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
            {!playing ? (
              <button className="btn" onClick={play}>Play</button>
            ) : (
              <button className="btn secondary" onClick={stop}>Pause</button>
            )}
            <button className="btn secondary" onClick={reset}>Reset</button>
          </div>
          <p className="pill" style={{ marginTop: "1rem" }}>{status}</p>
          {(served.white || served.black) && (
            <p className="muted">
              serving white≈{served.white && Math.round(served.white)} ·
              black≈{served.black && Math.round(served.black)}
            </p>
          )}
        </div>
      </div>
    </main>
  );
}
