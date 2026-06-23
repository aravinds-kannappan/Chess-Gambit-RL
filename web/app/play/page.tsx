"use client";

import { useCallback, useMemo, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";

export default function PlayPage() {
  const game = useMemo(() => new Chess(), []);
  const [fen, setFen] = useState(game.fen());
  const [status, setStatus] = useState("Your move - you are White.");
  const [thinking, setThinking] = useState(false);
  const [source, setSource] = useState<string>("");

  const refresh = useCallback(() => {
    setFen(game.fen());
    if (game.isCheckmate()) setStatus("Checkmate.");
    else if (game.isDraw()) setStatus("Draw.");
    else if (game.isCheck()) setStatus("Check.");
  }, [game]);

  const agentMove = useCallback(async () => {
    if (game.isGameOver()) return;
    setThinking(true);
    try {
      const res = await fetch("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: game.fen() }),
      });
      const data = await res.json();
      if (data.move) {
        game.move({
          from: data.move.slice(0, 2),
          to: data.move.slice(2, 4),
          promotion: data.move.slice(4) || undefined,
        });
        setSource(data.source);
        refresh();
        if (!game.isGameOver()) setStatus("Your move.");
      }
    } finally {
      setThinking(false);
    }
  }, [game, refresh]);

  const onDrop = useCallback(
    (sourceSquare: string, targetSquare: string) => {
      try {
        const move = game.move({ from: sourceSquare, to: targetSquare, promotion: "q" });
        if (!move) return false;
      } catch {
        return false;
      }
      refresh();
      void agentMove();
      return true;
    },
    [game, refresh, agentMove],
  );

  const reset = () => {
    game.reset();
    setSource("");
    setStatus("Your move - you are White.");
    setFen(game.fen());
  };

  return (
    <main className="container">
      <h1 className="title">Play the agent</h1>
      <p className="subtitle">{status}</p>
      <div className="grid cols-2">
        <div className="card">
          <Chessboard
            position={fen}
            onPieceDrop={onDrop}
            boardWidth={420}
            customBoardStyle={{ borderRadius: "8px" }}
          />
        </div>
        <div className="card">
          <h2>Agent</h2>
          <p className="muted">
            You play White; the agent replies as Black via <span className="mono">/api/move</span>.
          </p>
          <p className="pill">
            last move source:{" "}
            {source ? (
              <span className={source === "huggingface" ? "tag-hf" : "tag-fallback"}>{source}</span>
            ) : (
              "-"
            )}
          </p>
          <p className="pill">{thinking ? "agent is thinking…" : "ready"}</p>
          <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
            <button className="btn" onClick={reset}>New game</button>
          </div>
          <p className="muted mono" style={{ marginTop: "1rem", fontSize: "0.75rem", wordBreak: "break-all" }}>
            {fen}
          </p>
        </div>
      </div>
    </main>
  );
}
