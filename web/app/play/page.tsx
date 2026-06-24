"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";

function sessionId(): string {
  if (typeof window === "undefined") return "anon";
  let id = localStorage.getItem("sg_session");
  if (!id) { id = "s_" + Math.random().toString(36).slice(2, 10); localStorage.setItem("sg_session", id); }
  return id;
}

export default function PlayPage() {
  const game = useMemo(() => new Chess(), []);
  const session = useMemo(sessionId, []);
  const agentFens = useRef<string[]>([]);
  const agentMoves = useRef<string[]>([]);
  const [fen, setFen] = useState(game.fen());
  const [status, setStatus] = useState("Your move - you are White.");
  const [source, setSource] = useState("");
  const [adaptInfo, setAdaptInfo] = useState<string>("");
  const [history, setHistory] = useState<number[]>([]);
  const [competitive, setCompetitive] = useState(false);
  const [userElo, setUserElo] = useState(1200);
  const targetElo = competitive ? 2300 : userElo;

  useEffect(() => {
    setHistory(JSON.parse(localStorage.getItem("sg_winrates") || "[]"));
  }, []);

  const logIfOver = useCallback(async () => {
    if (!game.isGameOver()) return;
    // result for the agent (Black): +1 if White is checkmated.
    let result = 0;
    if (game.isCheckmate()) result = game.turn() === "w" ? 1 : -1;
    await fetch("/api/log-game", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: session, fens: agentFens.current, moves: agentMoves.current, result }),
    }).catch(() => {});
  }, [game, session]);

  const agentMove = useCallback(async () => {
    if (game.isGameOver()) { await logIfOver(); return; }
    const fenBefore = game.fen();
    try {
      const res = await fetch("/api/move", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: fenBefore, session, elo: targetElo }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setStatus(`Backend warming up - try again shortly. (${data.reason ?? res.status})`);
        return;
      }
      agentFens.current.push(fenBefore);
      agentMoves.current.push(data.move);
      game.move({ from: data.move.slice(0, 2), to: data.move.slice(2, 4), promotion: data.move.slice(4) || undefined });
      setSource(data.source);
      setFen(game.fen());
      if (game.isGameOver()) { setStatus("Game over."); await logIfOver(); }
      else setStatus("Your move.");
    } catch {
      setStatus("Backend unreachable.");
    }
  }, [game, session, logIfOver, targetElo]);

  const onDrop = useCallback((from: string, to: string) => {
    try {
      const mv = game.move({ from, to, promotion: "q" });
      if (!mv) return false;
    } catch { return false; }
    setFen(game.fen());
    void agentMove();
    return true;
  }, [game, agentMove]);

  const newGame = () => {
    game.reset(); agentFens.current = []; agentMoves.current = [];
    setSource(""); setStatus("Your move - you are White."); setFen(game.fen());
  };

  const retrain = async () => {
    setAdaptInfo("Fine-tuning on your games...");
    try {
      const res = await fetch("/api/adapt", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: session }),
      });
      const d = await res.json();
      if (!res.ok || d.error) { setAdaptInfo(`Could not adapt: ${d.error ?? d.reason ?? res.status}`); return; }
      const wr = d.agent_win_rate ?? 0;
      const next = [...history, Math.round(wr * 100)];
      setHistory(next); localStorage.setItem("sg_winrates", JSON.stringify(next));
      setAdaptInfo(`Adapted on ${d.n_games} games · loss ${d.loss_before} → ${d.loss_after} · its win-rate vs you so far: ${(wr * 100).toFixed(0)}%`);
    } catch { setAdaptInfo("Backend unreachable."); }
  };

  return (
    <main className="container">
      <h1 className="title">Play the adaptive agent</h1>
      <p className="subtitle">{status}</p>
      <div className="grid cols-2">
        <div className="card">
          <Chessboard position={fen} onPieceDrop={onDrop} boardWidth={420}
            customBoardStyle={{ borderRadius: "8px" }} />
        </div>
        <div className="card">
          <h2>This agent learns from you</h2>
          <p className="muted">
            You play White; the trained network replies as Black. Your games are
            logged to a personal checkpoint - press <b>Retrain</b> to fine-tune it
            on how you play.
          </p>
          <p className="pill">last move source: <span className={source === "personal" ? "tag-hf" : "tag-fallback"}>{source || "-"}</span></p>

          <div style={{ marginTop: "1rem", borderTop: "1px solid var(--border)", paddingTop: "1rem" }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>Competitive mode</b>
              <span
                className={`toggle ${competitive ? "on" : ""}`}
                onClick={() => setCompetitive((v) => !v)}
                role="button"
                tabIndex={0}
              >
                {competitive ? "🔥 Tournament" : "🤝 Adaptive"}
              </span>
            </div>
            {competitive ? (
              <p className="muted" style={{ marginTop: "0.5rem" }}>
                The agent plays at full tournament strength (~2300) - no mercy, no
                adapting down. Train here to prepare for real competition.
              </p>
            ) : (
              <div style={{ marginTop: "0.5rem" }}>
                <label className="muted">Match my level: <b>{userElo} Elo</b></label>
                <input type="range" min={600} max={2200} step={50} value={userElo}
                  onChange={(e) => setUserElo(Number(e.target.value))} style={{ width: "100%" }} />
                <p className="muted">The agent meets you at your level and adapts as you improve.</p>
              </div>
            )}
          </div>

          <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
            <button className="btn" onClick={newGame}>New game</button>
            <button className="btn secondary" onClick={retrain}>Retrain on my games</button>
          </div>
          {adaptInfo && <p className="muted" style={{ marginTop: "0.75rem" }}>{adaptInfo}</p>}
          {history.length > 0 && (
            <p className="muted">win-rate vs you over retrains: {history.map((h) => `${h}%`).join(" → ")}</p>
          )}
        </div>
      </div>
    </main>
  );
}
