"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Chessboard } from "react-chessboard";
import { agentMove, evalPawns } from "@/app/lib/engine";
import { gameStore, useGameVersion } from "@/app/lib/game";

function sessionId(): string {
  if (typeof window === "undefined") return "anon";
  let id = localStorage.getItem("sg_session");
  if (!id) { id = "s_" + Math.random().toString(36).slice(2, 10); localStorage.setItem("sg_session", id); }
  return id;
}

const START: Record<string, number> = { p: 8, n: 2, b: 2, r: 2, q: 1 };
function captured(fen: string, color: "w" | "b"): string {
  const board = fen.split(" ")[0];
  const counts: Record<string, number> = { p: 0, n: 0, b: 0, r: 0, q: 0 };
  for (const ch of board) {
    const lower = ch.toLowerCase();
    if (lower in counts) {
      const isWhite = ch === ch.toUpperCase();
      if ((color === "w" && isWhite) || (color === "b" && !isWhite)) counts[lower]++;
    }
  }
  const glyph: Record<string, string> = color === "w"
    ? { q: "♕", r: "♖", b: "♗", n: "♘", p: "♙" }
    : { q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
  let out = "";
  for (const k of ["q", "r", "b", "n", "p"]) out += glyph[k].repeat(Math.max(0, START[k] - counts[k]));
  return out;
}

export default function PlayPage() {
  useGameVersion();
  const session = useMemo(sessionId, []);
  const [competitive, setCompetitive] = useState(false);
  const [userElo, setUserElo] = useState(1200);
  const [status, setStatus] = useState("Your move - you are White.");
  const [thinking, setThinking] = useState(false);
  const [src, setSrc] = useState<{ source: string; route?: string; elo: number } | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const [adaptInfo, setAdaptInfo] = useState("");

  const targetElo = competitive ? 2300 : userElo;
  const fen = gameStore.fen();
  const evalP = evalPawns(fen);

  useEffect(() => { setHistory(JSON.parse(localStorage.getItem("sg_winrates") || "[]")); }, []);

  const logIfOver = useCallback(async () => {
    if (!gameStore.isGameOver()) return;
    const res = gameStore.result();
    const result = res === "1-0" ? -1 : res === "0-1" ? 1 : 0; // agent is Black
    await fetch("/api/log-game", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: session, fens: gameStore.agentFens, moves: gameStore.agentMoves, result }),
    }).catch(() => {});
  }, [session]);

  const reply = useCallback(async () => {
    if (gameStore.isGameOver()) { setStatus("Game over."); await logIfOver(); return; }
    setThinking(true);
    try {
      const mv = await agentMove(gameStore.fen(), targetElo, session);
      gameStore.applyUci(mv.uci);
      setSrc({ source: mv.source, route: mv.route, elo: mv.elo });
      setStatus(gameStore.isGameOver() ? "Game over." : "Your move.");
      if (gameStore.isGameOver()) await logIfOver();
    } catch {
      setStatus("No reply available.");
    } finally {
      setThinking(false);
    }
  }, [targetElo, session, logIfOver]);

  const onDrop = useCallback((from: string, to: string) => {
    if (gameStore.turn() !== "w" || thinking) return false;
    if (!gameStore.tryUserMove(from, to)) return false;
    void reply();
    return true;
  }, [reply, thinking]);

  const retrain = async () => {
    setAdaptInfo("Fine-tuning on your games...");
    try {
      const res = await fetch("/api/adapt", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: session }),
      });
      const d = await res.json();
      if (!res.ok || d.error) { setAdaptInfo(`Backend warming up (${d.reason ?? d.error ?? res.status}).`); return; }
      const wr = Math.round((d.agent_win_rate ?? 0) * 100);
      const next = [...history, wr];
      setHistory(next); localStorage.setItem("sg_winrates", JSON.stringify(next));
      setAdaptInfo(`Adapted on ${d.n_games} games. Win-rate vs you: ${wr}%`);
    } catch { setAdaptInfo("Backend unreachable."); }
  };

  const whiteHeight = Math.round(((Math.max(-6, Math.min(6, evalP)) + 6) / 12) * 100);
  const moves = gameStore.history();

  return (
    <main className="container">
      <div className="split">
        <div className="board-wrap">
          <div className="eval-row">
            <div className="evalbar" title="evaluation"><i style={{ height: `${whiteHeight}%` }} /></div>
            <div style={{ flex: 1 }}>
              <Chessboard
                position={fen}
                onPieceDrop={onDrop}
                boardWidth={460}
                arePiecesDraggable={gameStore.turn() === "w" && !thinking}
                customBoardStyle={{ borderRadius: "12px" }}
                customDarkSquareStyle={{ backgroundColor: "#2b3344" }}
                customLightSquareStyle={{ backgroundColor: "#cdd6e6" }}
              />
            </div>
          </div>
          <div className="row" style={{ justifyContent: "space-between", marginTop: "0.7rem" }}>
            <span className="captured" title="you captured">{captured(fen, "b")}</span>
            <span className="eval-num" style={{ color: evalP >= 0 ? "#eaf1f8" : "#f5a623" }}>
              {evalP >= 0 ? "+" : ""}{evalP.toFixed(1)}
            </span>
            <span className="captured" title="agent captured">{captured(fen, "w")}</span>
          </div>
        </div>

        <div className="panel-stack">
          <div className="card">
            <div className="vs">
              <div className="who">
                <div className="avatar" style={{ background: "rgba(109,176,255,0.12)" }}>🧑</div>
                <div className="nm">You</div>
                <div className="el">{userElo}</div>
              </div>
              <div className="mid">vs</div>
              <div className="who">
                <div className="avatar" style={{ background: "rgba(176,109,255,0.14)" }}>{competitive ? "🔥" : "🤖"}</div>
                <div className="nm">{competitive ? "Tournament" : "Adaptive"} agent</div>
                <div className="el">{competitive ? 2300 : targetElo}</div>
              </div>
            </div>
            <p className="pill" style={{ textAlign: "center", marginTop: "0.6rem" }}>
              {thinking ? "agent thinking..." : status}
              {src && <> · via <span className={src.source === "agent" ? "tag-hf" : "tag-fallback"}>{src.route ?? src.source}</span></>}
            </p>
          </div>

          <div className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>Strength</b>
              <span className={`toggle ${competitive ? "on" : ""}`} role="button" tabIndex={0}
                onClick={() => setCompetitive((v) => !v)}>
                {competitive ? "🔥 Tournament" : "🤝 Adaptive"}
              </span>
            </div>
            {competitive ? (
              <p className="muted" style={{ marginTop: "0.5rem" }}>
                Full tournament strength (~2300). No adapting down - train for real competition.
              </p>
            ) : (
              <div style={{ marginTop: "0.6rem" }}>
                <div className="dial"><div className="v">{userElo}</div></div>
                <input className="slider" type="range" min={600} max={2200} step={50} value={userElo}
                  onChange={(e) => setUserElo(Number(e.target.value))} />
                <p className="muted">The agent meets you at this level and adapts as you improve.</p>
              </div>
            )}
            <div className="row" style={{ marginTop: "0.8rem" }}>
              <button className="btn" onClick={() => gameStore.reset()}>New game</button>
              <button className="btn secondary" onClick={retrain}>Retrain on my games</button>
            </div>
            {adaptInfo && <p className="muted" style={{ marginTop: "0.6rem" }}>{adaptInfo}</p>}
            {history.length > 0 && <p className="muted">win-rate vs you: {history.map((h) => `${h}%`).join(" → ")}</p>}
          </div>

          <div className="card">
            <b>Moves</b>
            <div className="movelist" style={{ marginTop: "0.5rem" }}>
              {moves.length === 0 && <span className="no">no moves yet</span>}
              {moves.map((m, i) => (
                <span key={i} className="mv">{i % 2 === 0 && <span className="no">{i / 2 + 1}.</span>} {m}</span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
