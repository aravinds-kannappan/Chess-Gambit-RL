"use client";

import { type CSSProperties, useCallback, useEffect, useMemo, useState } from "react";
import { Chessboard } from "react-chessboard";
import { agentMove, evalPawns, recommend } from "@/app/lib/engine";
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

type Mode = "auto" | "manual" | "full";

export default function PlayPage() {
  useGameVersion();
  const session = useMemo(sessionId, []);
  const [mode, setMode] = useState<Mode>("auto");
  const [userElo, setUserElo] = useState(800);
  const [ceiling, setCeiling] = useState(1000);
  const [status, setStatus] = useState("Your move. You are White.");
  const [thinking, setThinking] = useState(false);
  const [src, setSrc] = useState<{ source: string; route?: string; elo: number; adapted?: boolean } | null>(null);
  const [adaptInfo, setAdaptInfo] = useState("");
  const [sel, setSel] = useState<string | null>(null);
  // rolling record of how far each of your moves fell short of best play
  const [losses, setLosses] = useState<number[]>([]);

  // the honest playable range comes from the backend's calibrated ceiling
  useEffect(() => {
    let on = true;
    fetch("/api/ladder").then((r) => r.json()).then((d) => {
      const c = Math.round(d?.ceiling ?? d?.calibrated_elo ?? d?.best_elo ?? 1000);
      if (on && c > 0) { setCeiling(c); setUserElo((v) => Math.min(v, c)); }
    }).catch(() => {});
    return () => { on = false; };
  }, []);

  // rough strength estimate from your recent move quality (centipawn loss)
  const acpl = losses.length >= 4 ? losses.reduce((a, b) => a + b, 0) / losses.length : null;
  const estimate = acpl === null ? null
    : Math.max(450, Math.min(ceiling, Math.round((1250 - 420 * acpl) / 25) * 25));

  const targetElo = mode === "full" ? ceiling : mode === "auto" ? (estimate ?? Math.min(800, ceiling)) : userElo;

  const logIfOver = useCallback(async () => {
    if (!gameStore.isGameOver()) return;
    const res = gameStore.result();
    const result = res === "1-0" ? -1 : res === "0-1" ? 1 : 0; // agent is Black
    try {
      const r = await fetch("/api/log-game", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: session, fens: gameStore.agentFens, moves: gameStore.agentMoves, result }),
      });
      const d = await r.json();
      if (r.ok && d.adapting) setAdaptInfo("Game recorded. The engine is fine-tuning on it now.");
    } catch { /* offline: nothing to record against */ }
  }, [session]);

  const reply = useCallback(async () => {
    if (gameStore.isGameOver()) { setStatus("Game over."); await logIfOver(); return; }
    setThinking(true);
    try {
      const mv = await agentMove(gameStore.fen(), targetElo, session);
      gameStore.applyUci(mv.uci);
      setSrc({ source: mv.source, route: mv.route, elo: mv.elo, adapted: mv.route === "personal" });
      setStatus(gameStore.isGameOver() ? "Game over." : "Your move.");
      if (gameStore.isGameOver()) await logIfOver();
    } catch {
      setStatus("No reply available.");
    } finally {
      setThinking(false);
    }
  }, [targetElo, session, logIfOver]);

  // score the user's move against the local engine's best before applying it
  const scoreUserMove = useCallback((from: string, to: string) => {
    const recs = recommend(gameStore.fen(), 10, 2);
    if (recs.length === 0) return;
    const uci = from + to;
    const chosen = recs.find((r) => r.uci.startsWith(uci));
    const loss = chosen ? Math.max(0, recs[0].score - chosen.score)
      : Math.max(0.8, recs[0].score - recs[recs.length - 1].score);
    setLosses((prev) => [...prev.slice(-15), Math.min(loss, 4)]);
  }, []);

  const applyUserMove = useCallback((from: string, to: string): boolean => {
    if (gameStore.turn() !== "w" || thinking) return false;
    scoreUserMove(from, to);
    if (!gameStore.tryUserMove(from, to)) return false;
    setSel(null);
    void reply();
    return true;
  }, [reply, thinking, scoreUserMove]);

  const onDrop = useCallback((from: string, to: string) => applyUserMove(from, to), [applyUserMove]);

  const onSquareClick = useCallback((square: string) => {
    if (gameStore.turn() !== "w" || thinking) return;
    if (sel && gameStore.movesFrom(sel).includes(square)) { applyUserMove(sel, square); return; }
    setSel(gameStore.movesFrom(square).length > 0 ? square : null);
  }, [sel, thinking, applyUserMove]);

  const onDragBegin = useCallback((_piece: string, square: string) => { setSel(square); }, []);

  const retrain = async () => {
    setAdaptInfo("Fine-tuning on your games...");
    try {
      const res = await fetch("/api/adapt", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: session }),
      });
      const d = await res.json();
      if (!res.ok || d.error) { setAdaptInfo(`Backend warming up (${d.reason ?? d.error ?? res.status}).`); return; }
      if (d.status === "adapting") { setAdaptInfo("Already fine-tuning in the background."); return; }
      const wr = d.agent_win_rate != null ? Math.round(d.agent_win_rate * 100) : null;
      setAdaptInfo(`Adapted on ${d.n_games} game${d.n_games === 1 ? "" : "s"}.` + (wr != null ? ` Engine win rate vs you: ${wr}%.` : ""));
    } catch { setAdaptInfo("Backend unreachable."); }
  };

  const fen = gameStore.fen();
  const evalP = evalPawns(fen);
  const whiteHeight = Math.round(((Math.max(-6, Math.min(6, evalP)) + 6) / 12) * 100);
  const moves = gameStore.history();

  // last-move highlight, selection, and legal-target dots
  const last = gameStore.lastMove();
  const squareStyles: Record<string, CSSProperties> = {};
  if (last) {
    squareStyles[last.from] = { background: "rgba(151, 116, 31, 0.25)" };
    squareStyles[last.to] = { background: "rgba(151, 116, 31, 0.45)" };
  }
  if (sel) {
    squareStyles[sel] = { background: "rgba(138, 51, 36, 0.4)" };
    for (const t of gameStore.movesFrom(sel)) {
      squareStyles[t] = { background: "radial-gradient(circle, rgba(138, 51, 36, 0.5) 22%, transparent 25%)" };
    }
  }

  const modeChip = (m: Mode, label: string) => (
    <span className={`toggle ${mode === m ? "on" : ""}`} role="button" tabIndex={0}
      onClick={() => setMode(m)}>{label}</span>
  );

  return (
    <main className="container">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline", marginBottom: "1.1rem" }}>
        <h1 className="title" style={{ margin: 0, fontSize: "1.9rem" }}>The <span>house table</span></h1>
        <span className="pill">engine ceiling: rated ~{ceiling}, Stockfish graded</span>
      </div>

      <div className="split">
        <div className="board-wrap">
          <div className="eval-row">
            <div className="evalbar" title="evaluation"><i style={{ height: `${whiteHeight}%` }} /></div>
            <div style={{ flex: 1 }}>
              <Chessboard
                position={fen}
                onPieceDrop={onDrop}
                onSquareClick={onSquareClick}
                onPieceDragBegin={onDragBegin}
                customSquareStyles={squareStyles}
                boardWidth={460}
                arePiecesDraggable={gameStore.turn() === "w" && !thinking}
                customBoardStyle={{ borderRadius: "3px" }}
                customDarkSquareStyle={{ backgroundColor: "#9a6b44" }}
                customLightSquareStyle={{ backgroundColor: "#e8d2a8" }}
              />
            </div>
          </div>
          <div className="row" style={{ justifyContent: "space-between", marginTop: "0.7rem" }}>
            <span className="captured" title="you captured">{captured(fen, "b")}</span>
            <span className="eval-num" style={{ color: "#e9dcbe" }}>
              {evalP >= 0 ? "+" : ""}{evalP.toFixed(1)}
            </span>
            <span className="captured" title="engine captured">{captured(fen, "w")}</span>
          </div>
        </div>

        <div className="panel-stack">
          <div className="card">
            <div className="vs">
              <div className="who">
                <div className="avatar">♔</div>
                <div className="nm">You</div>
                <div className="el">{estimate ?? "?"}</div>
              </div>
              <div className="mid">vs</div>
              <div className="who">
                <div className="avatar">♚</div>
                <div className="nm">House engine</div>
                <div className="el">{Math.round(Math.min(targetElo, ceiling))}</div>
              </div>
            </div>
            <p className="pill" style={{ textAlign: "center", marginTop: "0.6rem" }}>
              {thinking ? "engine thinking..." : status}
              {src && <> · via <span className={src.source === "agent" ? "tag-hf" : "tag-fallback"}>{src.adapted ? "adapted net" : src.route ?? src.source}</span></>}
            </p>
          </div>

          <div className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>Strength</b>
              <div className="row" style={{ gap: "0.4rem" }}>
                {modeChip("auto", "Match me")}
                {modeChip("manual", "Set level")}
                {modeChip("full", "Full")}
              </div>
            </div>
            {mode === "auto" && (
              <p className="muted" style={{ marginTop: "0.6rem" }}>
                The engine meets you where you play. {estimate
                  ? `From your last ${losses.length} moves you are playing around ${estimate} (rough estimate), so it plays ${Math.round(Math.min(targetElo, ceiling))}.`
                  : "Play a few moves and it will find your level."}
              </p>
            )}
            {mode === "manual" && (
              <div style={{ marginTop: "0.6rem" }}>
                <div className="dial"><div className="v">{userElo}</div></div>
                <input className="slider" type="range" min={450} max={ceiling} step={25} value={Math.min(userElo, ceiling)}
                  onChange={(e) => setUserElo(Number(e.target.value))} />
                <p className="muted">Capped at the engine&apos;s honest ceiling (~{ceiling}). Nightly training raises it.</p>
              </div>
            )}
            {mode === "full" && (
              <p className="muted" style={{ marginTop: "0.6rem" }}>
                Full strength: the whole book, the full search budget, rated ~{ceiling} by
                Stockfish. No fake numbers above what it can actually play.
              </p>
            )}
            <div className="row" style={{ marginTop: "0.8rem" }}>
              <button className="btn" onClick={() => { gameStore.reset(); setLosses([]); setStatus("Your move. You are White."); }}>New game</button>
              <button className="btn secondary" onClick={retrain}>Fine-tune now</button>
            </div>
            {adaptInfo && <p className="muted" style={{ marginTop: "0.6rem" }}>{adaptInfo}</p>}
            <p className="pill" style={{ marginTop: "0.5rem" }}>
              Finished games fine-tune a personal engine for this browser automatically.
            </p>
          </div>

          <div className="card">
            <b>Scoresheet</b>
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
