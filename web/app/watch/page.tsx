"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { eloMove, evalPawns } from "@/app/lib/engine";

export default function WatchPage() {
  const gameRef = useRef(new Chess());
  const playingRef = useRef(false);
  const [fen, setFen] = useState(gameRef.current.fen());
  const [whiteElo, setWhiteElo] = useState(1600);
  const [blackElo, setBlackElo] = useState(1000);
  const [playing, setPlaying] = useState(false);
  const [status, setStatus] = useState("Set each side's Elo and press Play.");
  const [src, setSrc] = useState("");

  const stop = useCallback(() => { playingRef.current = false; setPlaying(false); }, []);
  const reset = useCallback(() => {
    stop(); gameRef.current = new Chess(); setFen(gameRef.current.fen());
    setStatus("Set each side's Elo and press Play."); setSrc("");
  }, [stop]);

  const step = useCallback(async () => {
    const game = gameRef.current;
    if (!playingRef.current) return;
    if (game.isGameOver() || game.history().length > 200) {
      stop(); setStatus(game.isCheckmate() ? `Checkmate - ${game.turn() === "w" ? "Black" : "White"} wins.` : "Draw.");
      return;
    }
    const turn = game.turn();
    const elo = turn === "w" ? whiteElo : blackElo;
    let uci: string | null = null;
    let source = "engine";
    try {
      const res = await fetch("/api/watch-move", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: game.fen(), whiteElo, blackElo }),
      });
      const data = await res.json();
      if (res.ok && !data.error && data.move) { uci = data.move; source = "agent"; }
    } catch { /* fall through */ }
    if (!uci) uci = eloMove(game.fen(), elo);
    if (!uci) { stop(); return; }
    game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.slice(4) || undefined });
    setFen(game.fen());
    setSrc(source);
    setStatus(`${turn === "w" ? "White" : "Black"} (~${elo}) played ${uci}.`);
    setTimeout(step, 600);
  }, [whiteElo, blackElo, stop]);

  const play = useCallback(() => {
    if (gameRef.current.isGameOver()) reset();
    playingRef.current = true; setPlaying(true); void step();
  }, [step, reset]);

  useEffect(() => () => { playingRef.current = false; }, []);

  const evalP = evalPawns(fen);
  const whiteHeight = Math.round(((Math.max(-6, Math.min(6, evalP)) + 6) / 12) * 100);

  return (
    <main className="container">
      <h1 className="title" style={{ fontSize: "2rem" }}>Featured <span>duel</span></h1>
      <p className="subtitle" style={{ textAlign: "left", margin: "0.3rem 0 0.6rem" }}>
        Two agents at the Elos you choose. Trained-network moves when the backend is
        live, otherwise the Elo-scaled engine - so strength always tracks the dial.
      </p>
      <p className="muted" style={{ margin: "0 0 1.3rem", fontSize: "0.9rem" }}>
        Prefer a panorama? <a href="/tiers">Watch every Elo tier at once →</a>
      </p>

      <div className="card" style={{ marginBottom: "1.2rem" }}>
        <div className="vs">
          <div className="who">
            <div className="avatar" style={{ background: "rgba(234,241,248,0.1)" }}>♔</div>
            <div className="nm">White</div>
            <div className="el">{whiteElo}</div>
          </div>
          <div className="mid">vs</div>
          <div className="who">
            <div className="avatar" style={{ background: "rgba(20,24,33,0.8)" }}>♚</div>
            <div className="nm">Black</div>
            <div className="el">{blackElo}</div>
          </div>
        </div>
      </div>

      <div className="split">
        <div className="board-wrap">
          <div className="eval-row">
            <div className="evalbar"><i style={{ height: `${whiteHeight}%` }} /></div>
            <div style={{ flex: 1 }}>
              <Chessboard position={fen} arePiecesDraggable={false} boardWidth={460}
                customBoardStyle={{ borderRadius: "12px" }}
                customDarkSquareStyle={{ backgroundColor: "#2b3344" }}
                customLightSquareStyle={{ backgroundColor: "#cdd6e6" }} />
            </div>
          </div>
          <p className="pill" style={{ marginTop: "0.6rem" }}>
            {status}{src && <> · via <span className={src === "agent" ? "tag-hf" : "tag-fallback"}>{src}</span></>}
          </p>
        </div>

        <div className="panel-stack">
          <div className="card">
            <label className="muted">White Elo: <b>{whiteElo}</b></label>
            <input className="slider" type="range" min={600} max={2300} step={50} value={whiteElo}
              onChange={(e) => setWhiteElo(Number(e.target.value))} />
            <label className="muted" style={{ marginTop: "0.6rem", display: "block" }}>Black Elo: <b>{blackElo}</b></label>
            <input className="slider" type="range" min={600} max={2300} step={50} value={blackElo}
              onChange={(e) => setBlackElo(Number(e.target.value))} />
            <div className="row" style={{ marginTop: "1rem" }}>
              {!playing ? <button className="btn" onClick={play}>Play</button>
                : <button className="btn secondary" onClick={stop}>Pause</button>}
              <button className="btn secondary" onClick={reset}>Reset</button>
            </div>
          </div>
          <div className="card">
            <b>Evaluation</b>
            <p className="eval-num" style={{ marginTop: "0.4rem", color: evalP >= 0 ? "#eaf1f8" : "#f5a623" }}>
              {evalP >= 0 ? "+" : ""}{evalP.toFixed(1)} <span className="pill">(white advantage)</span>
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
