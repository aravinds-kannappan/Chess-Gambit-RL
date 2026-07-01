"use client";

import { useEffect, useMemo, useState } from "react";
import { Chessboard } from "react-chessboard";
import { recommend, evalPawns, type Rec } from "@/app/lib/engine";
import { gameStore, useGameVersion } from "@/app/lib/game";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export default function PredictPage() {
  useGameVersion(); // re-render as the live game advances
  const [mode, setMode] = useState<"live" | "custom">("live");
  const [customFen, setCustomFen] = useState(START_FEN);
  type Pred = { bestMove: string; wdl: { win: number; draw: number; loss: number }; rating: number; value: number };
  const [pred, setPred] = useState<Pred | null>(null);
  const [hover, setHover] = useState<string | null>(null);

  const liveFen = gameStore.fen();
  const fen = mode === "live" ? liveFen : customFen;

  const recs: Rec[] = useMemo(() => recommend(fen, 5, 3), [fen]);
  const evalP = useMemo(() => evalPawns(fen), [fen]);

  // Trained-network prediction for this position (no-op if backend warming up).
  useEffect(() => {
    let on = true;
    setPred(null);
    fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen }),
    }).then((r) => r.json()).then((d) => {
      const w = d?.wdl ?? d;
      if (on && w && typeof w.win === "number") {
        setPred({
          bestMove: d.best_move ?? d.bestMove ?? "",
          wdl: { win: w.win, draw: w.draw, loss: w.loss },
          rating: d.rating ?? 0,
          value: d.value ?? 0,
        });
      }
    }).catch(() => {});
    return () => { on = false; };
  }, [fen]);

  const turn = fen.split(" ")[1] === "w" ? "White" : "Black";
  const lo = recs.length ? recs[recs.length - 1].score : 0;
  const span = recs.length ? Math.max(0.6, recs[0].score - lo) : 1;
  const whiteHeight = Math.round(((Math.max(-6, Math.min(6, evalP)) + 6) / 12) * 100);

  // board arrows: claret = trained-net best move, green = hovered/top engine rec
  const arrows: [string, string, string?][] = [];
  if (pred?.bestMove && pred.bestMove.length >= 4)
    arrows.push([pred.bestMove.slice(0, 2), pred.bestMove.slice(2, 4), "#8a3324"]);
  const previewUci = hover ?? recs[0]?.uci;
  if (previewUci && previewUci.length >= 4 && previewUci !== pred?.bestMove)
    arrows.push([previewUci.slice(0, 2), previewUci.slice(2, 4), "#3f6d4e"]);

  return (
    <main className="container">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
        <h1 className="title" style={{ fontSize: "2rem" }}>Move <span>recommendations</span></h1>
        <div className="row">
          <span className="chip" style={{ cursor: "pointer", borderColor: mode === "live" ? "var(--accent)" : undefined, color: mode === "live" ? "var(--text)" : undefined }}
            onClick={() => setMode("live")}>Live game</span>
          <span className="chip" style={{ cursor: "pointer", borderColor: mode === "custom" ? "var(--accent)" : undefined, color: mode === "custom" ? "var(--text)" : undefined }}
            onClick={() => setMode("custom")}>Custom FEN</span>
        </div>
      </div>
      <p className="subtitle" style={{ textAlign: "left", margin: "0.3rem 0 1.3rem" }}>
        {mode === "live"
          ? "Reading your ongoing game - recommendations update with every move, and the game stays put when you switch tabs."
          : "Paste any position to analyze it on its own."}
      </p>

      <div className="split">
        <div className="board-wrap">
          <div className="eval-row">
            <div className="evalbar"><i style={{ height: `${whiteHeight}%` }} /></div>
            <div style={{ flex: 1 }}>
              <Chessboard position={fen} arePiecesDraggable={false} boardWidth={460}
                customArrows={arrows as never}
                customBoardStyle={{ borderRadius: "3px" }}
                customDarkSquareStyle={{ backgroundColor: "#9a6b44" }}
                customLightSquareStyle={{ backgroundColor: "#e8d2a8" }} />
            </div>
          </div>
          {mode === "custom" ? (
            <input className="fen" style={{ marginTop: "0.7rem" }} value={customFen}
              onChange={(e) => setCustomFen(e.target.value)} spellCheck={false} />
          ) : (
            <div className="row" style={{ marginTop: "0.7rem", justifyContent: "space-between" }}>
              <span className="pill mono">move {gameStore.moveCount()}</span>
              <button className="btn secondary" onClick={() => gameStore.reset()}>New game</button>
            </div>
          )}
        </div>

        <div className="panel-stack">
          <div className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>{turn} to move</b>
              <span className="eval-num" style={{ color: evalP >= 0 ? "var(--ink)" : "var(--accent)", fontSize: "1.1rem" }}>
                {evalP >= 0 ? "+" : ""}{evalP.toFixed(1)}
              </span>
            </div>
            <div style={{ marginTop: "0.7rem" }}>
              {recs.length === 0 && <p className="muted">No legal moves - game over.</p>}
              {recs.map((r, i) => (
                <div key={r.uci} className={`rec ${i === 0 ? "top" : ""}`} style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHover(r.uci)} onMouseLeave={() => setHover(null)}>
                  <span className="san">{r.san}</span>
                  <span className="meter"><span style={{ width: `${Math.round(((r.score - lo) / span) * 100)}%` }} /></span>
                  <span className="sc">{r.score >= 0 ? "+" : ""}{r.score.toFixed(2)}</span>
                </div>
              ))}
              {recs.length > 0 && (
                <p className="pill" style={{ marginTop: "0.5rem" }}>
                  hover a move to preview it · <span style={{ color: "#8a3324" }}>claret</span> = network&apos;s pick
                </p>
              )}
            </div>
          </div>

          <div className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>Trained network</b>
              <span className={`badge ${pred ? "live" : ""}`}>{pred ? "live" : "warming"}</span>
            </div>
            {pred ? (
              <>
                {pred.bestMove && (
                  <p style={{ margin: "0.6rem 0 0.2rem" }}>
                    best move <span className="mono" style={{ color: "var(--accent)", fontSize: "1.2rem" }}>{pred.bestMove}</span>
                    {pred.rating ? <span className="pill"> · plays like ~{Math.round(pred.rating)} Elo</span> : null}
                  </p>
                )}
                <div className="bar" style={{ marginTop: "0.6rem", height: 12 }}>
                  <span style={{ width: `${pred.wdl.win * 100}%`, background: "var(--win)" }} />
                  <span style={{ width: `${pred.wdl.draw * 100}%`, background: "var(--draw)" }} />
                  <span style={{ width: `${pred.wdl.loss * 100}%`, background: "var(--loss)" }} />
                </div>
                <p className="muted" style={{ marginTop: "0.5rem" }}>
                  win {Math.round(pred.wdl.win * 100)}% · draw {Math.round(pred.wdl.draw * 100)}% · loss {Math.round(pred.wdl.loss * 100)}%
                </p>
              </>
            ) : (
              <p className="muted" style={{ marginTop: "0.5rem" }}>
                Backend warming up - the recommendations above come from the offline engine.
                Once the Space is live, the trained network&apos;s best move and outcome show here.
              </p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
