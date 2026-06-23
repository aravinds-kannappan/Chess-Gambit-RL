"use client";

import { useState } from "react";

interface PredictResult {
  bestMove: string;
  wdl: { win: number; draw: number; loss: number };
  value: number;
  rating: number;
  policyEntropyBits: number;
  source: string;
  error?: string;
}

const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export default function PredictPage() {
  const [input, setInput] = useState(START);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    setResult(null);
    const isPgn = input.includes("1.") || input.includes("\n");
    const body = isPgn ? { pgn: input } : { fen: input.trim() };
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await res.json();
      if (!res.ok || d.error) {
        setResult({ error: d.error === "backend warming up" ? "Backend warming up - try again shortly." : (d.error ?? "error") } as PredictResult);
      } else {
        // Normalize the Space's snake_case response.
        setResult({
          bestMove: d.best_move ?? d.bestMove ?? "",
          wdl: d.wdl ?? { win: 0, draw: 0, loss: 0 },
          value: d.value ?? 0,
          rating: d.rating ?? 0,
          policyEntropyBits: d.policy_entropy_bits ?? 0,
          source: d.source ?? "model",
        });
      }
    } finally {
      setLoading(false);
    }
  };

  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

  return (
    <main className="container">
      <h1 className="title">Live prediction</h1>
      <p className="subtitle">
        Paste a FEN or PGN to get win/draw/loss, the best move, and an estimated
        player rating from the prediction model.
      </p>

      <div className="card">
        <textarea
          className="fen"
          rows={3}
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <div style={{ marginTop: "0.75rem", display: "flex", gap: "0.5rem" }}>
          <button className="btn" onClick={run} disabled={loading}>
            {loading ? "Predicting…" : "Predict"}
          </button>
          <button className="btn secondary" onClick={() => setInput(START)}>
            Reset to start
          </button>
        </div>
      </div>

      {result && !result.error && (
        <div className="grid cols-2" style={{ marginTop: "1.25rem" }}>
          <div className="card">
            <h2>Outcome (side to move)</h2>
            <div className="bar" style={{ margin: "0.75rem 0" }}>
              <span style={{ width: pct(result.wdl.win), background: "var(--win)" }} />
              <span style={{ width: pct(result.wdl.draw), background: "var(--draw)" }} />
              <span style={{ width: pct(result.wdl.loss), background: "var(--loss)" }} />
            </div>
            <p className="muted">
              Win {pct(result.wdl.win)} · Draw {pct(result.wdl.draw)} · Loss{" "}
              {pct(result.wdl.loss)}
            </p>
            <p className="muted">value (tanh): {result.value.toFixed(3)}</p>
          </div>
          <div className="card">
            <h2>Best move &amp; rating</h2>
            <p style={{ fontSize: "1.6rem", margin: "0.4rem 0" }} className="mono">
              {result.bestMove || "-"}
            </p>
            <p className="muted">estimated rating: ~{Math.round(result.rating)} Elo</p>
            <p className="muted">policy entropy: {result.policyEntropyBits.toFixed(2)} bits</p>
            <p className="pill">
              source:{" "}
              <span className={result.source === "huggingface" ? "tag-hf" : "tag-fallback"}>
                {result.source}
              </span>
            </p>
          </div>
        </div>
      )}
      {result?.error && (
        <p className="muted" style={{ marginTop: "1rem" }}>Error: {result.error}</p>
      )}
    </main>
  );
}
