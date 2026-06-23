"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Analysis {
  n_positions: number;
  outcome_prior_entropy_bits: number;
  feature_mutual_information_bits: Record<string, number>;
  policy_entropy?: { mean_bits: number; median_bits: number; p10_bits: number; p90_bits: number };
  game_curve?: { entropy: number[]; info_gain: number[]; total_info_bits: number; decisive_ply: number };
  error?: string;
}

export default function AnalysisPage() {
  const [data, setData] = useState<Analysis | null>(null);

  useEffect(() => {
    fetch("/api/analysis").then((r) => r.json()).then(setData).catch(() => setData(null));
  }, []);

  if (!data) return <main className="container"><p className="muted">Loading…</p></main>;
  if (data.error)
    return (
      <main className="container">
        <h1 className="title">Information dashboard</h1>
        <p className="muted">{data.error}</p>
      </main>
    );

  const miData = Object.entries(data.feature_mutual_information_bits)
    .map(([name, mi]) => ({ name, mi: Number(mi.toFixed(4)) }))
    .sort((a, b) => b.mi - a.mi);

  const curveData =
    data.game_curve?.entropy.map((e, i) => ({
      ply: i,
      entropy: e,
      infoGain: data.game_curve!.info_gain[i],
    })) ?? [];

  return (
    <main className="container">
      <h1 className="title">Information dashboard</h1>
      <p className="subtitle">
        Computed over {data.n_positions.toLocaleString()} real positions. Baseline
        outcome entropy: {data.outcome_prior_entropy_bits.toFixed(3)} bits.
      </p>

      <div className="card">
        <h2>Which features carry the result? - Mutual information I(feature; outcome)</h2>
        <p className="muted">Higher bits = the feature tells you more about who wins.</p>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={miData} layout="vertical" margin={{ left: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a313c" />
            <XAxis type="number" stroke="#8b949e" />
            <YAxis dataKey="name" type="category" stroke="#8b949e" width={110} />
            <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #2a313c" }} />
            <Bar dataKey="mi" fill="#f5a623" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid cols-2" style={{ marginTop: "1.25rem" }}>
        {data.policy_entropy && (
          <div className="card">
            <h2>Learned policy entropy</h2>
            <p className="muted">How undecided the trained policy is per position (bits).</p>
            <p style={{ fontSize: "2rem", margin: "0.3rem 0" }}>
              {data.policy_entropy.mean_bits.toFixed(2)}
              <span className="muted" style={{ fontSize: "1rem" }}> bits mean</span>
            </p>
            <p className="muted">
              p10 {data.policy_entropy.p10_bits.toFixed(2)} · median{" "}
              {data.policy_entropy.median_bits.toFixed(2)} · p90{" "}
              {data.policy_entropy.p90_bits.toFixed(2)}
            </p>
          </div>
        )}
        {data.game_curve && (
          <div className="card">
            <h2>Where the game was decided</h2>
            <p className="muted">
              Total information gained: {data.game_curve.total_info_bits.toFixed(2)} bits ·
              decisive ply: {data.game_curve.decisive_ply}
            </p>
          </div>
        )}
      </div>

      {curveData.length > 0 && (
        <div className="card" style={{ marginTop: "1.25rem" }}>
          <h2>Outcome-uncertainty collapse (one game)</h2>
          <p className="muted">Outcome entropy per ply; it falls as the result becomes clear.</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={curveData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a313c" />
              <XAxis dataKey="ply" stroke="#8b949e" />
              <YAxis stroke="#8b949e" />
              <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #2a313c" }} />
              <Line type="monotone" dataKey="entropy" stroke="#58a6ff" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </main>
  );
}
