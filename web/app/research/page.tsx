"use client";

import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

interface GenRow {
  gen: number;
  elo: number;
  loss_policy?: number;
  loss_value?: number;
  win_rate_vs_random?: number;
}
interface Ladder {
  generations: number;
  best_elo?: number;
  elo_curve?: GenRow[];
  gauntlet?: { gen: number; elo: number }[];
}
interface Analysis {
  feature_mutual_information_bits?: Record<string, number>;
  outcome_prior_entropy_bits?: number;
}

export default function ResearchPage() {
  const [ladder, setLadder] = useState<Ladder | null>(null);
  const [live, setLive] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch("/api/ladder");
        const d = await r.json();
        if (r.ok && d.elo_curve) { setLadder(d); setLive(true); return; }
      } catch {/* fall through to static */}
      try {
        const s = await fetch("/data/ladder.json");
        if (s.ok) setLadder(await s.json());
      } catch {/* none */}
    };
    load();
    fetch("/api/analysis").then((r) => r.json()).then(setAnalysis).catch(() => {});
    const t = setInterval(load, 30000); // poll for live training progress
    return () => clearInterval(t);
  }, []);

  const curve = ladder?.gauntlet ?? ladder?.elo_curve ?? [];
  const lossData = (ladder?.elo_curve ?? []).filter((r) => r.loss_policy != null);
  const mi = analysis?.feature_mutual_information_bits ?? {};
  const miData = Object.entries(mi).map(([name, v]) => ({ name, mi: Number(v.toFixed(4)) }))
    .sort((a, b) => b.mi - a.mi);

  return (
    <main className="container">
      <h1 className="title">Research</h1>
      <p className="subtitle">
        The agents train continuously by self-play; every generation is versioned
        and rated. {live ? "Live from the training backend." : "Latest published run."}
        {ladder ? ` ${ladder.generations} generations.` : ""}
      </p>

      <div className="card">
        <h2>Strength over generations (Elo)</h2>
        <p className="muted">
          Anchored Elo per self-play generation. On a small CPU budget this is
          noisy; GPU bursts and more generations smooth and lift it.
        </p>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={curve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a313c" />
            <XAxis dataKey="gen" stroke="#8b949e" />
            <YAxis stroke="#8b949e" domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #2a313c" }} />
            <Line type="monotone" dataKey="elo" stroke="#f5a623" strokeWidth={2} dot />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {lossData.length > 0 && (
        <div className="card" style={{ marginTop: "1.25rem" }}>
          <h2>Self-play training loss</h2>
          <p className="muted">Policy and value loss fall as the network fits its self-play targets.</p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a313c" />
              <XAxis dataKey="gen" stroke="#8b949e" />
              <YAxis stroke="#8b949e" />
              <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #2a313c" }} />
              <Line type="monotone" dataKey="loss_policy" stroke="#58a6ff" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="loss_value" stroke="#3fb950" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {miData.length > 0 && (
        <div className="card" style={{ marginTop: "1.25rem" }}>
          <h2>Information theory: I(feature; result)</h2>
          <p className="muted">Which board features carry the most information about who wins (bits).</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={miData} layout="vertical" margin={{ left: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a313c" />
              <XAxis type="number" stroke="#8b949e" />
              <YAxis dataKey="name" type="category" stroke="#8b949e" width={110} />
              <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #2a313c" }} />
              <Bar dataKey="mi" fill="#f5a623" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </main>
  );
}
