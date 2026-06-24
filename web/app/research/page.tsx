"use client";

import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

interface GenRow { gen: number; elo: number; loss_policy?: number; loss_value?: number; calibrated_elo?: number; acpl?: number; top1?: number }
interface Ladder { generations: number; best_elo?: number; calibrated_elo?: number; elo_curve?: GenRow[]; gauntlet?: { gen: number; elo: number }[] }
interface Analysis { feature_mutual_information_bits?: Record<string, number> }

const AGENTS = [
  { icon: "♛", name: "MDP", note: "exact endgames" },
  { icon: "♞", name: "PPO", note: "on-policy RL" },
  { icon: "♟", name: "Reward", note: "DQN, shaped" },
  { icon: "♚", name: "Neural", note: "self-play net" },
];

export default function DashboardPage() {
  const [ladder, setLadder] = useState<Ladder | null>(null);
  const [live, setLive] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch("/api/ladder");
        const d = await r.json();
        if (r.ok && d.elo_curve) { setLadder(d); setLive(true); return; }
      } catch { /* fall through */ }
      try { const s = await fetch("/data/ladder.json"); if (s.ok) { setLadder(await s.json()); } } catch { /* none */ }
    };
    load();
    fetch("/api/analysis").then((r) => r.json()).then(setAnalysis).catch(() => {});
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);

  const curve = ladder?.gauntlet ?? ladder?.elo_curve ?? [];
  const lossData = (ladder?.elo_curve ?? []).filter((r) => r.loss_policy != null);
  const mi = analysis?.feature_mutual_information_bits ?? {};
  const miData = Object.entries(mi).map(([name, v]) => ({ name, mi: Number(v.toFixed(4)) })).sort((a, b) => b.mi - a.mi);
  const rating = ladder?.calibrated_elo ?? ladder?.best_elo ?? null;

  return (
    <main className="container">
      <div className="card" style={{ marginBottom: "1.2rem" }}>
        <div className="split" style={{ gridTemplateColumns: "260px 1fr", alignItems: "center" }}>
          <div style={{ textAlign: "center" }}>
            <div className="label">{ladder?.calibrated_elo ? "Stockfish-calibrated Elo" : "Best ladder Elo"}</div>
            <div className="num" style={{ fontSize: "3.4rem" }}>{rating ? Math.round(rating) : "-"}</div>
            <span className={`badge ${live ? "live" : ""}`}>{live ? "training live" : "last published run"}</span>
          </div>
          <div>
            <h1 className="title" style={{ fontSize: "1.7rem", textAlign: "left", margin: "0 0 0.4rem" }}>
              Live <span>training dashboard</span>
            </h1>
            <p className="muted" style={{ margin: 0 }}>
              Agents improve by continuous self-play; Stockfish grades them on a real scale.
              Each position is handed to the agent that owns it:
            </p>
            <div className="chips" style={{ marginTop: "0.7rem" }}>
              {AGENTS.map((a) => (
                <span key={a.name} className="chip"><span style={{ fontSize: "1.1rem" }}>{a.icon}</span> <b>{a.name}</b> · {a.note}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid cols-4" style={{ marginBottom: "1.2rem" }}>
        <div className="card stat"><div className="num">{ladder?.generations ?? "-"}</div><div className="label">Generations</div></div>
        <div className="card stat"><div className="num">{curve.length ? Math.round(curve[curve.length - 1].elo) : "-"}</div><div className="label">Latest Elo</div></div>
        <div className="card stat"><div className="num">{ladder?.elo_curve?.find((r) => r.acpl != null)?.acpl ?? "-"}</div><div className="label">Centipawn loss</div></div>
        <div className="card stat"><div className="num">{miData.length ? miData[0].mi.toFixed(2) : "-"}</div><div className="label">Top feature MI (bits)</div></div>
      </div>

      <div className="card">
        <h2>Strength over generations</h2>
        <p className="muted">Anchored Elo per self-play generation. Noisy on a CPU budget; GPU bursts smooth and lift it.</p>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={curve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#222a36" />
            <XAxis dataKey="gen" stroke="#93a1b5" />
            <YAxis stroke="#93a1b5" domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: "#0b1018", border: "1px solid #2a313c", borderRadius: 10 }} />
            <Line type="monotone" dataKey="elo" stroke="#f5a623" strokeWidth={2.5} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid cols-2" style={{ marginTop: "1.2rem" }}>
        {lossData.length > 0 ? (
          <div className="card">
            <h2>Self-play training loss</h2>
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222a36" />
                <XAxis dataKey="gen" stroke="#93a1b5" />
                <YAxis stroke="#93a1b5" />
                <Tooltip contentStyle={{ background: "#0b1018", border: "1px solid #2a313c", borderRadius: 10 }} />
                <Line type="monotone" dataKey="loss_policy" stroke="#6db0ff" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="loss_value" stroke="#3fb950" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="card">
            <h2>How the Elo is earned</h2>
            <p className="muted">
              Stockfish is throttled to known Elo bands and the agent plays gauntlets
              against each; a Bradley-Terry fit turns the scores into one calibrated
              rating. The agents never use Stockfish to choose a move - it only grades them.
            </p>
          </div>
        )}

        {miData.length > 0 ? (
          <div className="card">
            <h2>Information theory: I(feature; result)</h2>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={miData} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222a36" />
                <XAxis type="number" stroke="#93a1b5" />
                <YAxis dataKey="name" type="category" stroke="#93a1b5" width={110} />
                <Tooltip contentStyle={{ background: "#0b1018", border: "1px solid #2a313c", borderRadius: 10 }} />
                <Bar dataKey="mi" fill="#b06dff" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="card">
            <h2>Information theory</h2>
            <p className="muted">Which board features carry the most information about who wins - published with the analysis run.</p>
          </div>
        )}
      </div>
    </main>
  );
}
