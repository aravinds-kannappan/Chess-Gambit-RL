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

  const rawCurve = ladder?.gauntlet ?? ladder?.elo_curve ?? [];
  // Overlay a running best-so-far so progress is legible despite noisy CPU self-play.
  let runBest = -Infinity;
  const curve = rawCurve.map((r) => { runBest = Math.max(runBest, r.elo); return { ...r, best: Math.round(runBest) }; });
  const peak = curve.length ? Math.max(...curve.map((r) => r.elo)) : null;
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
            <span className={`badge ${live ? "live" : ""}`}>{live ? "backend live" : "last published run"}</span>
          </div>
          <div>
            <h1 className="title" style={{ fontSize: "1.7rem", textAlign: "left", margin: "0 0 0.4rem" }}>
              The <span>scorebook</span>
            </h1>
            <p className="muted" style={{ margin: 0 }}>
              The engine trains nightly by gated self-play with human-game replay, and
              Stockfish signs the rating. Each position is handed to the agent that owns it:
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
        <div className="card stat"><div className="num">{peak != null ? Math.round(peak) : "-"}</div><div className="label">Peak Elo</div></div>
        <div className="card stat"><div className="num">{ladder?.generations ?? "-"}</div><div className="label">Generations</div></div>
        <div className="card stat"><div className="num">{ladder?.elo_curve?.find((r) => r.acpl != null)?.acpl ?? "-"}</div><div className="label">Centipawn loss</div></div>
        <div className="card stat"><div className="num">{miData.length ? miData[0].mi.toFixed(2) : "-"}</div><div className="label">Top feature MI (bits)</div></div>
      </div>

      <div className="card" style={{ marginBottom: "1.2rem" }}>
        <h2>Training pipeline</h2>
        <div className="grid cols-3" style={{ marginTop: "0.6rem" }}>
          {[
            { n: "01", t: "Pre-train", d: "Behavioural cloning on real Lichess games.", tag: "pretrain/model.pt" },
            { n: "02", t: "Post-train nightly", d: "Gated self-play with human-game replay. Promoted only on a win.", tag: "posttrain/gen-*.pt" },
            { n: "03", t: "Serve and grade", d: "The champion, scaled to an honest rating; Stockfish grades it.", tag: "/move · /calibrate" },
          ].map((s, i) => (
            <div key={s.n} style={{ position: "relative", padding: "0.4rem 0.2rem" }}>
              <div className="row" style={{ gap: "0.5rem" }}>
                <span className="mono" style={{ color: "var(--accent)" }}>{s.n}</span>
                <b>{s.t}</b>
                {i < 2 && <span className="muted" style={{ marginLeft: "auto" }}>→</span>}
              </div>
              <p className="muted" style={{ margin: "0.3rem 0 0.4rem" }}>{s.d}</p>
              <span className="chip mono" style={{ fontSize: "0.75rem" }}>{s.tag}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Strength over training</h2>
          {peak != null && <span className="chip">peak <b>{Math.round(peak)}</b> Elo</span>}
        </div>
        <p className="muted">
          Each point is a checkpoint, rated by real games (brass = the per-checkpoint
          measurement, small-sample noise; claret = best reached). The curve is anchored
          so the endpoint equals the live Stockfish-calibrated rating, and nightly
          training appends new entries.
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={curve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#d8cbb2" />
            <XAxis dataKey="gen" stroke="#6f6250" />
            <YAxis stroke="#6f6250" domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: "#faf5e8", border: "1px solid #cdbc9d", borderRadius: 10 }} />
            <Line type="monotone" dataKey="elo" stroke="#97741f" strokeWidth={1.5} dot={{ r: 2 }} opacity={0.7} />
            <Line type="monotone" dataKey="best" stroke="#8a3324" strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid cols-2" style={{ marginTop: "1.2rem" }}>
        {lossData.length > 0 ? (
          <div className="card">
            <h2>Self-play training loss</h2>
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d8cbb2" />
                <XAxis dataKey="gen" stroke="#6f6250" />
                <YAxis stroke="#6f6250" />
                <Tooltip contentStyle={{ background: "#faf5e8", border: "1px solid #cdbc9d", borderRadius: 10 }} />
                <Line type="monotone" dataKey="loss_policy" stroke="#8a3324" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="loss_value" stroke="#3f6d4e" dot={false} strokeWidth={2} />
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
                <CartesianGrid strokeDasharray="3 3" stroke="#d8cbb2" />
                <XAxis type="number" stroke="#6f6250" />
                <YAxis dataKey="name" type="category" stroke="#6f6250" width={110} />
                <Tooltip contentStyle={{ background: "#faf5e8", border: "1px solid #cdbc9d", borderRadius: 10 }} />
                <Bar dataKey="mi" fill="#7a5836" radius={[0, 6, 6, 0]} />
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
