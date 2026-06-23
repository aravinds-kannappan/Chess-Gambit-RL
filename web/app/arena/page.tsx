"use client";

import { useEffect, useState } from "react";

interface Row {
  agent: string;
  elo: number;
  points: number;
  games: number;
}
interface Replay {
  white: string;
  black: string;
  result: string;
}
interface Arena {
  leaderboard: Row[];
  n_games: number;
  replays: Replay[];
  error?: string;
}

export default function ArenaPage() {
  const [data, setData] = useState<Arena | null>(null);

  useEffect(() => {
    fetch("/api/arena").then((r) => r.json()).then(setData).catch(() => setData(null));
  }, []);

  if (!data) return <main className="container"><p className="muted">Loading…</p></main>;
  if (data.error)
    return (
      <main className="container">
        <h1 className="title">Agent arena</h1>
        <p className="muted">{data.error}</p>
      </main>
    );

  return (
    <main className="container">
      <h1 className="title">Agent arena</h1>
      <p className="subtitle">
        Elo from {data.n_games} round-robin games between the agents.
      </p>

      <div className="card">
        <table className="lb">
          <thead>
            <tr>
              <th>#</th>
              <th>Agent</th>
              <th>Elo</th>
              <th>Points</th>
              <th>Games</th>
            </tr>
          </thead>
          <tbody>
            {data.leaderboard.map((row, i) => (
              <tr key={row.agent}>
                <td>{i + 1}</td>
                <td><span className="badge">{row.agent}</span></td>
                <td className="mono">{row.elo.toFixed(0)}</td>
                <td>{row.points}</td>
                <td>{row.games}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.replays?.length > 0 && (
        <div className="card" style={{ marginTop: "1.25rem" }}>
          <h2>Recent games</h2>
          <table className="lb">
            <thead>
              <tr><th>White</th><th>Black</th><th>Result</th></tr>
            </thead>
            <tbody>
              {data.replays.slice(0, 12).map((r, i) => (
                <tr key={i}>
                  <td>{r.white}</td>
                  <td>{r.black}</td>
                  <td className="mono">{r.result}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
