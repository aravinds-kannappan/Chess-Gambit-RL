import Link from "next/link";

export default function Home() {
  return (
    <main className="container">
      <h1 className="title">
        Shannon&apos;s <span>Gambit</span>
      </h1>
      <p className="subtitle">
        A chess intelligence that trains itself. Agents improve by continuous
        self-play on Hugging Face, every generation is versioned and rated, and
        the model adapts to how you play.
      </p>

      <div className="grid cols-2">
        <Link href="/play" className="card" style={{ display: "block" }}>
          <h2>♟ Play an agent that learns</h2>
          <p className="muted">
            Play the trained network; your games train a personal checkpoint that
            adapts to your style. Press Retrain to fine-tune it on how you play.
          </p>
        </Link>
        <Link href="/watch" className="card" style={{ display: "block" }}>
          <h2>👀 Watch mode</h2>
          <p className="muted">
            Pair two agents at chosen Elo levels and watch them play. The backend
            serves the nearest ladder snapshot tuned to each target strength.
          </p>
        </Link>
        <Link href="/research" className="card" style={{ display: "block" }}>
          <h2>📈 Research</h2>
          <p className="muted">
            Live training graphs: Elo per self-play generation, falling loss
            curves, and the information-theoretic analysis of real games.
          </p>
        </Link>
        <Link href="/predict" className="card" style={{ display: "block" }}>
          <h2>🔮 Predict</h2>
          <p className="muted">
            Paste a FEN or PGN to get win/draw/loss, the best move, and a value
            estimate from the current best network.
          </p>
        </Link>
      </div>

      <div className="card" style={{ marginTop: "1.25rem" }}>
        <h2>End to end, owned</h2>
        <ul className="muted">
          <li><b>Pre-train</b> - behavioural cloning on real Lichess games.</li>
          <li><b>Self-play RL</b> - AlphaZero-style MCTS self-play improves the network generation over generation; each checkpoint is rated on a stable Elo ladder.</li>
          <li><b>Adapt</b> - live opponent-modeling plus genuine per-session fine-tuning of a personal checkpoint.</li>
          <li><b>Serve</b> - a Hugging Face Space trains and serves; the site is a thin client with no heuristic fallback.</li>
        </ul>
      </div>
    </main>
  );
}
