import Link from "next/link";

export default function Home() {
  return (
    <main className="container">
      <h1 className="title">
        Shannon&apos;s <span>Gambit</span>
      </h1>
      <p className="subtitle">
        Information-theoretic reinforcement learning for chess. Agents formalised
        as a Markov Decision Process, solved with Bellman optimality, and trained
        with deep RL on real Lichess games.
      </p>

      <div className="grid cols-2">
        <Link href="/play" className="card" style={{ display: "block" }}>
          <h2>♟ Play the agent</h2>
          <p className="muted">
            Play a full game against the trained network (or the built-in
            heuristic agent), served through the Hugging Face Inference API.
          </p>
        </Link>
        <Link href="/analysis" className="card" style={{ display: "block" }}>
          <h2>📊 Information dashboard</h2>
          <p className="muted">
            Move-choice entropy, mutual information between board features and the
            result, and where a game&apos;s outcome uncertainty collapses.
          </p>
        </Link>
        <Link href="/predict" className="card" style={{ display: "block" }}>
          <h2>🔮 Live prediction</h2>
          <p className="muted">
            Paste a FEN or PGN to get win/draw/loss, the best move, and an
            estimated player rating from the prediction model.
          </p>
        </Link>
        <Link href="/arena" className="card" style={{ display: "block" }}>
          <h2>🏆 Agent arena</h2>
          <p className="muted">
            Elo leaderboard from round-robin matches between the random, tabular,
            DQN, supervised, and AlphaZero-lite agents.
          </p>
        </Link>
      </div>

      <div className="card" style={{ marginTop: "1.25rem" }}>
        <h2>How it fits together</h2>
        <ul className="muted">
          <li>
            <b>Information theory</b> - Shannon entropy, KL/JS divergence, and
            mutual information quantify policies and positions.
          </li>
          <li>
            <b>MDP + Bellman</b> - endgames (KRvK/KQvK) are enumerated and solved
            exactly by value/policy iteration; tabular Q-learning recovers the
            optimum from experience.
          </li>
          <li>
            <b>Deep RL</b> - a DQN validated against the solved value table, and a
            small AlphaZero-lite (policy/value net + PUCT MCTS self-play)
            bootstrapped from supervised behavioural cloning.
          </li>
        </ul>
      </div>
    </main>
  );
}
