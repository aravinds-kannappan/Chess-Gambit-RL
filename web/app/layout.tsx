import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

const REPO = "https://github.com/aravinds-kannappan/Chess-Gambit-RL";

export const metadata: Metadata = {
  title: "Shannon's Gambit — a chess engine that learns",
  description:
    "A chess engine that learns: an opening book and a network trained on real games for the opening and middlegame, exact solvers for the endgame, graded by Stockfish on a real Elo scale. Play it, watch it, and see the data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="nav">
          <Link href="/" className="brand">
            Shannon&apos;s <span>Gambit</span>
          </Link>
          <Link href="/play" className="link">Play</Link>
          <Link href="/watch" className="link">Watch</Link>
          <Link href="/research" className="link">Data</Link>
          <Link href="/predict" className="link">Predict</Link>
          <span style={{ flex: 1 }} />
          <a href={REPO} className="link" target="_blank" rel="noreferrer">GitHub</a>
        </nav>
        {children}
        <footer className="band">
          <div
            className="band-inner"
            style={{ paddingTop: "2.2rem", paddingBottom: "2.2rem", display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}
          >
            <span className="mono muted" style={{ fontSize: "0.8rem" }}>
              Shannon&apos;s Gambit — a chess engine that learns.
            </span>
            <a href={REPO} className="mono" style={{ fontSize: "0.8rem", color: "var(--accent)" }} target="_blank" rel="noreferrer">
              source ↗
            </a>
          </div>
        </footer>
      </body>
    </html>
  );
}
