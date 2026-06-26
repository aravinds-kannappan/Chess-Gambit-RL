import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Shannon's Gambit - chess agents that learn",
  description:
    "A multi-agent chess intelligence (MDP, PPO, reward) routed by game phase, benchmarked by Stockfish, that learns as it plays. Watch tiers of agents compete and train against them.",
};

// Deterministic floating-piece field (no random at render -> no hydration drift).
// Kept sparse on purpose -- a few drifting glyphs read as ambient, a crowd reads
// as clutter.
const PIECES = [
  { g: "♞", left: "14%", delay: "0s", dur: "30s", size: "2.8rem" },
  { g: "♛", left: "52%", delay: "9s", dur: "34s", size: "3.4rem" },
  { g: "♚", left: "82%", delay: "4s", dur: "32s", size: "3rem" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="board-bg" aria-hidden />
        <div className="pieces" aria-hidden>
          {PIECES.map((p, i) => (
            <span
              key={i}
              style={{ left: p.left, animationDelay: p.delay, animationDuration: p.dur, fontSize: p.size }}
            >
              {p.g}
            </span>
          ))}
        </div>
        <nav className="nav">
          <Link href="/" className="brand">
            Shannon&apos;s <span>Gambit</span>
          </Link>
          <Link href="/play" className="link">Play</Link>
          <Link href="/watch" className="link">Watch</Link>
          <Link href="/research" className="link">Dashboard</Link>
          <Link href="/predict" className="link">Predict</Link>
          <span style={{ flex: 1 }} />
          <a
            href="https://github.com/aravinds-kannappan/Chess-Gambit-RL"
            className="link"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </nav>
        {children}
      </body>
    </html>
  );
}
