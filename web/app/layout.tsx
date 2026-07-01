import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

const REPO = "https://github.com/aravinds-kannappan/Chess-Gambit-RL";

export const metadata: Metadata = {
  title: "Shannon's Gambit: a chess engine that learns",
  description:
    "A classic chess club with a learning engine at the table. Play it at an honest rating, watch it play itself, and read the scorebook: nightly gated training, graded by Stockfish.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="nav">
          <Link href="/" className="brand">
            ♞ Shannon&apos;s <span>Gambit</span>
          </Link>
          <Link href="/play" className="link">Play</Link>
          <Link href="/watch" className="link">Watch</Link>
          <Link href="/research" className="link">Scorebook</Link>
          <Link href="/predict" className="link">Analysis</Link>
          <span style={{ flex: 1 }} />
          <a href={REPO} className="link" target="_blank" rel="noreferrer">GitHub</a>
        </nav>
        {children}
        <footer className="band">
          <div
            className="band-inner"
            style={{ paddingTop: "2rem", paddingBottom: "2rem", display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "baseline" }}
          >
            <span className="mono muted" style={{ fontSize: "0.8rem", letterSpacing: "0.08em" }}>
              SHANNON&apos;S GAMBIT · EST. BY SELF-PLAY · TRAINS NIGHTLY
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
