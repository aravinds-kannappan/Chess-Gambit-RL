import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Shannon's Gambit",
  description:
    "Information-theoretic reinforcement learning for chess - MDP/Bellman, deep RL, and real Lichess data.",
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
          <Link href="/predict" className="link">Predict</Link>
          <Link href="/research" className="link">Research</Link>
          <Link href="/arena" className="link">Ladder</Link>
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
