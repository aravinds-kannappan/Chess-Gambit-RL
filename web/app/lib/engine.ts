// A small client-side chess engine used as a graceful fallback when the Hugging
// Face backend is warming up (HTTP 503). Strength scales with a target Elo via
// search depth + blunder rate, so "agents play at their Elo" holds even offline.
// When the backend is live, its trained-network moves take precedence (see
// agentMove). This engine is labelled in the UI so it is never mistaken for the
// trained agent.

import { Chess } from "chess.js";

const VAL: Record<string, number> = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };

// Piece-square tables (white perspective, a8..h1 row-major), midgame-ish.
const PST: Record<string, number[]> = {
  p: [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10, 5,5,10,25,25,10,5,5, 0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5, 5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0],
  n: [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30, -30,5,15,20,20,15,5,-30, -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30, -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
  b: [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10, -10,5,5,10,10,5,5,-10, -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10, -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
  r: [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
  q: [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10, -5,0,5,5,5,5,0,-5, 0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10, -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
  k: [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10, 20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20],
};

// Static eval from the side-to-move's perspective (centipawns).
function evaluate(game: Chess): number {
  if (game.isCheckmate()) return -100000;
  if (game.isDraw() || game.isStalemate()) return 0;
  const board = game.board();
  let score = 0;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const sq = board[r][c];
      if (!sq) continue;
      const idx = r * 8 + c;
      const base = VAL[sq.type] + PST[sq.type][sq.color === "w" ? idx : 63 - idx];
      score += sq.color === "w" ? base : -base;
    }
  }
  return game.turn() === "w" ? score : -score;
}

function ordered(game: Chess) {
  return game.moves({ verbose: true }).sort((a, b) => {
    const av = (a.captured ? VAL[a.captured] : 0) - VAL[a.piece] / 10;
    const bv = (b.captured ? VAL[b.captured] : 0) - VAL[b.piece] / 10;
    return bv - av;
  });
}

function negamax(game: Chess, depth: number, alpha: number, beta: number): number {
  if (depth === 0 || game.isGameOver()) return evaluate(game);
  let best = -Infinity;
  for (const m of ordered(game)) {
    game.move(m);
    const val = -negamax(game, depth - 1, -beta, -alpha);
    game.undo();
    if (val > best) best = val;
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }
  return best;
}

export type Rec = { uci: string; san: string; score: number };

// Top-N ranked moves for a position (score in pawns, side-to-move perspective).
export function recommend(fen: string, n = 5, depth = 3): Rec[] {
  let game: Chess;
  try { game = new Chess(fen); } catch { return []; }
  const recs: Rec[] = [];
  for (const m of ordered(game)) {
    game.move(m);
    const score = -negamax(game, depth - 1, -Infinity, Infinity);
    game.undo();
    recs.push({ uci: m.from + m.to + (m.promotion ?? ""), san: m.san, score: score / 100 });
  }
  recs.sort((a, b) => b.score - a.score);
  return recs.slice(0, n);
}

export function evalPawns(fen: string): number {
  try {
    const g = new Chess(fen);
    const cp = evaluate(g);
    return (g.turn() === "w" ? cp : -cp) / 100; // always white-positive
  } catch { return 0; }
}

function depthForElo(elo: number): number {
  if (elo < 1000) return 1;
  if (elo < 1600) return 2;
  return 3;
}
function blunderForElo(elo: number): number {
  return Math.max(0, Math.min(0.5, (1750 - elo) / 2600));
}

// A move at the requested strength: best play, perturbed by an Elo-scaled
// blunder rate (weaker agents pick down their own preference list more often).
export function eloMove(fen: string, elo: number): string | null {
  const recs = recommend(fen, 8, depthForElo(elo));
  if (recs.length === 0) return null;
  if (Math.random() < blunderForElo(elo)) {
    const k = Math.min(recs.length - 1, 1 + Math.floor(Math.random() * 3));
    return recs[k].uci;
  }
  // tiny noise among near-best so games are not identical
  const top = recs.filter((r) => r.score >= recs[0].score - 0.15);
  return top[Math.floor(Math.random() * top.length)].uci;
}

// Unified move source: trained backend first, Elo-scaled engine as fallback.
export async function agentMove(
  fen: string,
  elo: number,
  session?: string,
): Promise<{ uci: string; source: "agent" | "engine"; route?: string; elo: number }> {
  try {
    const res = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen, elo, session }),
    });
    const data = await res.json();
    if (res.ok && !data.error && data.move) {
      return { uci: data.move, source: "agent", route: data.route, elo: data.calibrated_elo ?? data.elo ?? elo };
    }
  } catch {
    /* fall through */
  }
  const uci = eloMove(fen, elo);
  if (!uci) throw new Error("no legal move");
  return { uci, source: "engine", route: "engine", elo };
}
