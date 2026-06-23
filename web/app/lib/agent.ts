// Built-in TypeScript fallback agent (no external model needed).
//
// A one-ply material + mobility search over chess.js legal moves. Always legal
// and instant, so "play vs the agent" works on Vercel even when no Hugging Face
// endpoint is configured (or while it is cold). It also returns an
// information-theoretic read of the position: a softmax over move scores and its
// Shannon entropy (bits), mirroring the policy-entropy idea from the Python
// `infotheory` module.

import { Chess } from "chess.js";
import type { Prediction, WDL } from "./types";

const VALUE: Record<string, number> = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };

function material(game: Chess): number {
  // Centipawns from White's perspective.
  let score = 0;
  for (const row of game.board()) {
    for (const sq of row) {
      if (!sq) continue;
      const v = VALUE[sq.type];
      score += sq.color === "w" ? v : -v;
    }
  }
  return score;
}

function evaluate(game: Chess): number {
  // White-perspective static eval: material + small mobility term.
  if (game.isCheckmate()) return game.turn() === "w" ? -100000 : 100000;
  if (game.isDraw() || game.isStalemate()) return 0;
  const mat = material(game);
  const mover = game.turn();
  const mobility = game.moves().length;
  const mobilityTerm = mover === "w" ? mobility : -mobility;
  return mat + 2 * mobilityTerm;
}

function softmaxEntropyBits(scores: number[]): number {
  if (scores.length <= 1) return 0;
  const t = 80; // temperature in centipawns
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp((s - max) / t));
  const sum = exps.reduce((a, b) => a + b, 0);
  let h = 0;
  for (const e of exps) {
    const p = e / sum;
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

function wdlFromCp(cp: number): WDL {
  // Logistic mapping of centipawns to a win probability (white perspective).
  const win = 1 / (1 + Math.pow(10, -cp / 400));
  const draw = 0.30 * (1 - Math.abs(2 * win - 1));
  const w = win * (1 - draw);
  const l = (1 - win) * (1 - draw);
  return { win: w, draw, loss: l };
}

export function fallbackPredict(fen: string): Prediction {
  const game = new Chess(fen);
  const mover = game.turn();
  const moves = game.moves({ verbose: true });
  if (moves.length === 0) {
    const cp = evaluate(game);
    return {
      bestMove: "",
      wdl: wdlFromCp(cp),
      value: 0,
      rating: 1200,
      policyEntropyBits: 0,
      source: "fallback",
    };
  }

  const scored = moves.map((m) => {
    game.move(m);
    // Score from the mover's perspective (negamax depth 1).
    const whiteEval = evaluate(game);
    const moverEval = mover === "w" ? whiteEval : -whiteEval;
    game.undo();
    return { move: `${m.from}${m.to}${m.promotion ?? ""}`, score: moverEval };
  });

  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const entropy = softmaxEntropyBits(scored.map((s) => s.score));

  // Best move's resulting white-perspective eval -> WDL + value.
  const moverBestCp = best.score;
  const whiteCp = mover === "w" ? moverBestCp : -moverBestCp;
  const wdl = wdlFromCp(whiteCp);
  const value = Math.tanh((mover === "w" ? whiteCp : -whiteCp) / 400);

  // Crude rating proxy: lower policy entropy in sharp positions ~ stronger read.
  const rating = Math.round(1300 + (3 - entropy) * 120);

  return {
    bestMove: best.move,
    wdl,
    value,
    rating: Math.max(800, Math.min(2600, rating)),
    policyEntropyBits: entropy,
    source: "fallback",
  };
}
