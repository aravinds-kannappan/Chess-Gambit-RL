import { NextResponse } from "next/server";
import { Chess } from "chess.js";
import { getPrediction } from "@/app/lib/predict";

export const dynamic = "force-dynamic";

// POST { fen } -> the agent's move for that position.
export async function POST(req: Request) {
  let fen: string;
  try {
    ({ fen } = await req.json());
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }

  let game: Chess;
  try {
    game = new Chess(fen);
  } catch {
    return NextResponse.json({ error: "invalid FEN" }, { status: 400 });
  }
  if (game.isGameOver()) {
    return NextResponse.json({ error: "game is over", gameOver: true }, { status: 409 });
  }

  const pred = await getPrediction(fen);

  // Validate the model's move; if it isn't legal here, take the top legal move
  // from the fallback so the API never returns an illegal move.
  const legal = new Set(game.moves({ verbose: true }).map((m) => `${m.from}${m.to}${m.promotion ?? ""}`));
  let move = pred.bestMove;
  if (!legal.has(move)) {
    const first = game.moves({ verbose: true })[0];
    move = `${first.from}${first.to}${first.promotion ?? ""}`;
  }

  return NextResponse.json({
    move,
    source: pred.source,
    value: pred.value,
    wdl: pred.wdl,
    policyEntropyBits: pred.policyEntropyBits,
  });
}
