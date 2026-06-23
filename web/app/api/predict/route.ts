import { NextResponse } from "next/server";
import { Chess } from "chess.js";
import { getPrediction } from "@/app/lib/predict";

export const dynamic = "force-dynamic";

// POST { fen?, pgn? } -> outcome (W/D/L), best move, value, estimated rating.
export async function POST(req: Request) {
  let body: { fen?: string; pgn?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }

  let fen = body.fen;
  if (!fen && body.pgn) {
    try {
      const game = new Chess();
      game.loadPgn(body.pgn);
      fen = game.fen();
    } catch {
      return NextResponse.json({ error: "invalid PGN" }, { status: 400 });
    }
  }
  if (!fen) {
    return NextResponse.json({ error: "provide fen or pgn" }, { status: 400 });
  }
  try {
    new Chess(fen);
  } catch {
    return NextResponse.json({ error: "invalid FEN" }, { status: 400 });
  }

  const pred = await getPrediction(fen);
  return NextResponse.json({ fen, ...pred });
}
