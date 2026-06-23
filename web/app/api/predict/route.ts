import { NextResponse } from "next/server";
import { Chess } from "chess.js";
import { api, proxy } from "@/app/lib/api";

export const dynamic = "force-dynamic";

// POST { fen?, pgn? } -> trained-model prediction (no fallback).
export async function POST(req: Request) {
  const body = await req.json();
  let fen = body.fen;
  if (!fen && body.pgn) {
    try {
      const g = new Chess();
      g.loadPgn(body.pgn);
      fen = g.fen();
    } catch {
      return NextResponse.json({ error: "invalid PGN" }, { status: 400 });
    }
  }
  if (!fen) return NextResponse.json({ error: "provide fen or pgn" }, { status: 400 });
  const { status, body: out } = await proxy(() => api.predict(fen));
  return NextResponse.json({ fen, ...out }, { status });
}
