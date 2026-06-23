import { NextResponse } from "next/server";
import { api, proxy } from "@/app/lib/api";

export const dynamic = "force-dynamic";

// POST { fen, whiteElo, blackElo } -> move for the side to move at its level.
export async function POST(req: Request) {
  const { fen, whiteElo, blackElo } = await req.json();
  const { status, body } = await proxy(() => api.watchMove(fen, whiteElo, blackElo));
  return NextResponse.json(body, { status });
}
