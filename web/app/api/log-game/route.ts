import { NextResponse } from "next/server";
import { api, proxy } from "@/app/lib/api";

export const dynamic = "force-dynamic";

// POST { sessionId, fens, moves, result } -> log a finished game for adaptation.
export async function POST(req: Request) {
  const { sessionId, fens, moves, result } = await req.json();
  const { status, body } = await proxy(() => api.logGame(sessionId, fens, moves, result));
  return NextResponse.json(body, { status });
}
