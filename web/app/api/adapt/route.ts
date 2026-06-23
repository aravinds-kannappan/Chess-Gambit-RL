import { NextResponse } from "next/server";
import { api, proxy } from "@/app/lib/api";

export const dynamic = "force-dynamic";

// POST { sessionId } -> fine-tune a personal checkpoint on this session's games.
export async function POST(req: Request) {
  const { sessionId } = await req.json();
  const { status, body } = await proxy(() => api.adapt(sessionId));
  return NextResponse.json(body, { status });
}
