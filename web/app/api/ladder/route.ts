import { NextResponse } from "next/server";
import { api, proxy } from "@/app/lib/api";

export const dynamic = "force-dynamic";

// GET -> the live checkpoint ladder (Elo per generation) from the Space.
export async function GET() {
  const { status, body } = await proxy(() => api.ladder());
  return NextResponse.json(body, { status });
}
