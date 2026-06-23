import { NextResponse } from "next/server";
import { readDataFile } from "@/app/lib/data";

export const dynamic = "force-dynamic";

// Serves the exported Elo leaderboard + replays (public/data/arena.json).
export async function GET() {
  const data = await readDataFile("arena");
  if (!data) {
    return NextResponse.json(
      { error: "no arena export yet; run `sgambit arena` then `sgambit export --web`" },
      { status: 404 },
    );
  }
  return NextResponse.json(data);
}
