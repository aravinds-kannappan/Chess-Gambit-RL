import { NextResponse } from "next/server";
import { readDataFile } from "@/app/lib/data";

export const dynamic = "force-dynamic";

// Serves the exported information-theory report (public/data/analysis.json).
export async function GET() {
  const data = await readDataFile("analysis");
  if (!data) {
    return NextResponse.json(
      { error: "no analysis export yet; run `sgambit analyze` then `sgambit export --web`" },
      { status: 404 },
    );
  }
  return NextResponse.json(data);
}
