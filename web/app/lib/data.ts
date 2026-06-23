import { promises as fs } from "fs";
import path from "path";

// Read an exported JSON artifact from public/data (written by `sgambit export
// --web`). Returns null if it has not been generated yet.
export async function readDataFile<T>(name: string): Promise<T | null> {
  try {
    const file = path.join(process.cwd(), "public", "data", `${name}.json`);
    const raw = await fs.readFile(file, "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}
