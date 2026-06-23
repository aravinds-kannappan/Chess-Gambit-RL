import { fallbackPredict } from "./agent";
import { hfPredict } from "./hf";
import type { Prediction } from "./types";

// Try the trained model on Hugging Face first; fall back to the built-in agent.
export async function getPrediction(fen: string): Promise<Prediction> {
  const hf = await hfPredict(fen);
  return hf ?? fallbackPredict(fen);
}
