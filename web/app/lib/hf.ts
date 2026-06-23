// Hugging Face Inference proxy.
//
// When HF_ENDPOINT_URL is configured, the API routes POST the FEN to a
// Hugging Face Inference Endpoint serving the trained ChessNet (the endpoint's
// handler does the board encoding server-side using the Python package, so no
// JS/Python tensor parity is required here). If the call fails or is not
// configured, callers fall back to the built-in TypeScript agent.

import type { Prediction, WDL } from "./types";

export function hfConfigured(): boolean {
  return Boolean(process.env.HF_ENDPOINT_URL);
}

export async function hfPredict(fen: string): Promise<Prediction | null> {
  const url = process.env.HF_ENDPOINT_URL;
  if (!url) return null;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.HF_API_TOKEN
          ? { Authorization: `Bearer ${process.env.HF_API_TOKEN}` }
          : {}),
      },
      body: JSON.stringify({ inputs: { fen } }),
      // Endpoints can be cold; keep latency bounded so we fall back quickly.
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) return null;
    const data = await res.json();
    const wdl: WDL = data.wdl ?? { loss: 0.33, draw: 0.34, win: 0.33 };
    return {
      bestMove: data.best_move ?? data.bestMove ?? "",
      wdl,
      value: data.value ?? 0,
      rating: data.rating ?? 1500,
      policyEntropyBits: data.policy_entropy_bits ?? data.policyEntropyBits ?? 0,
      source: "huggingface",
    };
  } catch {
    return null;
  }
}
