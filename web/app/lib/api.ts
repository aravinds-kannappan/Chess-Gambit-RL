// Server-side client for the Hugging Face Space backend.
//
// Every move/prediction comes from a trained network served by the Space -
// there is NO heuristic fallback. If the Space is unreachable the API routes
// surface a clear "backend warming up" error instead of inventing a move.

const SPACE = process.env.HF_SPACE_URL;
const TOKEN = process.env.HF_SPACE_TOKEN;

export class BackendError extends Error {}

async function spaceFetch(path: string, body?: unknown): Promise<any> {
  if (!SPACE) throw new BackendError("backend-not-configured");
  let res: Response;
  try {
    res = await fetch(`${SPACE.replace(/\/$/, "")}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        "Content-Type": "application/json",
        ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(20000),
      cache: "no-store",
    });
  } catch {
    throw new BackendError("backend-unreachable");
  }
  if (!res.ok) throw new BackendError(`backend-${res.status}`);
  return res.json();
}

export const api = {
  health: () => spaceFetch("/healthz"),
  ladder: () => spaceFetch("/ladder"),
  move: (fen: string, elo?: number, session?: string) =>
    spaceFetch("/move", { fen, elo, session }),
  watchMove: (fen: string, white_elo: number, black_elo: number) =>
    spaceFetch("/watch-move", { fen, white_elo, black_elo }),
  predict: (fen: string) => spaceFetch("/predict", { fen }),
  logGame: (session_id: string, fens: string[], moves: string[], result: number) =>
    spaceFetch("/log_game", { session_id, fens, moves, result }),
  adapt: (session_id: string) => spaceFetch("/adapt", { session_id }),
};

// Helper for API routes: run an api call, mapping backend errors to a 503 body.
export async function proxy(fn: () => Promise<any>) {
  try {
    return { status: 200, body: await fn() };
  } catch (e) {
    const reason = e instanceof BackendError ? e.message : "backend-error";
    return { status: 503, body: { error: "backend warming up", reason } };
  }
}
