export interface WDL {
  loss: number;
  draw: number;
  win: number;
}

export interface MovePrediction {
  move: string; // UCI
  evalCp: number; // centipawns, white perspective
  source: "huggingface" | "fallback";
}

export interface Prediction {
  bestMove: string;
  wdl: WDL;
  value: number; // [-1, 1], side-to-move perspective
  rating: number; // estimated Elo
  policyEntropyBits: number;
  source: "huggingface" | "fallback";
}
