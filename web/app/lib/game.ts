// A single shared game, persisted to localStorage so the live game survives tab
// switches and is readable from both Play and Predict (Predict streams move
// recommendations for whatever position the ongoing game is in).

import { useSyncExternalStore } from "react";
import { Chess } from "chess.js";

const KEY = "sg_live_game_v1";

class GameStore {
  game = new Chess();
  agentFens: string[] = [];
  agentMoves: string[] = [];
  private listeners = new Set<() => void>();
  private version = 0;
  private loaded = false;

  private ensureLoaded() {
    if (this.loaded || typeof window === "undefined") return;
    this.loaded = true;
    const pgn = localStorage.getItem(KEY);
    if (pgn) {
      try { this.game.loadPgn(pgn); } catch { /* start fresh */ }
    }
  }

  subscribe = (cb: () => void) => {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  };

  getVersion = () => {
    this.ensureLoaded();
    return this.version;
  };

  private bump() {
    this.version++;
    if (typeof window !== "undefined") {
      try { localStorage.setItem(KEY, this.game.pgn()); } catch { /* quota */ }
    }
    this.listeners.forEach((l) => l());
  }

  fen() { this.ensureLoaded(); return this.game.fen(); }
  turn() { return this.game.turn(); }
  isGameOver() { return this.game.isGameOver(); }
  history() { return this.game.history(); }
  moveCount() { return this.game.history().length; }

  result(): string | null {
    if (!this.game.isGameOver()) return null;
    if (this.game.isCheckmate()) return this.game.turn() === "w" ? "0-1" : "1-0";
    return "1/2-1/2";
  }

  tryUserMove(from: string, to: string, promotion = "q"): boolean {
    try {
      const mv = this.game.move({ from, to, promotion });
      if (!mv) return false;
    } catch { return false; }
    this.bump();
    return true;
  }

  applyUci(uci: string): boolean {
    const fenBefore = this.game.fen();
    try {
      const mv = this.game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.slice(4) || undefined });
      if (!mv) return false;
    } catch { return false; }
    this.agentFens.push(fenBefore);
    this.agentMoves.push(uci);
    this.bump();
    return true;
  }

  reset() {
    this.game = new Chess();
    this.agentFens = [];
    this.agentMoves = [];
    this.bump();
  }
}

export const gameStore = new GameStore();

// Re-render hook: returns the mutation version so components refresh on change.
export function useGameVersion(): number {
  return useSyncExternalStore(gameStore.subscribe, gameStore.getVersion, () => 0);
}
