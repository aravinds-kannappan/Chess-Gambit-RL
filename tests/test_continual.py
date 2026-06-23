"""Tests for the continual-RL core: ladder, anchored Elo, strength, serving, adapt."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess

from shannons_gambit.agents.adaptive import adapt_to_games
from shannons_gambit.agents.alphazero.mcts import AlphaZeroAgent
from shannons_gambit.agents.ladder import Ladder
from shannons_gambit.eval.elo import estimate_rating
from shannons_gambit.models.net import ChessNet, save_model
from shannons_gambit.serve import ModelServer

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _seed_ladder(run_dir: str, elo: float = 900.0) -> None:
    ckpt = Path(run_dir) / "gen-0000.pt"
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    save_model(ChessNet(channels=8, blocks=1), str(ckpt), extra={"gen": 0, "elo": elo})
    ladder = Ladder(run_dir=run_dir)
    ladder.add(0, str(ckpt), elo, {"seeded": True})
    ladder.save()


class TestLadder(unittest.TestCase):
    def test_roundtrip_and_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            lad = Ladder(run_dir=tmp)
            lad.add(0, f"{tmp}/gen-0000.pt", 700, {})
            lad.add(1, f"{tmp}/gen-0001.pt", 950, {})
            lad.save()
            reloaded = Ladder.load(tmp)
            self.assertEqual(len(reloaded.entries), 2)
            self.assertEqual(reloaded.best().gen, 1)
            self.assertEqual(reloaded.nearest(720).gen, 0)
            self.assertEqual(reloaded.next_gen(), 2)


class TestAnchoredElo(unittest.TestCase):
    def test_monotonic_in_score(self):
        # vs the same anchor, more points => higher estimated rating.
        low = estimate_rating([(1000.0, 2.0, 10)])
        high = estimate_rating([(1000.0, 8.0, 10)])
        self.assertGreater(high, low)
        even = estimate_rating([(1000.0, 5.0, 10)])
        self.assertAlmostEqual(even, 1000.0, delta=30)


class TestStrengthKnobs(unittest.TestCase):
    def setUp(self):
        self.model = ChessNet(channels=8, blocks=1)

    def test_blunder_and_temperature_stay_legal(self):
        board = chess.Board()
        for kw in ({"blunder_rate": 1.0}, {"temperature": 1.0}, {}):
            agent = AlphaZeroAgent(self.model, simulations=4, **kw)
            self.assertIn(agent.select_move(board), board.legal_moves)


class TestServer(unittest.TestCase):
    def test_move_and_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_ladder(tmp)
            server = ModelServer(tmp, base_sims=4)
            out = server.move(START, elo=900)
            self.assertIn(chess.Move.from_uci(out["move"]), chess.Board(START).legal_moves)
            pred = server.predict(START)
            self.assertIn("best_move", pred)
            watch = server.watch_move(START, 900, 600)
            self.assertIn("move", watch)


class TestAdapt(unittest.TestCase):
    def test_adapt_runs_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.pt"
            save_model(ChessNet(channels=8, blocks=1), str(base), extra={})
            board = chess.Board()
            fens, moves = [], []
            for _ in range(4):
                mv = next(iter(board.legal_moves))
                fens.append(board.fen())
                moves.append(mv.uci())
                board.push(mv)
            games = [{"fens": fens, "moves": moves, "result": 1.0}]
            out = Path(tmp) / "personal.pt"
            res = adapt_to_games(str(base), games, str(out))
            self.assertTrue(out.exists())
            self.assertEqual(res["n_examples"], 4)
            self.assertEqual(res["agent_win_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
