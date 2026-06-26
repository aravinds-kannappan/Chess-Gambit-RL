"""Tests for the opening book, phase classification, and phase routing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess

from shannons_gambit.agents.opening_book import OpeningBook, OpeningBookAgent, build_book
from shannons_gambit.agents.random_agent import RandomAgent
from shannons_gambit.agents.router import PhaseRouter
from shannons_gambit.phases import PHASES, game_phase


class TestPhases(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(game_phase(chess.Board()), "opening")
        # few men => endgame regardless of move number
        self.assertEqual(game_phase(chess.Board("8/8/8/4k3/8/4K3/4R3/8 w - - 0 60")), "endgame")
        # full board past the opening window (ply >= 24) => middlegame
        mid = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 0 14")
        self.assertGreaterEqual(mid.ply(), 24)
        self.assertEqual(game_phase(mid), "middlegame")
        self.assertIn(game_phase(mid), PHASES)


class TestOpeningBook(unittest.TestCase):
    def setUp(self):
        # bundled sample PGN; relaxed thresholds so the small sample yields a book.
        self.book = build_book("", max_games=10_000, min_elo=0, max_ply=20,
                               min_count=1, top_k=4)

    def test_book_has_start_position(self):
        self.assertGreater(len(self.book), 0)
        moves = self.book.lookup(chess.Board())
        self.assertTrue(moves)
        for move, _w in moves:
            self.assertIn(move, chess.Board().legal_moves)

    def test_out_of_book_past_max_ply(self):
        book = OpeningBook(self.book.table, max_ply=0)
        self.assertEqual(book.lookup(chess.Board()), [])
        self.assertIsNone(book.sample(chess.Board()))

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "opening_book.json"
            self.book.save(str(path))
            reloaded = OpeningBook.load(str(path))
            self.assertEqual(len(reloaded), len(self.book))
            self.assertEqual(reloaded.max_ply, self.book.max_ply)

    def test_router_prefers_book_then_general(self):
        agent = OpeningBookAgent(self.book, seed=1)
        router = PhaseRouter(general=RandomAgent(seed=2), opening=agent)
        move = router.select_move(chess.Board())
        self.assertEqual(router.last_route, "opening")
        self.assertIn(move, chess.Board().legal_moves)
        # a deep position out of book falls through to the general agent.
        deep = chess.Board("8/8/8/4k3/8/4K3/4R3/8 w - - 0 60")
        router.select_move(deep)
        self.assertNotEqual(router.last_route, "opening")
        self.assertEqual(router.last_route, RandomAgent().name)


if __name__ == "__main__":
    unittest.main()
