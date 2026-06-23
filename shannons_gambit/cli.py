"""``sgambit`` command-line interface tying the pipeline together.

Subcommands: ``data`` (ingest real games), ``mdp`` (solve an endgame with
Bellman), ``train`` (supervised / dqn / alphazero / tabular), ``analyze``
(information theory), ``arena`` (Elo leaderboard), ``predict``, and ``export``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def _cmd_data(args: argparse.Namespace) -> None:
    from .config import DataConfig
    from .data.dataset import build_dataset

    cfg = DataConfig()
    if args.source:
        cfg = replace(cfg, url=args.source)
    cfg = replace(cfg, max_games=args.games, out_dir=args.out)
    print(json.dumps(build_dataset(cfg), indent=2))


def _cmd_mdp(args: argparse.Namespace) -> None:
    from .mdp.chess_mdp import EndgameMDP
    from .mdp.endgames import get_spec

    mdp = EndgameMDP(get_spec(args.endgame)).build()
    V, history = mdp.solve(gamma=args.gamma)
    print(f"endgame={args.endgame} states={len(mdp.states)} "
          f"sweeps={len(history)} final_delta={history[-1]:.2e} max_dtm={mdp.max_dtm}")
    rng = np.random.default_rng(0)
    mated = 0
    plies = []

    for _ in range(args.verify):
        state = mdp.sample_won_state(rng)
        board = mdp.board_from_state(state)
        n = 0
        while not board.is_game_over() and n < 80:
            board.push(mdp.optimal_move(board))
            n += 1
        if board.is_checkmate():
            mated += 1
            plies.append(n)
    print(f"optimal-policy verification: mated {mated}/{args.verify} "
          f"won positions, avg plies {np.mean(plies):.1f}")


def _cmd_train(args: argparse.Namespace) -> None:
    from .config import TabularQConfig, get_preset

    preset = get_preset(args.preset)
    if args.kind == "supervised":
        from .models.supervised import train_supervised

        res = train_supervised(preset["supervised"])
        print(json.dumps(res["history"][-1] if res["history"] else {}, indent=2))
    elif args.kind == "dqn":
        from .agents.dqn import DQNAgent
        from .mdp.chess_mdp import load_endgame

        cfg = preset["dqn"]
        mdp = load_endgame(cfg.endgame, gamma=cfg.gamma)
        agent = DQNAgent(mdp, cfg)
        agent.train()
        print(json.dumps(agent.evaluate(n=300, max_dtm=8), indent=2))
    elif args.kind == "alphazero":
        from .agents.alphazero.train import train_alphazero

        res = train_alphazero(preset["alphazero"])
        print(json.dumps(res["history"][-1] if res["history"] else {}, indent=2))
    elif args.kind == "tabular":
        from .agents.tabular_q import TabularQAgent
        from .mdp.chess_mdp import load_endgame

        cfg = TabularQConfig(episodes=args.episodes)
        mdp = load_endgame(cfg.endgame, gamma=cfg.gamma)
        agent = TabularQAgent(mdp, cfg)
        agent.train()
        for d in (4, 8, 12):
            print(json.dumps(agent.evaluate(n=400, max_dtm=d), indent=2))


def _cmd_analyze(args: argparse.Namespace) -> None:
    from . import reports
    from .data.dataset import load_records

    records = load_records(args.data)
    report = {
        "n_positions": len(records["fen"]),
        "outcome_prior_entropy_bits": round(reports.outcome_prior_entropy(records), 4),
        "feature_mutual_information_bits": {
            k: round(v, 4) for k, v in reports.feature_mi_report(records).items()
        },
    }
    if args.model and Path(args.model).exists():
        from .models.prediction import Predictor

        predictor = Predictor.from_checkpoint(args.model)
        report["policy_entropy"] = reports.policy_entropy_report(predictor, records["fen"])
        curve = reports.first_game_curve(predictor, args.source)
        if curve is not None:
            report["game_curve"] = curve
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _build_agents(device: str, az_sims: int):
    from .agents.random_agent import RandomAgent

    agents = [RandomAgent(seed=0)]
    sup = Path("runs/supervised/model.pt")
    az = Path("runs/alphazero/model.pt")
    if sup.exists():
        from .agents.neural import NeuralAgent, ValueAgent

        agents.append(NeuralAgent.from_checkpoint(str(sup), device=device))
        agents.append(ValueAgent(agents[-1].predictor))
        agents[-1].name = "value"
    if az.exists():
        from .agents.alphazero.mcts import AlphaZeroAgent

        agents.append(AlphaZeroAgent.from_checkpoint(str(az), device=device, simulations=az_sims))
    return agents


def _cmd_arena(args: argparse.Namespace) -> None:
    from .eval.arena import round_robin

    agents = _build_agents(args.device, args.sims)
    result = round_robin(agents, games_per_pair=args.games, max_moves=args.max_moves)
    print(json.dumps(result["leaderboard"], indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def _cmd_predict(args: argparse.Namespace) -> None:
    import chess

    from .models.prediction import Predictor

    predictor = Predictor.from_checkpoint(args.model, device=args.device)
    board = chess.Board(args.fen)
    print(json.dumps(predictor.predict(board).to_dict(), indent=2))


def _cmd_export(args: argparse.Namespace) -> None:
    from .export import model_card, push_model_to_hf, write_web_data

    if args.web:
        payloads = {}
        arena = Path("runs/arena/arena.json")
        analysis = Path("data/analysis.json")
        if arena.exists():
            payloads["arena"] = json.loads(arena.read_text())
        if analysis.exists():
            payloads["analysis"] = json.loads(analysis.read_text())
        print("wrote:", write_web_data(payloads))
    if args.hf:
        from .models.net import load_model

        _, extra = load_model(args.model)
        card = model_card(extra.get("final_metrics", {}), repo_id=args.hf)
        print("pushed:", push_model_to_hf(args.hf, args.model, card=card))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sgambit", description="Shannon's Gambit CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="ingest real Lichess games")
    d.add_argument("--source", default=None, help="PGN url or path (default: Lichess dump)")
    d.add_argument("--games", type=int, default=2000)
    d.add_argument("--out", default="data")
    d.set_defaults(func=_cmd_data)

    m = sub.add_parser("mdp", help="solve an endgame with Bellman value iteration")
    m.add_argument("--endgame", default="KRvK")
    m.add_argument("--gamma", type=float, default=0.99)
    m.add_argument("--verify", type=int, default=200)
    m.set_defaults(func=_cmd_mdp)

    t = sub.add_parser("train", help="train an agent / model")
    t.add_argument("kind", choices=["supervised", "dqn", "alphazero", "tabular"])
    t.add_argument("--preset", default="local_full")
    t.add_argument("--episodes", type=int, default=40000, help="tabular only")
    t.set_defaults(func=_cmd_train)

    a = sub.add_parser("analyze", help="information-theory report over the data")
    a.add_argument("--data", default="data")
    a.add_argument("--model", default="runs/supervised/model.pt")
    a.add_argument("--source", default=None, help="PGN for the info-gain curve")
    a.add_argument("--out", default="data/analysis.json")
    a.set_defaults(func=_cmd_analyze)

    ar = sub.add_parser("arena", help="round-robin Elo leaderboard")
    ar.add_argument("--games", type=int, default=2)
    ar.add_argument("--max-moves", type=int, default=120, dest="max_moves")
    ar.add_argument("--sims", type=int, default=32)
    ar.add_argument("--device", default="cpu")
    ar.add_argument("--out", default="runs/arena/arena.json")
    ar.set_defaults(func=_cmd_arena)

    pr = sub.add_parser("predict", help="predict from a FEN")
    pr.add_argument("--fen", default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    pr.add_argument("--model", default="runs/supervised/model.pt")
    pr.add_argument("--device", default="cpu")
    pr.set_defaults(func=_cmd_predict)

    ex = sub.add_parser("export", help="export web data / push model to HF")
    ex.add_argument("--web", action="store_true")
    ex.add_argument("--hf", default=None, help="HF repo id, e.g. user/shannons-gambit")
    ex.add_argument("--model", default="runs/supervised/model.pt")
    ex.set_defaults(func=_cmd_export)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
