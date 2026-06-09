"""Evaluate a trained supervised neural-network 2048 agent."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.supervised import NeuralNetworkAgent
from game.game import Game

_WATCH_TILES = [256, 512, 1024, 2048]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def run_single_game(
    agent: NeuralNetworkAgent,
    seed: int | None,
) -> dict[str, int]:
    game = Game(seed=seed)
    game.reset()
    agent.reset()

    steps = 0
    while True:
        valid_moves = game.board.get_valid_moves()
        if not valid_moves:
            break
        action = agent.get_action(game.board)
        if action not in valid_moves:
            action = valid_moves[0]
        _, _, done, info = game.step(action, return_state=False)
        steps += 1
        if info.get("invalid") or done:
            break

    return {
        "score": int(game.score),
        "max_tile": int(game.max_tile),
        "steps": int(steps),
    }


def evaluate(
    model_path: Path,
    games: int,
    seed: int,
    device: str,
    model_type: str | None,
) -> dict[str, object]:
    agent = NeuralNetworkAgent(
        model_path=model_path,
        model_type=model_type,
        device=device,
    )
    scores: list[int] = []
    max_tiles: list[int] = []
    steps_list: list[int] = []
    start_time = time.time()

    for index in range(games):
        started = time.time()
        result = run_single_game(
            agent=agent,
            seed=seed + index,
        )
        elapsed = time.time() - started
        scores.append(result["score"])
        max_tiles.append(result["max_tile"])
        steps_list.append(result["steps"])
        print(
            f"[{index + 1:>4}/{games}] {_fmt_time(elapsed):>6} | "
            f"Score: {result['score']:>7} | "
            f"Max: {result['max_tile']:>5} | "
            f"Steps: {result['steps']:>5}"
        )

    total_time = time.time() - start_time
    return {
        "scores": scores,
        "max_tiles": max_tiles,
        "steps_list": steps_list,
        "tile_cumulative": {
            tile: sum(max_tile >= tile for max_tile in max_tiles)
            for tile in _WATCH_TILES
        },
        "total_games": games,
        "total_time": total_time,
        "model_path": str(model_path),
        "device": device,
    }


def _std(values: list[int]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def print_report(results: dict[str, object]) -> None:
    scores = list(results["scores"])
    max_tiles = list(results["max_tiles"])
    steps_list = list(results["steps_list"])
    total = int(results["total_games"])
    elapsed = float(results["total_time"])

    print()
    print("=" * 60)
    print("Supervised neural-network evaluation")
    print("=" * 60)
    print(f"Games:           {total}")
    print(f"Model:           {results['model_path']}")
    print(f"Device:          {results['device']}")
    print(f"Total time:      {_fmt_time(elapsed)}")
    print(f"Average time:    {elapsed / max(1, total):.2f}s/game")
    print("-" * 60)
    print(f"Mean score:      {sum(scores) / max(1, total):>10.1f}")
    print(f"Max score:       {max(scores):>10}" if scores else "Max score:        N/A")
    print(f"Min score:       {min(scores):>10}" if scores else "Min score:        N/A")
    print(f"Score std:       {_std(scores):>10.1f}")
    print(f"Mean steps:      {sum(steps_list) / max(1, total):>10.1f}")
    print(f"Mean max tile:   {sum(max_tiles) / max(1, total):>10.1f}")
    print("-" * 60)

    tile_dist = Counter(max_tiles)
    for tile in sorted(tile_dist):
        count = tile_dist[tile]
        pct = 100 * count / max(1, total)
        bar = "#" * int(pct / 2)
        print(f"    {tile:>6}: {count:>4} games ({pct:5.1f}%) {bar}")

    print("-" * 60)
    cumulative = dict(results["tile_cumulative"])
    for tile in _WATCH_TILES:
        count = int(cumulative.get(tile, 0))
        pct = 100 * count / max(1, total)
        print(f"{tile:>5} rate:      {count:>4}/{total} ({pct:5.1f}%)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained supervised 2048 policy."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-type", choices=("mlp", "cnn"), default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--games", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    results = evaluate(
        model_path=args.model_path,
        games=args.games,
        seed=args.seed,
        device=args.device,
        model_type=args.model_type,
    )
    print_report(results)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(results, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
