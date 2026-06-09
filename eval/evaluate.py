"""Evaluate the 2048 heuristic agent over many independent games."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import multiprocess as mp
except ImportError:  # pragma: no cover - standard-library fallback
    import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import HeuristicWeights
from agent.heuristic import HeuristicAgent, run_heuristic_game, run_heuristic_games

_WATCH_TILES = [256, 512, 1024, 2048]


def resolve_worker_count(requested: int, task_count: int | None = None) -> int:
    """Return the worker count to use. 0 means all logical CPUs."""
    if requested < 0:
        raise ValueError("workers must be >= 0")
    if requested == 0:
        requested = os.cpu_count() or 1
    worker_count = max(1, requested)
    if task_count is not None:
        worker_count = min(worker_count, max(1, task_count))
    return worker_count


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m{secs:02d}s"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _game_task(
    task: tuple[int, list[float], int | None, int],
) -> tuple[int, float, dict[str, int]]:
    index, weights, seed, depth = task
    started = time.time()
    result = run_heuristic_game(weights=weights, seed=seed, max_depth=depth)
    return index, time.time() - started, result


def _game_batch_task(
    task: tuple[list[int], list[float], list[int | None], int],
) -> list[tuple[int, float, dict[str, int]]]:
    indices, weights, seeds, depth = task
    started = time.time()
    results = run_heuristic_games(weights=weights, seeds=seeds, max_depth=depth)
    elapsed = time.time() - started
    per_game_elapsed = elapsed / len(results) if results else 0.0
    return [
        (index, per_game_elapsed, result)
        for index, result in zip(indices, results)
    ]


def run_single_game(agent: HeuristicAgent, seed: int | None = None) -> dict[str, int]:
    """Backward-compatible single-game helper."""
    return run_heuristic_game(
        weights=agent.weights,
        seed=seed,
        max_depth=agent.max_depth,
    )


def _print_game_line(
    index: int,
    total: int,
    elapsed: float,
    result: dict[str, int],
) -> None:
    score = int(result["score"])
    max_tile = int(result["max_tile"])
    marks = {
        tile: 1 if max_tile >= tile else 0
        for tile in _WATCH_TILES
    }
    print(
        f"[{index + 1:>4}/{total}] {_fmt_time(elapsed):>6} | "
        f"Score: {score:>7} | "
        f"Max: {max_tile:>5} | "
        f"(256:{marks[256]}  512:{marks[512]}  "
        f"1024:{marks[1024]}  2048:{marks[2048]})"
    )


def evaluate(
    agent: HeuristicAgent,
    n_games: int = 100,
    base_seed: int | None = None,
    workers: int = 1,
) -> dict:
    if n_games <= 0:
        raise ValueError("n_games must be positive")

    scores: list[int] = []
    max_tiles: list[int] = []
    steps_list: list[int] = []
    results_by_index: list[dict[str, int] | None] = [None] * n_games

    worker_count = resolve_worker_count(workers, n_games)
    weights = agent.weights.to_list()
    depth = agent.max_depth
    start_time = time.time()

    if worker_count <= 1:
        for index in range(n_games):
            seed = (base_seed + index) if base_seed is not None else None
            started = time.time()
            result = run_heuristic_game(
                weights=weights,
                seed=seed,
                max_depth=depth,
            )
            elapsed = time.time() - started
            results_by_index[index] = result
            _print_game_line(index, n_games, elapsed, result)
    else:
        groups: list[list[int]] = [[] for _ in range(worker_count)]
        for index in range(n_games):
            groups[index % worker_count].append(index)
        tasks = [
            (
                indices,
                weights,
                [
                    (base_seed + index) if base_seed is not None else None
                    for index in indices
                ],
                depth,
            )
            for indices in groups
            if indices
        ]
        chunksize = max(1, len(tasks) // (worker_count * 4))
        with contextlib.ExitStack() as stack:
            context = mp.get_context("spawn")
            pool = stack.enter_context(context.Pool(processes=worker_count))
            for batch in pool.imap_unordered(
                _game_batch_task,
                tasks,
                chunksize=chunksize,
            ):
                for index, elapsed, result in batch:
                    results_by_index[index] = result
                    _print_game_line(index, n_games, elapsed, result)

    for result in results_by_index:
        if result is None:
            continue
        scores.append(int(result["score"]))
        max_tiles.append(int(result["max_tile"]))
        steps_list.append(int(result["steps"]))

    elapsed = time.time() - start_time
    tile_cumulative = {
        tile: sum(max_tile >= tile for max_tile in max_tiles)
        for tile in _WATCH_TILES
    }

    return {
        "scores": scores,
        "max_tiles": max_tiles,
        "steps_list": steps_list,
        "tile_cumulative": tile_cumulative,
        "total_games": n_games,
        "total_time": elapsed,
        "workers": worker_count,
    }


def _std(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return variance**0.5


def print_report(results: dict) -> None:
    scores = results["scores"]
    max_tiles = results["max_tiles"]
    steps_list = results["steps_list"]
    tile_cumulative = results["tile_cumulative"]
    total = results["total_games"]
    elapsed = results["total_time"]

    print()
    print("=" * 60)
    print("Evaluation report")
    print("=" * 60)
    print(f"Games:           {total}")
    print(f"Workers:         {results.get('workers', 1)}")
    print(f"Total time:      {_fmt_time(elapsed)}")
    if total > 0:
        print(f"Average time:    {elapsed / total:.1f}s/game")
    print("-" * 60)
    print(
        f"Mean score:      {sum(scores) / total:>10.1f}"
        if total > 0
        else "Mean score:       N/A"
    )
    print(f"Max score:       {max(scores):>10}" if scores else "Max score:        N/A")
    print(f"Min score:       {min(scores):>10}" if scores else "Min score:        N/A")
    print(f"Score std:       {_std(scores):>10.1f}")
    print(
        f"Mean steps:      {sum(steps_list) / total:>10.1f}"
        if total > 0
        else "Mean steps:       N/A"
    )
    print(
        f"Mean max tile:   {sum(max_tiles) / total:>10.1f}"
        if total > 0
        else "Mean max tile:    N/A"
    )
    print("-" * 60)

    tile_dist = Counter(max_tiles)
    if tile_dist:
        print("Max tile distribution")
        for tile in sorted(tile_dist.keys()):
            count = tile_dist[tile]
            pct = 100 * count / total if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {tile:>6}: {count:>4} games ({pct:5.1f}%) {bar}")

    print("-" * 60)
    for tile in _WATCH_TILES:
        count = tile_cumulative.get(tile, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"{tile:>5} rate:      {count:>4}/{total} ({pct:5.1f}%)")
    print("=" * 60)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the 2048 heuristic agent."
    )
    parser.add_argument("--games", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--depth",
        type=positive_int,
        default=2,
        help="Expectimax max depth. Higher values are much slower.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=5,
        default=None,
        metavar=("W_LOST", "W_EMPTY", "W_MERGES", "W_MONO", "W_SUM"),
    )
    parser.add_argument(
        "--workers",
        type=non_negative_int,
        default=1,
        help="Worker processes for game evaluation. Use 0 for all CPUs.",
    )
    args = parser.parse_args()

    if args.weights is not None:
        weights = HeuristicWeights.from_list(args.weights)
        print(f"Weights: {weights}")
    else:
        weights = HeuristicWeights()
        print(f"Weights: [default] {weights}")

    worker_count = resolve_worker_count(args.workers, args.games)
    agent = HeuristicAgent(weights=weights, max_depth=args.depth)

    print(
        f"Depth: {args.depth} | Games: {args.games} | "
        f"Seed: {args.seed} | Workers: {worker_count}"
    )
    print()

    results = evaluate(
        agent,
        n_games=args.games,
        base_seed=args.seed,
        workers=worker_count,
    )
    print_report(results)


if __name__ == "__main__":
    main()
