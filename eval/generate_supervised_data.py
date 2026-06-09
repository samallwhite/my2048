"""Generate supervised learning data from the heuristic 2048 expert."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import HeuristicWeights
from agent.heuristic import HeuristicAgent
from agent.supervised import valid_action_mask
from game.game import Game


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


def load_weights(path: str | None, values: list[float] | None) -> HeuristicWeights:
    if values is not None:
        return HeuristicWeights.from_list(values)
    if path is None:
        return HeuristicWeights()

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return HeuristicWeights.from_list([float(v) for v in data])
    if isinstance(data, dict):
        if "weights" in data and isinstance(data["weights"], list):
            return HeuristicWeights.from_list([float(v) for v in data["weights"]])
        names = HeuristicWeights().param_names
        if all(name in data for name in names):
            return HeuristicWeights.from_list([float(data[name]) for name in names])
    raise ValueError(f"Cannot parse heuristic weights from {path}")


def keep_sample(max_tile: int, rng: np.random.Generator) -> bool:
    """Stratified sampling to reduce easy early-game dominance."""
    if max_tile < 128:
        return rng.random() < 0.35
    if max_tile < 1024:
        return rng.random() < 0.75
    return True


def generate_dataset(
    output: Path,
    games: int,
    seed: int,
    depth: int,
    weights: HeuristicWeights,
    max_samples: int | None = None,
) -> dict[str, int | str]:
    rng = np.random.default_rng(seed)
    expert = HeuristicAgent(weights=weights, max_depth=depth)

    states: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    scores: list[int] = []
    max_tiles: list[int] = []
    empty_counts: list[int] = []
    steps_total = 0

    for game_index in range(games):
        game = Game(seed=seed + game_index)
        expert.reset()
        game.reset()

        while not game.board.is_game_over():
            board = game.board
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                break

            action = expert.get_action_bits(board.bits)
            if action not in valid_moves:
                action = valid_moves[0]

            if keep_sample(board.max_tile, rng):
                states.append(board.get_state().astype(np.uint8, copy=False).reshape(16))
                actions.append(int(action))
                masks.append(valid_action_mask(board))
                scores.append(int(board.score))
                max_tiles.append(int(board.max_tile))
                empty_counts.append(len(board.get_empty_cells()))

                if max_samples is not None and len(actions) >= max_samples:
                    break

            _, _, done, info = game.step(action, return_state=False)
            if info.get("invalid"):
                break
            steps_total += 1
            if done:
                break

        if (game_index + 1) % 50 == 0 or game_index + 1 == games:
            print(
                f"[{game_index + 1}/{games}] samples={len(actions)} "
                f"last_score={game.score} last_max={game.max_tile}"
            )

        if max_samples is not None and len(actions) >= max_samples:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=np.asarray(states, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        valid_masks=np.asarray(masks, dtype=np.bool_),
        scores=np.asarray(scores, dtype=np.int32),
        max_tiles=np.asarray(max_tiles, dtype=np.int32),
        empty_counts=np.asarray(empty_counts, dtype=np.int8),
        metadata=np.asarray(
            json.dumps(
                {
                    "games_requested": games,
                    "seed": seed,
                    "depth": depth,
                    "samples": len(actions),
                    "steps_total": steps_total,
                    "weights": weights.to_list(),
                },
                ensure_ascii=True,
            )
        ),
    )

    return {
        "output": str(output),
        "samples": len(actions),
        "games_requested": games,
        "seed": seed,
        "depth": depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate state/action samples from a heuristic 2048 expert."
    )
    parser.add_argument("--output", type=Path, default=Path("data/supervised_data.npz"))
    parser.add_argument("--games", type=positive_int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=positive_int, default=2)
    parser.add_argument("--max-samples", type=non_negative_int, default=0)
    parser.add_argument(
        "--weights",
        type=float,
        nargs=5,
        default=None,
        metavar=("W_LOST", "W_EMPTY", "W_MERGES", "W_MONO", "W_SUM"),
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        default=None,
        help="JSON list, {'weights': [...]}, or dict with HeuristicWeights fields.",
    )
    args = parser.parse_args()

    weights = load_weights(args.weights_file, args.weights)
    summary = generate_dataset(
        output=args.output,
        games=args.games,
        seed=args.seed,
        depth=args.depth,
        weights=weights,
        max_samples=args.max_samples or None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
