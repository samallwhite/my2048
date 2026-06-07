"""Genetic optimization for 2048 heuristic weights.

This script optimizes the five linear parameters exposed by
agent.config.HeuristicWeights. It intentionally keeps the first version
single-process and deterministic so experiment results are easy to reproduce.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import HeuristicWeights
from agent.heuristic import HeuristicAgent
from game.game import Game


@dataclass
class Metrics:
    games: int
    mean_score: float
    max_score: int
    min_score: int
    std_score: float
    mean_steps: float
    mean_max_tile: float
    rate_2048: float
    rate_4096: float
    rate_8192: float
    max_tile_distribution: dict[int, int]


@dataclass
class EvaluationRecord:
    weights: list[float]
    fitness: float
    metrics: Metrics
    stage: str


@dataclass
class GAConfig:
    population: int
    generations: int
    games_fast: int
    games_full: int
    depth_fast: int
    depth_full: int
    seed: int
    full_fraction: float
    elite_size: int
    tournament_size: int
    alpha: float
    mutation_rate: float
    mutation_scale: float
    output: str
    resume: str | None


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def normalize_distribution(distribution: dict[int, int]) -> dict[str, int]:
    return {str(tile): count for tile, count in sorted(distribution.items())}


def metrics_to_json(metrics: Metrics) -> dict:
    data = asdict(metrics)
    data["max_tile_distribution"] = normalize_distribution(
        metrics.max_tile_distribution
    )
    return data


def record_to_json(record: EvaluationRecord) -> dict:
    return {
        "weights": record.weights,
        "fitness": record.fitness,
        "stage": record.stage,
        "metrics": metrics_to_json(record.metrics),
    }


def fitness_from_metrics(metrics: Metrics) -> float:
    return (
        metrics.mean_score
        + 5000.0 * metrics.rate_2048
        + 12000.0 * metrics.rate_4096
        + 25000.0 * metrics.rate_8192
        - 0.05 * metrics.std_score
    )


def evaluate_weights(
    weights: list[float],
    games: int,
    seed: int,
    depth: int,
) -> EvaluationRecord:
    """Run several games and return aggregate metrics for one chromosome."""
    scores: list[int] = []
    steps_list: list[int] = []
    max_tiles: list[int] = []
    distribution: dict[int, int] = {}

    config = HeuristicWeights.from_list(weights)

    for offset in range(games):
        env = Game(seed=seed + offset)
        agent = HeuristicAgent(weights=config, max_depth=depth)
        agent.reset()
        env.reset()

        done = False
        steps = 0
        while not done:
            action = agent.get_action(env.board)
            _, _, done, _ = env.step(action)
            steps += 1

        score = int(env.score)
        max_tile = int(env.max_tile)
        scores.append(score)
        steps_list.append(steps)
        max_tiles.append(max_tile)
        distribution[max_tile] = distribution.get(max_tile, 0) + 1

    mean_score = sum(scores) / games
    variance = sum((score - mean_score) ** 2 for score in scores) / games
    std_score = math.sqrt(variance)

    metrics = Metrics(
        games=games,
        mean_score=mean_score,
        max_score=max(scores),
        min_score=min(scores),
        std_score=std_score,
        mean_steps=sum(steps_list) / games,
        mean_max_tile=sum(max_tiles) / games,
        rate_2048=sum(tile >= 2048 for tile in max_tiles) / games,
        rate_4096=sum(tile >= 4096 for tile in max_tiles) / games,
        rate_8192=sum(tile >= 8192 for tile in max_tiles) / games,
        max_tile_distribution=distribution,
    )

    return EvaluationRecord(
        weights=list(weights),
        fitness=fitness_from_metrics(metrics),
        metrics=metrics,
        stage="",
    )


def make_random_individual(
    rng: random.Random,
    bounds: list[tuple[float, float]],
) -> list[float]:
    return [rng.uniform(low, high) for low, high in bounds]


def initialize_population(
    rng: random.Random,
    population_size: int,
    bounds: list[tuple[float, float]],
) -> list[list[float]]:
    default_weights = HeuristicWeights().to_list()
    population = [default_weights]
    while len(population) < population_size:
        population.append(make_random_individual(rng, bounds))
    return population


def tournament_select(
    rng: random.Random,
    records: list[EvaluationRecord],
    tournament_size: int,
) -> list[float]:
    size = min(tournament_size, len(records))
    candidates = rng.sample(records, size)
    winner = max(candidates, key=lambda record: record.fitness)
    return list(winner.weights)


def blx_alpha_crossover(
    rng: random.Random,
    parent_a: list[float],
    parent_b: list[float],
    bounds: list[tuple[float, float]],
    alpha: float,
) -> list[float]:
    child: list[float] = []
    for a, b, (low, high) in zip(parent_a, parent_b, bounds):
        lower = min(a, b)
        upper = max(a, b)
        diff = upper - lower
        value = rng.uniform(lower - alpha * diff, upper + alpha * diff)
        child.append(clamp(value, low, high))
    return child


def mutate(
    rng: random.Random,
    individual: list[float],
    bounds: list[tuple[float, float]],
    mutation_rate: float,
    mutation_scale: float,
) -> list[float]:
    mutated = list(individual)
    for index, (low, high) in enumerate(bounds):
        if rng.random() >= mutation_rate:
            continue
        sigma = mutation_scale * (high - low)
        mutated[index] = clamp(mutated[index] + rng.gauss(0.0, sigma), low, high)
    return mutated


def evaluate_population(
    population: list[list[float]],
    config: GAConfig,
    generation: int,
) -> list[EvaluationRecord]:
    """Evaluate all individuals quickly, then full-evaluate top candidates."""
    fast_seed = config.seed + generation * 10000
    fast_records: list[EvaluationRecord] = []

    for individual in population:
        record = evaluate_weights(
            individual,
            games=config.games_fast,
            seed=fast_seed,
            depth=config.depth_fast,
        )
        record.stage = "fast"
        fast_records.append(record)

    full_count = max(
        config.elite_size,
        math.ceil(len(population) * config.full_fraction),
    )
    full_count = min(full_count, len(population))
    full_seed = config.seed + generation * 10000 + 5000

    records = list(fast_records)
    top_indices = sorted(
        range(len(fast_records)),
        key=lambda index: fast_records[index].fitness,
        reverse=True,
    )[:full_count]

    for index in top_indices:
        record = fast_records[index]
        full_record = evaluate_weights(
            record.weights,
            games=config.games_full,
            seed=full_seed,
            depth=config.depth_full,
        )
        full_record.stage = "full"
        records[index] = full_record

    return records


def make_next_population(
    rng: random.Random,
    records: list[EvaluationRecord],
    config: GAConfig,
    bounds: list[tuple[float, float]],
) -> list[list[float]]:
    ranked = sorted(records, key=lambda record: record.fitness, reverse=True)
    next_population = [
        list(record.weights) for record in ranked[: config.elite_size]
    ]

    while len(next_population) < config.population:
        parent_a = tournament_select(rng, ranked, config.tournament_size)
        parent_b = tournament_select(rng, ranked, config.tournament_size)
        child = blx_alpha_crossover(rng, parent_a, parent_b, bounds, config.alpha)
        child = mutate(
            rng,
            child,
            bounds,
            config.mutation_rate,
            config.mutation_scale,
        )
        next_population.append(child)

    return next_population


def write_generation_csv(path: Path, generation: int, records: list[EvaluationRecord]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "generation",
                    "rank",
                    "stage",
                    "fitness",
                    "mean_score",
                    "std_score",
                    "mean_steps",
                    "mean_max_tile",
                    "rate_2048",
                    "rate_4096",
                    "rate_8192",
                    "w_lost_penalty",
                    "w_empty",
                    "w_merges",
                    "w_monotonicity",
                    "w_sum",
                ]
            )

        ranked = sorted(records, key=lambda record: record.fitness, reverse=True)
        for rank, record in enumerate(ranked, start=1):
            writer.writerow(
                [
                    generation,
                    rank,
                    record.stage,
                    f"{record.fitness:.6f}",
                    f"{record.metrics.mean_score:.6f}",
                    f"{record.metrics.std_score:.6f}",
                    f"{record.metrics.mean_steps:.6f}",
                    f"{record.metrics.mean_max_tile:.6f}",
                    f"{record.metrics.rate_2048:.6f}",
                    f"{record.metrics.rate_4096:.6f}",
                    f"{record.metrics.rate_8192:.6f}",
                    *[f"{value:.12g}" for value in record.weights],
                ]
            )


def save_checkpoint(
    output_dir: Path,
    generation: int,
    next_generation: int,
    population: list[list[float]],
    records: list[EvaluationRecord],
    best: EvaluationRecord,
    config: GAConfig,
    rng: random.Random,
) -> None:
    checkpoint = {
        "generation": generation,
        "next_generation": next_generation,
        "config": asdict(config),
        "population": population,
        "records": [record_to_json(record) for record in records],
        "best": record_to_json(best),
        "random_state": rng.getstate(),
    }
    (output_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "best_weights.json").write_text(
        json.dumps(record_to_json(best), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def tupleize(value: object) -> object:
    if isinstance(value, list):
        return tuple(tupleize(item) for item in value)
    return value


def metrics_from_json(data: dict) -> Metrics:
    distribution = {
        int(tile): int(count)
        for tile, count in data["max_tile_distribution"].items()
    }
    return Metrics(
        games=int(data["games"]),
        mean_score=float(data["mean_score"]),
        max_score=int(data["max_score"]),
        min_score=int(data["min_score"]),
        std_score=float(data["std_score"]),
        mean_steps=float(data["mean_steps"]),
        mean_max_tile=float(data["mean_max_tile"]),
        rate_2048=float(data["rate_2048"]),
        rate_4096=float(data["rate_4096"]),
        rate_8192=float(data["rate_8192"]),
        max_tile_distribution=distribution,
    )


def record_from_json(data: dict) -> EvaluationRecord:
    return EvaluationRecord(
        weights=[float(value) for value in data["weights"]],
        fitness=float(data["fitness"]),
        metrics=metrics_from_json(data["metrics"]),
        stage=str(data["stage"]),
    )


def load_checkpoint(
    path: Path,
    rng: random.Random,
) -> tuple[int, list[list[float]], EvaluationRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rng.setstate(tupleize(data["random_state"]))
    population = [
        [float(value) for value in individual]
        for individual in data["population"]
    ]
    best = record_from_json(data["best"])
    return int(data.get("next_generation", data["generation"] + 1)), population, best


def run_evolution(config: GAConfig) -> EvaluationRecord:
    rng = random.Random(config.seed)
    bounds = HeuristicWeights().bounds
    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "generations.csv"

    start_generation = 0
    population = initialize_population(rng, config.population, bounds)
    best_record: EvaluationRecord | None = None
    if config.resume:
        start_generation, population, best_record = load_checkpoint(
            Path(config.resume),
            rng,
        )
        print(f"Resuming from generation {start_generation}")

    for generation in range(start_generation, config.generations):
        started = time.perf_counter()
        records = evaluate_population(population, config, generation)
        ranked = sorted(records, key=lambda record: record.fitness, reverse=True)
        generation_best = ranked[0]

        if best_record is None or generation_best.fitness > best_record.fitness:
            best_record = generation_best

        elapsed = time.perf_counter() - started
        print(
            "Gen {generation:03d} | best {fitness:.1f} | "
            "score {score:.1f} | max {max_tile:.1f} | "
            "stage {stage} | {elapsed:.1f}s | weights {weights}".format(
                generation=generation,
                fitness=generation_best.fitness,
                score=generation_best.metrics.mean_score,
                max_tile=generation_best.metrics.mean_max_tile,
                stage=generation_best.stage,
                elapsed=elapsed,
                weights=[round(value, 4) for value in generation_best.weights],
            )
        )

        next_population = make_next_population(rng, records, config, bounds)

        write_generation_csv(csv_path, generation, records)
        save_checkpoint(
            output_dir,
            generation,
            generation + 1,
            next_population,
            records,
            best_record,
            config,
            rng,
        )

        population = next_population

    assert best_record is not None
    return best_record


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def bounded_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def parse_args(argv: Iterable[str] | None = None) -> GAConfig:
    parser = argparse.ArgumentParser(
        description="Optimize 2048 heuristic weights with a genetic algorithm."
    )
    parser.add_argument("--population", type=positive_int, default=16)
    parser.add_argument("--generations", type=positive_int, default=20)
    parser.add_argument("--games-fast", type=positive_int, default=3)
    parser.add_argument("--games-full", type=positive_int, default=3)
    parser.add_argument("--depth-fast", type=positive_int, default=1)
    parser.add_argument("--depth-full", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-fraction", type=bounded_fraction, default=0.35)
    parser.add_argument("--elite-size", type=positive_int, default=2)
    parser.add_argument("--tournament-size", type=positive_int, default=3)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--mutation-rate", type=bounded_fraction, default=0.2)
    parser.add_argument("--mutation-scale", type=float, default=0.08)
    parser.add_argument("--output", default="evolution_results")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint.json file to resume from.",
    )

    args = parser.parse_args(argv)
    if args.elite_size > args.population:
        parser.error("--elite-size must be <= --population")
    if args.tournament_size > args.population:
        parser.error("--tournament-size must be <= --population")
    if args.mutation_scale < 0:
        parser.error("--mutation-scale must be >= 0")
    if args.alpha < 0:
        parser.error("--alpha must be >= 0")

    return GAConfig(
        population=args.population,
        generations=args.generations,
        games_fast=args.games_fast,
        games_full=args.games_full,
        depth_fast=args.depth_fast,
        depth_full=args.depth_full,
        seed=args.seed,
        full_fraction=args.full_fraction,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        alpha=args.alpha,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        output=args.output,
        resume=args.resume,
    )


def main(argv: Iterable[str] | None = None) -> None:
    config = parse_args(argv)
    best = run_evolution(config)
    print("\nBest weights:")
    print(" ".join(f"{value:.12g}" for value in best.weights))
    print(
        "Fitness: {fitness:.3f} | mean score: {score:.1f} | "
        "mean max tile: {tile:.1f}".format(
            fitness=best.fitness,
            score=best.metrics.mean_score,
            tile=best.metrics.mean_max_tile,
        )
    )


if __name__ == "__main__":
    main()
