"""Genetic optimization for 2048 heuristic weights.

This script optimizes the five linear parameters exposed by
agent.config.HeuristicWeights. It intentionally keeps the first version
single-process and deterministic so experiment results are easy to reproduce.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    import multiprocess as mp
except ImportError:  # pragma: no cover - standard-library fallback
    import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import HeuristicWeights
from agent.heuristic import run_heuristic_games


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
    validation_games: int
    depth_fast: int
    depth_full: int
    validation_seed: int
    seed: int
    full_fraction: float
    elite_size: int
    tournament_size: int
    alpha: float
    mutation_rate: float
    mutation_scale: float
    baseline_paired: bool
    bounds_mode: str
    fix_lost_penalty: bool
    local_init_fraction: float
    local_init_scale: float
    output: str
    resume: str | None
    workers: int
    validation_interval: int
    validate_improvements_only: bool
    racing_games: list[int] | None
    racing_fractions: list[float] | None
    racing_depths: list[int] | None


DEFAULT_WEIGHTS = HeuristicWeights().to_list()
DEFAULT_BOUNDS = HeuristicWeights().bounds
NARROW_BOUNDS = [
    (150000.0, 300000.0),  # w_lost_penalty
    (200.0, 700.0),        # w_empty
    (400.0, 1400.0),       # w_merges
    (40.0, 150.0),         # w_monotonicity
    (7.0, 25.0),           # w_sum
]
_BASELINE_CACHE: dict[tuple[int, int, int], EvaluationRecord] = {}


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


def _candidate_eval_task(
    task: tuple[int, list[float], list[int | None], int],
) -> tuple[int, list[dict[str, int]]]:
    index, weights, seeds, depth = task
    return index, run_heuristic_games(weights=weights, seeds=seeds, max_depth=depth)


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


def paired_fitness(metrics: Metrics, baseline: Metrics) -> float:
    """Score a candidate by improvement over default weights on same seeds."""
    return (
        metrics.mean_score
        - baseline.mean_score
        + 3000.0 * (metrics.rate_4096 - baseline.rate_4096)
        + 8000.0 * (metrics.rate_8192 - baseline.rate_8192)
        - 0.02 * (metrics.std_score - baseline.std_score)
    )


def get_baseline_record(
    games: int,
    seed: int,
    depth: int,
    pool: mp.pool.Pool | None = None,
    worker_count: int = 1,
) -> EvaluationRecord:
    key = (games, seed, depth)
    cached = _BASELINE_CACHE.get(key)
    if cached is not None:
        return cached

    if pool is None or worker_count <= 1:
        record = evaluate_weights(DEFAULT_WEIGHTS, games=games, seed=seed, depth=depth)
    else:
        record = evaluate_weight_batch(
            [DEFAULT_WEIGHTS],
            games=games,
            seed=seed,
            depth=depth,
            pool=pool,
            worker_count=worker_count,
        )[0]
    record.stage = "baseline"
    _BASELINE_CACHE[key] = record
    return record


def apply_paired_fitness(
    record: EvaluationRecord,
    games: int,
    seed: int,
    depth: int,
    pool: mp.pool.Pool | None = None,
    worker_count: int = 1,
) -> EvaluationRecord:
    baseline = get_baseline_record(
        games,
        seed,
        depth,
        pool=pool,
        worker_count=worker_count,
    )
    record.fitness = paired_fitness(record.metrics, baseline.metrics)
    return record


def evaluate_stage_records(
    candidates: list[list[float]],
    games: int,
    seed: int,
    depth: int,
    stage: str,
    baseline_paired: bool,
    pool: mp.pool.Pool | None = None,
    worker_count: int = 1,
) -> list[EvaluationRecord]:
    """Evaluate a stage, batching baseline with candidates when needed."""
    if not candidates:
        return []

    baseline: EvaluationRecord | None = None
    candidate_count = len(candidates)
    batch = candidates
    if baseline_paired:
        key = (games, seed, depth)
        baseline = _BASELINE_CACHE.get(key)
        if baseline is None:
            batch = [*candidates, DEFAULT_WEIGHTS]

    records = evaluate_weight_batch(
        batch,
        games=games,
        seed=seed,
        depth=depth,
        pool=pool,
        worker_count=worker_count,
    )

    if baseline_paired and baseline is None:
        baseline = records[-1]
        baseline.stage = "baseline"
        _BASELINE_CACHE[(games, seed, depth)] = baseline
        records = records[:candidate_count]

    if baseline_paired:
        assert baseline is not None
        for record in records:
            record.fitness = paired_fitness(record.metrics, baseline.metrics)

    for record in records:
        record.stage = stage

    return records


def make_record_from_results(
    weights: list[float],
    results: list[dict[str, int]],
) -> EvaluationRecord:
    """Aggregate per-game results into one chromosome evaluation."""
    games = len(results)
    if games <= 0:
        raise ValueError("at least one game is required")

    scores = [int(result["score"]) for result in results]
    steps_list = [int(result["steps"]) for result in results]
    max_tiles = [int(result["max_tile"]) for result in results]
    distribution: dict[int, int] = {}
    for max_tile in max_tiles:
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


def evaluate_weights(
    weights: list[float],
    games: int,
    seed: int,
    depth: int,
) -> EvaluationRecord:
    """Run several games serially and return metrics for one chromosome."""
    seeds = [seed + offset for offset in range(games)]
    results = run_heuristic_games(weights=weights, seeds=seeds, max_depth=depth)
    return make_record_from_results(weights, results)


def evaluate_weight_batch(
    candidates: list[list[float]],
    games: int,
    seed: int,
    depth: int,
    pool: mp.pool.Pool | None = None,
    worker_count: int = 1,
) -> list[EvaluationRecord]:
    """Evaluate many chromosomes, parallelizing individual games when possible."""
    if pool is None or worker_count <= 1 or len(candidates) <= 0:
        return [
            evaluate_weights(weights, games=games, seed=seed, depth=depth)
            for weights in candidates
        ]

    tasks = [
        (
            index,
            list(weights),
            [seed + offset for offset in range(games)],
            depth,
        )
        for index, weights in enumerate(candidates)
    ]
    chunksize = max(1, len(tasks) // (worker_count * 2))
    records: list[EvaluationRecord | None] = [None] * len(candidates)

    for index, results in pool.imap_unordered(
        _candidate_eval_task,
        tasks,
        chunksize=chunksize,
    ):
        records[index] = make_record_from_results(candidates[index], results)

    if any(record is None for record in records):
        raise RuntimeError("worker pool returned incomplete evaluation results")
    return [record for record in records if record is not None]


def make_random_individual(
    rng: random.Random,
    bounds: list[tuple[float, float]],
) -> list[float]:
    return [rng.uniform(low, high) for low, high in bounds]


def make_local_individual(
    rng: random.Random,
    bounds: list[tuple[float, float]],
    scale: float,
) -> list[float]:
    individual: list[float] = []
    for default, (low, high) in zip(DEFAULT_WEIGHTS, bounds):
        sigma = scale * (high - low)
        individual.append(clamp(default + rng.gauss(0.0, sigma), low, high))
    return individual


def initialize_population(
    rng: random.Random,
    population_size: int,
    bounds: list[tuple[float, float]],
    local_fraction: float,
    local_scale: float,
) -> list[list[float]]:
    population = [list(DEFAULT_WEIGHTS)]
    local_count = max(1, round((population_size - 1) * local_fraction))
    while len(population) < population_size:
        if len(population) <= local_count:
            population.append(make_local_individual(rng, bounds, local_scale))
        else:
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


def get_racing_schedule(config: GAConfig) -> list[tuple[int, float, int]]:
    """Build optional successive-halving schedule as (games, fraction, depth)."""
    if not config.racing_games:
        return []

    games = config.racing_games
    if any(value <= 0 for value in games):
        raise ValueError("racing games must be positive")

    final_fraction = max(config.full_fraction, config.elite_size / config.population)
    if config.racing_fractions is None:
        fractions = []
        for index in range(len(games)):
            if index == 0:
                fractions.append(1.0)
            elif index == len(games) - 1:
                fractions.append(final_fraction)
            else:
                fractions.append(max(final_fraction, 0.5 ** index))
    else:
        fractions = config.racing_fractions

    if config.racing_depths is None:
        depths = [
            config.depth_full if index == len(games) - 1 else config.depth_fast
            for index in range(len(games))
        ]
    else:
        depths = config.racing_depths

    if not (len(games) == len(fractions) == len(depths)):
        raise ValueError("racing games, fractions, and depths must have same length")
    if fractions[0] != 1.0:
        raise ValueError("first racing fraction must be 1.0")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("racing fractions must be in (0, 1]")
    if any(depth <= 0 for depth in depths):
        raise ValueError("racing depths must be positive")

    return list(zip(games, fractions, depths))


def evaluate_population_racing(
    population: list[list[float]],
    config: GAConfig,
    generation: int,
    schedule: list[tuple[int, float, int]],
    pool: mp.pool.Pool | None = None,
    worker_count: int = 1,
) -> list[EvaluationRecord]:
    """Evaluate population with successive halving across racing stages."""
    active_indices = list(range(len(population)))

    for stage_index, (games, fraction, depth) in enumerate(schedule):
        del fraction
        candidates = [population[index] for index in active_indices]
        stage_seed = config.seed + generation * 10000 + stage_index * 5000
        stage = "race" if stage_index == len(schedule) - 1 else f"race{stage_index + 1}"
        stage_records = evaluate_stage_records(
            candidates,
            games=games,
            seed=stage_seed,
            depth=depth,
            stage=stage,
            baseline_paired=config.baseline_paired,
            pool=pool,
            worker_count=worker_count,
        )

        if stage_index == len(schedule) - 1:
            return stage_records

        next_fraction = schedule[stage_index + 1][1]
        survivor_count = max(
            config.elite_size,
            math.ceil(config.population * next_fraction),
        )
        survivor_count = min(survivor_count, len(active_indices))
        ranked_local = sorted(
            range(len(stage_records)),
            key=lambda local_index: stage_records[local_index].fitness,
            reverse=True,
        )
        active_indices = [
            active_indices[local_index]
            for local_index in ranked_local[:survivor_count]
        ]

    return []


def evaluate_population(
    population: list[list[float]],
    config: GAConfig,
    generation: int,
    pool: mp.pool.Pool | None = None,
    worker_count: int = 1,
) -> list[EvaluationRecord]:
    """Evaluate all individuals quickly, then full-evaluate top candidates."""
    schedule = get_racing_schedule(config)
    if schedule:
        return evaluate_population_racing(
            population,
            config,
            generation,
            schedule,
            pool=pool,
            worker_count=worker_count,
        )

    fast_seed = config.seed + generation * 10000
    fast_records = evaluate_stage_records(
        population,
        games=config.games_fast,
        seed=fast_seed,
        depth=config.depth_fast,
        stage="fast",
        baseline_paired=config.baseline_paired,
        pool=pool,
        worker_count=worker_count,
    )

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

    full_candidates = [fast_records[index].weights for index in top_indices]
    full_records = evaluate_stage_records(
        full_candidates,
        games=config.games_full,
        seed=full_seed,
        depth=config.depth_full,
        stage="full",
        baseline_paired=config.baseline_paired,
        pool=pool,
        worker_count=worker_count,
    )

    for index, full_record in zip(top_indices, full_records):
        records[index] = full_record

    return records


def validate_record(
    record: EvaluationRecord,
    config: GAConfig,
    generation: int,
    pool: mp.pool.Pool | None = None,
    worker_count: int = 1,
) -> EvaluationRecord:
    validation_seed = config.validation_seed + generation * 10000
    validation = evaluate_stage_records(
        [record.weights],
        games=config.validation_games,
        seed=validation_seed,
        depth=config.depth_full,
        stage="validation",
        baseline_paired=config.baseline_paired,
        pool=pool,
        worker_count=worker_count,
    )[0]
    return validation


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


def get_search_bounds(config: GAConfig) -> list[tuple[float, float]]:
    bounds = list(NARROW_BOUNDS if config.bounds_mode == "narrow" else DEFAULT_BOUNDS)
    if config.fix_lost_penalty:
        default_lost = DEFAULT_WEIGHTS[0]
        bounds[0] = (default_lost, default_lost)
    return bounds


def should_validate_generation(
    generation: int,
    generation_best: EvaluationRecord,
    best_record: EvaluationRecord | None,
    config: GAConfig,
) -> bool:
    if config.validation_games <= 0:
        return False
    if generation == 0 or config.validation_interval <= 1:
        interval_due = True
    else:
        interval_due = generation % config.validation_interval == 0
    if not interval_due:
        return False
    if (
        config.validate_improvements_only
        and best_record is not None
        and generation_best.fitness <= best_record.fitness
    ):
        return False
    return True


def run_evolution(config: GAConfig) -> EvaluationRecord:
    rng = random.Random(config.seed)
    bounds = get_search_bounds(config)
    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "generations.csv"

    start_generation = 0
    population = initialize_population(
        rng,
        config.population,
        bounds,
        config.local_init_fraction,
        config.local_init_scale,
    )
    best_record: EvaluationRecord | None = None
    if config.resume:
        start_generation, population, best_record = load_checkpoint(
            Path(config.resume),
            rng,
        )
        print(f"Resuming from generation {start_generation}")

    worker_count = resolve_worker_count(config.workers)
    with contextlib.ExitStack() as stack:
        pool = None
        if worker_count > 1:
            context = mp.get_context("spawn")
            pool = stack.enter_context(context.Pool(processes=worker_count))
            print(f"Using {worker_count} worker processes")

        for generation in range(start_generation, config.generations):
            started = time.perf_counter()
            records = evaluate_population(
                population,
                config,
                generation,
                pool=pool,
                worker_count=worker_count,
            )
            ranked = sorted(records, key=lambda record: record.fitness, reverse=True)
            generation_best = ranked[0]
            validation_record: EvaluationRecord | None = None
            if should_validate_generation(
                generation,
                generation_best,
                best_record,
                config,
            ):
                validation_record = validate_record(
                    generation_best,
                    config,
                    generation,
                    pool=pool,
                    worker_count=worker_count,
                )

            if validation_record is not None:
                if best_record is None or validation_record.fitness > best_record.fitness:
                    best_record = validation_record
            elif config.validation_games <= 0:
                if best_record is None or generation_best.fitness > best_record.fitness:
                    best_record = generation_best

            elapsed = time.perf_counter() - started
            display_record = validation_record or generation_best
            valid_text = (
                f"{validation_record.fitness:.1f}"
                if validation_record is not None
                else ("off" if config.validation_games <= 0 else "skip")
            )
            print(
                "Gen {generation:03d} | train {train_fitness:.1f} | "
                "valid {valid_fitness} | score {score:.1f} | "
                "max {max_tile:.1f} | {elapsed:.1f}s | weights {weights}".format(
                    generation=generation,
                    train_fitness=generation_best.fitness,
                    valid_fitness=valid_text,
                    score=display_record.metrics.mean_score,
                    max_tile=display_record.metrics.mean_max_tile,
                    elapsed=elapsed,
                    weights=[round(value, 4) for value in display_record.weights],
                )
            )

            next_population = make_next_population(rng, records, config, bounds)

            csv_records = records + ([validation_record] if validation_record else [])
            write_generation_csv(csv_path, generation, csv_records)
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


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def bounded_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def parse_int_csv(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        return None
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def parse_fraction_csv(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        return None
    if any(not 0.0 < item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("all values must be in (0, 1]")
    return values


def parse_args(argv: Iterable[str] | None = None) -> GAConfig:
    parser = argparse.ArgumentParser(
        description="Optimize 2048 heuristic weights with a genetic algorithm."
    )
    parser.add_argument("--population", type=positive_int, default=16)
    parser.add_argument("--generations", type=positive_int, default=20)
    parser.add_argument("--games-fast", type=positive_int, default=3)
    parser.add_argument("--games-full", type=positive_int, default=8)
    parser.add_argument(
        "--validation-games",
        type=non_negative_int,
        default=10,
        help="Validation games for generation best. Use 0 to disable.",
    )
    parser.add_argument(
        "--validation-interval",
        type=positive_int,
        default=1,
        help="Validate every N generations. Generation 0 is always validated.",
    )
    parser.add_argument(
        "--validate-improvements-only",
        action="store_true",
        help="Only validate generations whose train fitness beats current best.",
    )
    parser.add_argument("--depth-fast", type=positive_int, default=1)
    parser.add_argument("--depth-full", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=10000)
    parser.add_argument("--full-fraction", type=bounded_fraction, default=0.35)
    parser.add_argument("--elite-size", type=positive_int, default=2)
    parser.add_argument("--tournament-size", type=positive_int, default=3)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--mutation-rate", type=bounded_fraction, default=0.2)
    parser.add_argument("--mutation-scale", type=float, default=0.08)
    parser.add_argument(
        "--baseline-paired",
        action="store_true",
        help="Use same-seed improvement over default weights as fitness.",
    )
    parser.add_argument(
        "--bounds-mode",
        choices=("narrow", "default"),
        default="narrow",
        help="Use narrowed report-driven bounds or original config bounds.",
    )
    parser.add_argument(
        "--fix-lost-penalty",
        action="store_true",
        help="Keep w_lost_penalty fixed at the default value.",
    )
    parser.add_argument("--local-init-fraction", type=bounded_fraction, default=0.75)
    parser.add_argument("--local-init-scale", type=float, default=0.12)
    parser.add_argument("--output", default="evolution_results")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint.json file to resume from.",
    )
    parser.add_argument(
        "--workers",
        type=non_negative_int,
        default=1,
        help="Worker processes for game evaluation. Use 0 for all CPUs.",
    )
    parser.add_argument(
        "--racing-games",
        type=parse_int_csv,
        default=None,
        help="Comma-separated games per racing stage, e.g. 1,3,8.",
    )
    parser.add_argument(
        "--racing-fractions",
        type=parse_fraction_csv,
        default=None,
        help="Comma-separated survivor fractions per stage, e.g. 1,0.5,0.25.",
    )
    parser.add_argument(
        "--racing-depths",
        type=parse_int_csv,
        default=None,
        help="Comma-separated search depths per racing stage, e.g. 1,1,2.",
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
    if args.local_init_scale < 0:
        parser.error("--local-init-scale must be >= 0")
    if args.racing_games is None and (
        args.racing_fractions is not None or args.racing_depths is not None
    ):
        parser.error("--racing-fractions/--racing-depths require --racing-games")
    if args.racing_games is not None:
        racing_len = len(args.racing_games)
        if args.racing_fractions is not None and len(args.racing_fractions) != racing_len:
            parser.error("--racing-fractions length must match --racing-games")
        if args.racing_depths is not None and len(args.racing_depths) != racing_len:
            parser.error("--racing-depths length must match --racing-games")
        if args.racing_fractions is not None and args.racing_fractions[0] != 1.0:
            parser.error("first --racing-fractions value must be 1.0")

    return GAConfig(
        population=args.population,
        generations=args.generations,
        games_fast=args.games_fast,
        games_full=args.games_full,
        validation_games=args.validation_games,
        depth_fast=args.depth_fast,
        depth_full=args.depth_full,
        validation_seed=args.validation_seed,
        seed=args.seed,
        full_fraction=args.full_fraction,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        alpha=args.alpha,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        baseline_paired=args.baseline_paired,
        bounds_mode=args.bounds_mode,
        fix_lost_penalty=args.fix_lost_penalty,
        local_init_fraction=args.local_init_fraction,
        local_init_scale=args.local_init_scale,
        output=args.output,
        resume=args.resume,
        workers=args.workers,
        validation_interval=args.validation_interval,
        validate_improvements_only=args.validate_improvements_only,
        racing_games=args.racing_games,
        racing_fractions=args.racing_fractions,
        racing_depths=args.racing_depths,
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
