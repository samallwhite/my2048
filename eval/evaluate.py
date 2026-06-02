"""2048 AI 智能体评估脚本。

对指定智能体进行多局游戏测试，统计：
  - 平均得分（主要指标）
  - 最大合成方块分布
  - 平均生存步数

用法：
  python eval/evaluate.py                      # 默认 100 局
  python eval/evaluate.py --games 500          # 自定义局数
  python eval/evaluate.py --seed 42 --verbose   # 指定种子 + 每局输出
"""

import argparse
import sys
import time
from pathlib import Path
from collections import Counter

# 添加项目根目录到 sys.path，确保从任意目录运行均可正确导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.game import Game
from agent.heuristic import HeuristicAgent
from agent.config import HeuristicWeights


def run_single_game(agent: HeuristicAgent, seed: int | None = None
                    ) -> dict:
    """运行一局完整的 2048 游戏并返回统计信息。

    Returns:
        dict with keys: score, max_tile, steps, seed
    """
    game = Game(seed=seed)
    agent.reset()
    game.reset()

    steps = 0
    while True:
        action = agent.get_action(game.board)
        _, _, done, info = game.step(action)

        if info.get("invalid"):
            # 如果智能体返回了无效动作，尝试其他方向
            valid_moves = game.board.get_valid_moves()
            if valid_moves:
                action = valid_moves[0]
                game.step(action)
            else:
                break

        steps += 1

        if done:
            break

    return {
        "score": game.score,
        "max_tile": int(game.board.max_tile),
        "steps": steps,
        "seed": seed,
    }


def evaluate(agent: HeuristicAgent,
             n_games: int = 100,
             base_seed: int | None = None,
             verbose: bool = False) -> dict:
    """评估智能体在 n_games 局游戏中的表现。

    Args:
        agent: 待评估的智能体实例。
        n_games: 测试局数。
        base_seed: 起始随机种子，每局自动 +1 以保证可复现性。
        verbose: 是否逐局输出结果。

    Returns:
        dict with keys:
            avg_score, max_score, min_score, std_score,
            avg_steps, avg_max_tile, max_tile_distribution,
            total_games, total_time
    """
    scores = []
    max_tiles = []
    steps_list = []
    tile_counter = Counter()

    start_time = time.time()

    for i in range(n_games):
        seed = (base_seed + i) if base_seed is not None else None
        result = run_single_game(agent, seed=seed)

        scores.append(result["score"])
        max_tiles.append(result["max_tile"])
        steps_list.append(result["steps"])
        tile_counter[result["max_tile"]] += 1

        if verbose:
            print(f"[{i+1:>4}/{n_games}] "
                  f"Score: {result['score']:>6}  "
                  f"Max: {result['max_tile']:>5}  "
                  f"Steps: {result['steps']:>4}")

    elapsed = time.time() - start_time

    return {
        "avg_score": sum(scores) / n_games,
        "max_score": max(scores),
        "min_score": min(scores),
        "std_score": _std(scores),
        "avg_steps": sum(steps_list) / n_games,
        "avg_max_tile": sum(max_tiles) / n_games,
        "max_tile_distribution": dict(sorted(tile_counter.items())),
        "total_games": n_games,
        "total_time": elapsed,
    }


def _std(values: list[float]) -> float:
    """计算样本标准差。"""
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    return variance ** 0.5


def print_report(results: dict) -> None:
    """格式化输出评估报告。"""
    print("\n" + "=" * 50)
    print("  2048 AI 评估报告")
    print("=" * 50)
    print(f"  总局数:       {results['total_games']}")
    print(f"  总耗时:       {results['total_time']:.2f}s")
    print(f"  平均耗时:     {results['total_time'] / results['total_games']:.3f}s/局")
    print("-" * 50)
    print(f"  平均得分:     {results['avg_score']:>10.1f}")
    print(f"  最高得分:     {results['max_score']:>10}")
    print(f"  最低得分:     {results['min_score']:>10}")
    print(f"  得分标准差:   {results['std_score']:>10.1f}")
    print(f"  平均步数:     {results['avg_steps']:>10.1f}")
    print(f"  平均最大方块: {results['avg_max_tile']:>10.1f}")
    print("-" * 50)
    print("  最大方块分布:")
    for tile, count in results["max_tile_distribution"].items():
        pct = 100 * count / results["total_games"]
        bar = "#" * int(pct / 2)
        print(f"    {tile:>6}: {count:>4} 局 ({pct:5.1f}%) {bar}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="2048 AI 启发式智能体评估工具"
    )
    parser.add_argument("--games", type=int, default=100,
                        help="测试局数（默认 100）")
    parser.add_argument("--seed", type=int, default=0,
                        help="起始随机种子（默认 0）")
    parser.add_argument("--depth", type=int, default=2,
                        help="Expectimax 最大搜索深度（默认 2）")
    parser.add_argument("--verbose", action="store_true",
                        help="逐局输出详细结果")
    parser.add_argument("--weights", type=float, nargs=4, default=None,
                        metavar=("W_EMPTY", "W_MONO", "W_SMOOTH", "W_CORNER"),
                        help="自定义权重（四个浮点数）")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Chance 节点采样数（默认 3）")
    parser.add_argument("--sample-threshold", type=int, default=None,
                        help="空位数超过此值触发采样（默认 4）")
    args = parser.parse_args()

    # 构建权重配置
    if args.weights is not None:
        weights = HeuristicWeights.from_list(args.weights)
        print(f"使用自定义权重: {weights}")
    else:
        weights = HeuristicWeights()
        print(f"使用默认权重: {weights}")

    # 创建智能体
    agent_kwargs = dict(weights=weights, max_depth=args.depth)
    if args.sample_size is not None:
        agent_kwargs["sample_size"] = args.sample_size
    if args.sample_threshold is not None:
        agent_kwargs["sample_threshold"] = args.sample_threshold
    agent = HeuristicAgent(**agent_kwargs)

    print(f"搜索深度: {args.depth} | 测试局数: {args.games} | 种子: {args.seed}")
    print(f"采样参数: size={agent.sample_size}, threshold={agent.sample_threshold}")
    print("开始评估...")

    results = evaluate(
        agent,
        n_games=args.games,
        base_seed=args.seed,
        verbose=args.verbose,
    )

    print_report(results)


if __name__ == "__main__":
    main()
