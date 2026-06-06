"""2048 AI 智能体评估脚本。

每局实时打印得分、最大方块和累积分布，全部完成后输出汇总报告。

用法：
  python eval/evaluate.py                      # 默认 100 局
  python eval/evaluate.py --games 500          # 自定义局数
  python eval/evaluate.py --seed 42            # 指定种子
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.game import Game
from agent.heuristic import HeuristicAgent
from agent.config import HeuristicWeights

# 关注的大方块阈值
_WATCH_TILES = [256, 512, 1024, 2048]


# ── 时间格式化 ──────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"


# ── 单局运行 ────────────────────────────────────────────────────────────

def run_single_game(agent: HeuristicAgent, seed: int | None = None) -> dict:
    game = Game(seed=seed)
    agent.reset()
    game.reset()

    steps = 0
    while True:
        action = agent.get_action(game.board)
        _, _, done, info = game.step(action)

        if info.get("invalid"):
            # 智能体不应返回无效动作。若发生，尝试任意合法动作保底。
            valid_moves = game.board.get_valid_moves()
            if not valid_moves:
                break
            _, _, done, _ = game.step(valid_moves[0])

        steps += 1
        if done:
            break

    return {
        "score": int(game.score),
        "max_tile": int(game.board.max_tile),
        "steps": steps,
    }


# ── 评估主函数 ──────────────────────────────────────────────────────────

def evaluate(agent: HeuristicAgent,
             n_games: int = 100,
             base_seed: int | None = None) -> dict:
    scores = []
    max_tiles = []
    steps_list = []

    # 累积计数，显式 int 键避免任何类型混淆
    cum_256 = 0
    cum_512 = 0
    cum_1024 = 0
    cum_2048 = 0

    start_time = time.time()

    for i in range(n_games):
        t0 = time.time()
        seed = (base_seed + i) if base_seed is not None else None
        result = run_single_game(agent, seed=seed)
        dt = time.time() - t0

        score = int(result["score"])
        max_t = int(result["max_tile"])
        steps = int(result["steps"])

        scores.append(score)
        max_tiles.append(max_t)
        steps_list.append(steps)

        # 累加：显式 Python int 比较，分条写死避免循环变量作用域隐患
        if max_t >= 256:
            cum_256 += 1
        if max_t >= 512:
            cum_512 += 1
        if max_t >= 1024:
            cum_1024 += 1
        if max_t >= 2048:
            cum_2048 += 1

        # 每局一行——括号内为本局标记（1=达成, 0=未达成）
        m256  = 1 if max_t >= 256  else 0
        m512  = 1 if max_t >= 512  else 0
        m1024 = 1 if max_t >= 1024 else 0
        m2048 = 1 if max_t >= 2048 else 0
        print(f"[{i+1:>4}/{n_games}] {_fmt_time(dt):>6} | "
              f"Score: {score:>7} | "
              f"Max: {max_t:>5} | "
              f"(256:{m256}  512:{m512}  1024:{m1024}  2048:{m2048})")

    elapsed = time.time() - start_time

    return {
        "scores": scores,
        "max_tiles": max_tiles,
        "steps_list": steps_list,
        "tile_cumulative": {256: cum_256, 512: cum_512,
                            1024: cum_1024, 2048: cum_2048},
        "total_games": n_games,
        "total_time": elapsed,
    }


# ── 报告输出 ────────────────────────────────────────────────────────────

def _std(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return variance ** 0.5


def print_report(results: dict) -> None:
    scores = results["scores"]
    max_tiles = results["max_tiles"]
    steps_list = results["steps_list"]
    tile_cumulative = results["tile_cumulative"]
    total = results["total_games"]
    elapsed = results["total_time"]

    print()
    print("=" * 60)
    print("                      评 估 报 告")
    print("=" * 60)
    print(f"总局数:        {total}")
    print(f"总耗时:        {_fmt_time(elapsed)}")
    if total > 0:
        print(f"平均耗时:      {elapsed / total:.1f}s/局")
    print("-" * 60)
    print(f"平均得分:      {sum(scores) / total:>10.1f}" if total > 0 else
          "平均得分:       N/A")
    print(f"最高得分:      {max(scores):>10}" if scores else
          "最高得分:       N/A")
    print(f"最低得分:      {min(scores):>10}" if scores else
          "最低得分:       N/A")
    print(f"得分标准差:    {_std(scores):>10.1f}")
    print(f"平均步数:      {sum(steps_list) / total:>10.1f}" if total > 0 else
          "平均步数:       N/A")
    print(f"平均最大方块:  {sum(max_tiles) / total:>10.1f}" if total > 0 else
          "平均最大方块:   N/A")
    print("-" * 60)

    # 最大方块分布（按实际出现的值统计）
    from collections import Counter
    tile_dist = Counter(max_tiles)
    if tile_dist:
        print("最大方块分布:")
        for tile in sorted(tile_dist.keys()):
            count = tile_dist[tile]
            pct = 100 * count / total if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {tile:>6}: {count:>4} 局 ({pct:5.1f}%) {bar}")

    print("-" * 60)
    for t in _WATCH_TILES:
        count = tile_cumulative.get(t, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"{t:>5} 达成率:  {count:>4}/{total} ({pct:5.1f}%)")
    print("=" * 60)


# ── 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2048 AI 启发式智能体评估工具"
    )
    parser.add_argument("--games", type=int, default=100,
                        help="测试局数（默认 100）")
    parser.add_argument("--seed", type=int, default=0,
                        help="起始随机种子（默认 0）")
    parser.add_argument("--depth", type=int, default=2,
                        help="Expectimax 最大搜索深度上限（默认 2，可调高但会明显变慢）")
    parser.add_argument("--weights", type=float, nargs=5, default=None,
                        metavar=("W_LOST", "W_EMPTY", "W_MERGES",
                                 "W_MONO", "W_SUM"),
                        help="自定义启发式线性权重（5 个浮点数）")
    args = parser.parse_args()

    # 权重配置
    if args.weights is not None:
        weights = HeuristicWeights.from_list(args.weights)
        print(f"权重: {weights}")
    else:
        weights = HeuristicWeights()
        print(f"权重: [默认] {weights}")

    # 智能体
    agent = HeuristicAgent(weights=weights, max_depth=args.depth)

    print(f"深度: {args.depth} | 局数: {args.games} | 种子: {args.seed}")
    print()

    results = evaluate(
        agent,
        n_games=args.games,
        base_seed=args.seed,
    )

    print_report(results)


if __name__ == "__main__":
    main()
