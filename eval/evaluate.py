"""2048 AI 智能体评估脚本。

特性：
  - 实时进度条 + 动态指标更新
  - 断点续训：自动保存检查点，中断后可恢复

用法：
  python eval/evaluate.py                           # 默认 100 局
  python eval/evaluate.py --games 500               # 自定义局数
  python eval/evaluate.py --seed 42                 # 指定种子
  python eval/evaluate.py --resume                  # 从最近检查点恢复
  python eval/evaluate.py --resume eval/checkpoints/my_run.json  # 指定检查点文件
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from collections import Counter

# 添加项目根目录到 sys.path，确保从任意目录运行均可正确导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.game import Game
from agent.heuristic import HeuristicAgent
from agent.config import HeuristicWeights

# ── 常量 ─────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
PROGRESS_INTERVAL = 1       # 每 N 局保存一次检查点
REPORT_INTERVAL = 10        # 每 N 局打印一次汇总行


# ── 进度显示 ──────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """将秒数格式化为可读时间。"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"


def print_progress(current: int, total: int, results: dict,
                   start_time: float, last_n: int = 1) -> None:
    """打印实时进度行（覆盖上一行）。

    显示：进度条、完成数/总数、最近N局均分、全局均分、最佳、ETA
    """
    elapsed = time.time() - start_time
    pct = current / total * 100 if total > 0 else 0
    bar_width = 20
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_width - filled)

    scores = results["scores"]
    avg_all = sum(scores) / len(scores) if scores else 0
    avg_recent = sum(scores[-last_n:]) / min(last_n, len(scores)) if scores else 0
    best = max(scores) if scores else 0

    # ETA
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = _fmt_time(eta)
    else:
        eta_str = "--"

    elapsed_str = _fmt_time(elapsed)

    # 最大方块分布（简写）
    tile_dist = results.get("tile_counter", Counter())
    dist_parts = []
    for t in sorted(tile_dist.keys()):
        dist_parts.append(f"{t}:{tile_dist[t]}")

    # 用 \r 覆盖当前行
    line = (f"\r[{bar}] {current:>4}/{total} | "
            f"最近{last_n}局均分: {avg_recent:>7.0f} | "
            f"全局均分: {avg_all:>7.0f} | "
            f"最佳: {best:>6} | "
            f"方块: {' '.join(dist_parts):<20} | "
            f"耗时: {elapsed_str} | "
            f"ETA: {eta_str}")
    # 填充到终端宽度以避免残留
    line = line.ljust(120)
    sys.stdout.write(line)
    sys.stdout.flush()


# ── 检查点管理 ────────────────────────────────────────────────────────────

def _make_checkpoint_filename(args: argparse.Namespace) -> str:
    """根据运行参数生成唯一检查点文件名。"""
    key = f"d{args.depth}_s{args.seed}_w{args.weights}_"
    key += f"ss{args.sample_size}_st{args.sample_threshold}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"eval_{h}.json"


def save_checkpoint(filepath: Path, results: dict,
                    completed: int, total: int,
                    agent_params: dict | None = None) -> None:
    """保存检查点到磁盘。

    Args:
        filepath: 检查点文件路径。
        results: 包含 scores, max_tiles, steps_list, tile_counter 的字典。
        completed: 已完成的局数。
        total: 目标总局数。
        agent_params: 智能体参数（用于恢复时验证配置一致性）。
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 将 Counter 转为普通 dict 以便 JSON 序列化
    data = {
        "completed": completed,
        "total": total,
        "scores": results["scores"],
        "max_tiles": results["max_tiles"],
        "steps_list": results["steps_list"],
        "tile_counter": dict(results["tile_counter"]),
        "start_time": results.get("start_time", time.time()),
        "agent_params": agent_params or {},
    }
    # 原子写入：先写临时文件再重命名
    tmp = filepath.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, filepath)


def load_checkpoint(filepath: Path) -> dict | None:
    """加载检查点，文件不存在或损坏时返回 None。"""
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON 会将整数 key 转为字符串，恢复为 int
        if "tile_counter" in data:
            data["tile_counter"] = {
                int(k): v for k, v in data["tile_counter"].items()
            }
        return data
    except (json.JSONDecodeError, KeyError):
        print(f"  [WARN] 检查点文件损坏，忽略: {filepath}")
        return None


def list_checkpoints() -> list[Path]:
    """列出所有检查点文件，按修改时间降序。"""
    if not CHECKPOINT_DIR.exists():
        return []
    files = sorted(CHECKPOINT_DIR.glob("eval_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files


# ── 单局运行 ──────────────────────────────────────────────────────────────

def run_single_game(agent: HeuristicAgent, seed: int | None = None) -> dict:
    """运行一局完整的 2048 游戏并返回统计信息。"""
    game = Game(seed=seed)
    agent.reset()
    game.reset()

    steps = 0
    while True:
        action = agent.get_action(game.board)
        _, _, done, info = game.step(action)

        if info.get("invalid"):
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


# ── 主评估函数 ────────────────────────────────────────────────────────────

def evaluate(agent: HeuristicAgent,
             n_games: int = 100,
             base_seed: int | None = None,
             resume_from: Path | None = None,
             checkpoint_file: Path | None = None,
             agent_params: dict | None = None,
             progress_callback=None) -> dict:
    """评估智能体在 n_games 局游戏中的表现（支持断点续训）。

    Args:
        agent: 待评估的智能体实例。
        n_games: 目标总局数。
        base_seed: 起始随机种子。
        resume_from: 检查点文件路径（恢复训练时使用）。
        checkpoint_file: 本次评估的检查点保存路径。
        progress_callback: 每完成一局后的回调函数。

    Returns:
        dict with keys: scores, max_tiles, steps_list, tile_counter,
                        completed, total, start_time, resumed
    """
    # ── 初始化或恢复数据 ──────────────────────────────────────────────
    if resume_from is not None:
        ckpt = load_checkpoint(resume_from)
        if ckpt is None:
            print(f"  [ERR] 无法加载检查点: {resume_from}")
            sys.exit(1)

        scores = ckpt["scores"]
        max_tiles = ckpt["max_tiles"]
        steps_list = ckpt["steps_list"]
        tile_counter = Counter(ckpt["tile_counter"])
        start_game = ckpt["completed"]
        start_time = ckpt.get("start_time", time.time())
        print(f"  [OK] 从检查点恢复: 已完成 {start_game}/{ckpt['total']} 局")
        print(f"    文件: {resume_from}")

        # 如果目标局数变更，以当前指定为准
        total = n_games
    else:
        scores = []
        max_tiles = []
        steps_list = []
        tile_counter = Counter()
        start_game = 0
        total = n_games
        start_time = time.time()

    results = {
        "scores": scores,
        "max_tiles": max_tiles,
        "steps_list": steps_list,
        "tile_counter": tile_counter,
        "start_time": start_time,
    }

    remaining = total - start_game
    if remaining <= 0:
        print("  [OK] 所有游戏已完成，无需继续。")
        return {**results, "completed": start_game, "total": total,
                "resumed": resume_from is not None}

    # ── 运行游戏 ──────────────────────────────────────────────────────
    for i in range(remaining):
        game_idx = start_game + i
        seed = (base_seed + game_idx) if base_seed is not None else None
        result = run_single_game(agent, seed=seed)

        scores.append(result["score"])
        max_tiles.append(result["max_tile"])
        steps_list.append(result["steps"])
        tile_counter[result["max_tile"]] += 1

        completed = game_idx + 1

        # 实时进度回调
        if progress_callback is not None:
            progress_callback(completed, total, results, start_time)

        # 每局后保存检查点
        if checkpoint_file is not None:
            save_checkpoint(checkpoint_file, results,
                            completed, total, agent_params)

        # 每 REPORT_INTERVAL 局打印汇总行
        if completed % REPORT_INTERVAL == 0 or completed == total:
            scores_slice = scores[-REPORT_INTERVAL:]
            avg_recent = sum(scores_slice) / len(scores_slice)
            avg_all = sum(scores) / len(scores)
            elapsed = time.time() - start_time
            eta = elapsed / completed * (total - completed) if completed > 0 else 0
            print(f"\n  [{completed:>4}/{total}] "
                  f"最近{REPORT_INTERVAL}局均分: {avg_recent:>7.0f} | "
                  f"全局均分: {avg_all:>7.0f} | "
                  f"最佳: {max(scores):>6} | "
                  f"ETA: {_fmt_time(eta)}")

    return {**results, "completed": total, "total": total,
            "resumed": resume_from is not None}


# ── 报告输出 ──────────────────────────────────────────────────────────────

def _std(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    return variance ** 0.5


def print_report(results: dict) -> None:
    scores = results["scores"]
    max_tiles = results["max_tiles"]
    steps_list = results["steps_list"]
    tile_counter = results["tile_counter"]
    total = len(scores)
    elapsed = time.time() - results.get("start_time", time.time())

    print("\n" + "=" * 56)
    print("  2048 AI 评估报告")
    if results.get("resumed"):
        print("  (从检查点恢复)")
    print("=" * 56)
    print(f"  总局数:       {total}")
    print(f"  总耗时:       {_fmt_time(elapsed)}")
    if total > 0:
        print(f"  平均耗时:     {elapsed / total:.2f}s/局")
    print("-" * 56)
    print(f"  平均得分:     {sum(scores) / total:>10.1f}" if total > 0 else "  平均得分:       N/A")
    print(f"  最高得分:     {max(scores):>10}" if scores else "  最高得分:       N/A")
    print(f"  最低得分:     {min(scores):>10}" if scores else "  最低得分:       N/A")
    print(f"  得分标准差:   {_std(scores):>10.1f}" if len(scores) > 1 else "  得分标准差:     N/A")
    print(f"  平均步数:     {sum(steps_list) / total:>10.1f}" if total > 0 else "  平均步数:       N/A")
    print(f"  平均最大方块: {sum(max_tiles) / total:>10.1f}" if total > 0 else "  平均最大方块:   N/A")
    print("-" * 56)
    print("  最大方块分布:")
    for tile in sorted(tile_counter.keys()):
        count = tile_counter[tile]
        pct = 100 * count / total if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {tile:>6}: {count:>4} 局 ({pct:5.1f}%) {bar}")
    print("=" * 56)


# ── 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2048 AI 启发式智能体评估工具（实时进度 + 断点续训）"
    )
    parser.add_argument("--games", type=int, default=100,
                        help="测试局数（默认 100）")
    parser.add_argument("--seed", type=int, default=0,
                        help="起始随机种子（默认 0）")
    parser.add_argument("--depth", type=int, default=2,
                        help="Expectimax 最大搜索深度（默认 2）")
    parser.add_argument("--weights", type=float, nargs=4, default=None,
                        metavar=("W_EMPTY", "W_MONO", "W_SMOOTH", "W_CORNER"),
                        help="自定义权重（四个浮点数）")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Chance 节点采样数（默认 3）")
    parser.add_argument("--sample-threshold", type=int, default=None,
                        help="空位数超过此值触发采样（默认 4）")

    # 断点续训相关
    parser.add_argument("--resume", nargs="?", const="__latest__",
                        metavar="CHECKPOINT_FILE",
                        help="从检查点恢复训练。不带参数则自动选择最近的检查点。")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="禁用检查点保存（默认启用）")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(CHECKPOINT_DIR),
                        help="检查点保存目录")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="列出所有可用的检查点")
    args = parser.parse_args()

    # 列出检查点
    if args.list_checkpoints:
        files = list_checkpoints()
        if not files:
            print("没有找到检查点文件。")
        else:
            print("可用的检查点（按时间降序）:")
            for f in files:
                ckpt = load_checkpoint(f)
                if ckpt:
                    print(f"  {f.name}  — 已完成 {ckpt['completed']}/{ckpt['total']} 局")
                else:
                    print(f"  {f.name}  — 损坏")
        return

    # ── 构建权重配置 ──────────────────────────────────────────────────
    if args.weights is not None:
        weights = HeuristicWeights.from_list(args.weights)
        print(f"权重: {weights}")
    else:
        weights = HeuristicWeights()
        print(f"权重: [默认] {weights}")

    # ── 创建智能体 ────────────────────────────────────────────────────
    agent_kwargs = dict(weights=weights, max_depth=args.depth)
    if args.sample_size is not None:
        agent_kwargs["sample_size"] = args.sample_size
    if args.sample_threshold is not None:
        agent_kwargs["sample_threshold"] = args.sample_threshold
    agent = HeuristicAgent(**agent_kwargs)

    # ── 智能体参数（用于检查点恢复时验证配置一致性） ──────────────────
    agent_params = {
        "depth": args.depth,
        "seed": args.seed,
        "weights": args.weights if args.weights else weights.to_list(),
        "sample_size": agent.sample_size,
        "sample_threshold": agent.sample_threshold,
    }

    print(f"深度: {args.depth} | 局数: {args.games} | 种子: {args.seed}")
    print(f"采样: size={agent.sample_size}, threshold={agent.sample_threshold}")

    # ── 断点续训逻辑 ──────────────────────────────────────────────────
    resume_from = None
    checkpoint_file = None

    if args.resume is not None:
        if args.resume == "__latest__":
            # 自动选择最近的检查点
            files = list_checkpoints()
            if not files:
                print("  [WARN] 没有找到检查点文件，将开始全新训练。")
            else:
                resume_from = files[0]
        else:
            resume_from = Path(args.resume)

    if not args.no_checkpoint:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_file = checkpoint_dir / _make_checkpoint_filename(args)

    print(f"检查点目录: {args.checkpoint_dir}")
    if args.no_checkpoint:
        print("检查点保存: 已禁用")
    else:
        print(f"检查点文件: {checkpoint_file.name}")
    print()

    # ── 开始评估 ──────────────────────────────────────────────────────
    print("开始评估...")
    print(f"(每 {REPORT_INTERVAL} 局输出汇总，每局自动保存检查点，Ctrl+C 可随时中断)\n")

    def progress_fn(completed, total, results, start_time):
        print_progress(completed, total, results, start_time, last_n=5)

    try:
        results = evaluate(
            agent,
            n_games=args.games,
            base_seed=args.seed,
            resume_from=resume_from,
            checkpoint_file=checkpoint_file if not args.no_checkpoint else None,
            agent_params=agent_params,
            progress_callback=progress_fn,
        )
        # 换行，避免进度条残留
        print()
        print_report(results)

        # 完成后删除检查点（可选）
        if checkpoint_file and checkpoint_file.exists() and not args.no_checkpoint:
            checkpoint_file.unlink()
            print(f"\n(检查点已自动清理: {checkpoint_file.name})")

    except KeyboardInterrupt:
        print("\n\n[WARN] 用户中断 (Ctrl+C)")
        if checkpoint_file and checkpoint_file.exists():
            ckpt = load_checkpoint(checkpoint_file)
            if ckpt:
                print(f"  检查点已保存: {checkpoint_file}")
                print(f"  已完成: {ckpt['completed']}/{args.games} 局")
                print(f"  恢复命令: python eval/evaluate.py --resume {checkpoint_file}")
        sys.exit(0)


if __name__ == "__main__":
    main()
