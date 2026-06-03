"""基于 Expectimax（期望极小极大）启发式搜索的 2048 AI 智能体。

搜索树结构（depth=2 示例）：
  Max（根：选动作）→ Chance（spawn 2/4）→ Max（选动作）→ Chance（spawn 2/4）→ 评价

关键设计：
  - Max 节点：遍历四个合法移动方向，取最高期望得分。
  - Chance 节点：枚举所有空位 × {2(90%), 4(10%)}，按概率加权求期望。
  - 动态深度：空格多时 depth=2（快），空格少时 depth=3（深）。
  - 全枚举为主：depth=2 时开销约 32×E 次评价/步，不需采样。
  - 权重参数独立配置，为第 3 部分遗传算法预留接口。
"""

import random
import numpy as np
from game.board import Board, UP, DOWN, LEFT, RIGHT
from agent.base import Agent
from agent.config import HeuristicWeights

_DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


class HeuristicAgent(Agent):
    """使用 Expectimax 搜索 + 多特征启发式评价函数的 2048 智能体。

    性能特征（depth=2，全枚举）：
      - 每步评价次数：~32×E（E 为空格数），最大约 448 次
      - 单步耗时：< 0.05 秒
      - 每局游戏：~2-5 秒（取决于存活步数）
    """

    def __init__(self,
                 weights: HeuristicWeights | None = None,
                 max_depth: int = 2,
                 sample_threshold: int = 4,
                 sample_size: int = 3):
        """
        Args:
            weights: 启发式权重配置。第 3 部分遗传算法将传入优化后的权重。
            max_depth: 最大搜索深度（模拟的游戏步数），默认 2。
            sample_threshold: 空位数超过此阈值时触发随机采样（默认 4）。
                              设为 4 意味着 E>4 时每层只采样 4 个空位（8 分支）。
            sample_size: 采样数量，默认 3（每层约 6 个分支）。
        """
        self.weights = weights if weights is not None else HeuristicWeights()
        self.max_depth = max_depth
        self.sample_threshold = sample_threshold
        self.sample_size = sample_size

    def reset(self) -> None:
        """每局游戏开始时重置内部状态。"""
        pass

    # ── 顶层接口 ──────────────────────────────────────────────────────

    def get_action(self, board: Board) -> int:
        """返回当前棋盘下的最佳动作。

        模拟 4 个方向，对每个方向的移动结果调用 Chance 节点求期望，
        选择期望值最高的动作。
        """
        best_score = -float('inf')
        best_action = UP

        # 根据当前空格数动态调整深度
        empty_count = len(board.get_empty_cells())
        dynamic_depth = self._calc_depth(empty_count)

        for direction in _DIRECTIONS:
            new_board = board.execute_move(direction)

            # 无效移动跳过
            if np.array_equal(new_board.grid, board.grid):
                continue

            score = self._chance_node(new_board, dynamic_depth)

            if score > best_score:
                best_score = score
                best_action = direction

        return best_action

    def _calc_depth(self, empty_count: int) -> int:
        """根据空格数量动态计算搜索深度。

        空格 ≥ 5 → depth 2（快，~256 次评价）
        空格 < 5 → depth 3（深，~3K-15K 次评价）
        """
        if empty_count >= 5:
            return min(2, self.max_depth)
        else:
            return min(3, self.max_depth)

    # ── 搜索树节点 ────────────────────────────────────────────────────

    def _chance_node(self, board: Board, remaining_depth: int) -> float:
        """Chance 节点：枚举所有可能的随机 spawn，按概率求期望值。

        Args:
            board: 已执行移动但尚未 spawn 的棋盘。
            remaining_depth: 还能模拟的 Max 层数（≥1）。
        """
        empty = board.get_empty_cells()
        if not empty:
            return self._evaluate(board)

        # 空位过多时随机采样（默认阈值 10，一般深度 2 不会触发）
        if len(empty) > self.sample_threshold:
            empty = random.sample(empty, self.sample_size)

        expected_value = 0.0
        prob_per_cell = 1.0 / len(empty)

        for r, c in empty:
            # 放 2（概率 90%）→ 进入 Max 节点
            board_2 = board.copy()
            board_2.grid[r, c] = 2
            value_2 = self._max_node(board_2, remaining_depth - 1)

            # 放 4（概率 10%）→ 进入 Max 节点
            board_4 = board.copy()
            board_4.grid[r, c] = 4
            value_4 = self._max_node(board_4, remaining_depth - 1)

            expected_value += prob_per_cell * (0.9 * value_2 + 0.1 * value_4)

        return expected_value

    def _max_node(self, board: Board, remaining_depth: int) -> float:
        """Max 节点：遍历四个方向，取最高评价值。

        Args:
            board: 已 spawn 新方块的棋盘。
            remaining_depth: 还能模拟的 Max 层数。
        """
        if remaining_depth == 0:
            # 到达深度上限，直接评价
            return self._evaluate(board)

        best_score = -float('inf')
        has_valid = False

        for direction in _DIRECTIONS:
            new_board = board.execute_move(direction)
            if np.array_equal(new_board.grid, board.grid):
                continue
            has_valid = True
            score = self._chance_node(new_board, remaining_depth)
            if score > best_score:
                best_score = score

        if not has_valid:
            return self._evaluate(board)

        return best_score

    # ── 启发式评价函数（nneonneo/2048-ai 风格 + NumPy 向量化） ─────

    def _evaluate(self, board: Board) -> float:
        """逐行启发式评价函数（NumPy 向量化实现）。

        将棋盘拆为 4 行 + 4 列（共 8 个行向量），每条指令同时处理全部 8 行，
        用 C 底层循环替代 Python for + pow()。

        公式（每行）:
          row_score = w_lost_penalty
                    + w_empty * empty_count
                    + w_merges * merge_clusters
                    - w_monotonicity * min(mono_left, mono_right)
                    - w_sum * sum(rank ^ sum_power)

        参考: nneonneo/2048-ai (github.com/nneonneo/2048-ai)
        """
        w = self.weights
        grid = board.grid

        # ── 构造 rank 矩阵 ─────────────────────────────────────────
        ranks = np.zeros((4, 4), dtype=np.float64)
        mask = grid > 0
        ranks[mask] = np.log2(grid[mask].astype(np.float64))

        # 拼接为 (8, 4) 矩阵：前 4 行 = 原始行，后 4 行 = 列（转置后当行处理）
        rows = np.empty((8, 4), dtype=np.float64)
        rows[0:4] = ranks
        rows[4:8] = ranks.T

        # ── ① 空格计数（向量化）───────────────────────────────────
        empties = np.sum(rows == 0, axis=1)              # (8,) 每行空格数

        # ── ② sum 惩罚（向量化）──────────────────────────────────
        nonzero = rows > 0
        sum_powered = np.where(nonzero, rows ** w.sum_power, 0.0)
        sum_penalties = np.sum(sum_powered, axis=1)      # (8,) 每行的 Σ rank^3.5

        # ── ③ 单调性（向量化）────────────────────────────────────
        # diff[i,j] = rows[i,j+1] - rows[i,j]，shape (8, 3)
        diff = np.diff(rows, axis=1)
        pow_rows = rows ** w.mono_power                   # (8, 4) rank^4

        # mono_left:  行从左到右递减时累积（diff < 0）
        left_mask = diff < 0
        left_contrib = np.where(
            left_mask,
            pow_rows[:, :-1] - pow_rows[:, 1:],          # pow(prev) - pow(curr)
            0.0
        )
        mono_left = np.sum(left_contrib, axis=1)          # (8,)

        # mono_right: 行从左到右递增或相等时累积（diff >= 0）
        right_mask = diff >= 0
        right_contrib = np.where(
            right_mask,
            pow_rows[:, 1:] - pow_rows[:, :-1],          # pow(curr) - pow(prev)
            0.0
        )
        mono_right = np.sum(right_contrib, axis=1)        # (8,)

        # 取两者中较小者作为惩罚（非单调行两边都会累积，min > 0）
        mono_penalties = np.minimum(mono_left, mono_right)

        # ── ④ 合并簇（小循环：8 行 × 4 列，开销可忽略）─────────
        merges = np.zeros(8, dtype=np.float64)
        for i in range(8):
            prev = 0
            counter = 0
            for k in range(4):
                r = int(rows[i, k])
                if r == 0:
                    continue
                if prev == r:
                    counter += 1
                else:
                    if counter > 0:
                        merges[i] += 1 + counter
                        counter = 0
                    prev = r
            if counter > 0:
                merges[i] += 1 + counter

        # ── ⑤ 汇总（向量化）──────────────────────────────────────
        row_scores = (w.w_lost_penalty
                      + w.w_empty * empties
                      + w.w_merges * merges
                      - w.w_monotonicity * mono_penalties
                      - w.w_sum * sum_penalties)

        return float(np.sum(row_scores))

    # ── 公开接口（供外部调用或调试） ──────────────────────────────────

    def evaluate_board(self, board: Board) -> float:
        """对外暴露：返回当前棋盘的启发式评分。"""
        return self._evaluate(board)

    def get_feature_values(self, board: Board) -> dict[str, float]:
        """对外暴露：返回逐行评价的汇总指标（用于调试和 GUI 显示）。"""
        w = self.weights
        grid = board.grid

        ranks = np.zeros((4, 4), dtype=np.float64)
        mask = grid > 0
        ranks[mask] = np.log2(grid[mask].astype(np.float64))

        rows = np.empty((8, 4), dtype=np.float64)
        rows[0:4] = ranks
        rows[4:8] = ranks.T

        # 用向量化方式汇总每个特征
        empties = np.sum(rows == 0, axis=1)
        total_empty = int(np.sum(empties))

        nonzero = rows > 0
        sum_powered = np.where(nonzero, rows ** w.sum_power, 0.0)
        total_sum_penalty = float(np.sum(sum_powered))

        # 合并簇
        merges = np.zeros(8, dtype=np.float64)
        for i in range(8):
            prev = 0
            counter = 0
            for k in range(4):
                r = int(rows[i, k])
                if r == 0:
                    continue
                if prev == r:
                    counter += 1
                else:
                    if counter > 0:
                        merges[i] += 1 + counter
                        counter = 0
                    prev = r
            if counter > 0:
                merges[i] += 1 + counter
        total_merges = int(np.sum(merges))

        # 单调性惩罚
        diff = np.diff(rows, axis=1)
        pow_rows = rows ** w.mono_power
        left_contrib = np.where(
            diff < 0,
            pow_rows[:, :-1] - pow_rows[:, 1:], 0.0)
        right_contrib = np.where(
            diff >= 0,
            pow_rows[:, 1:] - pow_rows[:, :-1], 0.0)
        mono_penalties = np.minimum(
            np.sum(left_contrib, axis=1),
            np.sum(right_contrib, axis=1))
        total_mono_penalty = float(np.sum(mono_penalties))

        return {
            "total_score": self._evaluate(board),
            "empty_cells": total_empty,
            "merge_clusters": total_merges,
            "mono_penalty": total_mono_penalty,
            "sum_penalty": total_sum_penalty,
        }
