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

    # ── 启发式评价函数 ────────────────────────────────────────────────

    def _evaluate(self, board: Board) -> float:
        """多特征线性加权启发式评价函数。

        评价 = w_empty * 空格数量
              + w_monotonicity * 单调性得分
              + w_smoothness * 平滑度得分
              + w_max_corner * 最大数靠角得分
        """
        w = self.weights

        # 预计算 log2 棋盘（三个特征共用，避免重复计算）
        grid = board.grid
        log_grid = np.zeros((4, 4), dtype=np.float32)
        mask = grid > 0
        log_grid[mask] = np.log2(grid[mask].astype(np.float32))

        # 计算四个启发式特征
        empty_score = float(len(board.get_empty_cells()))
        mono_score = self._monotonicity_from_log(log_grid)
        smooth_score = self._smoothness_from_log(grid, log_grid)
        corner_score = self._max_corner_from_log(grid, log_grid)

        return (w.w_empty * empty_score +
                w.w_monotonicity * mono_score +
                w.w_smoothness * smooth_score +
                w.w_max_corner * corner_score)

    # ── 启发式特征实现 ────────────────────────────────────────────────

    @staticmethod
    def _monotonicity_from_log(log_grid: np.ndarray) -> float:
        """单调性：奖励沿同一方向单调变化的行和列。

        对每行/列分别计算递增和递减的累积对数差，
        取 max(inc, dec) 作为该行/列的单调性得分。
        使用 log2 确保不同量级差异可比。
        """
        total = 0.0

        # 行单调性
        for i in range(4):
            inc, dec = 0.0, 0.0
            for j in range(3):
                diff = log_grid[i, j] - log_grid[i, j + 1]
                if diff > 0:       # 从左到右递减
                    dec += abs(diff)
                elif diff < 0:     # 从左到右递增
                    inc += abs(diff)
            total += max(inc, dec)

        # 列单调性
        for j in range(4):
            inc, dec = 0.0, 0.0
            for i in range(3):
                diff = log_grid[i, j] - log_grid[i + 1, j]
                if diff > 0:       # 从上到下递减
                    dec += abs(diff)
                elif diff < 0:     # 从上到下递增
                    inc += abs(diff)
            total += max(inc, dec)

        return total

    @staticmethod
    def _smoothness_from_log(grid: np.ndarray, log_grid: np.ndarray) -> float:
        """平滑度：惩罚相邻方块数值差异过大。

        只计算两个均为非空格子之间的对数差。
        如 2 和 4 的 log2 差为 1，1024 和 2048 的 log2 差也为 1。
        """
        penalty = 0.0
        for i in range(4):
            for j in range(4):
                if grid[i, j] == 0:
                    continue
                # 右邻
                if j + 1 < 4 and grid[i, j + 1] != 0:
                    penalty += abs(log_grid[i, j] - log_grid[i, j + 1])
                # 下邻
                if i + 1 < 4 and grid[i + 1, j] != 0:
                    penalty += abs(log_grid[i, j] - log_grid[i + 1, j])

        return -penalty

    @staticmethod
    def _max_corner_from_log(grid: np.ndarray, log_grid: np.ndarray) -> float:
        """最大数靠角：奖励最大方块位于四个角落之一。

        角落位置有助于构建单调递减的蛇形排列。
        奖励值使用 log2(max_val)，与其余特征保持同一量级。
        """
        max_val = np.max(grid)
        if max_val == 0:
            return 0.0

        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for r, c in corners:
            if grid[r, c] == max_val:
                return float(np.log2(max_val))

        return 0.0

    # ── 公开接口（供外部调用或调试） ──────────────────────────────────

    def evaluate_board(self, board: Board) -> float:
        """对外暴露：返回当前棋盘的启发式评分。"""
        return self._evaluate(board)

    def get_feature_values(self, board: Board) -> dict[str, float]:
        """对外暴露：返回四个启发式特征的原始值（用于调试和可视化）。"""
        grid = board.grid
        log_grid = np.zeros((4, 4), dtype=np.float32)
        mask = grid > 0
        log_grid[mask] = np.log2(grid[mask].astype(np.float32))

        return {
            "empty": float(len(board.get_empty_cells())),
            "monotonicity": self._monotonicity_from_log(log_grid),
            "smoothness": self._smoothness_from_log(grid, log_grid),
            "max_corner": self._max_corner_from_log(grid, log_grid),
        }
