"""基于 Expectimax（期望极小极大）启发式搜索的 2048 AI 智能体。

搜索树结构（depth=2 示例）：
  Max（根：选动作）→ Chance（spawn 2/4）→ Max（选动作）→ Chance（spawn 2/4）→ 评价

关键设计：
  - Max 节点：遍历四个合法移动方向，取最高期望得分。
  - Chance 节点：枚举所有空位 × {2(90%), 4(10%)}，按概率加权求期望。
  - 动态深度：采用 nneonneo/2048-ai 风格的 distinct_tiles - 2。
  - 概率剪枝：累计概率低于阈值时直接用启发式评价。
  - 权重参数独立配置，为第 3 部分遗传算法预留接口。
"""

import numpy as np
from game.board import Board, UP, DOWN, LEFT, RIGHT
from agent.base import Agent
from agent.config import HeuristicWeights

_DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


class HeuristicAgent(Agent):
    """使用 Expectimax 搜索 + 多特征启发式评价函数的 2048 智能体。

    搜索深度采用 nneonneo/2048-ai 风格的自适应策略：
      depth = max(3, distinct_tiles - 2)，上限由 max_depth 控制。
    内置置换表（transposition table），缓存已评价棋盘避免重复计算。
    """

    # 置换表缓存深度上限（C++ 参考值 15，远大于实际搜索深度）
    _CACHE_DEPTH_LIMIT = 15
    _CPROB_THRESH_BASE = 0.0001
    _ROW_MASK = 0xffff

    def __init__(self,
                 weights: HeuristicWeights | None = None,
                 max_depth: int = 2):
        """
        Args:
            weights: 启发式权重配置。第 3 部分遗传算法将传入优化后的权重。
            max_depth: 最大搜索深度上限。默认 2，保证 Python 版评测可用；
                需要更强但更慢的搜索时可调高。
        """
        self.weights = weights if weights is not None else HeuristicWeights()
        self.max_depth = max_depth

        # 预建查表：65536 种行的启发式得分（一次计算，终生使用）
        self._row_table = self._build_table()
        self._row_scores = self._row_table.tolist()
        self._row_left_table, self._row_right_table = self._build_move_tables()

        # 置换表：每步 get_action 时重建，跨 root 方向共享
        self._trans_table: dict = {}

    def reset(self) -> None:
        """每局游戏开始时重置内部状态。"""
        self._trans_table = {}

    # ── 顶层接口 ──────────────────────────────────────────────────────

    def get_action(self, board: Board) -> int:
        """返回当前棋盘下的最佳动作。

        使用 C++ 风格的自适应深度 depth = max(3, distinct_tiles - 2)，
        通过置换表缓存已评价棋盘，避免不同路径到达相同状态时重复计算。
        """
        best_score = -float('inf')
        best_action = UP

        # C++ 风格自适应深度：方块种类越多越深入
        distinct = self._count_distinct_tiles(board)
        dynamic_depth = min(self.max_depth, max(3, distinct - 2))

        # 每步重建置换表（与 C++ score_toplevel_move 创建 eval_state 一致）
        self._trans_table = {}

        rank_board = self._board_to_rank_int(board)

        for direction in _DIRECTIONS:
            new_board = self._execute_move(direction, rank_board)

            if new_board == rank_board:
                continue

            score = self._chance_node(new_board,
                                      curdepth=0,
                                      depth_limit=dynamic_depth,
                                      cprob=1.0)

            if score > best_score:
                best_score = score
                best_action = direction

        return best_action

    def _count_distinct_tiles(self, board: Board) -> int:
        """统计棋盘上不同方块 rank 的种类数（不含空格）。

        用于自适应深度计算，与 C++ count_distinct_tiles 等价。
        """
        grid = board.grid
        mask = grid > 0
        if not mask.any():
            return 0
        ranks = np.log2(grid[mask].astype(np.uint64))
        return int(len(np.unique(ranks)))

    # ── 搜索树节点 ────────────────────────────────────────────────────

    def _chance_node(self,
                     board: int,
                     curdepth: int,
                     depth_limit: int,
                     cprob: float) -> float:
        """Chance 节点：全枚举所有空位 × {2, 4}，按概率求期望值。

        内置置换表：缓存已评价的 (board, curdepth) 避免重复计算。
        与 C++ score_tilechoose_node + trans_table 逻辑等价。

        Args:
            board: 已执行移动但尚未 spawn 的 rank bitboard。
            curdepth: 当前已经模拟的 Max 层数。
            depth_limit: 本次搜索的 Max 层数上限。
            cprob: 到达当前节点的累计概率。
        """
        if cprob < self._CPROB_THRESH_BASE or curdepth >= depth_limit:
            return self._evaluate(board)

        # ── 置换表查询 ──────────────────────────────────────────
        if curdepth < self._CACHE_DEPTH_LIMIT:
            cached = self._trans_table.get(board)
            if cached is not None and cached[0] <= curdepth:
                return cached[1]

        empty_shifts = []
        tmp = board
        for shift in range(0, 64, 4):
            if (tmp & 0xf) == 0:
                empty_shifts.append(shift)
            tmp >>= 4

        if not empty_shifts:
            return self._evaluate(board)

        expected_value = 0.0
        num_open = len(empty_shifts)
        child_cprob = cprob / num_open

        for shift in empty_shifts:
            # 放 2（概率 90%）→ 进入 Max 节点
            board_2 = board | (1 << shift)
            value_2 = self._max_node(board_2,
                                     curdepth=curdepth,
                                     depth_limit=depth_limit,
                                     cprob=child_cprob * 0.9)

            # 放 4（概率 10%）→ 进入 Max 节点
            board_4 = board | (2 << shift)
            value_4 = self._max_node(board_4,
                                     curdepth=curdepth,
                                     depth_limit=depth_limit,
                                     cprob=child_cprob * 0.1)

            expected_value += 0.9 * value_2 + 0.1 * value_4

        expected_value /= num_open

        # ── 置换表写入 ──────────────────────────────────────────
        if curdepth < self._CACHE_DEPTH_LIMIT:
            self._trans_table[board] = (curdepth, expected_value)

        return expected_value

    def _max_node(self,
                  board: int,
                  curdepth: int,
                  depth_limit: int,
                  cprob: float) -> float:
        """Max 节点：遍历四个方向，取最高评价值。

        Args:
            board: 已 spawn 新方块的 rank bitboard。
            curdepth: 当前已经模拟的 Max 层数。
            depth_limit: 本次搜索的 Max 层数上限。
            cprob: 到达当前节点的累计概率。
        """
        best_score = 0.0
        next_depth = curdepth + 1

        for direction in _DIRECTIONS:
            new_board = self._execute_move(direction, board)
            if new_board == board:
                continue
            score = self._chance_node(new_board,
                                      curdepth=next_depth,
                                      depth_limit=depth_limit,
                                      cprob=cprob)
            if score > best_score:
                best_score = score

        return best_score

    # ── 搜索用 bitboard 工具 ───────────────────────────────────────────

    @staticmethod
    def _board_to_rank_int(board: Board) -> int:
        """将外部 Board 转成 C++ 风格的 64-bit rank bitboard。"""
        packed = 0
        shift = 0
        for value in board.grid.reshape(16):
            v = int(value)
            if v:
                packed |= (v.bit_length() - 1) << shift
            shift += 4
        return packed

    @staticmethod
    def _reverse_row(row: int) -> int:
        return ((row >> 12) |
                ((row >> 4) & 0x00f0) |
                ((row << 4) & 0x0f00) |
                ((row << 12) & 0xf000))

    @staticmethod
    def _pack_line(line: list[int]) -> int:
        return (line[0] |
                (line[1] << 4) |
                (line[2] << 8) |
                (line[3] << 12))

    @staticmethod
    def _slide_rank_line_left(line: list[int]) -> list[int]:
        tiles = [v for v in line if v != 0]
        merged = []
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged.append(min(tiles[i] + 1, 0xf))
                i += 2
            else:
                merged.append(tiles[i])
                i += 1
        return merged + [0] * (4 - len(merged))

    def _build_move_tables(self) -> tuple[list[int], list[int]]:
        """预计算每种 16-bit 行向左/向右移动后的 XOR diff。"""
        row_left = [0] * 65536
        row_right = [0] * 65536

        for row in range(65536):
            line = [
                (row >> 0) & 0xf,
                (row >> 4) & 0xf,
                (row >> 8) & 0xf,
                (row >> 12) & 0xf,
            ]
            result = self._pack_line(self._slide_rank_line_left(line))
            rev_row = self._reverse_row(row)
            rev_result = self._reverse_row(result)

            row_left[row] = row ^ result
            row_right[rev_row] = rev_row ^ rev_result

        return row_left, row_right

    @staticmethod
    def _transpose(board: int) -> int:
        a1 = board & 0xF0F00F0FF0F00F0F
        a2 = board & 0x0000F0F00000F0F0
        a3 = board & 0x0F0F00000F0F0000
        a = a1 | (a2 << 12) | (a3 >> 12)
        b1 = a & 0xFF00FF0000FF00FF
        b2 = a & 0x00FF00FF00000000
        b3 = a & 0x00000000FF00FF00
        return b1 | (b2 >> 24) | (b3 << 24)

    def _execute_rows(self, board: int, table: list[int]) -> int:
        ret = board
        ret ^= table[(board >> 0) & self._ROW_MASK] << 0
        ret ^= table[(board >> 16) & self._ROW_MASK] << 16
        ret ^= table[(board >> 32) & self._ROW_MASK] << 32
        ret ^= table[(board >> 48) & self._ROW_MASK] << 48
        return ret

    def _execute_move(self, direction: int, board: int) -> int:
        if direction == UP:
            return self._transpose(
                self._execute_rows(self._transpose(board), self._row_left_table)
            )
        if direction == DOWN:
            return self._transpose(
                self._execute_rows(self._transpose(board), self._row_right_table)
            )
        if direction == LEFT:
            return self._execute_rows(board, self._row_left_table)
        if direction == RIGHT:
            return self._execute_rows(board, self._row_right_table)
        raise ValueError(f"Invalid direction: {direction}")

    def _evaluate_rank_board(self, board: int) -> float:
        scores = self._row_scores
        transposed = self._transpose(board)
        return float(
            scores[(board >> 0) & self._ROW_MASK] +
            scores[(board >> 16) & self._ROW_MASK] +
            scores[(board >> 32) & self._ROW_MASK] +
            scores[(board >> 48) & self._ROW_MASK] +
            scores[(transposed >> 0) & self._ROW_MASK] +
            scores[(transposed >> 16) & self._ROW_MASK] +
            scores[(transposed >> 32) & self._ROW_MASK] +
            scores[(transposed >> 48) & self._ROW_MASK]
        )

    # ── 查表构建 ──────────────────────────────────────────────────────

    def _build_table(self) -> np.ndarray:
        """预计算所有 65536 种行的启发式得分，存入查表数组。

        一行 4 格，每格 rank 值 0-15 占 4 bit，共 16 bit = 65536 种可能。
        游戏开始前一次性计算完毕，运行时仅需 8 次数组查表 + 求和。

        耗时约 80ms（仅初始化时执行一次）。
        """
        w = self.weights
        codes = np.arange(65536, dtype=np.uint64)

        # 解码所有行：将 16-bit 编码展开为 (65536, 4) 的 rank 矩阵
        ranks = np.zeros((65536, 4), dtype=np.float64)
        ranks[:, 0] = (codes >> 0) & 0xf
        ranks[:, 1] = (codes >> 4) & 0xf
        ranks[:, 2] = (codes >> 8) & 0xf
        ranks[:, 3] = (codes >> 12) & 0xf

        # ── ① 空格计数 ────────────────────────────────────────────
        empties = np.sum(ranks == 0, axis=1)

        # ── ② sum 惩罚 ────────────────────────────────────────────
        nonzero = ranks > 0
        sum_p = np.where(nonzero, ranks ** w.sum_power, 0.0)
        sum_penalties = np.sum(sum_p, axis=1)

        # ── ③ 单调性 ──────────────────────────────────────────────
        diff = np.diff(ranks, axis=1)
        pow_ranks = ranks ** w.mono_power
        left_c = np.where(diff < 0,
                          pow_ranks[:, :-1] - pow_ranks[:, 1:], 0.0)
        right_c = np.where(diff >= 0,
                           pow_ranks[:, 1:] - pow_ranks[:, :-1], 0.0)
        mono_p = np.minimum(np.sum(left_c, axis=1),
                            np.sum(right_c, axis=1))

        # ── ④ 合并簇 ────────────────────────────────────────────
        # 状态机逻辑难以完全向量化，用 Python 循环计算 65536 行。
        # 每行仅 4 个元素，总迭代 262K 次，约 50ms。
        merges = np.zeros(65536, dtype=np.float64)
        for code in range(65536):
            prev = 0
            counter = 0
            for k in range(4):
                r = int(ranks[code, k])
                if r == 0:
                    continue
                if prev == r:
                    counter += 1
                else:
                    if counter > 0:
                        merges[code] += 1 + counter
                        counter = 0
                    prev = r
            if counter > 0:
                merges[code] += 1 + counter

        # ── ⑤ 汇总 ────────────────────────────────────────────────
        table = (w.w_lost_penalty
                 + w.w_empty * empties
                 + w.w_merges * merges
                 - w.w_monotonicity * mono_p
                 - w.w_sum * sum_penalties)

        return table.astype(np.float64)

    # ── 行编码工具 ─────────────────────────────────────────────────────

    @staticmethod
    def _pack_rows(ranks: np.ndarray) -> np.ndarray:
        """将 (4, 4) rank 矩阵的每一行打包为 16-bit 整数。

        rank[i,j] 占 4 bit，一行 4 格 = 16 bit。
        """
        r = ranks.astype(np.uint64)
        return (r[:, 0] | (r[:, 1] << 4) |
                (r[:, 2] << 8) | (r[:, 3] << 12))

    # ── 启发式评价函数（nneonneo/2048-ai 风格 + 查表）──────────────

    def _evaluate(self, board: Board | int) -> float:
        """查表式启发式评价。

        将棋盘 4 行 + 4 列分别编码为 16-bit 整数，
        查预计算表取出对应行的得分，8 次求和即得总评价值。
        单次评价约 200ns（仅数组索引 + 加法）。
        """
        if isinstance(board, int):
            return self._evaluate_rank_board(board)

        grid = board.grid

        # tile 值 → rank（log2）
        ranks = np.zeros((4, 4), dtype=np.uint64)
        mask = grid > 0
        ranks[mask] = np.log2(grid[mask].astype(np.uint64))

        # 4 行编码 + 4 列编码，各查表求和
        row_codes = self._pack_rows(ranks)       # (4,) uint64
        col_codes = self._pack_rows(ranks.T)     # (4,) uint64

        return float(np.sum(self._row_table[row_codes]) +
                     np.sum(self._row_table[col_codes]))

    # ── 公开接口（供外部调用或调试） ──────────────────────────────────

    def evaluate_board(self, board: Board) -> float:
        """对外暴露：返回当前棋盘的启发式评分。"""
        return self._evaluate(board)

    def get_feature_values(self, board: Board) -> dict[str, float]:
        """对外暴露：返回逐行评价的汇总指标（用于调试和 GUI 显示）。

        注意：此函数为调试用途，未使用查表优化，单次调用约 20μs。
        """
        w = self.weights
        grid = board.grid

        ranks = np.zeros((4, 4), dtype=np.float64)
        mask = grid > 0
        ranks[mask] = np.log2(grid[mask].astype(np.float64))

        rows = np.empty((8, 4), dtype=np.float64)
        rows[0:4] = ranks
        rows[4:8] = ranks.T

        empties = np.sum(rows == 0, axis=1)
        total_empty = int(np.sum(empties))

        nonzero = rows > 0
        sum_powered = np.where(nonzero, rows ** w.sum_power, 0.0)
        total_sum_penalty = float(np.sum(sum_powered))

        merges_arr = np.zeros(8, dtype=np.float64)
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
                        merges_arr[i] += 1 + counter
                        counter = 0
                    prev = r
            if counter > 0:
                merges_arr[i] += 1 + counter
        total_merges = int(np.sum(merges_arr))

        diff = np.diff(rows, axis=1)
        pow_rows = rows ** w.mono_power
        left_contrib = np.where(diff < 0,
                                pow_rows[:, :-1] - pow_rows[:, 1:], 0.0)
        right_contrib = np.where(diff >= 0,
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
