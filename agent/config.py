"""启发式评价函数的权重配置。
第 3 部分（进化计算/遗传算法）将对此配置进行自动调参优化。

当前使用 nneonneo/2048-ai 风格的逐行评价函数（详见 agent/heuristic.py）。
"""

from dataclasses import dataclass


@dataclass
class HeuristicWeights:
    """启发式评价函数的权重系数。

    评价方式：将棋盘拆为 4 行 + 4 列，每行独立计算：
      row_score = w_lost_penalty
                + w_empty * empty_count
                + w_merges * merge_clusters
                - w_monotonicity * min(mono_left, mono_right)
                - w_sum * sum(pow(rank, sum_power))

    总分 = Σ 行得分 + Σ 列得分（共 8 个行向量）
    """

    # ── 线性权重（GA 可优化） ─────────────────────────────────────────
    w_lost_penalty: float = 200000.0   # 基础惩罚（常数值，提供评分基线）
    w_empty: float = 270.0            # 空格奖励（每行每空位加分）
    w_merges: float = 700.0           # 合并簇奖励（相邻同值方块聚集）
    w_monotonicity: float = 47.0      # 单调性惩罚（非单调行减分）
    w_sum: float = 11.0               # 大值惩罚（高 rank 方块减分）

    # ── 非线性指数（亦可作为 GA 优化对象） ───────────────────────────
    mono_power: float = 4.0           # 单调性计算中的幂指数
    sum_power: float = 3.5            # sum 惩罚中的幂指数

    # ── GA 接口（第 3 部分使用）──────────────────────────────────────

    def to_list(self) -> list[float]:
        """将线性权重转为列表，方便 GA 编码为染色体。"""
        return [self.w_lost_penalty, self.w_empty, self.w_merges,
                self.w_monotonicity, self.w_sum]

    @classmethod
    def from_list(cls, values: list[float]) -> "HeuristicWeights":
        """从列表解码权重，方便 GA 染色体解码。"""
        return cls(
            w_lost_penalty=values[0],
            w_empty=values[1],
            w_merges=values[2],
            w_monotonicity=values[3],
            w_sum=values[4],
        )

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """各线性权重的搜索范围，供 GA 初始化种群使用。"""
        return [
            (100000.0, 500000.0),   # w_lost_penalty
            (50.0, 1000.0),         # w_empty
            (100.0, 2000.0),        # w_merges
            (10.0, 200.0),          # w_monotonicity
            (1.0, 50.0),            # w_sum
        ]

    @property
    def param_names(self) -> list[str]:
        """线性权重参数名称列表。"""
        return ["w_lost_penalty", "w_empty", "w_merges",
                "w_monotonicity", "w_sum"]
