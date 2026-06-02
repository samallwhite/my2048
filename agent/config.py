"""启发式评价函数的权重配置。
第 3 部分（进化计算/遗传算法）将对此配置进行自动调参优化。
"""

from dataclasses import dataclass, asdict


@dataclass
class HeuristicWeights:
    """启发式评价函数的线性加权系数。

    评价函数 = w_empty * 空格数
              + w_monotonicity * 单调性得分
              + w_smoothness * 平滑度得分
              + w_max_corner * 最大数靠角得分
    """

    w_empty: float = 12.0       # 空格数量权重
    w_monotonicity: float = 1.5  # 行列单调性权重
    w_smoothness: float = 1.0    # 相邻方块平滑度权重
    w_max_corner: float = 8.0    # 最大数靠角奖励权重

    # ── GA 接口（第 3 部分使用）──────────────────────────────────────

    def to_list(self) -> list[float]:
        """将权重转为列表，方便 GA 编码为染色体。"""
        return [self.w_empty, self.w_monotonicity,
                self.w_smoothness, self.w_max_corner]

    @classmethod
    def from_list(cls, values: list[float]) -> "HeuristicWeights":
        """从列表解码权重，方便 GA 染色体解码。"""
        return cls(
            w_empty=values[0],
            w_monotonicity=values[1],
            w_smoothness=values[2],
            w_max_corner=values[3],
        )

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """各权重的搜索范围，供 GA 初始化种群使用。"""
        return [
            (0.0, 50.0),    # w_empty
            (0.0, 10.0),    # w_monotonicity
            (0.0, 10.0),    # w_smoothness
            (0.0, 50.0),    # w_max_corner
        ]

    @property
    def param_names(self) -> list[str]:
        """权重参数名称列表。"""
        return ["w_empty", "w_monotonicity", "w_smoothness", "w_max_corner"]
