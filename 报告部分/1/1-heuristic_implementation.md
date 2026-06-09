# Heuristic 实现路径

本文档只说明当前 heuristic agent 的实现流程，便于实验报告描述。

## 1. 总体思路

当前智能体使用 `Expectimax` 搜索选择动作：

1. `Max` 节点表示玩家回合，遍历上下左右四个方向，选择后续期望分数最高的动作。
2. `Chance` 节点表示随机生成新方块，枚举所有空格，并分别考虑生成 `2` 和 `4` 两种情况。
3. 当搜索达到深度上限，或当前路径累计概率过低时，不再继续展开，而是调用 heuristic 评价函数估计棋盘价值。

最终，顶层对四个合法移动分别计算期望价值，返回分数最高的方向。

## 2. 棋盘表示

为了提高搜索速度，搜索内部不直接使用 `4 x 4` 数组，而是把棋盘压缩成一个 64 位整数：

- 每个格子占 4 bit。
- 存储的不是方块原值，而是 `rank = log2(tile)`。
- 空格的 rank 为 `0`，例如 `2 -> 1`，`4 -> 2`，`2048 -> 11`。

这样每一行刚好是 16 bit，一共有 `65536` 种可能。程序可以预先计算所有行的移动结果和 heuristic 分数，后续搜索时直接查表。

## 3. 行查表预计算

初始化 `HeuristicAgent` 时会构建三类表：

1. `row_score_table`：保存每一种 16 bit 行编码的 heuristic 分数。
2. `row_left_table`：保存每一种行向左移动后的 XOR 差分。
3. `row_right_table`：保存每一种行向右移动后的 XOR 差分。

左右移动时，直接对四行查移动表；上下移动时，先转置棋盘，再复用左右移动表，最后转置回来。

## 4. Heuristic 评分函数

棋盘总分由 4 行和 4 列的分数相加得到。每一行的评分公式为：

```text
row_score =
    w_lost_penalty
  + w_empty * empty_count
  + w_merges * merge_clusters
  - w_monotonicity * monotonicity_penalty
  - w_sum * sum_penalty
```

当前默认权重为：

```text
w_lost_penalty = 200000
w_empty        = 270
w_merges       = 700
w_monotonicity = 47
w_sum          = 11
mono_power     = 4.0
sum_power      = 3.5
```

各项含义如下：

- `empty_count`：空格数量。空格越多，棋盘越灵活，因此加分。
- `merge_clusters`：同一行中相邻且数值相同的方块簇。可合并机会越多，加分越高。
- `monotonicity_penalty`：单调性惩罚。若一行更接近从小到大或从大到小排列，惩罚较小；反之惩罚较大。
- `sum_penalty`：大数惩罚。对非空 rank 计算 `rank ^ sum_power` 后求和，用于控制高 rank 分布，避免只追求局部大数。

评价一个棋盘时，程序把 4 行和 4 列都编码为 16 bit，然后查 `row_score_table` 并求和。这样单次评价只需要 8 次查表和加法。

## 5. Expectimax 搜索流程

顶层 `get_action(board)` 的执行流程如下：

1. 统计当前棋盘中不同非空 rank 的数量。
2. 计算搜索深度：

```text
dynamic_depth = min(max_depth, max(3, distinct_tiles - 2))
```

当前构造函数默认 `max_depth = 2`，因此默认评估时深度上限为 2；如果调高 `max_depth`，后期棋盘会根据不同方块种类自动增加搜索深度。

3. 清空本轮搜索的置换表缓存。
4. 将棋盘转成 64 位 rank bitboard。
5. 枚举四个方向：
   - 如果移动后棋盘不变，说明该方向非法，跳过。
   - 否则进入 `Chance` 节点，计算该动作后的期望价值。
6. 返回期望价值最高的合法方向。

## 6. Chance 节点

`Chance` 节点用于模拟游戏随机生成新方块：

1. 找出当前棋盘所有空格。
2. 对每个空格分别模拟：
   - 以 `90%` 概率生成 `2`。
   - 以 `10%` 概率生成 `4`。
3. 每个生成结果继续进入 `Max` 节点。
4. 将所有结果按概率加权平均，得到该随机节点的期望价值。

若满足以下任一条件，则直接返回 heuristic 评分：

- 当前深度达到 `depth_limit`。
- 当前路径累计概率低于 `0.0001`。
- 棋盘已经没有空格。

## 7. Max 节点

`Max` 节点模拟玩家下一步选择：

1. 枚举上下左右四个方向。
2. 跳过不能改变棋盘的非法移动。
3. 对每个合法移动进入下一层 `Chance` 节点。
4. 返回所有合法移动中的最高期望价值。

## 8. 置换表缓存

搜索过程中可能通过不同路径到达同一个棋盘。为了避免重复计算，`Chance` 节点使用置换表缓存已评价过的棋盘：

- key 为压缩后的 64 位棋盘。
- value 保存该棋盘对应的搜索深度和期望分数。
- 每次顶层选动作时重新创建缓存，因此缓存只在一次决策搜索树内部共享。

该缓存可以显著减少重复状态的展开，提高 Expectimax 的搜索效率。

## 9. 对外接口

`HeuristicAgent` 提供两个主要调试接口：

- `evaluate_board(board)`：返回当前棋盘的 heuristic 总分。
- `get_feature_values(board)`：返回空格数、合并簇、单调性惩罚、sum 惩罚和总分，便于 GUI 或实验分析展示。

