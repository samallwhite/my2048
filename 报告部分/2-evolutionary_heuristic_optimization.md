# 第二部分：Evolutionary Heuristic Optimization 实现说明

本文档记录第二部分中已经完成的进化式参数优化实现。第一部分已经完成了基于 bitboard 和 Expectimax 的 heuristic agent，本部分不重新设计游戏智能体，而是在现有 heuristic 评估函数上自动搜索更优权重，并将训练、评估和结果保存流程工程化。

相关实现主要位于：

- `agent/config.py`
- `agent/heuristic.py`
- `eval/evaluate.py`
- `eval/evolve_weights.py`

## 1. 优化目标

当前 heuristic agent 的核心决策仍然是 Expectimax。Evolution 部分优化的是叶子节点启发式评估函数中的线性权重，使同一搜索深度下的策略更稳定、更容易达到高阶方块。

优化目标包括：

- 提高平均得分。
- 提高 2048、4096、8192 达成率。
- 避免只根据单局最高分选择参数。
- 保持 `HeuristicAgent` 接口不变，使优化后的权重可以直接用于评估脚本、GUI 和后续神经网络/强化学习模块。

## 2. 染色体设计

启发式评估函数采用逐行评分方式。棋盘的 4 行和 4 列分别被编码为 16-bit row code，然后查表得到每一行/列的评分。

行评分公式为：

```text
row_score =
    w_lost_penalty
  + w_empty * empty_count
  + w_merges * merge_clusters
  - w_monotonicity * monotonicity_penalty
  - w_sum * sum_penalty
```

进化算法直接优化 5 个线性权重：

```text
chromosome = [
    w_lost_penalty,
    w_empty,
    w_merges,
    w_monotonicity,
    w_sum,
]
```

默认权重为：

```text
[200000.0, 270.0, 700.0, 47.0, 11.0]
```

实现中通过 `HeuristicWeights.to_list()` 和 `HeuristicWeights.from_list()` 在配置对象与染色体列表之间转换。因此 evolution 代码不需要了解 agent 内部细节，只需要传入权重列表即可完成评估。

## 3. 搜索范围

代码保留了两套边界：

```text
default bounds:
  w_lost_penalty: 100000.0 ~ 500000.0
  w_empty:             50.0 ~   1000.0
  w_merges:           100.0 ~   2000.0
  w_monotonicity:      10.0 ~    200.0
  w_sum:                1.0 ~     50.0

narrow bounds:
  w_lost_penalty: 150000.0 ~ 300000.0
  w_empty:           200.0 ~    700.0
  w_merges:          400.0 ~   1400.0
  w_monotonicity:     40.0 ~    150.0
  w_sum:               7.0 ~     25.0
```

正式 v2 实验使用了 `narrow` 搜索范围，并固定 `w_lost_penalty = 200000.0`。原因是 `w_lost_penalty` 是逐行常数项，在许多候选动作比较中影响较弱；固定它可以减少搜索维度，把评估预算集中在 `empty`、`merges`、`monotonicity` 和 `sum` 四个更直接影响策略排序的参数上。

## 4. 种群初始化

`eval/evolve_weights.py` 中的初始化策略包含两类个体：

- 默认权重个体：保证初始种群包含当前可用 baseline。
- 局部扰动个体：围绕默认权重按比例加入高斯噪声。
- 随机个体：在搜索边界内均匀采样，用于提供多样性。

正式实现默认使用较高比例的局部初始化：

```text
local_init_fraction = 0.75
local_init_scale = 0.12
```

这样做是因为当前 heuristic 已经是可用策略，完全随机搜索容易浪费大量评估预算；围绕已知有效参数做局部搜索更适合本项目当前规模。

## 5. 遗传算法流程

每一代的基本流程如下：

1. 对当前种群进行评估。
2. 根据 fitness 对个体排序。
3. 保留前 `elite_size` 个精英个体。
4. 使用锦标赛选择产生父代。
5. 使用 BLX-alpha 实数交叉生成子代。
6. 对子代执行按参数范围缩放的高斯变异。
7. 将所有基因裁剪回合法边界。
8. 保存当代统计、最佳权重和 checkpoint。

交叉公式为：

```text
child_gene = uniform(
    min(parent_a, parent_b) - alpha * diff,
    max(parent_a, parent_b) + alpha * diff
)
```

变异公式为：

```text
gene = gene + normal(0, mutation_scale * parameter_range)
```

这种实数编码方式比二进制编码更直接，也更符合当前权重参数本身的连续数值特征。

## 6. 适应度函数

基础 fitness 使用多局统计指标综合计算：

```text
fitness =
    mean_score
  + 5000  * rate_2048
  + 12000 * rate_4096
  + 25000 * rate_8192
  - 0.05  * std_score
```

其中：

- `mean_score` 表示平均得分。
- `rate_2048`、`rate_4096`、`rate_8192` 表示达到对应方块的比例。
- `std_score` 用于惩罚结果波动。

正式 v2 实验进一步启用了 `--baseline-paired`。该模式会在同一组随机种子上同时评估默认权重，并用候选权重相对默认权重的提升作为排序依据：

```text
paired_fitness =
    candidate.mean_score - baseline.mean_score
  + 3000 * (candidate.rate_4096 - baseline.rate_4096)
  + 8000 * (candidate.rate_8192 - baseline.rate_8192)
  - 0.02 * (candidate.std_score - baseline.std_score)
```

paired fitness 的作用是降低 2048 随机生成方块带来的噪声。同一代中候选权重面对相同 seed 集合，因此相对提升比绝对分数更适合作为训练时的选择信号。

## 7. 分阶段评估

由于 depth 2 的 Expectimax 单局成本较高，进化过程中不能对所有个体都进行大量完整评估。因此实现采用分阶段评估：

```text
fast stage:
  games_fast
  depth_fast

full stage:
  games_full
  depth_full
  只评估 fast stage 排名前 full_fraction 的个体

validation stage:
  validation_games
  depth_full
  用独立 validation seed 验证当前最优个体
```

正式 v2 训练配置为：

```text
population = 12
generations = 20
games_fast = 3
games_full = 8
validation_games = 10
depth_fast = 1
depth_full = 2
full_fraction = 0.35
baseline_paired = true
bounds_mode = narrow
fix_lost_penalty = true
```

此外，代码中还实现了可选的 racing schedule，也就是 successive halving。它允许把评估过程拆成多轮，先用低成本配置评估全部个体，再逐步提高局数或深度，只让排名靠前的候选进入下一轮。

## 8. 并行化实现

为了提升训练和评估速度，本阶段对 evaluation 和 evolution 都加入了多进程并行能力。代码优先导入 `multiprocess`，若环境中不可用则回退到 Python 标准库 `multiprocessing`：

```python
try:
    import multiprocess as mp
except ImportError:
    import multiprocessing as mp
```

并行粒度分为两层：

- `eval/evaluate.py`：一个评估任务内部可以用 `--workers` 将多局游戏分发给多个 worker process。
- `eval/evolve_weights.py`：训练时可以用 `--workers` 并行评估多个候选权重。

在 Windows 环境下使用 `spawn` context 创建进程池，避免 fork 语义不可用的问题：

```python
context = mp.get_context("spawn")
pool = context.Pool(processes=worker_count)
```

为了减少进程间通信开销，`evaluate.py` 并不是把每一局都单独提交给 worker，而是先按 worker 数量把游戏编号分组，每个 worker 在本进程内连续运行一批游戏。`agent/heuristic.py` 中对应提供了：

```text
run_heuristic_game(...)
run_heuristic_games(...)
```

`run_heuristic_games` 会在同一个 worker 内复用同一个 `HeuristicAgent`，避免每局都重新构建 row table，从而降低批量评估开销。

## 9. 结果保存与恢复

训练输出目录中保存三类文件：

```text
evolution_results_v2/
  best_weights.json
  checkpoint.json
  generations.csv
```

其中：

- `best_weights.json` 保存当前验证阶段最优权重及其指标。
- `checkpoint.json` 保存种群、当前代数、配置、随机状态和最佳个体，可用于中断后恢复。
- `generations.csv` 保存每代所有候选的排序、fitness、分数、达成率和权重，便于报告分析。

如果训练过程中 terminal 被关闭，可以使用：

```powershell
python eval/evolve_weights.py --output evolution_results_v2 --resume evolution_results_v2/checkpoint.json
```

实际恢复时仍应补齐与原训练一致的参数，例如 `population`、`generations`、`games-fast`、`games-full`、`depth-fast`、`depth-full` 和 `workers`，以保持配置一致。

## 10. 最终评估接口

最终对照实验使用 `eval/evaluate.py` 完成。默认权重命令为：

```powershell
python eval/evaluate.py --games 300 --seed 20260608 --depth 2 --workers 20
```

进化权重命令为：

```powershell
python eval/evaluate.py --games 300 --seed 20260608 --depth 2 --workers 20 --weights 200000 459.779989788 800.537950377 41.1437746442 11.214963378
```

二者使用相同游戏数、相同起始 seed、相同搜索深度和相同 worker 数量。这样可以把差异主要归因于 heuristic 权重，而不是评估环境或随机性。

## 11. 与后续工作的关系

本阶段完成后，项目获得了一个可复用的自动调参流程。后续神经网络或强化学习部分可以复用以下成果：

- 更强的 heuristic baseline。
- 并行化评估框架。
- 多 seed 评估和结果统计逻辑。
- checkpoint 和实验记录格式。
- 通过 `run_heuristic_games` 批量生成游戏结果的接口。

因此 evolution 部分不仅提升了当前 Expectimax agent 的性能，也为后续大规模训练和评估提供了工程基础。
