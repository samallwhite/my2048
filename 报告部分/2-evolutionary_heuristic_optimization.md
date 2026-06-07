# 第二部分：基于进化计算的 Heuristic 参数优化方案

本文档用于记录项目第二部分的工作纲要：在第一部分已经完成的
Expectimax + heuristic 智能体基础上，使用进化计算方法自动优化启发式评估函数中的参数。

当前 `agent/heuristic.py` 已经将 2048 棋盘底层操作切换为 bitboard，并通过行查表方式加速评估函数计算；
`agent/config.py` 中的 `HeuristicWeights` 也已经为权重列表化、边界约束和后续遗传算法编码预留了接口。
因此第二部分的重点不是重新设计智能体结构，而是在现有接口上构建一个可复现、可比较、可检查点恢复的参数搜索流程。

## 1. 优化目标

本阶段目标是通过进化计算寻找一组更优的 heuristic 线性权重，使智能体在固定 Expectimax 深度下取得更稳定的游戏表现。

主要优化指标包括：

- 平均得分提高。
- 2048、4096、8192 等大方块达成率提高。
- 多随机种子下表现更稳定，避免只对少数局面过拟合。
- 保持当前 `HeuristicAgent` 的接口不变，使优化后的权重可以直接用于评估脚本、GUI 或后续学习模块。

需要注意的是，2048 的游戏结果具有较强随机性，单局得分不能代表参数优劣。
因此优化过程必须使用多局固定随机种子评估，并在最终验证阶段使用独立种子集合。

## 2. 当前 Heuristic 参数结构

当前启发式评估函数采用 nneonneo/2048-ai 风格的逐行评分方式。
棋盘被拆成 4 行和 4 列，共 8 个长度为 4 的行向量，每个行向量独立计算评分后求和。

当前行评分形式为：

```text
row_score =
    w_lost_penalty
  + w_empty * empty_count
  + w_merges * merge_clusters
  - w_monotonicity * mono_penalty
  - w_sum * sum_penalty
```

其中当前第一阶段建议优化的 5 个线性参数为：

```text
w_lost_penalty
w_empty
w_merges
w_monotonicity
w_sum
```

这 5 个参数已经由 `HeuristicWeights.to_list()` 和 `HeuristicWeights.from_list()` 暴露为列表形式，
适合直接作为进化算法中的染色体。

`mono_power` 和 `sum_power` 当前建议暂时固定。
原因是这两个参数会改变非线性惩罚的尺度，一开始同时优化会扩大搜索空间，增加评估成本。
当 5 个线性权重的优化流程稳定后，可以将它们扩展为第二阶段的 7 维染色体。

## 3. 染色体编码与搜索范围

本项目建议使用实数编码遗传算法，而不是二进制编码。

染色体定义如下：

```text
chromosome = [
    w_lost_penalty,
    w_empty,
    w_merges,
    w_monotonicity,
    w_sum,
]
```

每个基因的取值范围直接使用 `HeuristicWeights.bounds`：

```text
w_lost_penalty: 100000.0 ~ 500000.0
w_empty:          50.0 ~   1000.0
w_merges:        100.0 ~   2000.0
w_monotonicity:   10.0 ~    200.0
w_sum:             1.0 ~     50.0
```

初始化种群时应将当前默认权重作为一个保留个体加入初始种群：

```text
[200000.0, 270.0, 700.0, 47.0, 11.0]
```

这样可以保证进化搜索至少不会完全脱离现有可用基线，并且后续每代结果都能和默认权重进行直接比较。

## 4. 进化算法设计

建议第一版采用标准实数编码遗传算法，结构如下：

- 种群规模：16 到 24。
- 代数：20 到 40。
- 精英保留：每代保留前 2 个个体。
- 选择方法：锦标赛选择，锦标赛规模为 3。
- 交叉方法：BLX-alpha 或模拟二进制交叉。
- 变异方法：按参数范围进行高斯扰动。
- 边界处理：变异后将参数截断回合法范围。

其中 BLX-alpha 的实现较简单，适合作为第一版：

```text
child_gene = uniform(
    min(parent_a, parent_b) - alpha * diff,
    max(parent_a, parent_b) + alpha * diff
)
```

然后对每个 gene 执行边界截断。

变异可以按如下方式执行：

```text
gene = gene + normal(0, mutation_scale * parameter_range)
```

初始建议：

```text
alpha = 0.3
mutation_rate = 0.2
mutation_scale = 0.08
```

如果连续多代最优适应度无明显提升，可以适当提高变异尺度；
如果结果波动过大，则降低变异尺度并增加复评局数。

## 5. 适应度函数

适应度函数不能只使用单局得分，也不能只使用最高得分。
建议综合平均得分、大方块达成率和稳定性。

第一版适应度函数建议为：

```text
fitness =
    mean_score
  + 5000  * rate_2048
  + 12000 * rate_4096
  + 25000 * rate_8192
  - 0.05  * std_score
```

其中：

- `mean_score` 表示多局平均分。
- `rate_2048` 表示达到 2048 的比例。
- `rate_4096` 表示达到 4096 的比例。
- `rate_8192` 表示达到 8192 的比例。
- `std_score` 表示得分标准差，用于惩罚不稳定策略。

这个适应度函数的目的不是替代最终报告指标，而是为进化算法提供一个排序依据。
最终结果仍应使用完整评估报告展示平均分、最高分、最低分、标准差、平均最大方块和最大方块分布。

## 6. 分阶段评估策略

当前 `eval/evaluate.py` 已经支持通过命令行传入权重：

```text
python eval/evaluate.py --games 50 --seed 0 --depth 2 --weights ...
```

但实际测试中，`depth=2` 的单局运行时间已经较长。
因此进化算法不能对每个个体都进行大量完整评估，否则计算成本会非常高。

建议采用分阶段评估：

### 6.1 快速筛选

快速筛选阶段用于淘汰明显较差的个体。

建议配置：

```text
depth = 1
games = 2 ~ 4
```

该阶段只用于粗略判断，不作为最终结果依据。

### 6.2 正式选择

正式选择阶段用于评估每代候选中的较优个体。

建议配置：

```text
depth = 2
games = 3 ~ 5
```

每代可以先用快速筛选评估全部个体，然后只对排名靠前的 25% 到 40% 个体进行正式评估。

### 6.3 独立验证

独立验证阶段用于验证当前最优个体是否真的优于默认权重。

建议配置：

```text
depth = 2
games = 20 ~ 50
seed = validation_seed_start
```

验证种子不能与进化训练阶段完全相同，否则容易高估优化效果。

## 7. 随机种子控制

为了保证不同个体之间比较公平，每一代内部应使用相同的随机种子集合。
这可以减少 2048 随机生成方块带来的噪声。

例如第 `g` 代可以使用：

```text
seeds = [base_seed + g * 1000 + i for i in range(games)]
```

如果希望进一步降低噪声，也可以固定一组训练种子：

```text
train_seeds = [0, 1, 2, 3, 4]
validation_seeds = [10000, 10001, ..., 10049]
```

训练种子用于进化选择，验证种子只用于最终比较。

## 8. 建议文件结构

第一版实现可以保持轻量，直接新增一个进化优化入口：

```text
eval/evolve_weights.py
```

如果后续功能继续扩展，再拆分为独立目录：

```text
evolution/
    genetic.py
    fitness.py
    checkpoint.py
    results/
```

其中建议职责如下：

```text
genetic.py      # 种群初始化、选择、交叉、变异、精英保留
fitness.py      # 多局游戏评估、适应度计算
checkpoint.py   # 保存和恢复种群、最优权重、随机状态
results/        # 保存每代统计结果和最终权重
```

为了和当前项目结构保持一致，第一版优先实现 `eval/evolve_weights.py` 即可。
当第二部分逻辑稳定后，再考虑拆分为独立包。

## 9. 命令行接口设计

建议优化脚本支持以下命令行参数：

```text
python eval/evolve_weights.py \
  --population 20 \
  --generations 30 \
  --games-fast 3 \
  --games-full 5 \
  --depth-fast 1 \
  --depth-full 2 \
  --seed 0 \
  --output evolution_results
```

关键参数说明：

- `--population` 控制种群规模。
- `--generations` 控制进化代数。
- `--games-fast` 控制快速筛选局数。
- `--games-full` 控制正式评估局数。
- `--depth-fast` 控制快速筛选的 Expectimax 深度。
- `--depth-full` 控制正式评估的 Expectimax 深度。
- `--seed` 控制进化过程随机性。
- `--output` 控制结果保存目录。

每代建议输出：

```text
generation
best_fitness
mean_fitness
best_weights
mean_score
std_score
max_tile_distribution
elapsed_seconds
```

## 10. 检查点与结果保存

由于评估耗时较长，优化脚本必须支持检查点保存。

每代结束后至少保存：

```text
generation
population
fitness_values
best_weights
best_metrics
random_state
config
```

建议保存为 JSON 或 pickle。
其中 JSON 适合保存最终结果和报告数据，pickle 适合保存可恢复的完整运行状态。

最终应额外保存一个简洁的最优权重文件：

```text
best_weights.json
```

示例内容：

```json
{
  "weights": [200000.0, 270.0, 700.0, 47.0, 11.0],
  "fitness": 0.0,
  "metrics": {
    "mean_score": 0.0,
    "std_score": 0.0,
    "rate_2048": 0.0,
    "rate_4096": 0.0,
    "rate_8192": 0.0
  }
}
```

## 11. 实验对照

最终报告中至少需要包含以下对照实验：

### 11.1 默认权重基线

使用当前 `HeuristicWeights()` 默认参数，在独立验证种子上评估。

### 11.2 进化权重结果

使用 GA 得到的最优参数，在相同独立验证种子上评估。

### 11.3 消融实验

建议至少进行一次消融：

```text
固定 w_lost_penalty，只优化其余 4 个权重
```

因为 `w_lost_penalty` 是逐行常数项，在部分搜索比较场景中可能会被抵消。
通过消融可以判断它是否真的需要参与优化。

## 12. 风险点

### 12.1 评估成本过高

Expectimax 搜索本身计算量较大。
如果直接使用大种群、多局数和较高搜索深度，优化过程会非常慢。

应对方式：

- 使用分阶段评估。
- 先小规模 smoke test。
- 保存检查点。
- 优先优化 5 个线性权重。

### 12.2 随机噪声导致误判

2048 的随机生成方块会导致单局结果波动较大。

应对方式：

- 每个个体使用多局平均结果。
- 同一代个体使用相同随机种子集合。
- 最终使用独立验证种子。

### 12.3 对训练种子过拟合

如果长期使用固定少量种子，GA 可能只优化这些种子下的表现。

应对方式：

- 训练阶段可以轮换种子集合。
- 验证阶段必须使用未参与进化的种子。
- 报告中同时展示均值和标准差。

### 12.4 参数尺度差异较大

不同权重的数量级差异明显，例如 `w_lost_penalty` 是十万级，而 `w_sum` 是十级。

应对方式：

- 变异按参数范围的比例执行，而不是使用统一绝对扰动。
- 交叉后进行边界截断。
- 后续可以考虑归一化编码到 `[0, 1]` 区间。

## 13. 实施步骤

### Step 1: 封装权重评估函数

实现：

```text
evaluate_weights(weights, games, seed, depth) -> metrics
```

该函数内部创建：

```text
HeuristicWeights.from_list(weights)
HeuristicAgent(weights=..., max_depth=depth)
Game(seed=...)
```

并返回多局统计结果。

### Step 2: 实现 GA 基础流程

实现：

- 初始化种群。
- 适应度评估。
- 锦标赛选择。
- 实数交叉。
- 高斯变异。
- 精英保留。
- 每代统计输出。

### Step 3: 加入检查点

每代结束保存当前状态。
如果程序中断，可以从最近一代继续运行。

### Step 4: 小规模 smoke test

先运行：

```text
python eval/evolve_weights.py --population 4 --generations 2 --games-fast 1 --games-full 1 --depth-fast 1 --depth-full 1
```

确认流程、日志、结果保存和权重传递都正确。

### Step 5: 正式小规模搜索

建议初始正式配置：

```text
python eval/evolve_weights.py --population 16 --generations 20 --games-fast 3 --games-full 3 --depth-fast 1 --depth-full 2
```

### Step 6: 独立验证

将默认权重和最优进化权重放到相同验证种子上比较：

```text
python eval/evaluate.py --games 50 --seed 10000 --depth 2 --weights ...
```

最终以独立验证结果作为报告结论。

## 14. 预期交付

第二部分完成后应包含：

- 一个可运行的进化优化脚本。
- 一个保存最优权重的结果文件。
- 每代进化统计日志。
- 默认权重与进化权重的对照评估结果。
- 对优化效果、随机性和局限性的说明。

本阶段完成后，项目将具备自动调参能力。
后续监督学习或强化学习模块可以直接复用优化后的 heuristic 权重作为更强的搜索基线。



• 你现在可以直接运行我新增的脚本来开始优化。

  先建议跑一个小规模确认流程：

  python eval/evolve_weights.py --population 4 --generations 2 --games-fast 1 --games-full 1 --depth-fast 1 --depth-full 1 --elite-size 1 --tournament-size 2 --seed 0 --output
  evolution_results_test

  确认没问题后，跑正式一点的初始实验：

  python eval/evolve_weights.py --population 16 --generations 20 --games-fast 3 --games-full 3 --depth-fast 1 --depth-full 2 --seed 0 --output evolution_results

  运行过程中它会输出每代最优结果，并保存：

  evolution_results/generations.csv
  evolution_results/checkpoint.json
  evolution_results/best_weights.json

  如果中途中断，可以恢复：

  python eval/evolve_weights.py --population 16 --generations 20 --games-fast 3 --games-full 3 --depth-fast 1 --depth-full 2 --seed 0 --output evolution_results --resume
  evolution_results/checkpoint.json

  优化完成后，打开 evolution_results/best_weights.json，里面的 weights 就是最优参数。然后用现有评估脚本做独立验证，例如：

  python eval/evaluate.py --games 50 --seed 10000 --depth 2 --weights 437768.74061 770.056682793 899.086003579 59.1941825557 26.0524613471

  再和默认权重对照：

  python eval/evaluate.py --games 50 --seed 10000 --depth 2

  重点看平均得分、最大方块分布、2048/4096/8192 达成率。不要只看 GA 训练过程里的单次最优，因为 2048 随机性很强，最终结论要以独立 seed 的 50 局左右评估为准。
