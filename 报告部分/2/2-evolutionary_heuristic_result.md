# 第二部分：Evolutionary Heuristic Optimization 实验结果

本文档记录第二部分 evolution 训练和最终对照评估结果。训练产物来自 `evolution_results_v2/`，正式对照测试输出来自根目录 `evolution_comparament.txt`。

## 1. 实验设置

正式对照实验使用相同游戏数、相同随机种子、相同搜索深度和相同并行 worker 数量：

```text
games = 300
seed = 20260608
depth = 2
workers = 20
```

默认权重评估命令：

```powershell
python eval/evaluate.py --games 300 --seed 20260608 --depth 2 --workers 20
```

进化权重评估命令：

```powershell
python eval/evaluate.py --games 300 --seed 20260608 --depth 2 --workers 20 --weights 200000 459.779989788 800.537950377 41.1437746442 11.214963378
```

## 2. 进化训练配置

`evolution_results_v2/checkpoint.json` 中记录的训练配置如下：

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

该配置使用较小种群和分阶段评估控制训练成本。训练阶段启用 paired baseline，即同一组 seed 下比较候选权重相对默认权重的提升，从而降低随机方块生成带来的噪声。

## 3. 最优权重

`evolution_results_v2/best_weights.json` 中保存的最终最优权重为：

```text
w_lost_penalty = 200000.000000000
w_empty        =    459.779989788
w_merges       =    800.537950377
w_monotonicity =     41.143774644
w_sum          =     11.214963378
```

对应命令行传参形式为：

```text
200000 459.779989788 800.537950377 41.1437746442 11.214963378
```

与默认权重相比：

```text
default = [200000, 270, 700, 47, 11]
evolved = [200000, 459.78, 800.54, 41.14, 11.21]
```

变化主要体现在：

- `w_empty` 明显提高，策略更重视保留空格。
- `w_merges` 提高，策略更重视可合并结构。
- `w_monotonicity` 略降低，说明在当前搜索深度下，过强的单调性惩罚不一定最优。
- `w_sum` 基本保持接近默认值。
- `w_lost_penalty` 在正式训练中固定为默认值。

## 4. 训练阶段观察

`generations.csv` 中的较优记录显示，训练过程中多代候选在小样本 full stage 中取得了较高得分。例如：

| Generation | Stage | Fitness | Mean Score | Mean Max Tile | 2048 Rate | 4096 Rate | 8192 Rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | full | 36000.63 | 86883.0 | 4608.0 | 100.0% | 100.0% | 12.5% |
| 8 | full | 33484.20 | 86703.5 | 4864.0 | 100.0% | 87.5% | 25.0% |
| 18 | full | 33368.33 | 100932.5 | 5632.0 | 100.0% | 100.0% | 37.5% |
| 10 | validation | 31617.93 | 79371.6 | 5222.4 | 90.0% | 80.0% | 40.0% |

需要注意的是，训练阶段每个候选的评估局数较少，因此这些数据不能直接作为最终结论。最终结论应以 300 局独立对照测试为准。

## 5. 300 局正式对照结果

正式测试结果如下：

| 指标 | 默认权重 | Evolution 权重 | 变化 |
|---|---:|---:|---:|
| Games | 300 | 300 | - |
| Workers | 20 | 20 | - |
| Total Time | 14m23s | 18m19s | +3m56s |
| Average Time | 2.9s/game | 3.7s/game | +0.8s/game |
| Mean Score | 66813.9 | 70851.0 | +4037.1 |
| Max Score | 175580 | 173740 | -1840 |
| Min Score | 7168 | 6448 | -720 |
| Score Std | 32718.5 | 33834.3 | +1115.8 |
| Mean Steps | 3065.9 | 3227.4 | +161.5 |
| Mean Max Tile | 3991.9 | 4152.3 | +160.4 |
| 2048 Rate | 94.3% | 95.3% | +1.0 pct |

平均得分提升为：

```text
(70851.0 - 66813.9) / 66813.9 = 6.04%
```

平均最大方块提升为：

```text
(4152.3 - 3991.9) / 3991.9 = 4.02%
```

## 6. 最大方块分布

最大方块分布对比如下：

| Max Tile | 默认权重 | Evolution 权重 | 变化 |
|---:|---:|---:|---:|
| 512 | 3 games (1.0%) | 3 games (1.0%) | 0 |
| 1024 | 14 games (4.7%) | 11 games (3.7%) | -3 games |
| 2048 | 77 games (25.7%) | 68 games (22.7%) | -9 games |
| 4096 | 162 games (54.0%) | 169 games (56.3%) | +7 games |
| 8192 | 44 games (14.7%) | 49 games (16.3%) | +5 games |

可以看到，evolution 权重减少了停留在 1024 和 2048 的局数，并增加了达到 4096 和 8192 的局数。这说明提升并不只是平均分波动造成的，而是最大方块分布整体向更高方块移动。

## 7. 结果分析

从 300 局对照结果看，evolution 权重相对默认权重有明确正收益：

- 平均分从 `66813.9` 提升到 `70851.0`，提升约 `6.0%`。
- 2048 达成率从 `94.3%` 提升到 `95.3%`。
- 4096 比例从 `54.0%` 提升到 `56.3%`。
- 8192 比例从 `14.7%` 提升到 `16.3%`。
- 平均最大方块从 `3991.9` 提升到 `4152.3`。

这说明 evolutionary optimization 成功找到了比默认 heuristic 更强的一组权重，并且提升不仅体现在得分均值上，也体现在高阶方块达成率上。

## 8. 副作用与局限

这组权重也带来了一些副作用：

- 分数标准差从 `32718.5` 增加到 `33834.3`，说明结果波动略有增大。
- 最低分从 `7168` 降到 `6448`，最差局表现略差。
- 最高分从 `175580` 降到 `173740`，但最高分受随机性影响较大，不作为主要判断标准。
- evolution 权重平均耗时更高，正式评估中从 `2.9s/game` 增加到 `3.7s/game`。

因此结论应表述为：在当前测试配置下，evolution 权重提升了总体表现和高阶方块达成率，但稳定性和运行时间略有代价。

## 9. 实验结论

本阶段可以得出以下结论：

1. Evolutionary optimization 流程成功执行，并产出了可复用的 heuristic 权重。
2. 优化后的权重可以被 `eval/evaluate.py` 正常加载，并在 20 worker 并行评估中稳定运行。
3. 在 `300 games / seed 20260608 / depth 2 / workers 20` 的正式对照实验中，evolution 权重将平均分提升约 `6.0%`。
4. 高阶方块达成率同步提升，2048、4096、8192 指标均优于默认权重。
5. 该结果表明 evolution 方法确实提升了当前 heuristic agent 的成功率和整体性能。

更严谨地说，该实验已经证明 evolution 在当前 held-out seed 配置下有效。若要进一步证明泛化能力，后续可以使用更多独立 seed 进行重复评估。
