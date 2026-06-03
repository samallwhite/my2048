# 2048 AI 项目 — 工作记录

## 项目概述

本项目是《人工智能概论》课程实践，分 5 个部分实现 AI 玩 2048 游戏的完整技术栈。当前已完成**第 1 部分：2048 游戏系统**和**第 2 部分：启发式搜索决策策略**。

## 已完成工作

### 第 1 部分：2048 游戏引擎

实现了完整的 2048 游戏核心逻辑和双模式交互。

#### 项目结构

```
2048/
├── game/                  # 游戏引擎（核心）
│   ├── __init__.py
│   ├── board.py           # Board 类：棋盘逻辑、移动合并、游戏结束判定
│   └── game.py            # Game 类：Gym 风格环境，管理随机生成和游戏循环
├── agent/                 # 智能体（后续部分使用）
│   ├── __init__.py
│   └── base.py            # Agent 抽象基类，定义 get_action / reset 接口
├── tests/
│   ├── __init__.py
│   └── test_board.py      # 32 个单元测试（全通过）
├── main.py                # CLI 交互入口（程序/训练用）
├── gui.py                 # GUI 交互入口（人类游玩用）
├── requirements.txt       # numpy>=1.24, pytest>=7.0
├── project_demand.md      # 课程原始要求
└── outline.md             # 初始实现方案大纲
```

#### 架构设计

采用 **Board + Game 双层分离**：

- **`Board`**：纯逻辑层，无随机性。move 操作返回新 Board 实例，不变异原对象。适合 expectimax 树搜索。
- **`Game`**：Gym 风格环境层。封装 `reset()` / `step(action)` / `clone()`，管理方块随机生成和游戏循环。

#### 核心算法：`slide_row_left`

整个引擎简化为一个 row-level 函数。四个方向通过行列变换组合实现：

| 方向 | 变换方式 |
|------|---------|
| Left  | `slide_row_left(row)` |
| Right | `reverse(slide_row_left(reverse(row)))` |
| Up    | `slide_row_left(col)` |
| Down  | `reverse(slide_row_left(reverse(col)))` |

合并规则：`i += 2` 跳过已合并项，正确实现"每步最多合并一次"。

#### Board API 摘要

| 方法 | 说明 |
|------|------|
| `slide_row_left(row)` | 静态方法，核心合并逻辑。返回 `(new_row, score, changed)` |
| `execute_move(direction)` | 返回新 Board，方向 0=up 1=down 2=left 3=right |
| `get_empty_cells()` | 返回 `[(r, c), ...]` 空格列表 |
| `get_valid_moves()` | 返回当前可移动方向列表 |
| `is_game_over()` | 满盘且无合法移动时为 True |
| `copy()` | 深拷贝，用于搜索树 |
| `get_state()` | 返回 log2 编码的 4×4 float32 数组（空=0，2→1，4→2，...）用于 NN 输入 |
| `max_tile` / `score` | 属性 |

#### Game API 摘要（Gym 风格）

| 方法 | 说明 |
|------|------|
| `reset()` | 清空棋盘，生成 2 个初始方块，返回 `state` |
| `step(action)` | 执行动作，返回 `(state, reward, done, info)` |
| `clone()` | 深拷贝（含 RNG 状态），expectimax 关键接口 |
| `render()` | 控制台输出棋盘 |

#### 测试覆盖

`tests/test_board.py` — 32 个测试用例全部通过：

- `slide_row_left`：15 组参数化（空行、滑动、单合并、双合并、四格配对、不合并）
- 四方向移动测试（各方向合并正确性）
- 不可变性测试（原 Board 不被修改）
- 积分累加测试（单行 + 多行）
- `is_game_over` 测试（空盘/满盘有合并/满盘无合并）
- `get_valid_moves` 测试
- `copy` 深拷贝验证
- `get_state` 编码正确性
- 非法方向参数异常测试

#### 验证结果

- 生成概率：10000 局中 4 出现率 10.2%（目标 10%）
- 100 局随机移动无崩溃

---

## 两种交互模式

### CLI 模式（`main.py`）— 程序交互 / AI 训练

```bash
python main.py          # 默认随机种子
python main.py --seed 42  # 固定种子
```

- WASD + 回车 输入
- 返回结构化数据 `(state, reward, done, info)`
- **后续第 2-5 部分所有 AI 训练和评估统一使用此模式**

### GUI 模式（`gui.py`）— 人类游玩

```bash
python gui.py
```

- 方向键 / WASD 控制
- 经典 2048 配色，支持 2 → 8192+ 共 12 级颜色
- 自动缩放字体，分数实时显示，Game Over 遮罩
- New Game 按钮
- 底层复用同一套 `game/` 引擎，零修改

---

## 后续工作规划

### ⚠️ 重要约定

**所有后续学习、训练、评估、数据采集均基于 CLI 模式（`Game` 类），GUI 仅面向人类用户。**

各部分的 Agent 实现均继承 `agent/base.py` 的 `Agent` 抽象类，通过 `Game.step()` 与引擎交互。

### 依赖关系

```
第1部分（游戏引擎）✅ 已完成
    ↓
第2部分（启发式搜索）✅ 已完成 → 第3部分（进化计算优化参数）
    ↓
    生成训练数据
    ↓
第4部分（监督学习 NN）
    ↓
第5部分（RL 优化）
```

### 评价指标

| 指标 | 优先级 | 说明 |
|------|--------|------|
| 平均得分 | 主要 | 每方法跑 100 局取平均 |
| 最大合成方块 | 辅助 | 是否稳定达到 1024/2048 |
| 平均生存步数 | 辅助 | 策略维持游戏长度的能力 |

---

## 第 2 部分：启发式搜索决策策略

### 阶段一：游戏本体审查

在实现 AI 策略前，首先对第 1 部分的游戏引擎进行了全面审查，确认以下四项均严格符合要求：

| 审查项 | 要求 | 实现位置 | 结果 |
|--------|------|---------|------|
| 棋盘尺寸 | 4×4 | `board.py:13` — `np.zeros((4, 4))` | ✅ |
| 生成概率 | 2(90%) / 4(10%) | `game.py:54` — `random() < 0.1` | ✅ |
| 结束判定 | 无空格且无合法移动 | `board.py:99-102` | ✅ |
| 模拟克隆 | 不可变移动 + 深拷贝 | `board.py:48` + `game.py:56` | ✅ |

**结论**：游戏引擎的不可变设计（`execute_move` 返回新 Board）天然支持 Expectimax 虚拟模拟，无需修改任何游戏本体代码。

### 阶段二：Expectimax 搜索 + 启发式评价

#### 新增文件

```
agent/
├── __init__.py       # 更新：导出 HeuristicAgent, HeuristicWeights
├── config.py          # 新增：HeuristicWeights 权重配置（★ 为第 3 部分 GA 预留）
└── heuristic.py       # 新增：Expectimax 搜索 + 四特征启发式评价
eval/
├── __init__.py        # 新增
└── evaluate.py        # 新增：多局评估脚本（平均分/最大方块/生存步数）
```

#### 搜索树结构

```
Max（根节点：选动作）
  ├─ UP    → Chance（枚举空位 × {2(90%), 4(10%)}）→ Max → Chance → 评价
  ├─ DOWN  → Chance → Max → Chance → 评价
  ├─ LEFT  → Chance → Max → Chance → 评价
  └─ RIGHT → Chance → Max → Chance → 评价
```

- **Max 节点**：遍历 4 个合法移动方向，取最高期望得分
- **Chance 节点**：枚举（或采样）所有空位 × {2, 4}，概率加权求数学期望

#### 动态深度控制

| 空格数 E | 搜索深度 | 单步评价次数 | 说明 |
|----------|---------|-------------|------|
| E ≥ 5 | depth 2 | ~576 次 | 早期游戏，分支多，浅搜 |
| E < 5 | depth 3 | ~3K-12K 次 | 终局阶段，分支少，深搜 |

#### Chance 节点采样策略

当空位数超过 `sample_threshold`（默认 4）时，随机采样 `sample_size`（默认 3）个空位计算期望值，避免组合爆炸。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_threshold` | 4 | 空位数超过此值时触发采样 |
| `sample_size` | 3 | 每层采样空位数（×2 值 = 6 分支） |

#### 启发式评价函数

评价函数 = w_empty × 空格数量 + w_monotonicity × 单调性 + w_smoothness × 平滑度 + w_max_corner × 最大数靠角

| 特征 | 权重参数 | 默认值 | 计算方式 |
|------|---------|--------|---------|
| 空格数量 | `w_empty` | 12.0 | 直接计数空位 |
| 单调性 | `w_monotonicity` | 1.5 | 每行/列取 max(递增log差, 递减log差)，求和 |
| 平滑度 | `w_smoothness` | 1.0 | -∑ 相邻非空格 log2 差 |
| 最大数靠角 | `w_max_corner` | 8.0 | 若 max 在四角之一则返回 log2(max)，否则 0 |

所有特征使用 **log2 缩放**（空格除外），确保不同数量级的差异可比（2→4 和 1024→2048 的 log2 差均为 1）。

#### ★ 为第 3 部分遗传算法预留的接口

`HeuristicWeights` 是一个 `@dataclass`，提供：

```python
# 权重 ⇄ 染色体 互转（GA 直接使用）
weights.to_list()                        # → [12.0, 1.5, 1.0, 8.0]
HeuristicWeights.from_list([15, 2, 1.5, 10])  # → HeuristicWeights 实例

# 搜索范围（GA 初始化种群）
weights.bounds  # → [(0, 50), (0, 10), (0, 10), (0, 50)]
```

权重不硬编码在评价函数中，以 `HeuristicWeights` 实例注入 `HeuristicAgent`，GA 可直接替换权重对象。

#### 运行方式

```bash
# 默认参数评估（100 局，depth=2）
python eval/evaluate.py

# 自定义局数和种子
python eval/evaluate.py --games 50 --seed 42 --verbose

# 调整搜索深度和采样
python eval/evaluate.py --depth 3 --sample-size 4 --sample-threshold 5

# 使用自定义权重（为 GA 调优准备）
python eval/evaluate.py --weights 15.0 2.0 1.5 10.0
```

#### 评估结果（默认参数，20 局，seed=42）

| 指标 | 数值 |
|------|------|
| 平均得分 | **5,771** |
| 最高得分 | 10,388 |
| 最低得分 | 2,100 |
| 得分标准差 | 2,445 |
| 平均生存步数 | 410 |
| 平均最大方块 | 371 |

| 最大方块分布 | 局数 | 占比 |
|-------------|------|------|
| 128 | 2 | 10% |
| 256 | 8 | 40% |
| 512 | 10 | 50% |

#### 性能特征

| 指标 | 数值 |
|------|------|
| 单步搜索耗时 | ~0.12s（depth=2, E≈10） |
| 单局总耗时 | ~49s（~410 步） |
| 100 局评估预估 | ~80 分钟 |

可通过减小 `--sample-size` 或降低 `--depth` 换取速度（会牺牲一定得分）。

#### HeuristicAgent API

| 方法 | 说明 |
|------|------|
| `get_action(board)` | 返回最佳动作 0/1/2/3 |
| `evaluate_board(board)` | 返回当前棋盘的启发式评分 |
| `get_feature_values(board)` | 返回四个特征的原始值字典（调试用） |
| `reset()` | 重置内部状态 |

### GUI AI 可视化

`gui.py` 新增 **AI 自动模式**，可在图形界面中实时观察智能体的决策过程：

- **AI 开关按钮**：点击启动/停止 AI 自动游戏
- **速度滑条**：50ms（快）⇄ 600ms（慢）可拖拽调节
- **信息栏**：实时显示步数、方向、四个启发式特征值
- AI 运行时自动禁用键盘输入，停止后可手动接管

```bash
python gui.py
# 点击 "▶ AI 自动" 按钮即可观看
```

### 评估脚本增强

`eval/evaluate.py` 新增**实时进度**和**断点续训**功能：

- **实时进度条**：显示完成进度、最近N局均分、全局均分、最佳分、方块分布、耗时、ETA
- **检查点自动保存**：每完成一局自动写入 `eval/checkpoints/`，Ctrl+C 中断后可从断点恢复
- **断点续训**：`--resume` 自动选择最近检查点，或指定文件路径
- **检查点管理**：`--list-checkpoints` 列出所有可用检查点
- **自动清理**：评估正常结束后自动删除检查点文件

```bash
python eval/evaluate.py --games 200           # 200局评估，自动保存检查点
python eval/evaluate.py --resume              # 从最近检查点继续
python eval/evaluate.py --list-checkpoints    # 查看可用检查点
python eval/evaluate.py --no-checkpoint       # 禁用检查点
```

---

## 环境信息

| 项目 | 版本 |
|------|------|
| Python | 3.12.7 |
| numpy | 1.26.4 |
| pytest | 7.4.4 |
| 平台 | Windows 11 |
