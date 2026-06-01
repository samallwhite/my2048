# 2048 AI 项目 — 工作记录

## 项目概述

本项目是《人工智能概论》课程实践，分 5 个部分实现 AI 玩 2048 游戏的完整技术栈。当前已完成**第 1 部分：2048 游戏系统**。

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
第2部分（启发式搜索）────→ 第3部分（进化计算优化参数）
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

### 后续部分预期文件

```
agent/
├── heuristic.py   # 第2部分：expectimax + 启发式评价函数
├── genetic.py     # 第3部分：遗传算法优化权重
├── supervised.py  # 第4部分：CNN/MLP 监督学习
└── rl.py          # 第5部分：DQN 强化学习
train/             # 训练脚本
eval/              # 评估脚本
```

---

## 环境信息

| 项目 | 版本 |
|------|------|
| Python | 3.12.7 |
| numpy | 1.26.4 |
| pytest | 7.4.4 |
| 平台 | Windows 11 |
