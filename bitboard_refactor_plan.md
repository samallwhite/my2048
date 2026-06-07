# Bitboard 全局重构计划

本文档用于记录下一阶段的大规模改造方案：将 2048 项目的棋盘底层实现统一切换为位运算 bitboard，同时保留清晰的上层接口，方便 heuristic、GA、监督学习和强化学习共同复用。

## 1. 重构目标

当前最终版 heuristic 已经在 `agent/heuristic.py` 内部使用 64-bit rank bitboard 做快速搜索。下一步目标是把这套能力提升到 `game` 层，使整个项目长期使用统一的位运算棋盘。

目标不是把位运算逻辑写进 `main.py`。`main.py` 应继续只作为程序入口。真正的结构应为：

```text
game/bitboard.py   # 底层位运算、查表、移动、编码解码
game/board.py      # Board 对外接口，内部使用 bitboard 存储
game/game.py       # 游戏流程环境，调用 Board
agent/heuristic.py # 搜索和评分，复用 game.bitboard
```

## 2. 为什么仍然需要 board.py

即使底层全部改成 bitboard，仍然需要 `Board` 这个抽象。

原因：

- GUI、评估脚本、监督学习数据生成、RL 环境都需要统一的棋盘接口。
- 直接在各处传裸 `int` 会让状态转换、计分、最大方块、空格查询等逻辑散落在项目中。
- `Board` 可以隐藏底层实现，让上层代码继续使用 `board.execute_move()`、`board.get_state()`、`board.max_tile` 等稳定接口。

因此，应该保留 `game/board.py` 文件，但把它从 numpy 存储后端改成 bitboard 存储后端。

## 3. 推荐最终文件职责

### game/bitboard.py

只负责底层高性能棋盘操作，不依赖 agent，也不包含 heuristic 评分函数。

建议包含：

```python
ROW_MASK = 0xffff

board_to_bits(grid) -> int
bits_to_grid(bits) -> np.ndarray
execute_move(bits, direction) -> tuple[int, int, bool]
get_empty_shifts(bits) -> list[int]
spawn_tile(bits, shift, rank) -> int
count_distinct_tiles(bits) -> int
max_tile(bits) -> int
get_state(bits) -> np.ndarray
```

其中 `execute_move()` 必须返回：

```python
new_bits, score_gained, changed
```

因为游戏环境需要正确累计合并得分，而 heuristic 搜索也需要判断移动是否合法。

### game/board.py

保留 `Board` 类和现有公开接口，但内部字段从 `grid` 改成 `bits`。

建议接口保持：

```python
class Board:
    bits: int
    score: int

    @property
    def grid(self) -> np.ndarray
    def execute_move(self, direction: int) -> "Board"
    def get_empty_cells(self) -> list[tuple[int, int]]
    def get_valid_moves(self) -> list[int]
    def is_game_over(self) -> bool
    def get_state(self) -> np.ndarray
    def copy(self) -> "Board"
    @property
    def max_tile(self) -> int
```

`grid` 可以保留为属性，但应由 `bits` 临时解码得到，不再作为真实存储。

### game/game.py

理论上尽量少改。它继续管理：

- 初始化游戏
- 随机生成初始方块
- 每步执行动作
- 有效移动后生成新方块
- 返回 reward、done、info

如果 `Board` 接口保持稳定，`game.py` 改动应很小。

### agent/heuristic.py

改造后不再维护自己的 bitboard 移动工具函数。

应删除或迁移这些内部函数：

```python
_board_to_rank_int
_reverse_row
_pack_line
_slide_rank_line_left
_build_move_tables
_transpose
_execute_rows
_execute_move
```

然后直接复用：

```python
from game import bitboard
```

`get_action()` 中直接使用：

```python
rank_board = board.bits
new_board = bitboard.execute_move(rank_board, direction)[0]
```

heuristic 评分表仍留在 `agent/heuristic.py`，因为它和启发式权重绑定，不属于通用游戏规则。

## 4. 迁移步骤

### Step 1: 新建 game/bitboard.py

先把位运算工具独立出来，完整实现：

- 64-bit rank 编码
- 16-bit 行编码
- 行向左/向右移动表
- 行移动得分表
- 棋盘转置
- 上下左右移动
- 空格枚举
- 新方块生成
- bitboard 到 numpy grid 的解码

注意：当前 heuristic 内部移动只关心新棋盘，但全局游戏环境必须关心得分，所以移动表需要同时保存移动后的行和本行合并得分。

### Step 2: 写 tests/test_bitboard.py

在修改 `Board` 之前，先用现有 numpy `Board` 作为对照，验证 bitboard 行为一致。

至少覆盖：

- 编码和解码一致性
- 上、下、左、右移动结果一致
- 每次移动获得的合并分数一致
- 无效移动时 `changed == False`
- 空格数量一致
- 最大方块一致
- 不同非空 rank 种类统计一致

### Step 3: 重写 game/board.py 的内部实现

保留文件和类名，保留上层接口，但内部改为：

```python
class Board:
    def __init__(self, grid: np.ndarray | None = None, score: int = 0, bits: int | None = None):
        if bits is not None:
            self.bits = bits
        else:
            self.bits = bitboard.board_to_bits(grid or empty_grid)
        self.score = int(score)
```

这样可以兼容旧代码中用 `Board(grid=...)` 构造测试棋盘的方式，也支持未来直接用 `Board(bits=...)`。

### Step 4: 修改 agent/heuristic.py

将搜索内部的棋盘移动统一改为调用 `game.bitboard`。

保留：

- Expectimax
- Chance 节点
- Max 节点
- heuristic row score table
- 置换表
- `evaluate_board()`
- `get_feature_values()`

删除重复的移动表、转置、行滑动等底层实现。

### Step 5: 回归测试

先运行：

```text
python -m compileall game agent eval tests
pytest
```

再运行小规模 smoke test：

```text
python eval/evaluate.py --games 1 --seed 0 --depth 2
```

确认无误后，再恢复正式测评。

## 5. 风险点

### 计分

这是最大风险。原来的 heuristic 内部移动不关心分数，但正式游戏环境必须正确累计每次合并得分。

因此 `bitboard.execute_move()` 必须同时返回 `score_gained`。

### grid 兼容

GUI、测试和数据生成可能仍访问 `board.grid`。

建议保留 `Board.grid` 属性，但每次从 `bits` 解码生成 numpy 数组。这样接口不破坏，底层仍是 bitboard。

### 循环依赖

`game.bitboard` 不应 import `agent`。

允许：

```text
game.bitboard -> game.board 中的方向常量
agent.heuristic -> game.bitboard
```

但不要让 `game` 层依赖 `agent` 层。

### main.py

不要把位运算逻辑写进 `main.py`。

`main.py` 只负责入口和流程组织，不应承担棋盘实现。

## 6. 预期收益

完成后：

- 游戏环境、heuristic、GA、监督学习、RL 都可以共享同一套高速棋盘操作。
- heuristic 不再重复维护 bitboard 工具函数。
- 后续大量模拟时，不需要反复从 numpy 棋盘转换到 bitboard。
- 项目结构更清楚：`bitboard.py` 管性能，`board.py` 管接口，`agent` 管策略。

