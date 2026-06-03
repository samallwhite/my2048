"""2048 GUI — Tkinter 图形界面。
支持人类游玩和 AI 自动模式两种交互方式。
"""

import tkinter as tk
from game.game import Game
from game.board import UP, DOWN, LEFT, RIGHT

# ── constants ────────────────────────────────────────────────────────────────

GRID_SIZE = 4
CELL_SIZE = 120
PADDING = 12
FONT = ("Helvetica", 36, "bold")
SMALL_FONT = ("Helvetica", 28, "bold")
TINY_FONT = ("Helvetica", 20, "bold")

BG_COLOR = "#bbada0"
EMPTY_COLOR = "#cdc1b4"

TILE_COLORS: dict[int, tuple[str, str]] = {
    0:     ("#cdc1b4", "#cdc1b4"),
    2:     ("#eee4da", "#776e65"),
    4:     ("#ede0c8", "#776e65"),
    8:     ("#f2b179", "#f9f6f2"),
    16:    ("#f59563", "#f9f6f2"),
    32:    ("#f67c5f", "#f9f6f2"),
    64:    ("#f65e3b", "#f9f6f2"),
    128:   ("#edcf72", "#f9f6f2"),
    256:   ("#edcc61", "#f9f6f2"),
    512:   ("#edc850", "#f9f6f2"),
    1024:  ("#edc53f", "#f9f6f2"),
    2048:  ("#edc22e", "#f9f6f2"),
    4096:  ("#3c3a32", "#f9f6f2"),
    8192:  ("#3c3a32", "#f9f6f2"),
}

_DIR_NAMES = {UP: "↑ 上", DOWN: "↓ 下", LEFT: "← 左", RIGHT: "→ 右"}


def tile_bg_fg(value: int) -> tuple[str, str]:
    return TILE_COLORS.get(value, ("#3c3a32", "#f9f6f2"))


def font_for(value: int):
    if value < 100:
        return FONT
    if value < 1000:
        return SMALL_FONT
    return TINY_FONT


# ── app ──────────────────────────────────────────────────────────────────────

class App2048:
    def __init__(self):
        self.game = Game()
        self.game.reset()

        # AI 相关状态
        self.ai_enabled = False
        self.ai_job_id = None       # root.after() 返回的 job ID
        self.ai_speed = 200         # AI 每步间隔（毫秒），越小越快
        self.ai_agent = None        # 延迟初始化，避免导入耗时影响启动
        self.ai_step_count = 0      # AI 已执行步数

        self.root = tk.Tk()
        self.root.title("2048 — AI 可视化")
        self.root.resizable(False, False)
        self.root.configure(bg="#faf8ef")

        # header
        header = tk.Frame(self.root, bg="#faf8ef")
        header.pack(padx=20, pady=(20, 0), fill="x")

        tk.Label(
            header, text="2048", font=("Helvetica", 42, "bold"),
            fg="#776e65", bg="#faf8ef"
        ).pack(side="left")

        self.score_label = tk.Label(
            header, text="Score: 0", font=("Helvetica", 18, "bold"),
            fg="#776e65", bg="#faf8ef"
        )
        self.score_label.pack(side="right", pady=10)

        # 状态信息栏
        self.info_label = tk.Label(
            header, text="", font=("Helvetica", 11),
            fg="#9e948a", bg="#faf8ef"
        )
        self.info_label.pack(side="right", padx=(0, 15))

        # canvas
        canvas_size = CELL_SIZE * GRID_SIZE + PADDING * (GRID_SIZE + 1)
        self.canvas = tk.Canvas(
            self.root, width=canvas_size, height=canvas_size,
            bg=BG_COLOR, highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=10)
        self.canvas.focus_set()

        # key bindings（人类模式）
        self.canvas.bind("<Key>", self._on_key)

        # ── 控制按钮区 ──────────────────────────────────────────────────
        ctrl_frame = tk.Frame(self.root, bg="#faf8ef")
        ctrl_frame.pack(padx=20, pady=(0, 5), fill="x")

        # New Game 按钮
        tk.Button(
            ctrl_frame, text="新游戏", font=("Helvetica", 13, "bold"),
            fg="#776e65", bg="#cdc1b4", activebackground="#bbada0",
            relief="flat", padx=16, pady=6, command=self._new_game
        ).pack(side="left", padx=(0, 10))

        # AI 开关按钮
        self.ai_btn = tk.Button(
            ctrl_frame, text="▶ AI 自动", font=("Helvetica", 13, "bold"),
            fg="#f9f6f2", bg="#8f7a66", activebackground="#9f8b76",
            relief="flat", padx=16, pady=6, command=self._toggle_ai
        )
        self.ai_btn.pack(side="left", padx=(0, 10))

        # 速度标签
        tk.Label(
            ctrl_frame, text="速度:", font=("Helvetica", 11),
            fg="#776e65", bg="#faf8ef"
        ).pack(side="left", padx=(0, 4))

        # 速度滑条（慢 ← → 快）
        self.speed_scale = tk.Scale(
            ctrl_frame, from_=600, to=50, orient="horizontal",
            length=140, showvalue=False,
            bg="#faf8ef", fg="#776e65", troughcolor="#cdc1b4",
            highlightthickness=0, borderwidth=0,
            command=self._on_speed_change
        )
        self.speed_scale.set(self.ai_speed)
        self.speed_scale.pack(side="left")

        tk.Label(
            ctrl_frame, text="慢", font=("Helvetica", 9),
            fg="#9e948a", bg="#faf8ef"
        ).pack(side="left")
        tk.Label(
            ctrl_frame, text="快", font=("Helvetica", 9),
            fg="#9e948a", bg="#faf8ef"
        ).pack(side="left")

        self._draw()

    # ── 人类输入 ─────────────────────────────────────────────────────────────

    def _on_key(self, event: tk.Event) -> None:
        # AI 模式下禁用键盘输入
        if self.ai_enabled:
            return

        key_map = {
            "Up": UP, "Down": DOWN, "Left": LEFT, "Right": RIGHT,
            "w": UP, "s": DOWN, "a": LEFT, "d": RIGHT,
        }
        direction = key_map.get(event.keysym)
        if direction is None:
            return

        _, reward, _, info = self.game.step(direction)
        if not info.get("invalid"):
            self._draw()

        if self.game.board.is_game_over():
            self._draw_game_over()

    # ── AI 模式 ──────────────────────────────────────────────────────────────

    def _init_ai_agent(self):
        """延迟初始化 AI 智能体（避免启动时卡顿）。"""
        if self.ai_agent is None:
            from agent.heuristic import HeuristicAgent
            from agent.config import HeuristicWeights
            self.ai_agent = HeuristicAgent(
                weights=HeuristicWeights(),
                max_depth=2,
            )

    def _toggle_ai(self) -> None:
        """切换 AI 自动模式。"""
        if self.ai_enabled:
            self._stop_ai()
        else:
            self._start_ai()

    def _start_ai(self) -> None:
        """启动 AI 自动模式。"""
        if self.game.board.is_game_over():
            self._new_game()

        self._init_ai_agent()
        self.ai_enabled = True
        self.ai_step_count = 0
        self.ai_btn.config(text="⏸ 停止 AI", bg="#e84c3d",
                           activebackground="#f06050")
        self.info_label.config(text="AI 运行中...")
        self._schedule_ai_step()

    def _stop_ai(self) -> None:
        """停止 AI 自动模式。"""
        self.ai_enabled = False
        if self.ai_job_id is not None:
            self.root.after_cancel(self.ai_job_id)
            self.ai_job_id = None
        self.ai_btn.config(text="▶ AI 自动", bg="#8f7a66",
                           activebackground="#9f8b76")
        self.info_label.config(text=f"AI 已停止（共 {self.ai_step_count} 步）")

    def _schedule_ai_step(self) -> None:
        """调度下一次 AI 移动。"""
        if not self.ai_enabled:
            return
        self.ai_job_id = self.root.after(self.ai_speed, self._ai_step)

    def _ai_step(self) -> None:
        """执行一次 AI 决策并步进游戏。"""
        if not self.ai_enabled:
            return

        # 检查游戏是否结束
        if self.game.board.is_game_over():
            self._stop_ai()
            self._draw_game_over()
            self.info_label.config(
                text=f"游戏结束 | 得分: {self.game.score} | "
                     f"最大: {self.game.max_tile} | 步数: {self.ai_step_count}"
            )
            return

        # AI 决策
        action = self.ai_agent.get_action(self.game.board)
        _, reward, _, info = self.game.step(action)

        if not info.get("invalid"):
            self.ai_step_count += 1
            self._draw()

            # 更新信息栏
            feats = self.ai_agent.get_feature_values(self.game.board)
            self.info_label.config(
                text=f"#{self.ai_step_count} {_DIR_NAMES[action]} | "
                     f"评分:{feats['total_score']:.0f} "
                     f"空:{feats['empty_cells']:.0f} "
                     f"簇:{feats['merge_clusters']:.0f} "
                     f"单罚:{feats['mono_penalty']:.1f} "
                     f"和罚:{feats['sum_penalty']:.1f}"
            )
        else:
            self.info_label.config(
                text=f"#{self.ai_step_count} ⚠ 无效移动，跳过"
            )

        # 继续下一轮
        self._schedule_ai_step()

    def _on_speed_change(self, value: str) -> None:
        """速度滑条回调。"""
        self.ai_speed = int(value)
        # 如果 AI 正在运行，不打断，下一轮自动生效

    # ── 绘制 ─────────────────────────────────────────────────────────────────

    def _draw(self):
        self.canvas.delete("all")

        grid = self.game.board.grid
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x = PADDING + c * (CELL_SIZE + PADDING)
                y = PADDING + r * (CELL_SIZE + PADDING)
                val = int(grid[r, c])

                bg, fg = tile_bg_fg(val)
                self.canvas.create_rectangle(
                    x, y, x + CELL_SIZE, y + CELL_SIZE,
                    fill=bg, width=0
                )

                if val != 0:
                    self.canvas.create_text(
                        x + CELL_SIZE / 2, y + CELL_SIZE / 2,
                        text=str(val), font=font_for(val), fill=fg
                    )

        self.score_label.config(text=f"Score: {self.game.score}")

    def _draw_game_over(self):
        w = int(self.canvas["width"])
        h = int(self.canvas["height"])
        self.canvas.create_rectangle(0, 0, w, h, fill="#ffffff77", width=0)
        self.canvas.create_text(
            w / 2, h / 2 - 10,
            text="Game Over", font=("Helvetica", 36, "bold"), fill="#776e65"
        )
        self.canvas.create_text(
            w / 2, h / 2 + 25,
            text=f"Score: {self.game.score}",
            font=("Helvetica", 16), fill="#776e65"
        )

    def _new_game(self):
        # 先停止 AI
        if self.ai_enabled:
            self._stop_ai()
        self.game.reset()
        self.ai_step_count = 0
        self.info_label.config(text="")
        self._draw()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App2048().run()
