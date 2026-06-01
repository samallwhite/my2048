"""2048 GUI — Tkinter-based graphical interface.

Reuses the existing game/ engine (Board + Game) without modification.
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

        self.root = tk.Tk()
        self.root.title("2048")
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

        # canvas
        canvas_size = CELL_SIZE * GRID_SIZE + PADDING * (GRID_SIZE + 1)
        self.canvas = tk.Canvas(
            self.root, width=canvas_size, height=canvas_size,
            bg=BG_COLOR, highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=10)
        self.canvas.focus_set()

        # key bindings
        self.canvas.bind("<Key>", self._on_key)

        # button
        tk.Button(
            self.root, text="New Game", font=("Helvetica", 14, "bold"),
            fg="#776e65", bg="#cdc1b4", activebackground="#bbada0",
            relief="flat", padx=20, pady=8, command=self._new_game
        ).pack(pady=(0, 20))

        self._draw()

    # ── input ─────────────────────────────────────────────────────────────

    def _on_key(self, event: tk.Event) -> None:
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

    # ── drawing ───────────────────────────────────────────────────────────

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
        self.game.reset()
        self._draw()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App2048().run()
