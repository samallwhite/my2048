import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class Board:
    """4x4 2048 board. Immutable move operations return new Board instances."""

    def __init__(self, grid: np.ndarray | None = None, score: int = 0):
        self.grid = (
            grid.astype(np.uint32)
            if grid is not None
            else np.zeros((4, 4), dtype=np.uint32)
        )
        self.score = int(score)

    # ── core row operation ────────────────────────────────────────────

    @staticmethod
    def slide_row_left(row: list[int]) -> tuple[list[int], int, bool]:
        """
        Slide and merge a 4-element row to the left.

        Returns (new_row, score_gained, changed).
        """
        tiles = [v for v in row if v != 0]

        score = 0
        merged = []
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged.append(tiles[i] * 2)
                score += tiles[i] * 2
                i += 2
            else:
                merged.append(tiles[i])
                i += 1

        new_row = merged + [0] * (4 - len(merged))
        return new_row, score, new_row != list(row)

    # ── move composition ─────────────────────────────────────────────────

    def execute_move(self, direction: int) -> "Board":
        """Return a new Board with the move applied (no spawn)."""
        new_grid = self.grid.copy()
        total_score = 0

        if direction == UP:
            for j in range(4):
                col = list(new_grid[:, j])
                new_col, s, _ = self.slide_row_left(col)
                new_grid[:, j] = new_col
                total_score += s

        elif direction == DOWN:
            for j in range(4):
                col = list(reversed(new_grid[:, j]))
                new_col, s, _ = self.slide_row_left(col)
                new_grid[:, j] = list(reversed(new_col))
                total_score += s

        elif direction == LEFT:
            for i in range(4):
                row = list(new_grid[i, :])
                new_row, s, _ = self.slide_row_left(row)
                new_grid[i, :] = new_row
                total_score += s

        elif direction == RIGHT:
            for i in range(4):
                row = list(reversed(new_grid[i, :]))
                new_row, s, _ = self.slide_row_left(row)
                new_grid[i, :] = list(reversed(new_row))
                total_score += s

        else:
            raise ValueError(f"Invalid direction: {direction}")

        return Board(grid=new_grid, score=self.score + total_score)

    # ── queries ──────────────────────────────────────────────────────────

    def get_empty_cells(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(4) for c in range(4) if self.grid[r, c] == 0]

    def get_valid_moves(self) -> list[int]:
        valid = []
        for d in (UP, DOWN, LEFT, RIGHT):
            moved = self._would_change(d)
            if moved:
                valid.append(d)
        return valid

    def is_game_over(self) -> bool:
        if self.get_empty_cells():
            return False
        return len(self.get_valid_moves()) == 0

    def _would_change(self, direction: int) -> bool:
        """Check if a direction move would modify the grid (fast path)."""
        if direction == LEFT:
            for i in range(4):
                _, _, changed = self.slide_row_left(list(self.grid[i, :]))
                if changed:
                    return True
        elif direction == RIGHT:
            for i in range(4):
                _, _, changed = self.slide_row_left(
                    list(reversed(self.grid[i, :]))
                )
                if changed:
                    return True
        elif direction == UP:
            for j in range(4):
                _, _, changed = self.slide_row_left(list(self.grid[:, j]))
                if changed:
                    return True
        elif direction == DOWN:
            for j in range(4):
                _, _, changed = self.slide_row_left(
                    list(reversed(self.grid[:, j]))
                )
                if changed:
                    return True
        return False

    # ── utilities ────────────────────────────────────────────────────────

    def copy(self) -> "Board":
        return Board(grid=self.grid.copy(), score=self.score)

    def get_state(self) -> np.ndarray:
        """Log2-encoded board for NN consumption (0 for empty cells)."""
        state = np.zeros((4, 4), dtype=np.float32)
        mask = self.grid > 0
        state[mask] = np.log2(self.grid[mask].astype(np.float32))
        return state

    @property
    def max_tile(self) -> int:
        return int(np.max(self.grid))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return np.array_equal(self.grid, other.grid) and self.score == other.score
