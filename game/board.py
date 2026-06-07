import numpy as np
from game import bitboard

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class Board:
    """4x4 2048 board. Immutable move operations return new Board instances."""

    def __init__(self,
                 grid: np.ndarray | None = None,
                 score: int = 0,
                 bits: int | None = None):
        self.bits = int(bits) if bits is not None else bitboard.board_to_bits(grid)
        self.score = int(score)

    @property
    def grid(self) -> np.ndarray:
        return bitboard.bits_to_grid(self.bits)

    @grid.setter
    def grid(self, value: np.ndarray) -> None:
        self.bits = bitboard.board_to_bits(value)

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
        new_bits, score_gained, _ = bitboard.execute_move(self.bits, direction)
        return Board(bits=new_bits, score=self.score + score_gained)

    # ── queries ──────────────────────────────────────────────────────────

    def get_empty_cells(self) -> list[tuple[int, int]]:
        cells = []
        for shift in bitboard.get_empty_shifts(self.bits):
            index = shift // 4
            cells.append((index // 4, index % 4))
        return cells

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
        _, _, changed = bitboard.execute_move(self.bits, direction)
        return changed

    # ── utilities ────────────────────────────────────────────────────────

    def copy(self) -> "Board":
        return Board(bits=self.bits, score=self.score)

    def get_state(self) -> np.ndarray:
        """Log2-encoded board for NN consumption (0 for empty cells)."""
        return bitboard.get_state(self.bits)

    @property
    def max_tile(self) -> int:
        return bitboard.max_tile(self.bits)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return self.bits == other.bits and self.score == other.score
