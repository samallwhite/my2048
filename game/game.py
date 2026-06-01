import numpy as np
from game.board import Board, UP, DOWN, LEFT, RIGHT


class Game:
    """Gym-like 2048 environment."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.board = Board()

    def reset(self) -> np.ndarray:
        """Reset the game: empty board + 2 initial tiles. Returns state."""
        self.board = Board()
        self._spawn_tile()
        self._spawn_tile()
        return self.board.get_state()

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        """
        Execute an action.

        Returns (state, reward, done, info).
        If the action is invalid (no change), reward=0 and state is unchanged
        but a new tile is NOT spawned. done is checked after the move.
        """
        prev_score = self.board.score
        new_board = self.board.execute_move(action)

        if np.array_equal(new_board.grid, self.board.grid):
            return self.board.get_state(), 0, self.board.is_game_over(), {
                "score": self.board.score,
                "max_tile": self.board.max_tile,
                "invalid": True,
            }

        self.board = new_board
        reward = self.board.score - prev_score
        self._spawn_tile()

        done = self.board.is_game_over()
        return self.board.get_state(), reward, done, {
            "score": self.board.score,
            "max_tile": self.board.max_tile,
            "invalid": False,
        }

    def _spawn_tile(self) -> None:
        empty = self.board.get_empty_cells()
        if not empty:
            return
        idx = self.rng.integers(len(empty))
        r, c = empty[idx]
        self.board.grid[r, c] = 4 if self.rng.random() < 0.1 else 2

    def clone(self) -> "Game":
        """Deep copy including RNG state (essential for expectimax search)."""
        g = Game.__new__(Game)
        g.board = self.board.copy()
        g.rng = np.random.default_rng()
        g.rng.bit_generator.state = self.rng.bit_generator.state
        return g

    def render(self) -> None:
        """Print the board to console."""
        print()
        for r in range(4):
            row = []
            for c in range(4):
                v = self.board.grid[r, c]
                row.append(f"{v:>5}" if v else "    .")
            print(" ".join(row))
        print(f"\nScore: {self.board.score}  Max: {self.board.max_tile}")
        print()

    @property
    def score(self) -> int:
        return self.board.score

    @property
    def max_tile(self) -> int:
        return self.board.max_tile
