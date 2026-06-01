from game.board import Board


class Agent:
    """Base class for all 2048 AI agents."""

    def get_action(self, board: Board) -> int:
        """Return 0=up, 1=down, 2=left, 3=right."""
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the start of each game."""
        pass
