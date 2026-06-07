import numpy as np
import pytest
from game.board import Board, UP, DOWN, LEFT, RIGHT


# ── slide_row_left ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "input_row, expected_row, expected_score, expected_changed",
    [
        ([0, 0, 0, 0], [0, 0, 0, 0], 0, False),
        ([2, 0, 0, 0], [2, 0, 0, 0], 0, False),
        ([0, 2, 0, 0], [2, 0, 0, 0], 0, True),
        ([2, 2, 0, 0], [4, 0, 0, 0], 4, True),
        ([0, 2, 2, 0], [4, 0, 0, 0], 4, True),
        ([0, 0, 2, 2], [4, 0, 0, 0], 4, True),
        ([2, 0, 2, 0], [4, 0, 0, 0], 4, True),
        ([2, 2, 4, 4], [4, 8, 0, 0], 12, True),
        ([2, 2, 2, 2], [4, 4, 0, 0], 8, True),
        ([2, 2, 2, 0], [4, 2, 0, 0], 4, True),
        ([4, 2, 2, 0], [4, 4, 0, 0], 4, True),
        ([2, 4, 8, 16], [2, 4, 8, 16], 0, False),
        ([4, 4, 8, 8], [8, 16, 0, 0], 24, True),
        ([8, 8, 8, 8], [16, 16, 0, 0], 32, True),
        ([16, 0, 16, 0], [32, 0, 0, 0], 32, True),
    ],
)
def test_slide_row_left(input_row, expected_row, expected_score, expected_changed):
    new_row, score, changed = Board.slide_row_left(input_row)
    assert new_row == expected_row, f"row mismatch: {new_row} != {expected_row}"
    assert score == expected_score, f"score mismatch: {score} != {expected_score}"
    assert changed == expected_changed, f"changed mismatch: {changed} != {expected_changed}"


# ── direction moves ───────────────────────────────────────────────────

def _make_board(grid_data):
    """Helper: create Board from nested list."""
    return Board(grid=np.array(grid_data, dtype=np.uint32))


def test_move_left_merge():
    b = _make_board([[2, 2, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    b2 = b.execute_move(LEFT)
    assert list(b2.grid[0, :]) == [4, 8, 0, 0]


def test_move_right_merge():
    b = _make_board([[2, 2, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    b2 = b.execute_move(RIGHT)
    assert list(b2.grid[0, :]) == [0, 0, 4, 8]


def test_move_up_merge():
    b = _make_board([[2, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]])
    b2 = b.execute_move(UP)
    assert list(b2.grid[:, 0]) == [4, 8, 0, 0]


def test_move_down_merge():
    b = _make_board([[2, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]])
    b2 = b.execute_move(DOWN)
    assert list(b2.grid[:, 0]) == [0, 0, 4, 8]


def test_move_does_not_mutate_original():
    b = _make_board([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    b.execute_move(UP)
    assert list(b.grid[:, 0]) == [2, 2, 0, 0]


def test_score_accumulates():
    b = _make_board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    b2 = b.execute_move(LEFT)
    assert b2.score == 4


def test_score_accumulates_multiple_rows():
    b = _make_board([[2, 2, 0, 0], [4, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    b2 = b.execute_move(LEFT)
    assert b2.score == 12  # 4 + 8


# ── valid moves / game over ───────────────────────────────────────────

def test_board_with_tile_has_valid_moves():
    b = _make_board([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert len(b.get_valid_moves()) >= 1


def test_not_game_over_with_empty_cells():
    b = _make_board([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert not b.is_game_over()


def test_not_game_over_with_merge_possible():
    # full board but two adjacent 2s in row 0 can merge
    b = _make_board([[2, 2, 4, 8], [16, 8, 4, 2], [2, 4, 8, 16], [16, 8, 4, 2]])
    assert not b.is_game_over()


def test_game_over_full_no_moves():
    b = _make_board([[2, 4, 8, 16], [16, 8, 4, 2], [2, 4, 8, 16], [16, 8, 4, 2]])
    assert b.is_game_over()


def test_valid_moves_detection():
    b = _make_board([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    valid = b.get_valid_moves()
    # can move up, down, left, right — all should change the board
    # actually with one tile at (0,0): left and up won't change (already at edge)
    assert UP not in valid
    assert LEFT not in valid
    assert DOWN in valid
    assert RIGHT in valid


# ── utilities ─────────────────────────────────────────────────────────

def test_copy_is_deep():
    b = _make_board([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    b2 = b.copy()
    grid = b2.grid
    grid[0, 1] = 8
    b2.grid = grid
    assert b.grid[0, 1] == 0
    assert b2.grid[0, 1] == 8


def test_get_state_encoding():
    b = _make_board([[2, 4, 0, 0], [0, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    state = b.get_state()
    assert state[0, 0] == 1.0  # log2(2) = 1
    assert state[0, 1] == 2.0  # log2(4) = 2
    assert state[1, 1] == 3.0  # log2(8) = 3
    assert state[0, 2] == 0.0  # empty


def test_max_tile():
    b = _make_board([[2, 4, 0, 0], [0, 64, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert b.max_tile == 64


def test_empty_cells():
    b = _make_board([[2, 0, 4, 0], [0, 8, 0, 16], [0, 0, 0, 0], [2, 0, 0, 0]])
    assert len(b.get_empty_cells()) == 11


def test_invalid_direction_raises():
    b = Board()
    with pytest.raises(ValueError):
        b.execute_move(99)
