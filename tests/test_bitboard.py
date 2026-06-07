import numpy as np
import pytest

from game import bitboard
from game.board import UP, DOWN, LEFT, RIGHT


def _slide_row_left(row):
    tiles = [v for v in row if v != 0]
    score = 0
    merged = []
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            value = tiles[i] * 2
            merged.append(value)
            score += value
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    new_row = merged + [0] * (4 - len(merged))
    return new_row, score, new_row != list(row)


def _reference_move(grid, direction):
    new_grid = grid.copy()
    score = 0

    if direction == UP:
        for col in range(4):
            moved, gained, _ = _slide_row_left(list(new_grid[:, col]))
            new_grid[:, col] = moved
            score += gained
    elif direction == DOWN:
        for col in range(4):
            moved, gained, _ = _slide_row_left(list(reversed(new_grid[:, col])))
            new_grid[:, col] = list(reversed(moved))
            score += gained
    elif direction == LEFT:
        for row in range(4):
            moved, gained, _ = _slide_row_left(list(new_grid[row, :]))
            new_grid[row, :] = moved
            score += gained
    elif direction == RIGHT:
        for row in range(4):
            moved, gained, _ = _slide_row_left(list(reversed(new_grid[row, :])))
            new_grid[row, :] = list(reversed(moved))
            score += gained
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return new_grid, score, not np.array_equal(new_grid, grid)


@pytest.mark.parametrize(
    "grid",
    [
        np.zeros((4, 4), dtype=np.uint32),
        np.array([[2, 0, 4, 0],
                  [0, 8, 0, 16],
                  [32, 0, 64, 0],
                  [0, 128, 0, 256]], dtype=np.uint32),
        np.array([[2, 2, 4, 4],
                  [8, 0, 8, 0],
                  [16, 32, 64, 128],
                  [0, 0, 2, 2]], dtype=np.uint32),
    ],
)
def test_encode_decode_round_trip(grid):
    bits = bitboard.board_to_bits(grid)
    np.testing.assert_array_equal(bitboard.bits_to_grid(bits), grid)


@pytest.mark.parametrize(
    "grid",
    [
        np.array([[2, 2, 4, 4],
                  [0, 2, 0, 2],
                  [4, 0, 4, 0],
                  [8, 16, 32, 64]], dtype=np.uint32),
        np.array([[2, 4, 8, 16],
                  [32, 64, 128, 256],
                  [512, 1024, 2048, 4096],
                  [0, 2, 2, 0]], dtype=np.uint32),
        np.array([[2, 4, 8, 16],
                  [16, 8, 4, 2],
                  [2, 4, 8, 16],
                  [16, 8, 4, 2]], dtype=np.uint32),
    ],
)
@pytest.mark.parametrize("direction", [UP, DOWN, LEFT, RIGHT])
def test_execute_move_matches_reference(grid, direction):
    bits = bitboard.board_to_bits(grid)
    new_bits, score, changed = bitboard.execute_move(bits, direction)
    expected_grid, expected_score, expected_changed = _reference_move(
        grid, direction
    )

    np.testing.assert_array_equal(bitboard.bits_to_grid(new_bits), expected_grid)
    assert score == expected_score
    assert changed == expected_changed


def test_invalid_move_reports_unchanged():
    grid = np.array([[2, 4, 8, 16],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]], dtype=np.uint32)
    bits = bitboard.board_to_bits(grid)
    new_bits, score, changed = bitboard.execute_move(bits, LEFT)

    assert new_bits == bits
    assert score == 0
    assert changed is False


def test_empty_distinct_and_max_tile_queries():
    grid = np.array([[2, 0, 4, 0],
                     [0, 8, 0, 16],
                     [0, 0, 0, 0],
                     [2, 0, 0, 0]], dtype=np.uint32)
    bits = bitboard.board_to_bits(grid)

    assert len(bitboard.get_empty_shifts(bits)) == 11
    assert bitboard.count_distinct_tiles(bits) == 4
    assert bitboard.max_tile(bits) == 16


def test_spawn_tile_sets_rank():
    bits = 0
    bits = bitboard.spawn_tile(bits, 0, 1)
    bits = bitboard.spawn_tile(bits, 4, 2)

    np.testing.assert_array_equal(
        bitboard.bits_to_grid(bits),
        np.array([[2, 4, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.uint32),
    )
