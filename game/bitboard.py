"""Bitboard primitives for 2048 boards.

The board is stored as sixteen 4-bit ranks in row-major order. Rank 0 is an
empty cell, rank 1 is tile 2, rank 2 is tile 4, and so on.
"""

from __future__ import annotations

import numpy as np

ROW_MASK = 0xffff
BOARD_SIZE = 4
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
MAX_RANK = 0xf


def _reverse_row(row: int) -> int:
    return ((row >> 12) |
            ((row >> 4) & 0x00f0) |
            ((row << 4) & 0x0f00) |
            ((row << 12) & 0xf000))


def _pack_line(line: list[int]) -> int:
    return (line[0] |
            (line[1] << 4) |
            (line[2] << 8) |
            (line[3] << 12))


def _slide_rank_line_left(line: list[int]) -> tuple[list[int], int]:
    tiles = [v for v in line if v != 0]
    merged: list[int] = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            new_rank = min(tiles[i] + 1, MAX_RANK)
            merged.append(new_rank)
            score += 1 << new_rank
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    return merged + [0] * (BOARD_SIZE - len(merged)), score


def _build_move_tables() -> tuple[list[int], list[int], list[int], list[int]]:
    row_left = [0] * 65536
    row_right = [0] * 65536
    score_left = [0] * 65536
    score_right = [0] * 65536

    for row in range(65536):
        line = [
            (row >> 0) & 0xf,
            (row >> 4) & 0xf,
            (row >> 8) & 0xf,
            (row >> 12) & 0xf,
        ]
        moved, score = _slide_rank_line_left(line)
        result = _pack_line(moved)

        rev_row = _reverse_row(row)
        rev_result = _reverse_row(result)

        row_left[row] = row ^ result
        row_right[rev_row] = rev_row ^ rev_result
        score_left[row] = score
        score_right[rev_row] = score

    return row_left, row_right, score_left, score_right


ROW_LEFT_TABLE, ROW_RIGHT_TABLE, SCORE_LEFT_TABLE, SCORE_RIGHT_TABLE = (
    _build_move_tables()
)


def board_to_bits(grid: np.ndarray | list[list[int]] | None) -> int:
    if grid is None:
        return 0

    arr = np.asarray(grid)
    if arr.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError("grid must have shape (4, 4)")

    bits = 0
    shift = 0
    for value in arr.reshape(CELL_COUNT):
        tile = int(value)
        if tile < 0:
            raise ValueError("tile values must be non-negative")
        if tile:
            if tile & (tile - 1):
                raise ValueError("tile values must be powers of two")
            rank = tile.bit_length() - 1
            if rank > MAX_RANK:
                raise ValueError("tile rank exceeds 4-bit bitboard capacity")
            bits |= rank << shift
        shift += 4
    return bits


def bits_to_grid(bits: int) -> np.ndarray:
    grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint32)
    for index in range(CELL_COUNT):
        rank = (bits >> (index * 4)) & 0xf
        if rank:
            grid[index // BOARD_SIZE, index % BOARD_SIZE] = 1 << rank
    return grid


def get_state(bits: int) -> np.ndarray:
    state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for index in range(CELL_COUNT):
        rank = (bits >> (index * 4)) & 0xf
        if rank:
            state[index // BOARD_SIZE, index % BOARD_SIZE] = float(rank)
    return state


def transpose(bits: int) -> int:
    a1 = bits & 0xF0F00F0FF0F00F0F
    a2 = bits & 0x0000F0F00000F0F0
    a3 = bits & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _execute_rows(bits: int, table: list[int],
                  score_table: list[int]) -> tuple[int, int]:
    ret = bits
    score = 0

    row = (bits >> 0) & ROW_MASK
    ret ^= table[row] << 0
    score += score_table[row]

    row = (bits >> 16) & ROW_MASK
    ret ^= table[row] << 16
    score += score_table[row]

    row = (bits >> 32) & ROW_MASK
    ret ^= table[row] << 32
    score += score_table[row]

    row = (bits >> 48) & ROW_MASK
    ret ^= table[row] << 48
    score += score_table[row]

    return ret, score


def execute_move(bits: int, direction: int) -> tuple[int, int, bool]:
    if direction == 0:
        moved, score = _execute_rows(transpose(bits),
                                     ROW_LEFT_TABLE,
                                     SCORE_LEFT_TABLE)
        new_bits = transpose(moved)
    elif direction == 1:
        moved, score = _execute_rows(transpose(bits),
                                     ROW_RIGHT_TABLE,
                                     SCORE_RIGHT_TABLE)
        new_bits = transpose(moved)
    elif direction == 2:
        new_bits, score = _execute_rows(bits, ROW_LEFT_TABLE, SCORE_LEFT_TABLE)
    elif direction == 3:
        new_bits, score = _execute_rows(bits, ROW_RIGHT_TABLE,
                                        SCORE_RIGHT_TABLE)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return new_bits, score, new_bits != bits


def get_empty_shifts(bits: int) -> list[int]:
    return [
        shift
        for shift in range(0, 64, 4)
        if ((bits >> shift) & 0xf) == 0
    ]


def spawn_tile(bits: int, shift: int, rank: int) -> int:
    if shift < 0 or shift >= 64 or shift % 4 != 0:
        raise ValueError(f"Invalid bit shift: {shift}")
    if rank < 0 or rank > MAX_RANK:
        raise ValueError(f"Invalid tile rank: {rank}")
    if ((bits >> shift) & 0xf) != 0:
        raise ValueError("Cannot spawn tile into a non-empty cell")
    return bits | (rank << shift)


def count_distinct_tiles(bits: int) -> int:
    ranks: set[int] = set()
    for shift in range(0, 64, 4):
        rank = (bits >> shift) & 0xf
        if rank:
            ranks.add(rank)
    return len(ranks)


def max_tile(bits: int) -> int:
    max_rank = 0
    for shift in range(0, 64, 4):
        rank = (bits >> shift) & 0xf
        if rank > max_rank:
            max_rank = rank
    return 0 if max_rank == 0 else 1 << max_rank
