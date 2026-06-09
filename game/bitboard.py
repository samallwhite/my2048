"""Bitboard primitives for 2048 boards.

The board is stored as sixteen 4-bit ranks in row-major order. Rank 0 is an
empty cell, rank 1 is tile 2, rank 2 is tile 4, and so on.
"""

from __future__ import annotations

import os
import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    njit = None

ROW_MASK = 0xffff
BOARD_SIZE = 4
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
MAX_RANK = 0xf
USE_NUMBA = njit is not None and os.environ.get("PY2048_USE_NUMBA") == "1"


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


def _build_query_tables() -> tuple[list[tuple[int, ...]], list[int], list[int]]:
    empty_shifts: list[tuple[int, ...]] = [()] * 65536
    max_ranks = [0] * 65536
    rank_masks = [0] * 65536

    for row in range(65536):
        shifts: list[int] = []
        max_rank = 0
        rank_mask = 0
        for shift in range(0, 16, 4):
            rank = (row >> shift) & 0xf
            if rank == 0:
                shifts.append(shift)
                continue
            if rank > max_rank:
                max_rank = rank
            rank_mask |= 1 << rank

        empty_shifts[row] = tuple(shifts)
        max_ranks[row] = max_rank
        rank_masks[row] = rank_mask

    return empty_shifts, max_ranks, rank_masks


ROW_LEFT_TABLE, ROW_RIGHT_TABLE, SCORE_LEFT_TABLE, SCORE_RIGHT_TABLE = (
    _build_move_tables()
)
ROW_EMPTY_SHIFTS_TABLE, ROW_MAX_RANK_TABLE, ROW_RANK_MASK_TABLE = (
    _build_query_tables()
)
ROW_LEFT_ARRAY = np.asarray(ROW_LEFT_TABLE, dtype=np.uint64)
ROW_RIGHT_ARRAY = np.asarray(ROW_RIGHT_TABLE, dtype=np.uint64)
SCORE_LEFT_ARRAY = np.asarray(SCORE_LEFT_TABLE, dtype=np.uint64)
SCORE_RIGHT_ARRAY = np.asarray(SCORE_RIGHT_TABLE, dtype=np.uint64)


if njit is not None:
    @njit
    def _transpose_numba(bits: np.uint64) -> np.uint64:
        a1 = bits & np.uint64(0xF0F00F0FF0F00F0F)
        a2 = bits & np.uint64(0x0000F0F00000F0F0)
        a3 = bits & np.uint64(0x0F0F00000F0F0000)
        a = a1 | (a2 << np.uint64(12)) | (a3 >> np.uint64(12))
        b1 = a & np.uint64(0xFF00FF0000FF00FF)
        b2 = a & np.uint64(0x00FF00FF00000000)
        b3 = a & np.uint64(0x00000000FF00FF00)
        return b1 | (b2 >> np.uint64(24)) | (b3 << np.uint64(24))


    @njit
    def _execute_rows_numba(
        bits: np.uint64,
        table: np.ndarray,
        score_table: np.ndarray,
    ) -> tuple[np.uint64, int]:
        row_mask = np.uint64(0xffff)
        ret = bits
        score = 0

        row = int((bits >> np.uint64(0)) & row_mask)
        ret ^= table[row] << np.uint64(0)
        score += int(score_table[row])

        row = int((bits >> np.uint64(16)) & row_mask)
        ret ^= table[row] << np.uint64(16)
        score += int(score_table[row])

        row = int((bits >> np.uint64(32)) & row_mask)
        ret ^= table[row] << np.uint64(32)
        score += int(score_table[row])

        row = int((bits >> np.uint64(48)) & row_mask)
        ret ^= table[row] << np.uint64(48)
        score += int(score_table[row])

        return ret, score


    @njit
    def _execute_move_numba(
        bits: np.uint64,
        direction: int,
        row_left: np.ndarray,
        row_right: np.ndarray,
        score_left: np.ndarray,
        score_right: np.ndarray,
    ) -> tuple[np.uint64, int, bool]:
        if direction == 0:
            moved, score = _execute_rows_numba(
                _transpose_numba(bits),
                row_left,
                score_left,
            )
            new_bits = _transpose_numba(moved)
        elif direction == 1:
            moved, score = _execute_rows_numba(
                _transpose_numba(bits),
                row_right,
                score_right,
            )
            new_bits = _transpose_numba(moved)
        elif direction == 2:
            new_bits, score = _execute_rows_numba(bits, row_left, score_left)
        else:
            new_bits, score = _execute_rows_numba(bits, row_right, score_right)

        return new_bits, score, new_bits != bits


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
    if direction not in (0, 1, 2, 3):
        raise ValueError(f"Invalid direction: {direction}")

    if USE_NUMBA:
        new_bits, score, changed = _execute_move_numba(
            np.uint64(bits),
            int(direction),
            ROW_LEFT_ARRAY,
            ROW_RIGHT_ARRAY,
            SCORE_LEFT_ARRAY,
            SCORE_RIGHT_ARRAY,
        )
        return int(new_bits), int(score), bool(changed)

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

    return new_bits, score, new_bits != bits


def get_empty_shifts(bits: int) -> list[int]:
    row_mask = ROW_MASK
    row0 = bits & row_mask
    row1 = (bits >> 16) & row_mask
    row2 = (bits >> 32) & row_mask
    row3 = (bits >> 48) & row_mask

    shifts = list(ROW_EMPTY_SHIFTS_TABLE[row0])
    shifts.extend(shift + 16 for shift in ROW_EMPTY_SHIFTS_TABLE[row1])
    shifts.extend(shift + 32 for shift in ROW_EMPTY_SHIFTS_TABLE[row2])
    shifts.extend(shift + 48 for shift in ROW_EMPTY_SHIFTS_TABLE[row3])
    return shifts


def spawn_tile(bits: int, shift: int, rank: int) -> int:
    if shift < 0 or shift >= 64 or shift % 4 != 0:
        raise ValueError(f"Invalid bit shift: {shift}")
    if rank < 0 or rank > MAX_RANK:
        raise ValueError(f"Invalid tile rank: {rank}")
    if ((bits >> shift) & 0xf) != 0:
        raise ValueError("Cannot spawn tile into a non-empty cell")
    return bits | (rank << shift)


def count_distinct_tiles(bits: int) -> int:
    row_mask = ROW_MASK
    rank_mask = (
        ROW_RANK_MASK_TABLE[bits & row_mask]
        | ROW_RANK_MASK_TABLE[(bits >> 16) & row_mask]
        | ROW_RANK_MASK_TABLE[(bits >> 32) & row_mask]
        | ROW_RANK_MASK_TABLE[(bits >> 48) & row_mask]
    )
    return rank_mask.bit_count()


def max_tile(bits: int) -> int:
    row_mask = ROW_MASK
    max_rank = max(
        ROW_MAX_RANK_TABLE[bits & row_mask],
        ROW_MAX_RANK_TABLE[(bits >> 16) & row_mask],
        ROW_MAX_RANK_TABLE[(bits >> 32) & row_mask],
        ROW_MAX_RANK_TABLE[(bits >> 48) & row_mask],
    )
    return 0 if max_rank == 0 else 1 << max_rank
