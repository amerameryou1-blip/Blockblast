"""Policy distillation scaffold (Gen-2, next iteration).

v1/v2 value nets failed on mean-regression: final-score targets have enormous
luck-driven variance, so the net collapsed to the data mean and search with it
degenerated to greedy play. The fix is the AlphaGo move — imitate the teacher
(beam search) directly with per-candidate softmax over legal placements.

Data contract per sample:
    board:int64, p0/p1/p2:int  -> teacher_slot, teacher_ar, teacher_ac

Training (tomorrow): for each sampled state, recompute legal_actions(),
featurize every candidate as (board plane + placement-mask plane), score each
with the small CNN, cross-entropy against the teacher's choice. Inference:
evaluate all candidates, argmax — optionally reported with beam depth 1.

This file just provides the candidate featurizer shared by train & eval.
"""

from __future__ import annotations

from simulator.engine import Game, PLACEMENTS, GRID


def candidate_planes(board: int, pid: int, anchor: tuple[int, int],
                     xp) -> "xp.ndarray":
    """2×8×8 float array: [board plane, placement-mask plane]."""
    out = xp.zeros((2, GRID, GRID), dtype="float32")
    for c in range(64):
        if board >> c & 1:
            out[0, c // 8, c % 8] = 1.0
    mask = PLACEMENTS[pid][anchor]
    for c in range(64):
        if mask >> c & 1:
            out[1, c // 8, c % 8] = 1.0
    return out


def teacher_rows(game: Game, action: tuple[int, int, int]) -> dict:
    return {
        "board": game.board, "p0": game.pieces[0], "p1": game.pieces[1],
        "p2": game.pieces[2], "teacher_slot": action[0],
        "teacher_ar": action[1], "teacher_ac": action[2],
    }
