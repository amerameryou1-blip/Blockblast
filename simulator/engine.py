"""Core Block Blast engine: 8x8 bitboard, pure-int state, zero dependencies.

Board representation: a 64-bit int; bit (r*8 + c) set == cell filled.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .pieces import PIECES, Piece, PieceSpawner

GRID = 8
N_CELLS = 64
FULL = (1 << 64) - 1

ROW_MASKS = tuple(sum(1 << (r * 8 + c) for c in range(GRID)) for r in range(GRID))
COL_MASKS = tuple(sum(1 << (r * 8 + c) for r in range(GRID)) for c in range(GRID))


def _placement_masks(piece: Piece) -> dict[tuple[int, int], int]:
    masks: dict[tuple[int, int], int] = {}
    for ar in range(GRID - piece.rows + 1):
        for ac in range(GRID - piece.cols + 1):
            mask = 0
            for dr, dc in piece.cells:
                mask |= 1 << ((ar + dr) * 8 + (ac + dc))
            masks[(ar, ac)] = mask
    return masks


PLACEMENTS: list[dict[tuple[int, int], int]] = [_placement_masks(p) for p in PIECES]


@dataclass(frozen=True)
class ScoringConfig:
    """Calibratable scoring model (real constants unknown) — see ARCHITECTURE.md."""

    line_base: int = 10          # points per line, base
    multi_line: bool = True      # lines^2 bonus for simultaneous clears
    combo_match: bool = True     # streak multiplier for consecutive clearing rounds


class Game:
    __slots__ = ("board", "pieces", "score", "streak", "turns", "done",
                 "rng", "scoring", "_round_cleared", "_spawner_mode", "_spawner")

    def __init__(self, seed: int | None = None, scoring: ScoringConfig | None = None,
                 spawner_mode: str = "fixed"):
        self.rng = random.Random(seed)
        self.scoring = scoring or ScoringConfig()
        self._spawner_mode = spawner_mode
        self._spawner = PieceSpawner(self.rng, spawner_mode)
        self.reset()

    def reset(self) -> None:
        self.board = 0
        self.pieces = self._spawner.draw()
        self.score = 0
        self.streak = 0
        self.turns = 0
        self.done = False
        self._round_cleared = False

    # ---- queries ---------------------------------------------------------

    def legal_actions(self) -> list[tuple[int, int, int]]:
        """(slot, anchor_row, anchor_col) for every legal placement."""
        out: list[tuple[int, int, int]] = []
        board = self.board
        for slot, pid in enumerate(self.pieces):
            if pid < 0:
                continue
            for (r, c), mask in PLACEMENTS[pid].items():
                if mask & board == 0:
                    out.append((slot, r, c))
        return out

    def has_legal_move(self) -> bool:
        board = self.board
        for pid in self.pieces:
            if pid < 0:
                continue
            for mask in PLACEMENTS[pid].values():
                if mask & board == 0:
                    return True
        return False

    # ---- mutation --------------------------------------------------------

    def step(self, action: tuple[int, int, int]) -> tuple[float, bool]:
        if self.done:
            raise RuntimeError("game over")
        slot, r, c = action
        pid = self.pieces[slot]
        if pid < 0:
            raise ValueError(f"slot {slot} already used")
        mask = PLACEMENTS[pid].get((r, c))
        if mask is None or mask & self.board:
            raise ValueError(f"illegal placement of {PIECES[pid].name} at {(r, c)}")

        before = self.score
        piece = PIECES[pid]
        self.board |= mask
        self.score += piece.n_cells

        clears = 0
        clear_mask = 0
        for m in ROW_MASKS + COL_MASKS:
            if self.board & m == m:
                clears += 1
                clear_mask |= m
        if clears:
            self.board &= FULL ^ clear_mask
            pts = self.scoring.line_base
            if self.scoring.multi_line:
                pts *= clears
            if self.scoring.combo_match:
                pts *= self.streak + 1
            self.score += pts * clears
            self._round_cleared = True

        self.pieces = tuple(
            -1 if i == slot else p for i, p in enumerate(self.pieces)
        )
        self.turns += 1

        if all(p < 0 for p in self.pieces):
            self.streak = self.streak + 1 if self._round_cleared else 0
            self._round_cleared = False
            self.pieces = self._spawner.draw()

        self.done = not self.has_legal_move()
        return float(self.score - before), self.done

    # ---- helpers ---------------------------------------------------------

    def clone(self) -> "Game":
        """Deterministic copy for search: private RNG derived from state."""
        g = Game.__new__(Game)
        g.board = self.board
        g.pieces = self.pieces
        g.score = self.score
        g.streak = self.streak
        g.turns = self.turns
        g.done = self.done
        g.scoring = self.scoring
        g._round_cleared = self._round_cleared
        g._spawner_mode = self._spawner_mode
        g.rng = random.Random(hash((self.board, self.pieces, self.turns)))
        g._spawner = PieceSpawner(g.rng, g._spawner_mode)
        return g

    def snapshot(self) -> tuple:
        return (self.board, self.pieces, self.score, self.streak,
                self.turns, self.done)

    def render(self) -> str:
        lines = []
        for r in range(GRID):
            row = []
            for c in range(GRID):
                row.append("[]" if self.board >> (r * 8 + c) & 1 else ". ")
            lines.append("".join(row))
        tray = " | ".join(
            PIECES[p].name if p >= 0 else "-" for p in self.pieces
        )
        return "\n".join(lines) + f"\n  tray: {tray}\n  score: {self.score}"
