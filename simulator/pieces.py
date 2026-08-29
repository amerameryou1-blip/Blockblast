"""Block Blast piece catalog.

Shapes are fixed-orientation (the real game has no rotation). Offsets are
(row, col) with top-left anchor. The exact pool and spawn distribution of the
real app are not public: parents and children make peace with uncertainty by
keeping this list declarative and by letting the spawner randomize weights
(see `PieceSpawner.randomized`) so trained policies are robust either way.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def shape(art: str) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    for r, line in enumerate(art.strip("\n").splitlines()):
        for c, ch in enumerate(line):
            if ch == "#":
                cells.append((r, c))
    return tuple(cells)


@dataclass(frozen=True)
class Piece:
    id: int
    name: str
    cells: tuple[tuple[int, int], ...]
    weight: float

    @property
    def rows(self) -> int:
        return max(r for r, _ in self.cells) + 1

    @property
    def cols(self) -> int:
        return max(c for _, c in self.cells) + 1

    @property
    def n_cells(self) -> int:
        return len(self.cells)


_DEFS: list[tuple[str, str, float]] = [
    # name, art, base spawn weight (heuristic defaults; calibrate from real games)
    ("single", "#", 1.0),
    ("bar2h", "##", 1.0),
    ("bar2v", "#\n#", 1.0),
    ("bar3h", "###", 0.9),
    ("bar3v", "#\n#\n#", 0.9),
    ("bar4h", "####", 0.6),
    ("bar4v", "#\n#\n#\n#", 0.6),
    ("bar5h", "#####", 0.35),
    ("bar5v", "#\n#\n#\n#\n#", 0.35),
    ("sq2", "##\n##", 0.8),
    ("sq3", "###\n###\n###", 0.3),
    ("tri_l_ne", "##\n#.", 0.8),
    ("tri_l_nw", "##\n.#", 0.8),
    ("tri_l_se", "#.\n##", 0.8),
    ("tri_l_sw", ".#\n##", 0.8),
    ("l4_a", "#.\n#.\n##", 0.6),
    ("l4_b", "###\n#..", 0.6),
    ("l4_c", "##\n.#\n.#", 0.6),
    ("l4_d", "..#\n###", 0.6),
    ("j4_a", "##\n#.\n#.", 0.6),
    ("j4_b", "#..\n###", 0.6),
    ("j4_c", ".#\n.#\n##", 0.6),
    ("j4_d", "###\n..#", 0.6),
    ("t4_up", "###\n.#.", 0.5),
    ("t4_right", ".#\n##\n.#", 0.5),
    ("t4_down", ".#.\n###", 0.5),
    ("t4_left", "#.\n##\n#.", 0.5),
    ("s4_h", ".##\n##.", 0.5),
    ("z4_h", "##.\n.##", 0.5),
    ("s4_v", "#.\n##\n.#", 0.5),
    ("z4_v", ".#\n##\n#.", 0.5),
    ("l5_se", "#..\n#..\n###", 0.35),
    ("l5_wn", "###\n#..\n#..", 0.35),
    ("l5_ws", "###\n..#\n..#", 0.35),
    ("l5_en", "..#\n..#\n###", 0.35),
    ("plus", ".#.\n###\n.#.", 0.25),
    ("rect23", "##\n##\n##", 0.4),
    ("rect32", "###\n###", 0.4),
]

PIECES: tuple[Piece, ...] = tuple(
    Piece(i, name, shape(art), w) for i, (name, art, w) in enumerate(_DEFS)
)
BY_NAME = {p.name: p for p in PIECES}


class PieceSpawner:
    """Weighted sampler for rounds of 3 pieces.

    mode="fixed"     -> use catalog weights (deterministic given seed)
    mode="randomized"-> resample weights each round (distribution randomization:
                        keeps the policy robust to the unknown true spawn rates)
    """

    def __init__(self, rng: random.Random, mode: str = "fixed"):
        self.rng = rng
        self.mode = mode

    def draw(self) -> tuple[int, int, int]:
        if self.mode == "randomized":
            weights = [self.rng.random() for _ in PIECES]
        else:
            weights = [p.weight for p in PIECES]
        picked = self.rng.choices(PIECES, weights=weights, k=3)
        return (picked[0].id, picked[1].id, picked[2].id)
