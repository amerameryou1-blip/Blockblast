"""Terminal play for humans — the "playable mode" used to sanity-check rules.

    python -m simulator.cli [seed]

Controls: pick `slot row col` e.g. `0 3 4`. `q` quits.
"""

from __future__ import annotations

import sys

from .engine import Game
from .pieces import PIECES


def _art(pid: int) -> str:
    p = PIECES[pid]
    grid = [["."] * p.cols for _ in range(p.rows)]
    for dr, dc in p.cells:
        grid[dr][dc] = "#"
    return "\n".join("".join(r) for r in grid)


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    g = Game(seed=seed)
    while not g.done:
        print("\n" + g.render())
        for i, p in enumerate(g.pieces):
            if p >= 0:
                print(f"  [{i}] {PIECES[p].name}\n" + _art(p))
        raw = input("place> ").strip().lower()
        if raw in {"q", "quit"}:
            return
        try:
            slot, r, c = (int(x) for x in raw.split())
        except ValueError:
            print("usage: <slot> <row> <col>")
            continue
        try:
            reward, _ = g.step((slot, r, c))
            print(f"+{reward:.0f}")
        except ValueError as e:
            print(f"nope: {e}")
    print("\nGAME OVER")
    print(g.render())
    print(f"final score: {g.score} in {g.turns} moves")


if __name__ == "__main__":
    main()
