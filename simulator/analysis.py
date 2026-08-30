"""Piece-quality analysis: which shapes are dangerous, rare, or overfed?

Plays N games with a random policy and reports per-piece stats:
  - offered: how often the piece showed up
  - at_death: how often the piece was in hand when the game ended
    (high ratio => the piece crowds the board / is hard to place)

    python -m simulator.analysis [games]
"""

from __future__ import annotations

import random
import sys
from collections import Counter

from .engine import Game
from .pieces import PIECES


def run(n_games: int = 2000, seed: int = 0) -> None:
    rng = random.Random(seed)
    offered: Counter[int] = Counter()
    at_death: Counter[int] = Counter()
    scores: list[int] = []
    for _ in range(n_games):
        g = Game(seed=rng.randrange(1 << 31))
        for p in g.pieces:
            offered[p] += 1
        while not g.done:
            acts = g.legal_actions()
            g.step(acts[rng.randrange(len(acts))])
        for p in g.pieces:
            if p >= 0:
                at_death[p] += 1
        scores.append(g.score)

    print(f"{n_games} random games  |  mean score {sum(scores) / len(scores):.0f}  "
          f"max {max(scores)}")
    print(f"{'piece':<10} {'cells':>5} {'w':>5} {'offered':>8} {'death%':>7}")
    rows = []
    for p in PIECES:
        off = offered[p.id] or 1
        death_rate = at_death[p.id] / off
        rows.append((death_rate, p.name, p.n_cells, p.weight, offered[p.id]))
    rows.sort(reverse=True)
    for death_rate, name, cells, w, off in rows[:12]:
        print(f"{name:<10} {cells:>5} {w:>5} {off:>8} {death_rate * 100:>6.1f}%")
    print("...")
    for death_rate, name, cells, w, off in rows[-4:]:
        print(f"{name:<10} {cells:>5} {w:>5} {off:>8} {death_rate * 100:>6.1f}%")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
