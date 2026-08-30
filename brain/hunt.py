"""Record-hunt helper: sweep seeds with a given beam config, log best score.

    python -m brain.hunt <seed_lo> <seed_hi> [width] [depth]
"""

from __future__ import annotations

import sys
import time

from .beam_search import BeamBrain
from simulator.engine import Game


def main() -> None:
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    d = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    brain = BeamBrain(w, d)
    best = 0
    for s in range(lo, hi):
        g = Game(seed=s)
        t0 = time.time()
        while not g.done:
            g.step(brain.choose(g))
        best = max(best, g.score)
        print(f"hunt[{lo}-{hi}] seed={s}: score={g.score} "
              f"best={best} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
