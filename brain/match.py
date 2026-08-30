"""Head-to-head: heuristic beam vs value-net beam over fixed seeds.

    python -m brain.match checkpoints/value_<ts>.pt [--games 20]

Reports mean/min/max for both so we only adopt the net when it actually wins.
"""

from __future__ import annotations

import argparse
import sys

from .beam_search import BeamBrain
from .net_eval import make_eval_fn
from simulator.engine import Game


def play(brain: BeamBrain, seed: int) -> tuple[int, int]:
    g = Game(seed=seed)
    while not g.done:
        g.step(brain.choose(g))
    return g.score, g.turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    heur = BeamBrain(args.width, args.depth)
    bp1 = BeamBrain(args.width, 1, eval_fn=make_eval_fn(args.checkpoint))

    results = {}
    for name, brain in (("heuristic", heur), ("net+1ply", bp1)):
        scores = [play(brain, s)[0] for s in range(args.games)]
        results[name] = (sum(scores) / len(scores), min(scores), max(scores))
        print(f"{name:9s}  mean={results[name][0]:8.0f}  min={results[name][1]:6}  max={results[name][2]:8}", flush=True)

    print(f"\nnet improvement: {(results['net+1ply'][0] - results['heuristic'][0]) / results['heuristic'][0] * 100:+.1f}%")


if __name__ == "__main__":
    sys.exit(main())
