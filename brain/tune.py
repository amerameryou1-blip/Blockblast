"""Heuristic weight tuning for the Gen-1 evaluator.

sweep(): tries candidate offsets on top of DEFAULT_W over fixed seeds and
prints the leaderboard. Kept tiny on purpose — the point is a stable v0.x
heuristic until the learned evaluator takes over.

    python -m brain.tune
"""

from __future__ import annotations

from .beam_search import DEFAULT_W, evaluate
from simulator.engine import Game

SEEDS = list(range(12))
CANDIDATES: list[dict[str, float]] = []
for holes in (-8.0, -10.0, -12.0):
    for near in (2.0, 3.0, 5.0):
        for mob in (4.0, 6.0):
            w = dict(DEFAULT_W, holes=holes, near=near, mobility=mob)
            CANDIDATES.append(w)


def play(g: Game, w: dict[str, float], width: int = 6, depth: int = 2) -> int:
    while not g.done:
        acts = g.legal_actions()
        beam = []
        for a in acts:
            g2 = g.clone()
            g2.step(a)
            beam.append((evaluate(g2, w), a, g2))
        beam.sort(reverse=True, key=lambda t: t[0])
        beam = beam[:width]
        for _ in range(depth - 1):
            nxt = []
            for _, a, g2 in beam:
                if g2.done:
                    continue
                for a2 in g2.legal_actions():
                    g3 = g2.clone()
                    g3.step(a2)
                    nxt.append((evaluate(g3, w), a, g3))
            if not nxt:
                break
            nxt.sort(reverse=True, key=lambda t: t[0])
            beam = nxt[:width]
        g.step(beam[0][1])
    return g.score


def sweep() -> None:
    table = []
    for w in CANDIDATES:
        scores = [play(Game(seed=s), w) for s in SEEDS]
        table.append((sum(scores) / len(scores), min(scores), w))
    table.sort(reverse=True, key=lambda t: t[0])
    for avg, mn, w in table[:6]:
        print(f"avg={avg:7.0f}  min={mn:6}  holes={w['holes']} near={w['near']} mob={w['mobility']}")


if __name__ == "__main__":
    sweep()
