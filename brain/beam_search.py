"""Gen-1 brain: beam search over placements with a hand-tuned state evaluation.

The evaluator favours boards that (a) keep options open, (b) are close to
clears, (c) don't fragment into unusable micro-gaps. Weights are v0 hand
heuristics — Gen-2 replaces this scoring with the learned network, but the
search machinery stays.

Run a benchmark:
    python -m brain.beam_search --games 25 --width 24 --depth 3
"""

from __future__ import annotations

import argparse
import time

from simulator.engine import Game, PLACEMENTS

DEFAULT_W = {
    "score": 1.0,          # current points (rewards realized clears)
    "holes": -8.0,         # empty cell surrounded by filled cells/borders
    "frag": -1.2,          # filled<->empty transitions per row/col
    "region": 0.6,         # largest contiguous empty region (room for big pieces)
    "near": 3.0,           # rows/cols missing <= 2 cells (clears within reach)
    "mobility": 4.0,       # min legal placements over offered pieces (survival)
}

NEI = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bits(board: int, r: int, c: int) -> int:
    return (board >> (r * 8 + c)) & 1


def evaluate(game: Game, w: dict[str, float] | None = None) -> float:
    w = w or DEFAULT_W
    b = game.board

    holes = 0
    frag = 0
    region_best = 0
    near = 0

    # holes: empty cell whose orthogonal neighbours are all filled/off-board
    for r in range(8):
        for c in range(8):
            if _bits(b, r, c):
                continue
            ok = True
            for dr, dc in NEI:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and not _bits(b, nr, nc):
                    ok = False
                    break
            if ok:
                holes += 1

    # fragmentation: count fill-state transitions along each line
    for r in range(8):
        prev = _bits(b, r, 0)
        for c in range(1, 8):
            cur = _bits(b, r, c)
            if cur != prev:
                frag += 1
            prev = cur
    for c in range(8):
        prev = _bits(b, 0, c)
        for r in range(1, 8):
            cur = _bits(b, r, c)
            if cur != prev:
                frag += 1
            prev = cur

    # largest empty region (flood fill on <=64 cells — cheap)
    seen = 0
    for r in range(8):
        for c in range(8):
            bit = 1 << (r * 8 + c)
            if b & bit or seen & bit:
                continue
            size = 0
            stack = [(r, c)]
            seen |= bit
            while stack:
                cr, cc = stack.pop()
                size += 1
                for dr, dc in NEI:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        nb = 1 << (nr * 8 + nc)
                        if not b & nb and not seen & nb:
                            seen |= nb
                            stack.append((nr, nc))
            region_best = max(region_best, size)

    # near-complete lines
    for r in range(8):
        cnt = sum(_bits(b, r, c) for c in range(8))
        if cnt >= 6:
            near += 1
    for c in range(8):
        cnt = sum(_bits(b, r, c) for r in range(8))
        if cnt >= 6:
            near += 1

    # mobility: worst-offered-piece placement count
    mobility = min(64, min(
        (
            sum(1 for m in PLACEMENTS[pid].values() if not m & b)
            for pid in game.pieces if pid >= 0
        ),
        default=0,
    ))

    return (
        w["score"] * game.score
        + w["holes"] * holes
        + w["frag"] * frag
        + w["region"] * region_best
        + w["near"] * near
        + w["mobility"] * mobility
    )


class BeamBrain:
    def __init__(self, width: int = 16, depth: int = 3,
                 weights: dict[str, float] | None = None,
                 eval_fn=None):
        self.width = width
        self.depth = depth
        self.w = weights
        self.eval_fn = eval_fn  # None => use hand-tuned heuristic

    def _evaluate(self, g: Game) -> float:
        return self.eval_fn(g) if self.eval_fn else evaluate(g, self.w)

    def choose(self, game: Game) -> tuple[int, int, int]:
        # candidates: (value, root_action, game_after)
        beam: list[tuple[float, tuple[int, int, int], Game]] = []
        for a in game.legal_actions():
            g = game.clone()
            g.step(a)
            beam.append((self._evaluate(g), a, g))
        if not beam:
            raise RuntimeError("no legal actions")
        beam.sort(reverse=True, key=lambda t: t[0])
        beam = beam[: self.width]

        for _ in range(self.depth - 1):
            nxt: list[tuple[float, tuple[int, int, int], Game]] = []
            for _, root, g in beam:
                if g.done:
                    continue
                for a in g.legal_actions():
                    gg = g.clone()
                    gg.step(a)
                    nxt.append((self._evaluate(gg), root, gg))
            if not nxt:
                break
            nxt.sort(reverse=True, key=lambda t: t[0])
            beam = nxt[: self.width]
        return beam[0][1]


def bench(games: int, width: int, depth: int, seed0: int = 0) -> None:
    brain = BeamBrain(width, depth)
    scores, turns, t0 = [], [], time.perf_counter()
    for i in range(games):
        g = Game(seed=seed0 + i)
        while not g.done:
            g.step(brain.choose(g))
        scores.append(g.score)
        turns.append(g.turns)
        print(f"  game {i}: score={g.score} turns={g.turns}")
    dt = time.perf_counter() - t0
    print(f"\n{games} games in {dt:.1f}s ({games / dt:.2f} games/s)")
    print(f"score  mean={sum(scores) / games:.0f} "
          f"min={min(scores)} max={max(scores)}")
    print(f"turns  mean={sum(turns) / games:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()
    bench(args.games, args.width, args.depth)


if __name__ == "__main__":
    main()
