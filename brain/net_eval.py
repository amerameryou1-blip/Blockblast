"""Load a trained value checkpoint and expose it as a beam-search evaluator.

`make_eval_fn(path)` returns a callable Game -> float that the BeamBrain uses
in place of the hand-tuned heuristic. Torch import stays lazy so the engine,
tests and workers never touch torch.
"""

from __future__ import annotations

from typing import Callable


def make_eval_fn(checkpoint_path: str) -> Callable:
    import torch

    from simulator.engine import Game, PLACEMENTS
    from swarm.train import ValueNet

    model = ValueNet.make()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def featurize(game: Game) -> "torch.Tensor":
        x = torch.zeros((1, 4, 8, 8))
        b = game.board
        for c in range(64):
            if b >> c & 1:
                x[0, 0, c // 8, c % 8] = 1.0
        for slot, pid in enumerate(game.pieces):
            if pid < 0:
                continue
            mask = PLACEMENTS[pid][(0, 0)]
            for c in range(64):
                if mask >> c & 1:
                    x[0, 1 + slot, c // 8, c % 8] = 1.0
        return x

    @torch.no_grad()
    def eval_fn(game: Game) -> float:
        # rank = current score + predicted remaining (log1p inverse)
        pred = float(model(featurize(game))[0, 0])
        return game.score + float(torch.expm1(torch.tensor(pred)))

    return eval_fn
