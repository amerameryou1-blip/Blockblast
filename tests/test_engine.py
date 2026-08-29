"""Mechanics that must never break. Run: pytest tests/ -q"""

import random

import pytest

from simulator import Game, PIECES, BY_NAME, PLACEMENTS
from simulator.engine import ROW_MASKS, COL_MASKS


def test_grid_masks():
    assert len(ROW_MASKS) == 8 and len(COL_MASKS) == 8
    assert all(bin(m).count("1") == 8 for m in ROW_MASKS + COL_MASKS)
    assert ROW_MASKS[0] == 0xFF
    assert COL_MASKS[0] == sum(1 << (8 * r) for r in range(8))


def test_new_game_state():
    g = Game(seed=1)
    assert g.board == 0 and g.score == 0 and not g.done
    assert len(g.pieces) == 3 and all(p >= 0 for p in g.pieces)


def test_place_marks_cells():
    g = Game(seed=1)
    g.pieces = (BY_NAME["bar3h"].id, -1, -1)
    before = g.score
    reward, done = g.step((0, 0, 0))
    assert g.board == 0b111
    assert reward == 3 and not done  # 3 cells, no clear yet


def test_illegal_overlap_rejected():
    g = Game(seed=1)
    g.pieces = (BY_NAME["sq2"].id, -1, -1)
    g.step((0, 0, 0))
    g.pieces = (BY_NAME["sq2"].id, -1, -1)
    with pytest.raises(ValueError):
        g.step((0, 1, 1))  # overlaps the first square


def test_row_clear():
    g = Game(seed=1)
    g.pieces = (BY_NAME["bar5h"].id, BY_NAME["bar3h"].id, -1)
    g.step((0, 0, 0))
    reward, _ = g.step((1, 0, 5))  # completes row 0
    assert g.board == 0                       # row cleared, nothing left
    assert reward == 3 + 10 * 1 * 1          # 3 cells + 1 clear line at streak 0


def test_row_and_column_clear_together():
    g = Game(seed=1)
    # fill row 0 except (0,0); fill col 0 except (0,0); then drop single at (0,0)
    g.board = 0
    for c in range(1, 8):
        g.board |= 1 << c              # row 0 minus first cell
    for r in range(1, 8):
        g.board |= 1 << (r * 8)        # col 0 minus first cell
    g.pieces = (BY_NAME["single"].id, -1, -1)
    reward, done = g.step((0, 0, 0))
    assert g.board == 0
    assert reward == 1 + 10 * 2 * 2    # 1 cell + 2 lines with multi-line bonus


def test_combo_streak_multiplies():
    from simulator import ScoringConfig

    g = Game(seed=1, scoring=ScoringConfig())
    g.streak = 2
    g.board = 0
    for c in range(1, 8):
        g.board |= 1 << c
    g.pieces = (BY_NAME["single"].id, -1, -1)
    reward, _ = g.step((0, 0, 0))
    assert reward == 1 + 10 * 3        # streak (2) + this round (1) = x3


def test_no_rotation_l_shape_fixed():
    g = Game(seed=1)
    l4 = BY_NAME["l4_a"]              # "#./#./##" — foot points right only
    assert (0, 0) in PLACEMENTS[l4.id]
    # at (0,0): occupies (0,0) (1,0) (2,0) (2,1); a rotated variant must NOT exist
    mask = PLACEMENTS[l4.id][(0, 0)]
    bits = {(r, c) for r in range(8) for c in range(8) if mask >> (r * 8 + c) & 1}
    assert bits == {(0, 0), (1, 0), (2, 0), (2, 1)}


def test_refill_after_three_placements():
    g = Game(seed=1)
    g.pieces = (BY_NAME["single"].id,) * 3
    for i, c in enumerate([0, 2, 4]):
        g.step((i, 0, c))
    assert all(p >= 0 for p in g.pieces)
    assert g.streak == 0               # no clears happened


def test_game_over_when_nothing_fits():
    g = Game(seed=1)
    g.board = (1 << 64) - 1            # completely full
    g.pieces = (BY_NAME["single"].id, -1, -1)
    # engine rechecks after step only, but has_legal_move is the same predicate
    assert not g.has_legal_move()


def test_deterministic_seed():
    a, b = Game(seed=42), Game(seed=42)
    seq_a, seq_b = [], []
    for g, out in ((a, seq_a), (b, seq_b)):
        while not g.done:
            acts = g.legal_actions()
            act = acts[0]
            out.append(act)
            g.step(act)
    assert seq_a == seq_b and a.score == b.score


def test_random_games_terminate():
    rng = random.Random(7)
    scores = []
    for _ in range(50):
        g = Game(seed=rng.randrange(1 << 30), spawner_mode="randomized")
        while not g.done:
            acts = g.legal_actions()
            g.step(acts[rng.randrange(len(acts))])
        scores.append(g.score)
    assert all(s > 0 for s in scores)
    assert max(scores) > 100           # sanity: some games went somewhere


def test_clone_is_deterministic_for_search():
    g = Game(seed=3)
    g.step(g.legal_actions()[0])
    c1, c2 = g.clone(), g.clone()
    assert c1.snapshot() == c2.snapshot()
