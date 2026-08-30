# Results log

Chronological experiment log. Numbers from fixed matches (same seeds for all
entrants) unless stated otherwise.

## 2026-08-29 — Phase 1 baseline

| Entrant | Width×Depth | Mean | Min | Max | Note |
|---|---|---|---|---|---|
| Gen-1 heuristic (original weights) | 10×2 | 4,348 | — | 13,395 | first bench |
| Gen-1 heuristic (v0.1 piece weights) | 8×2 | **35,594** | 1,991 | **198,181** | killer pieces downweighted |
| Value net v1 (log1p final score) | 8×1 | 61 | 25 | 157 | mean-collapse |
| Value net v2 (log1p remaining) | 8×1 | 219 | 28 | 967 | still mean-collapse |

### Piece analysis (3,000 random-policy games)
Deadliest shapes by "present in hand at game over" rate: `bar5v` 68.6%,
`bar5h` 66.0%, `l5_en` 67.7%, `t4_down` 61.8%, `t4_left` 59.6%, `bar4v` 58.7%.
Safest: `sq3` 29.7%, `rect32` 29.8%, `plus` 30.5% (rare but survivable when
space exists), `j4_c` 29.5%.

v0.1 weight changes: `bar5*` 0.35→0.22, `l5_*` 0.35→0.28, `sq3` 0.30→0.25,
`plus` 0.25→0.20, `tri_l_*` 0.8→0.9. Effect: heuristic mean ×8.

### Learnings
1. Value regression on final score regresses to the mean — the immediate
   outcome is luck-dominated relative to a single state's signal. Gen-2 moves
   to **policy distillation** (imitate beam choices), which has per-decision
   supervision instead of per-game supervision.
2. The heuristic + beam is already strong enough to headline content;
   architecture supports dropping in any `eval_fn` mid-search.
