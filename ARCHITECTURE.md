# Architecture

Full-stack plan for an AI that learns to play Block Blast well enough to break human-scale records,
with everything built to run on free infrastructure.

## The Game (verified)

- 8×8 grid, 64 cells. Three pieces offered per round; all three are placed before a new set of three.
- Pieces are fixed-orientation (no rotation). Families: single cell, bars (2–5), 2×2/3×3 squares,
  L/J, T, S/Z, large L (5-cell), plus/rectangle variants.
- A placement scores points per cell; completing full rows/columns clears them (multi-line and
  consecutive-round combos multiply points). Game over when none of the offered pieces fits.
- Input on the real app is drag-and-drop (press piece → slide → release).
- The exact scoring constants and piece spawn distribution are not public. Both are treated as
  *calibratable parameters* (`simulator/pieces.py` weights, `simulator/engine.py` scoring config),
  and the training pipeline randomizes spawn distributions so the policy is robust to the real one.

## Organs

### Brain (strategy)
- Lives purely on state arrays, never pixels. State = 8×8 bitboard + offered piece ids.
- Gen 1 — "Calculator": beam search over the three-piece round with a hand-tuned state evaluation
  (`brain/beam_search.py`). Verifies the simulator and sets the human-surpassing baseline.
- Gen 2 — "Learner": value/policy network (`swarm/train.py`) trained on self-play data; the beam
  bot acts as teacher and sparring partner.

### Eyes (screen reader) — phase 2 (Android)
- Screenshot (MediaProjection) → grid area detection → 8×8 cell occupancy + tray piece shapes →
  the same integer state the Brain consumes. Trained on synthetic renders + real recordings.

### Hands (drag-and-drop) — phase 2 (Android)
- Accessibility-service gestures: press tray piece center → curved waypoint path → hover → release.
- One-time visual offset calibration; board re-read after every placement so mistakes self-heal.

## Data & Compute

### Kaggle swarm (data factory)
- Up to 5 CPU sessions per account × 2 accounts. Sessions are hard-killed at 12 h, therefore every
  worker is *stateless*: it streams self-play games, writes them to uniquely-named parquet shards,
  and pushes to Hugging Face every ~15 minutes. Losing a session loses ≤15 min of one worker.
- 9 workers + 1 aggregator + 1 GPU trainer per week (P100, ~30 h/wk quota is far more than needed).

### Hugging Face (memory)
- Dataset repo `blockblast-selfplay`: append-only shard store. Shard path
  `shards/run={ts}/worker={id}/{seq}.parquet`. Nothing is ever overwritten; HF is git-backed.
- Model repo `blockblast-brain`: versioned checkpoints + `metrics.json` per training run.
- Aggregator maintains `manifest.json` (shard list, dedupe, per-worker stats, game benchmarks).

## Training loop

1. Workers self-play with the current best brain (Gen-1 first, then whatever checkpoint is latest)
   and log full trajectories.
2. Aggregator deduplicates shards and updates stats.
3. Weekly GPU session pulls the pinned dataset revision, trains the net, pushes a new checkpoint
   with metrics. Only checkpoints that beat the previous benchmark average become "latest".

## Repo map

```
simulator/   rules engine (pure ints, no deps) + terminal play
brain/       Gen-1 beam search + benchmarks
swarm/       Kaggle worker / aggregator / trainer
tests/       mechanics that must never break
.github/     CI
```

## Non-negotiable rules

1. Single-player/offline only. No interaction with other players' games, ever.
2. Every training artifact exists on Hugging Face; Kaggle sessions are disposable.
3. The simulator is the source of truth — scoring/piece changes happen there, never in callers.
4. Fast paths must stay dependency-light: the engine is pure-standard-library integers.
