# Blockblast

Training an AI to crush Block Blast (the 8×8 block puzzle) — built entirely on free infrastructure:
Kaggle for compute, Hugging Face for storage, GitHub Actions for builds.

Read **[ARCHITECTURE.md](ARCHITECTURE.md)** first — it's the source of truth for the design.

## Layout

| Path | Purpose |
|---|---|
| `simulator/` | Exact rules engine (8×8 bitboard, ~40 piece shapes, combos) |
| `brain/` | Gen-1 beam search (heuristic); Gen-2 learned net plugs in later |
| `swarm/` | Kaggle self-play workers, HF aggregator, GPU trainer |
| `tests/` | Mechanics that must never break (`pytest tests/`) |

## Quickstart

```bash
pip install pytest          # engine needs zero deps; pytest only for tests
python -m pytest tests/ -q  # verify the rules
python -m simulator.cli     # play it yourself in the terminal
python -m brain.beam_search --games 10   # watch the Gen-1 brain benchmark
```

## Roadmap

1. ✅ Simulator + beam search + Kaggle swarm skeleton
2. ⬜ Calibrate scoring/piece distribution from real game recordings
3. ⬜ Eyes: Android screen capture → board state
4. ⬜ Hands: Accessibility drag-and-drop gestures
5. ⬜ Gen-2 trained value net replaces heuristic evaluation
6. ⬜ Content: "AI learns Block Blast" series