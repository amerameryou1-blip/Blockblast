"""Kaggle CPU worker: plays self-play games and pushes data shards to Hugging Face.

Designed to be *disposable*: Kaggle kills sessions at ~12h, so the worker
streams data to HF every SYNC_MINUTES minutes and can die at any moment
losing at most one sync window of games.

Kaggle notebook cell:

    %%bash
    export WORKER_ID=0 HF_TOKEN=... HF_DATASET=you/blockblast-selfplay
    cd /kaggle/working && git clone https://github.com/you/Blockblast repo
    cd repo && pip install -q pandas pyarrow huggingface_hub
    python -m swarm.worker

Env:
  WORKER_ID      0..8 (unique across both Kaggle accounts)
  HF_TOKEN       Hugging Face write token
  HF_DATASET     dataset repo id, e.g. "you/blockblast-selfplay"
  SYNC_MINUTES   upload cadence (default 15)
  MAX_MINUTES    self-terminate before Kaggle's 12h kill (default 690 = 11.5h)
  BEAM_WIDTH     beam size for self-play (default 12, fast)
  EPSILON        random-move exploration rate for state diversity (default 0.05)
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone

from brain.beam_search import BeamBrain
from simulator.engine import Game

WORKER_ID = os.environ.get("WORKER_ID", "0")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET = os.environ.get("HF_DATASET", "")
SYNC_MINUTES = float(os.environ.get("SYNC_MINUTES", "15"))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "690"))
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "12"))
EPSILON = float(os.environ.get("EPSILON", "0.05"))

BUF_LIMIT = 50_000  # rows


def _hf():
    from huggingface_hub import HfApi

    return HfApi(token=HF_TOKEN)


def _push(api, local_path: str, repo_path: str) -> None:
    for attempt in range(3):
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=HF_DATASET,
                repo_type="dataset",
                commit_message=f"worker {WORKER_ID}: {repo_path}",
            )
            return
        except Exception as e:  # noqa: BLE001 — network on Kaggle is flaky; retry
            print(f"  upload retry {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to upload {repo_path}")


def main() -> None:
    import pandas as pd

    if not HF_TOKEN or not HF_DATASET:
        raise SystemExit("set HF_TOKEN and HF_DATASET")

    api = _hf()
    api.create_repo(HF_DATASET, repo_type="dataset", exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rng = random.Random(hash((WORKER_ID, run_id)) & 0xFFFFFFFF)
    brain = BeamBrain(width=BEAM_WIDTH, depth=3)

    os.makedirs("out", exist_ok=True)
    buf: list[dict] = []
    seq = 0
    games = 0
    score_hist: list[int] = []
    t0 = time.monotonic()
    last_sync = t0
    print(f"[worker {WORKER_ID}] run={run_id} beam={BEAM_WIDTH} eps={EPSILON}")

    def flush() -> None:
        nonlocal seq, buf, games, score_hist
        if not buf:
            return
        df = pd.DataFrame(buf)
        shard = f"shards/run={run_id}/worker={WORKER_ID}/{seq:05d}.parquet"
        lp = f"out/{seq:05d}.parquet"
        df.to_parquet(lp, index=False)
        _push(api, lp, shard)
        meta = {
            "worker": WORKER_ID, "run": run_id, "seq": seq,
            "shard": shard, "rows": len(df), "games": len(score_hist),
            "score_mean": sum(score_hist) / max(1, len(score_hist)),
            "score_max": max(score_hist, default=0),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        mp = f"out/{seq:05d}.json"
        with open(mp, "w") as f:
            json.dump(meta, f)
        _push(api, mp, f"meta/worker={WORKER_ID}/{run_id}_{seq:05d}.json")
        print(f"[worker {WORKER_ID}] pushed shard {seq} "
              f"({len(df)} rows, {games} games total)")
        os.remove(lp)
        os.remove(mp)
        seq += 1
        buf = []
        score_hist = []

    while True:
        g = Game(seed=rng.randrange(1 << 31), spawner_mode="randomized")
        game_id = f"{WORKER_ID}-{run_id}-{games}"
        rows_this: list[dict] = []
        while not g.done:
            acts = g.legal_actions()
            act = rng.choice(acts) if rng.random() < EPSILON else brain.choose(g)
            turn = g.turns
            reward, done = g.step(act)
            rows_this.append({
                "game_id": game_id, "turn": turn, "board": g.board,
                "p0": g.pieces[0], "p1": g.pieces[1], "p2": g.pieces[2],
                "slot": act[0], "ar": act[1], "ac": act[2],
                "reward": reward, "done": done,
            })
        for row in rows_this:  # attach outcome label for supervised targets
            row["final_score"] = g.score
        buf.extend(rows_this)
        score_hist.append(g.score)
        games += 1

        now = time.monotonic()
        if (now - last_sync) >= SYNC_MINUTES * 60 or len(buf) >= BUF_LIMIT:
            flush()
            last_sync = now
        if (now - t0) >= MAX_MINUTES * 60:
            flush()
            print(f"[worker {WORKER_ID}] done: {games} games in "
                  f"{(now - t0) / 3600:.2f}h")
            return


if __name__ == "__main__":
    main()
