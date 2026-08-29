"""Gen-2 trainer (Kaggle GPU session): learn a value net from self-play shards.

Pulls the dataset snapshot from HF, trains a small CNN that maps
(board, offered pieces) -> log final score, pushes a versioned checkpoint
to the HF model repo. The beam-search brain will later call this net as its
evaluation function (plug-in replacement for the heuristic weights).

    %%bash
    export HF_TOKEN=... HF_DATASET=you/blockblast-selfplay HF_MODEL=you/blockblast-brain
    cd /kaggle/working/repo && pip install -q torch pandas pyarrow huggingface_hub
    python -m swarm.train
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET = os.environ.get("HF_DATASET", "")
HF_MODEL = os.environ.get("HF_MODEL", "")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "500000"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))


def build_tensors(rows):
    import torch

    boards = torch.zeros((len(rows), 4, 8, 8), dtype=torch.float32)
    target = torch.zeros((len(rows), 1), dtype=torch.float32)
    from simulator.engine import PLACEMENTS

    for i, row in enumerate(rows):
        b = int(row["board"])
        for c in range(64):
            if b >> c & 1:
                boards[i, 0, c // 8, c % 8] = 1.0
        for slot, key in ((0, "p0"), (1, "p1"), (2, "p2")):
            pid = int(row[key])
            if pid < 0:
                continue
            some = PLACEMENTS[pid][(0, 0)]  # shape alone, at origin
            for c in range(64):
                if some >> c & 1:
                    boards[i, 1 + slot, c // 8, c % 8] = 1.0
        target[i, 0] = float(row["final_score"])
    return boards, target.log1p()


class ValueNet:  # thin wrapper so torch import stays module-level optional
    @staticmethod
    def make():
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 64, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )


def main() -> None:
    import pandas as pd
    import torch
    from huggingface_hub import HfApi, snapshot_download

    if not (HF_TOKEN and HF_DATASET and HF_MODEL):
        raise SystemExit("set HF_TOKEN, HF_DATASET, HF_MODEL")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_MODEL, repo_type="model", exist_ok=True)

    snap = snapshot_download(HF_DATASET, repo_type="dataset", token=HF_TOKEN,
                             allow_patterns="shards/**")
    rows = fn = None
    import glob

    files = sorted(glob.glob(f"{snap}/shards/**/*.parquet", recursive=True))
    dfs = []
    total = 0
    for f in files:
        dfs.append(pd.read_parquet(f))
        total += len(dfs[-1])
        if total >= MAX_ROWS:
            break
    df = pd.concat(dfs, ignore_index=True).head(MAX_ROWS)
    print(f"training on {len(df)} turns from {len(files)} shards")

    x, y = build_tensors(df.to_dict("records"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ValueNet.make().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    lossf = torch.nn.MSELoss()

    n = len(x)
    idx = list(range(n))
    for epoch in range(EPOCHS):
        random.shuffle(idx)
        tot = 0.0
        for s in range(0, n, 512):
            j = torch.tensor(idx[s:s + 512])
            loss = lossf(model(x[j].to(device)), y[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"epoch {epoch}: mse={tot:.2f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    torch.save({"state_dict": model.state_dict(), "rows": n}, "checkpoint.pt")
    meta = {"rows": n, "epochs": EPOCHS, "final_mse": tot, "ts": stamp,
            "games_source": HF_DATASET}
    with open("metrics.json", "w") as f:
        json.dump(meta, f, indent=1)
    for name in ("checkpoint.pt", "metrics.json"):
        api.upload_file(path_or_fileobj=name, path_in_repo=name,
                        repo_id=HF_MODEL, commit_message=f"train {stamp} ({n} rows)")
    print("pushed checkpoint + metrics to", HF_MODEL)


if __name__ == "__main__":
    main()
