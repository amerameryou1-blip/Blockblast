"""Kaggle aggregator session (the "+1").

Scans the HF dataset repo, deduplicates shards by path, merges per-shard
meta JSON, and publishes manifest.json + stats.json. Run on a cron-ish loop;
safe to kill at any moment (everything is re-derivable).

    %%bash
    export HF_TOKEN=... HF_DATASET=you/blockblast-selfplay
    cd /kaggle/working/repo && pip install -q huggingface_hub
    python -m swarm.aggregator
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET = os.environ.get("HF_DATASET", "")
INTERVAL_MINUTES = float(os.environ.get("INTERVAL_MINUTES", "20"))


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    if not HF_TOKEN or not HF_DATASET:
        raise SystemExit("set HF_TOKEN and HF_DATASET")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_DATASET, repo_type="dataset", exist_ok=True)

    while True:
        files = api.list_repo_files(HF_DATASET, repo_type="dataset")
        shards = sorted(f for f in files if f.startswith("shards/") and f.endswith(".parquet"))
        metas = sorted(f for f in files if f.startswith("meta/") and f.endswith(".json"))

        workers: dict[str, dict] = {}
        games = rows = 0
        for m in metas:
            try:
                lp = hf_hub_download(HF_DATASET, m, repo_type="dataset", token=HF_TOKEN)
                with open(lp) as f:
                    d = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            games += d.get("games", 0)
            rows += d.get("rows", 0)
            w = workers.setdefault(d["worker"], {"runs": set(), "games": 0, "rows": 0})
            w["runs"].add(d["run"])
            w["games"] += d.get("games", 0)
            w["rows"] += d.get("rows", 0)

        manifest = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "n_shards": len(shards),
            "shards": shards,
        }
        stats = {
            "updated": manifest["updated"],
            "games_total": games,
            "rows_total": rows,
            "workers": {
                w: {"runs": len(v["runs"]), "games": v["games"], "rows": v["rows"]}
                for w, v in sorted(workers.items())
            },
        }
        for payload, name in ((manifest, "manifest.json"), (stats, "stats.json")):
            with open(name, "w") as f:
                json.dump(payload, f, indent=1)
            api.upload_file(
                path_or_fileobj=name,
                path_in_repo=name,
                repo_id=HF_DATASET,
                repo_type="dataset",
                commit_message=f"aggregator: {name} ({games} games, {len(shards)} shards)",
            )
        print(f"[aggregator] {len(shards)} shards, {games} games, {rows} rows")
        time.sleep(INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
