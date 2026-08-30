"""Kaggle fleet spawner: pushes worker kernels across multiple Kaggle accounts.

Each worker is a private script kernel with internet access that clones the
repo (foundation branch until Phase 1 merges), runs `swarm.worker`, and
writes parquet shards to /kaggle/working (kernel output). With an HF_TOKEN
Kaggle Secret added to the account, workers push shards straight to HF — the
kernel code below already honors HF_TOKEN/HF_DATASET when present.

    python -m swarm.kaggle_spawn account1 5     # spawn w0-w4 on account 1
    python -m swarm.kaggle_spawn account2 5 --first 5   # w5-w9 on account 2
    python -m swarm.kaggle_spawn status         # list kernels on all accounts
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

ACCOUNTS = {
    "account1": {"user": "amer38", "token": "KGAT_a68d572c53a44c864afa2126d8f50be2"},
    "account2": {"user": "amerameryou", "token": "KGAT_ba540944662660ab36a12fb5124bafab"},
}
BRANCH = os.environ.get("BLOCKBLAST_BRANCH", "foundation")
REPO_URL = f"https://github.com/amerameryou1-blip/Blockblast"

KERNEL_SCRIPT = r'''import os, subprocess, sys

REPO = "{REPO}"
BRANCH = "{BRANCH}"

if not os.path.exists("repo"):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, "repo"], check=True)
os.chdir("repo")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow", "huggingface_hub"], check=False)

env = dict(os.environ)
env.update(dict(
    WORKER_ID=str({WID}),
    LOCAL_DATA_DIR="/kaggle/working/data",
    SYNC_MINUTES="10",
    MAX_MINUTES="600",
    BEAM_WIDTH="8",
    EPSILON="0.05",
))
# If HF_TOKEN (Kaggle Secret) is set in the kernel environment, workers push
# shards to HF directly; otherwise they land in this kernel's output.
if os.environ.get("HF_TOKEN") and os.environ.get("HF_DATASET"):
    env.pop("LOCAL_DATA_DIR", None)
subprocess.run([sys.executable, "-m", "swarm.worker"], env=env)
print("worker done")
'''


def _set_token(token: str) -> None:
    kg = os.path.expanduser("~/.kaggle")
    os.makedirs(kg, exist_ok=True)
    path = os.path.join(kg, "access_token")
    with open(path, "w") as f:
        f.write(token)
    os.chmod(path, 0o600)


def _cli(argv: list[str]) -> int:
    from kaggle.cli import main

    old = sys.argv
    sys.argv = ["kaggle"] + argv
    try:
        main()
        return 0
    except SystemExit as e:
        return int(e.code or 0)
    finally:
        sys.argv = old


def _kernel_dir(user: str, slug: str, wid: int, gpu: bool) -> str:
    d = tempfile.mkdtemp(prefix=f"bb-{wid}-")
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    code = (KERNEL_SCRIPT
            .replace("{WID}", str(wid))
            .replace("{REPO}", REPO_URL)
            .replace("{BRANCH}", BRANCH))
    with open(os.path.join(d, "main.py"), "w") as f:
        f.write(code)
    return d


def push(account_key: str, wid: int, gpu: bool = False) -> None:
    acc = ACCOUNTS[account_key]
    slug = f"blockblast-w{wid}"
    _set_token(acc["token"])
    d = _kernel_dir(acc["user"], slug, wid, gpu)
    rc = _cli(["kernels", "push", "-p", d])
    print(f"[{account_key}] push {slug}: rc={rc}")


def status() -> None:
    for key, acc in ACCOUNTS.items():
        _set_token(acc["token"])
        print(f"=== {key} ({acc['user']}) ===")
        _cli(["kernels", "list", "--mine", "--page-size", "20"])


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] == "status":
        status()
        return
    account_key = args[0]
    count = int(args[1]) if len(args) > 1 else 5
    gpu = "--gpu" in args
    first = 0
    if "--first" in args:
        first = int(args[args.index("--first") + 1])
    for i in range(first, first + count):
        push(account_key, i, gpu)
        time.sleep(3)


if __name__ == "__main__":
    main()
