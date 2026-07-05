"""Split results/judge_queue.json into batch files for external judging.

Usage: judge_batches.py split [batch_size]   -> results/judge_batches/batch_NN.json
       judge_batches.py merge                -> results/judge_verdicts.json

Each batch file is a JSON array of {key, prompt}. A judge writes
results/judge_batches/verdicts_NN.json as {key: verdict} for its batch.
merge concatenates all verdict files (later files win on key collisions).
"""

import json
import sys
from pathlib import Path

import os
BDIR = Path(os.environ.get("SC_JUDGE_BDIR", "results/judge_batches"))
QUEUE = os.environ.get("SC_JUDGE_QUEUE", "results/judge_queue.json")
VERDICTS = os.environ.get("SC_JUDGE_VERDICTS", "results/judge_verdicts.json")


def split(batch_size=150):
    queue = json.load(open(QUEUE))
    BDIR.mkdir(parents=True, exist_ok=True)
    for old in BDIR.glob("batch_*.json"):
        old.unlink()
    for i in range(0, len(queue), batch_size):
        n = i // batch_size
        items = [{"key": q["key"], "prompt": q["prompt"]}
                 for q in queue[i : i + batch_size]]
        json.dump(items, open(BDIR / f"batch_{n:02d}.json", "w"), indent=1,
                  ensure_ascii=False)
    n_batches = (len(queue) + batch_size - 1) // batch_size
    print(f"{len(queue)} items -> {n_batches} batches in {BDIR}")


def merge():
    verdicts = {}
    for vf in sorted(BDIR.glob("verdicts_*.json")):
        verdicts.update(json.load(open(vf)))
    json.dump(verdicts, open(VERDICTS, "w"), indent=1)
    queue = json.load(open(QUEUE))
    missing = [q["key"] for q in queue if q["key"] not in verdicts]
    print(f"{len(verdicts)} verdicts merged; {len(missing)} missing")
    if missing:
        print("missing keys (first 10):", missing[:10])


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split(int(sys.argv[2]) if len(sys.argv) > 2 else 150)
    else:
        merge()
