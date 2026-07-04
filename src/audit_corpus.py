"""Audit composed conversations for plant contamination.

For each plant, keywords must appear in its own middle_user message (and
optionally in middle assistant replies — the middle is evicted, that's fine)
and must NOT appear anywhere in the early section or the tail section
(retained text would make the probe answerable by every arm).

Writes an audit report and annotates each conversation JSON in place with
per-plant `contaminated_early` / `contaminated_tail` flags.
"""

import json
import sys
from pathlib import Path


def find_hits(needle, messages, lo, hi):
    hits = []
    for i in range(lo, hi):
        if needle.lower() in messages[i]["content"].lower():
            hits.append(i)
    return hits


def audit_file(path):
    conv = json.load(open(path))
    msgs = conv["messages"]
    e, m = conv["sections"]["early_end_msg"], conv["sections"]["middle_end_msg"]
    report = []
    for plant in conv["plants"]:
        early_bad, tail_bad = [], []
        in_middle = False
        for kw in plant["keywords"]:
            early_bad += [(kw, i) for i in find_hits(kw, msgs, 0, e)]
            tail_bad += [(kw, i) for i in find_hits(kw, msgs, m, len(msgs))]
            if find_hits(kw, msgs, e, m):
                in_middle = True
        plant["contaminated_early"] = sorted({i for _, i in early_bad})
        plant["contaminated_tail"] = sorted({i for _, i in tail_bad})
        plant["keywords_present_in_middle"] = in_middle
        if early_bad or tail_bad or not in_middle:
            report.append((plant["id"], early_bad, tail_bad, in_middle))
    json.dump(conv, open(path, "w"), indent=1, ensure_ascii=False)
    return conv, report


def main():
    paths = sorted(Path("data/synthetic").glob("c*.json"))
    total = clean = 0
    for p in paths:
        conv, report = audit_file(p)
        total += len(conv["plants"])
        clean += len(conv["plants"]) - len(report)
        for pid, eb, tb, im in report:
            print(f"{p.name} {pid}: early={eb} tail={tb} in_middle={im}")
    print(f"\n{clean}/{total} plants clean across {len(paths)} conversations")


if __name__ == "__main__":
    main()
