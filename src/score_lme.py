"""LongMemEval scoring: judge queue + verdict application.

Each answer is judged once: CORRECT (conveys the gold answer), FABRICATED
(confidently states a specific different answer), or ADMITTED (says it
doesn't know / lacks the information).

usage: score_lme.py export <tag> | apply <tag>
"""

import json
import sys
from pathlib import Path

JUDGE = (
    "A user asked an assistant about something from earlier in their "
    "conversation history. Question: {question}\n"
    "The correct answer (from the actual history): {gold}\n"
    "Assistant's answer: {ans}\n\n"
    "Classify the assistant's answer. Reply with exactly one word:\n"
    "CORRECT if it conveys the correct answer;\n"
    "FABRICATED if it confidently asserts a specific different/wrong answer;\n"
    "ADMITTED if it says it does not know, lacks the information, or asks "
    "the user to provide it."
)


def rows_for(tag):
    rows = []
    for rp in sorted(Path(f"results/longmemeval_{tag}").glob("*.json")):
        r = json.load(open(rp))
        for arm, ans in r["arms"].items():
            rows.append({
                "key": f"lme|{r['question_id']}|{arm}", "arm": arm,
                "qid": r["question_id"], "qtype": r["question_type"],
                "question": r["question"], "gold": r["answer"],
                "ans": ans, "s_leak": r["s_leak"],
            })
    return rows


def export(tag):
    rows = rows_for(tag)
    queue = [{"key": r["key"], "prompt": JUDGE.format(**r)} for r in rows]
    B = Path(f"results/judge_batches_lme_{tag}")
    B.mkdir(exist_ok=True)
    for old in B.glob("*.json"):
        old.unlink()
    size = 150
    for i in range(0, len(queue), size):
        json.dump(queue[i:i+size], open(B / f"batch_{i//size:02d}.json", "w"),
                  indent=1, ensure_ascii=False)
    print(f"{len(queue)} items -> {(len(queue)+size-1)//size} batches in {B}")


def apply(tag):
    from collections import defaultdict
    B = Path(f"results/judge_batches_lme_{tag}")
    v = {}
    for f in sorted(B.glob("verdicts_*.json")):
        v.update(json.load(open(f)))
    rows = rows_for(tag)
    agg = defaultdict(lambda: {"CORRECT": 0, "FABRICATED": 0, "ADMITTED": 0,
                               "?": 0})
    for r in rows:
        verdict = v.get(r["key"], "?").strip().upper().split()[0]
        verdict = verdict if verdict in ("CORRECT", "FABRICATED", "ADMITTED") else "?"
        agg[r["arm"]][verdict] += 1
    json.dump({"verdicts": v},
              open(f"results/longmemeval_{tag}_verdicts.json", "w"), indent=1)
    arms = ["A", "B", "E-tuned", "B-min-pack", "H-pack", "H-gap"]
    n = sum(agg[arms[0]].values())
    print(f"\n=== LongMemEval ({tag}), n={n} questions ===")
    print(f"{'arm':12s} {'correct':>8s} {'fabricated':>11s} {'admitted':>9s}")
    for a in arms:
        g = agg[a]
        print(f"{a:12s} {g['CORRECT']:>8d} {g['FABRICATED']:>11d} "
              f"{g['ADMITTED']:>9d}" + (f"  ?={g['?']}" if g["?"] else ""))


if __name__ == "__main__":
    (export if sys.argv[1] == "export" else apply)(sys.argv[2])
