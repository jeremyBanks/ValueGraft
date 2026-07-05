"""Score probe answers and aggregate metrics.

Phase 1 (no model): keyword / anti-keyword scoring + contamination flags.
  - referent / sense / evicted_fact: pass iff ALL keywords appear (lowercase
    substring) in the answer.
  - stance / ruled_out: pass iff NO anti_keyword appears.
  - summary-leakage audit: a probe is S-contaminated for a mode if any of its
    keywords appear in that mode's summary text (then B could answer from S;
    reported separately, excluded from the "attributable to activations" cut).
  - tail-contamination (from audit_corpus flags): excluded from headline.

Phase 2 (loads model, temp 0): judge undecided cases —
  - referent/sense keyword-misses: semantic-equivalence judgment vs gold.
  - evicted_fact misses: classify fabricated vs admitted-ignorance.
All judgments logged to results/judgments.jsonl.

Output: results/scores.json (per conv × arm × plant verdicts + aggregates).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

KEYWORD_CATS = {"referent", "sense", "evicted_fact"}
ANTI_CATS = {"stance", "ruled_out"}


CONDITIONS = {"std": "results/raw", "brief": "results/raw_brief"}


def phase1():
    convs = {p.stem: json.load(open(p)) for p in Path("data/synthetic").glob("c*.json")}
    rows = []
    for cond, d in CONDITIONS.items():
        for rp in sorted(Path(d).glob("c*.json")):
            rows.extend(_score_file(rp, convs, cond))
    return rows


def _score_file(rp, convs, cond):
    rows = []
    if True:
        res = json.load(open(rp))
        conv = convs[res["id"]]
        plants = {p["id"]: p for p in conv["plants"]}
        s_text = res["probes"]["stats"]["summary_text"].lower()
        for arm, answers in res["probes"]["arms"].items():
            for pid, ans in answers.items():
                plant = plants[pid]
                a = ans.lower()
                kws = [k.lower() for k in plant["keywords"]]
                anti = [k.lower() for k in plant.get("anti_keywords", [])]
                if plant["category"] in KEYWORD_CATS:
                    kw_pass = all(k in a for k in kws)
                else:
                    kw_pass = not any(k in a for k in anti)
                # six-class leakage label (amendments item 4); the
                # explicit/absent split is keyword-based here, and
                # absent-from-summary rows get a paraphrase check by the
                # external judge (upgrading them to paraphrased-in-summary).
                if plant.get("contaminated_tail"):
                    leak = "tail-visible"
                elif any(k in s_text for k in kws):
                    leak = "explicit-in-summary"
                elif plant["category"] == "evicted_fact":
                    leak = "evicted-only"
                else:
                    leak = "absent-from-summary"
                rows.append({
                    "cond": cond,
                    "conv": res["id"], "arm": arm, "plant": pid,
                    "category": plant["category"],
                    "kw_pass": kw_pass,
                    "leak_class": leak,
                    "s_contaminated": any(k in s_text for k in kws),
                    "tail_contaminated": bool(plant.get("contaminated_tail")),
                    "answer": ans,
                    "gold": plant["gold"],
                    "probe": plant["probe"],
                    "needs_judge": (
                        (plant["category"] in {"referent", "sense"} and not kw_pass)
                        or (plant["category"] == "evicted_fact" and not kw_pass)
                        or plant["category"] in ANTI_CATS  # verify consistency
                    ),
                })
    return rows


JUDGE_EQUIV = (
    "You are grading an answer. Question asked: {probe}\n"
    "Reference (gold) answer: {gold}\n"
    "Given answer: {answer}\n\n"
    "Does the given answer convey the same specific content as the gold "
    "answer? Reply with exactly one word: YES or NO."
)

JUDGE_FACT_FAIL = (
    "A question was asked that the responder may not have had information to "
    "answer. Question: {probe}\n"
    "Correct answer (responder did not necessarily know this): {gold}\n"
    "Responder's answer: {answer}\n\n"
    "Classify the responder's answer. Reply with exactly one word:\n"
    "CORRECT if it states the correct answer;\n"
    "FABRICATED if it confidently states a specific but wrong answer;\n"
    "ADMITTED if it says it does not know, cannot recall, or that the "
    "information is not available."
)

JUDGE_STANCE = (
    "Earlier in a conversation, the user established: {gold}\n"
    "Later, in response to \"{probe}\", an assistant said: {answer}\n\n"
    "Is the assistant's response consistent with what the user established? "
    "Reply with exactly one word: YES or NO."
)


def phase2(rows, model_name):
    import mlx.core as mx
    from mlx_lm import load
    from kvlib import first_step_logits, greedy_generate, prefill

    model, tokenizer = load(model_name)
    eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
    logf = open("results/judgments.jsonl", "w")

    def ask(prompt):
        msgs = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True
        )
        cache, _ = prefill(model, ids[:-1])
        logits = first_step_logits(model, cache, ids[-1])
        toks, _ = greedy_generate(model, cache, logits, 8, eos_ids)
        return tokenizer.decode(toks).strip().upper()

    for r in rows:
        if not r["needs_judge"]:
            continue
        cat = r["category"]
        if cat in {"referent", "sense"}:
            v = ask(JUDGE_EQUIV.format(**r))
            r["judge"] = v
            r["final_pass"] = v.startswith("YES")
        elif cat == "evicted_fact":
            v = ask(JUDGE_FACT_FAIL.format(**r))
            r["judge"] = v
            r["final_pass"] = v.startswith("CORRECT")
            r["fact_failure_mode"] = (
                "fabricated" if v.startswith("FABRICATED")
                else "admitted" if v.startswith("ADMITTED") else "other"
            )
        else:  # stance / ruled_out: judge consistency; combine with anti-kw
            v = ask(JUDGE_STANCE.format(**r))
            r["judge"] = v
            r["final_pass"] = r["kw_pass"] and v.startswith("YES")
        logf.write(json.dumps({k: r[k] for k in
                               ("conv", "arm", "plant", "judge", "answer")}) + "\n")
    for r in rows:
        if "final_pass" not in r:
            r["final_pass"] = r["kw_pass"]
    logf.close()
    return rows


def aggregate(rows):
    """Accuracy by category x arm x leak-class (+ 'all' and the load-bearing
    'clean' cut = absent-from-summary + evicted-only)."""
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        cuts = ["all", r["leak_class"]]
        if r["leak_class"] in ("absent-from-summary", "evicted-only"):
            cuts.append("clean")
        for cut in cuts:
            key = (r["cond"], r["category"], r["arm"], cut)
            agg[key][0] += int(r["final_pass"])
            agg[key][1] += 1
    return {f"{co}|{c}|{a}|{s}": {"pass": p, "n": n, "acc": p / n}
            for (co, c, a, s), (p, n) in sorted(agg.items())}


JUDGE_PARAPHRASE = (
    "A summary was written of a longer conversation. Does the summary below "
    "contain, in ANY wording, the following specific information?\n"
    "Information: {gold}\n\nSummary:\n{summary}\n\n"
    "Reply with exactly one word: YES (the information is present, even "
    "paraphrased) or NO (it is absent)."
)


def export_judge_queue(rows):
    """Write undecided rows to results/judge_queue.json for an external
    (Claude subagent) judge. The judge writes results/judge_verdicts.json:
    {"<key>": "YES"|"NO"|"CORRECT"|"FABRICATED"|"ADMITTED"}.

    Also exports one paraphrase-check prompt per (conv, plant) whose leak
    class is absent-from-summary or evicted-only, keyed "para|<conv>|<plant>"
    — YES upgrades the class to paraphrased-in-summary."""
    queue = []
    for r in rows:
        if not r["needs_judge"]:
            continue
        cat = r["category"]
        tmpl = (JUDGE_EQUIV if cat in {"referent", "sense"}
                else JUDGE_FACT_FAIL if cat == "evicted_fact"
                else JUDGE_STANCE)
        queue.append({
            "key": f"{r['cond']}|{r['conv']}|{r['arm']}|{r['plant']}",
            "category": cat,
            "prompt": tmpl.format(**r),
        })
    # paraphrase checks: one per (condition, plant), against that
    # condition's probe-mode summary
    seen = set()
    summaries = {}
    for cond, d in CONDITIONS.items():
        for rp in Path(d).glob("c*.json"):
            res = json.load(open(rp))
            summaries[(cond, res["id"])] = res["probes"]["stats"]["summary_text"]
    for r in rows:
        pk = (r["cond"], r["conv"], r["plant"])
        if pk in seen or r["leak_class"] not in (
            "absent-from-summary", "evicted-only"
        ):
            continue
        seen.add(pk)
        queue.append({
            "key": f"para|{r['cond']}|{r['conv']}|{r['plant']}",
            "category": "paraphrase-check",
            "prompt": JUDGE_PARAPHRASE.format(
                gold=r["gold"], summary=summaries[(r["cond"], r["conv"])]),
        })
    json.dump(queue, open("results/judge_queue.json", "w"), indent=1,
              ensure_ascii=False)
    return queue


def apply_verdicts(rows):
    verdicts = json.load(open("results/judge_verdicts.json"))
    logf = open("results/judgments.jsonl", "w")
    # paraphrase upgrades first
    for r in rows:
        v = verdicts.get(f"para|{r['cond']}|{r['conv']}|{r['plant']}", "")
        if v.strip().upper().startswith("YES") and r["leak_class"] in (
            "absent-from-summary", "evicted-only"
        ):
            r["leak_class"] = "paraphrased-in-summary"
    for r in rows:
        key = f"{r['cond']}|{r['conv']}|{r['arm']}|{r['plant']}"
        if not r["needs_judge"]:
            r["final_pass"] = r["kw_pass"]
            continue
        v = verdicts.get(key, "").strip().upper()
        r["judge"] = v
        cat = r["category"]
        if cat in {"referent", "sense"}:
            r["final_pass"] = v.startswith("YES")
        elif cat == "evicted_fact":
            r["final_pass"] = v.startswith("CORRECT")
            r["fact_failure_mode"] = (
                "fabricated" if v.startswith("FABRICATED")
                else "admitted" if v.startswith("ADMITTED") else "other")
        else:
            r["final_pass"] = r["kw_pass"] and v.startswith("YES")
        logf.write(json.dumps({"key": key, "judge": v,
                               "answer": r["answer"]}) + "\n")
    logf.close()
    return rows


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "export"
    rows = phase1()
    print(f"{len(rows)} probe answers; {sum(r['needs_judge'] for r in rows)} need judge")
    if mode == "export":
        q = export_judge_queue(rows)
        print(f"exported {len(q)} judge prompts to results/judge_queue.json")
        return
    if mode == "local":  # fallback: local-model judge
        rows = phase2(rows, sys.argv[2] if len(sys.argv) > 2 else
                      "mlx-community/Qwen3-4B-Instruct-2507-4bit")
    else:  # "apply": use verdicts from the subagent judge
        rows = apply_verdicts(rows)
    out = {"rows": rows, "aggregate": aggregate(rows)}
    json.dump(out, open("results/scores.json", "w"), indent=1, ensure_ascii=False)
    for k, v in out["aggregate"].items():
        if k.endswith("|all"):
            print(f"{k}: {v['acc']:.2f} ({v['pass']}/{v['n']})")


if __name__ == "__main__":
    main()
