"""Repair pass: B-causal CONT scores.

supplement_arms re-rendered the B-causal context with add_generation_prompt=
True, which moves the template's empty <think> block INTO the trailing
assistant note (it attaches to the assistant message after the last user
turn), breaking prefix extension — so CONT was skipped. The fix: append the
constant generation-prompt token suffix directly to the canonical context ids
(identical treatment to arms A/B).
"""

import json
import sys
import time
from pathlib import Path

from mlx_lm import load

sys.path.insert(0, "src")
from arms import canonical_ids, generate_summary, render
from kvlib import batched_teacher_forced, prefill
from run_arms import MODEL
from supplement_arms import build_bcausal_messages


def main():
    model, tokenizer = load(MODEL)
    for p in sorted(Path("results/raw").glob("c*.json")):
        raw = json.load(open(p))
        arms_d = raw["cont"]["arms"]
        if "B-causal" in arms_d and "mean_logprob" in arms_d["B-causal"]:
            print(f"{raw['id']}: already fixed")
            continue
        if "B-causal" not in arms_d:
            print(f"{raw['id']}: not yet supplemented, skipping")
            continue
        conv = json.load(open(f"data/synthetic/{raw['id']}.json"))
        msgs = conv["messages"]
        ctx = msgs[:-1]
        tail_start_msg = conv["sections"]["middle_end_msg"]
        t0 = time.time()
        summ = generate_summary(model, tokenizer, ctx)
        assert summ["text"] == raw["cont"]["stats"]["summary_text"], \
            f"{raw['id']}: summary not reproduced"
        ctx_ids = canonical_ids(tokenizer, ctx)
        gp_suffix = render(tokenizer, ctx, True)[len(ctx_ids):]

        bc_msgs = build_bcausal_messages(ctx, summ["text"], tail_start_msg)
        bc_ids = canonical_ids(tokenizer, bc_msgs)
        cache, _ = prefill(model, bc_ids)
        cont_ids = tokenizer.encode(msgs[-1]["content"], add_special_tokens=False)
        lps = batched_teacher_forced(
            model, cache, gp_suffix + cont_ids[:-1], cont_ids)
        arms_d["B-causal"] = {"mean_logprob": sum(lps) / len(lps),
                              "logprobs": lps}
        json.dump(raw, open(p, "w"), indent=1, ensure_ascii=False)
        print(f"{raw['id']}: B-causal CONT {sum(lps)/len(lps):.4f} "
              f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
