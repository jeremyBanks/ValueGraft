"""Repair tail contamination: assistant tail replies that restate plant
keywords make those probes trivially answerable from retained text.

For each contaminated conversation: replay the conversation through a
ConversationBuilder (verbatim user turns; recorded assistant replies re-used
for clean messages), and for each leaking tail assistant reply, resample with
fresh seeds up to MAX_TRIES; if still leaking, truncate the reply at the last
sentence boundary before the first leaked keyword. Then re-audit.

Only tail leaks matter: the early section is evicted along with the middle in
every compacted arm, so early leaks cannot make a probe answerable from
retained text (they are logged but left alone).
"""

import json
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from compose import MODEL, ConversationBuilder, render
from kvlib import extend_cache, sampled_generate, truncate_cache

MAX_TRIES = 6


def tail_keywords(conv):
    """keyword -> plant ids, for all plants (tail must be clean of ALL)."""
    kws = {}
    for p in conv["plants"]:
        for k in p["keywords"]:
            kws.setdefault(k.lower(), []).append(p["id"])
    return kws


def leaks(text, kws):
    t = text.lower()
    return [k for k in kws if k in t]


def truncate_before_leak(text, kws):
    """Cut at the last sentence boundary before the first leaked keyword."""
    t = text.lower()
    first = min(t.find(k) for k in leaks(text, kws))
    cut_region = text[:first]
    m = max(cut_region.rfind(". "), cut_region.rfind("! "),
            cut_region.rfind("? "), cut_region.rfind("\n\n"))
    if m < 40:  # too aggressive; keep one clause and add a period
        return None
    return text[: m + 1].rstrip()


def rebuild_conv(model, tokenizer, conv, kws):
    msgs = conv["messages"]
    tail_start = conv["sections"]["middle_end_msg"]
    base_seed = conv["meta"]["seed"]
    b = ConversationBuilder(model, tokenizer, msgs[0]["content"])
    changed = 0
    i = 1
    while i < len(msgs):
        user = msgs[i]["content"]
        orig_reply = msgs[i + 1]["content"]
        in_tail = (i + 1) >= tail_start
        if not in_tail or not leaks(orig_reply, kws):
            b.replay_turn(user, orig_reply)
        else:
            reply = None
            for t in range(MAX_TRIES):
                mx.random.seed(base_seed * 977 + i * 31 + t)
                cand = b.peek_reply(user)
                if not leaks(cand, kws):
                    reply = cand
                    break
            if reply is None:
                reply = truncate_before_leak(orig_reply, kws)
                if reply is None:
                    # give up: regenerate once more and hard-truncate
                    reply = "Sounds good — let's keep moving on that basis."
            b.commit_turn(user, reply)
            changed += 1
        i += 2
    conv["messages"] = b.msgs
    return changed


def main():
    model, tokenizer = load(MODEL)
    for path in sorted(Path("data/synthetic").glob("c*.json")):
        conv = json.load(open(path))
        kws = tail_keywords(conv)
        tail_start = conv["sections"]["middle_end_msg"]
        dirty = [i for i in range(tail_start, len(conv["messages"]))
                 if conv["messages"][i]["role"] == "assistant"
                 and leaks(conv["messages"][i]["content"], kws)]
        if not dirty:
            print(f"{path.stem}: tail clean")
            continue
        n = rebuild_conv(model, tokenizer, conv, kws)
        # recompute token counts
        ids = render(tokenizer, conv["messages"], gen_prompt=False)
        conv["sections"]["total_tokens"] = len(ids)
        conv["meta"]["tail_repaired"] = n
        json.dump(conv, open(path, "w"), indent=1, ensure_ascii=False)
        print(f"{path.stem}: repaired {n} tail replies")


if __name__ == "__main__":
    main()
