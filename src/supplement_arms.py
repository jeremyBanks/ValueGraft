"""Supplement pass: add amendment arms to existing results/raw files.

Arms added (amendments doc items 1-2):
  B-causal   : fresh prefill of system + TAIL + SUMMARY (text order matched to
               Arm C's causal order tail->summary). C vs B-causal is the clean
               mechanism contrast; C vs B is deployment-flavored only.
  E-wrongconv: Arm E-post a=1.0 but the old values come from a DIFFERENT
               conversation's cache at the same positions. If this helps any
               metric, that metric is contaminated (negative control).
  E-shuffled : Arm E-post a=1.0 with the correct conversation's old values
               permuted across aligned positions (within layer). Tests whether
               alignment matters at all (negative control).

Reconstruction: summaries are greedy temp-0, so rebuilding ArmSet reproduces
the identical summary; asserted against the stored summary_text. Existing raw
files are updated in place (new arms merged under cont.arms / probes.arms).
"""

import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from arms import arm_e_snapshot, build_b_messages, canonical_ids, render
from kvlib import (
    batched_teacher_forced,
    extend_cache,
    first_step_logits,
    greedy_generate,
    prefill,
    rebuild_cache,
    snapshot_cache,
)
from run_arms import MODEL, PROBE_MAX_TOKENS, ArmSet, clear
import os
RAW_DIR = os.environ.get("SC_SUPP_RAW", "results/raw")
SUPP_ARMS = set(os.environ.get("SC_SUPP_ARMS", "B-causal,E-shuffled,E-wrongconv").split(","))


def build_bcausal_messages(msgs, summary_text, tail_start_msg):
    note = (
        "[Context note] Earlier parts of this conversation were compacted. "
        "Summary of what came before:\n\n" + summary_text
    )
    return [
        msgs[0],
        *msgs[tail_start_msg:],
        {"role": "assistant", "content": note},
    ]


def shuffled_pairs(pairs, seed):
    rng = random.Random(seed)
    olds = [o for _, o in pairs]
    rng.shuffle(olds)
    return [(n, o) for (n, _), o in zip(pairs, olds)]


def wrongconv_pairs(pairs, other_len):
    """Map each new position to a pseudo-random position in the other
    conversation's cache (deterministic, in-range, sinks excluded)."""
    out = []
    for i, (n, _) in enumerate(pairs):
        out.append((n, 4 + (n * 7919 + i * 104729) % (other_len - 8)))
    return out


def supplement_conversation(model, tokenizer, conv, other_summary_snap,
                            other_len, raw):
    msgs = conv["messages"]
    tail_start_msg = conv["sections"]["middle_end_msg"]

    for mode in ("cont", "probes"):
        ctx = msgs[:-1] if mode == "cont" else msgs
        aset = ArmSet(model, tokenizer, ctx, tail_start_msg)
        stored = raw[mode]["stats"]["summary_text"]
        assert aset.summary["text"] == stored, \
            f"{conv['id']} {mode}: summary not reproduced; refusing to merge"

        variants = []
        # B-causal
        bc_msgs = build_bcausal_messages(ctx, aset.summary["text"], tail_start_msg)
        bc_ids = canonical_ids(tokenizer, bc_msgs)
        bc_cache, _ = prefill(model, bc_ids)
        bc_snap = snapshot_cache(bc_cache)
        if "B-causal" in SUPP_ARMS:
            variants.append(("B-causal", lambda s=bc_snap: rebuild_cache(s),
                             bc_ids, bc_msgs))
        # negative controls on E-post a=1
        sh_pairs = shuffled_pairs(aset.pairs, seed=hash(conv["id"]) & 0xFFFF)
        sh_snap = arm_e_snapshot(aset.b_snap(), aset.summary["snapshot"],
                                 sh_pairs, 1.0)
        if "E-shuffled" in SUPP_ARMS:
            variants.append(("E-shuffled", lambda s=sh_snap: rebuild_cache(s),
                             aset.b_ids, aset.b_msgs))
        wc_pairs = wrongconv_pairs(aset.pairs, other_len)
        wc_snap = arm_e_snapshot(aset.b_snap(), other_summary_snap,
                                 wc_pairs, 1.0)
        if "E-wrongconv" in SUPP_ARMS:
            variants.append(("E-wrongconv", lambda s=wc_snap: rebuild_cache(s),
                             aset.b_ids, aset.b_msgs))

        eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
        for name, mk, ctx_ids, render_msgs in variants:
            t0 = time.time()
            if mode == "cont":
                cont_ids = tokenizer.encode(msgs[-1]["content"],
                                            add_special_tokens=False)
                gp = render(tokenizer, render_msgs, True)
                if gp[: len(ctx_ids)] != ctx_ids:
                    raw["cont"]["arms"][name] = {"error": "render_prefix_mismatch"}
                    continue
                cache = mk()
                lps = batched_teacher_forced(
                    model, cache, gp[len(ctx_ids):] + cont_ids[:-1], cont_ids)
                raw["cont"]["arms"][name] = {
                    "mean_logprob": sum(lps) / len(lps), "logprobs": lps}
                print(f"    CONT {name}: {sum(lps)/len(lps):.4f} "
                      f"({time.time()-t0:.0f}s)")
            else:
                answers = {}
                for plant in conv["plants"]:
                    full = render(tokenizer, render_msgs +
                                  [{"role": "user", "content": plant["probe"]}],
                                  True)
                    assert full[: len(ctx_ids)] == ctx_ids
                    cache = mk()
                    extend_cache(model, cache, full[len(ctx_ids):-1])
                    logits = first_step_logits(model, cache, full[-1])
                    toks, _ = greedy_generate(model, cache, logits,
                                              PROBE_MAX_TOKENS, eos_ids)
                    answers[plant["id"]] = tokenizer.decode(toks).strip()
                    clear(cache)
                raw["probes"]["arms"][name] = answers
                print(f"    PROBE {name}: {len(answers)} probes "
                      f"({time.time()-t0:.0f}s)")
        clear(aset)
    return raw


def main():
    model, tokenizer = load(MODEL)
    convs = {p.stem: json.load(open(p))
             for p in sorted(Path("data/synthetic").glob("c*.json"))}
    ids = sorted(convs)
    only = set(sys.argv[1:])
    for i, cid in enumerate(ids):
        if only and cid not in only:
            continue
        rp = Path(RAW_DIR) / f"{cid}.json"
        if not rp.exists():
            print(f"{cid}: no raw results yet, skipping")
            continue
        raw = json.load(open(rp))
        if all(a in raw["cont"]["arms"] for a in SUPP_ARMS):
            print(f"{cid}: already supplemented")
            continue
        # wrong-conversation donor: next conversation (cyclic), probe-mode
        # summary snapshot as the value source
        donor = convs[ids[(i + 1) % len(ids)]]
        # donor values need no summary: a plain prefill of the donor
        # conversation supplies foreign values at arbitrary positions
        d_ids = canonical_ids(tokenizer, donor["messages"])
        d_cache, _ = prefill(model, d_ids)
        donor_snap = snapshot_cache(d_cache)
        donor_len = donor_snap[0][0].shape[2]
        print(f"== supplement {cid} (donor {donor['id']})")
        t0 = time.time()
        try:
            raw = supplement_conversation(model, tokenizer, convs[cid],
                                          donor_snap, donor_len, raw)
        except AssertionError as e:
            print(f"== {cid} FAILED: {e}")
            continue
        json.dump(raw, open(rp, "w"), indent=1, ensure_ascii=False)
        print(f"== {cid} supplemented in {time.time()-t0:.0f}s")
        clear(d_cache, donor_snap)


if __name__ == "__main__":
    main()
