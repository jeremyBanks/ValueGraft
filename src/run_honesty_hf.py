"""Honesty suite at bf16: fabrication vs admission under decoy + evicted-fact
probes, arms B / E-tuned / B-min-pack / H-pack.

Decoy probes ask about facts NEVER discussed (data/decoy_probes.json);
evicted-fact probes ask about facts that were discussed but evicted (from
each synthetic conversation's probes list, classes evicted-fact / ruled-out).
The interesting contrast (from 4-bit runs): packed arms admit far more and
fabricate far less than B. This runner replicates at 30B bf16.

Env: SC_HF_MODEL, SC_HON_TAG (default 30b_bf16). Output:
results/honesty_<tag>/<conv>.json  — {probe_id: {arm: answer}}
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from arms_common import (
    SUMMARY_REQUEST,
    build_alignment,
    build_b_messages,
    bmin_pack_ids,
    canonical_ids,
    message_token_starts,
    render_hf,
)
from arms_hf import (
    answer_hf,
    arm_h_pack_snapshot_hf,
    generate_summary_hf,
    hf_prefill_ids,
    rope_base,
)
from kvlib_hf import blend_values

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
TAG = os.environ.get("SC_HON_TAG", "30b_bf16")
E_ALPHA = float(os.environ.get("SC_E_ALPHA", "0.75"))


def probe_suffix(tokenizer, msgs, probe):
    full = render_hf(tokenizer, msgs + [{"role": "user", "content": probe}],
                     True)
    canon = canonical_ids(tokenizer, msgs, renderer=render_hf)
    assert full[: len(canon)] == canon
    return full[len(canon):]


_PACK_FRAME = {}


def packed_suffix(tokenizer, probe):
    if not _PACK_FRAME:
        m = [{"role": "system", "content": "x"}]
        a = render_hf(tokenizer,
                      m + [{"role": "user", "content": "\x00Q\x00"}], True)
        b = render_hf(tokenizer, m, False)
        sfx = a[len(b):]
        mark = tokenizer("\x00Q\x00", add_special_tokens=False).input_ids
        for i in range(len(sfx) - len(mark) + 1):
            if sfx[i:i + len(mark)] == mark:
                _PACK_FRAME["pre"] = sfx[:i]
                _PACK_FRAME["post"] = sfx[i + len(mark):]
                break
    return (_PACK_FRAME["pre"]
            + tokenizer(probe, add_special_tokens=False).input_ids
            + _PACK_FRAME["post"])


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    base = rope_base(model)

    decoys = json.load(open("data/decoy_probes.json"))
    by_conv = {}
    for d in decoys:
        by_conv.setdefault(d["conv"], []).append((d["id"], d["probe"]))
    outdir = Path(f"results/honesty_{TAG}")
    outdir.mkdir(parents=True, exist_ok=True)

    for p in sorted(Path("data/synthetic").glob("c*.json")):
        c = json.load(open(p))
        cid = c["id"]
        outfile = outdir / f"{cid}.json"
        if outfile.exists():
            print(f"{cid}: done"); continue
        probes = list(by_conv.get(cid, []))
        for pl in c.get("plants", []):
            if (pl.get("category") in ("evicted_fact", "ruled_out")
                    and not pl.get("contaminated_early")
                    and not pl.get("contaminated_tail")):
                probes.append((pl["id"], pl["probe"]))
        if not probes:
            continue
        t0 = time.time()
        msgs = c["messages"][:-1]
        tsm = c["sections"]["middle_end_msg"]
        ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
        starts = message_token_starts(tokenizer, ids, len(msgs))
        summary = generate_summary_hf(model, tokenizer, msgs,
                                      request=SUMMARY_REQUEST)
        b_msgs = build_b_messages(msgs, summary["text"], tsm)
        b_ids = canonical_ids(tokenizer, b_msgs, renderer=render_hf)
        b_starts = message_token_starts(tokenizer, b_ids, len(b_msgs))
        regions = [
            ((b_starts[2], len(b_ids)), (starts[tsm], summary["conv_end"])),
            ((b_starts[1], b_starts[2]),
             (summary["s_start"], summary["s_end"])),
        ]
        pairs = build_alignment(b_ids, summary["old_ids"],
                                set(tokenizer.all_special_ids), regions)
        out = {}
        b_snap, _ = hf_prefill_ids(model, b_ids)
        e_snap = blend_values(b_snap, summary["snapshot"], pairs, E_ALPHA)
        bmp = bmin_pack_ids(summary, ids)
        bmp_snap, _ = hf_prefill_ids(model, bmp)
        hp = arm_h_pack_snapshot_hf(summary, base)
        for pid, probe in probes:
            out[pid] = {
                "B": answer_hf(model, tokenizer, b_snap,
                               probe_suffix(tokenizer, b_msgs, probe),
                               len(b_ids)),
                "E-tuned": answer_hf(model, tokenizer, e_snap,
                                     probe_suffix(tokenizer, b_msgs, probe),
                                     len(b_ids)),
                "B-min-pack": answer_hf(model, tokenizer, bmp_snap,
                                        packed_suffix(tokenizer, probe),
                                        len(bmp)),
                "H-pack": answer_hf(model, tokenizer, hp,
                                    packed_suffix(tokenizer, probe),
                                    hp[0][0].shape[2]),
            }
        del b_snap, e_snap, bmp_snap, hp, summary
        torch.cuda.empty_cache()
        tmp = outfile.with_suffix(".tmp")
        json.dump({"conv": cid, "model": MODEL, "dtype": "bfloat16",
                   "answers": out}, open(tmp, "w"), indent=1,
                  ensure_ascii=False)
        tmp.rename(outfile)
        print(f"== {cid} done in {time.time()-t0:.0f}s "
              f"({len(probes)} probes)", flush=True)


if __name__ == "__main__":
    main()
