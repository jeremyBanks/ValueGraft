"""Phase 2 runner: H-pack vs B-min-pack fabrication experiment.

Per conversation (brief/terse summary condition only):
  arms: A (oracle), B (text compaction), B-min-pack (packed summary text,
  fresh encode), H-pack (packed summary write-time entries), H-pack-wrongS
  (donor conversation's summary entries — content control), H-gap (link to
  the pilot).
  items: 2 evicted-fact plants + 2 decoy probes (never-discussed specifics;
  gold = admit) + 2 referent + 2 sense plants = 8 greedy probe answers/arm.

HP-0 identity (first conversation): H-pack built with pack_offset = original
s_start must produce byte-identical retained arrays to H-gap's.

Output: results/phase2_<tag>/<cid>.json
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from arms import (
    SUMMARY_REQUEST_BRIEF,
    arm_c_snapshot,
    arm_h_pack_snapshot,
    bmin_pack_ids,
    build_b_messages,
    canonical_ids,
    gapped_cache_from,
    generate_summary,
    render,
)
from kvlib import (
    extend_cache,
    first_step_logits,
    greedy_generate,
    max_abs_diff,
    prefill,
    rebuild_cache,
    snapshot_cache,
)

MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
TAG = os.environ.get("SC_P2_TAG", "4b")
PROBE_MAX_TOKENS = 160


def probe_suffix_ids(tokenizer, msgs, probe_text):
    """Token suffix for appending a probe turn: verified against the real
    template once per process, then built by direct encoding."""
    full = render(tokenizer, msgs + [{"role": "user", "content": probe_text}], True)
    canon = canonical_ids(tokenizer, msgs)
    assert full[: len(canon)] == canon
    return full[len(canon):]


_SUFFIX_TEMPLATE = {}


def packed_probe_suffix(tokenizer, probe_text):
    """Probe suffix for packed arms (no renderable message list). Uses the
    same token pattern as the real template's user-turn suffix."""
    if "pre" not in _SUFFIX_TEMPLATE:
        # derive the frame once from a dummy render
        msgs = [{"role": "system", "content": "x"}]
        a = render(tokenizer, msgs + [{"role": "user", "content": "\x00Q\x00"}], True)
        b = render(tokenizer, msgs, False)
        suffix = a[len(b):]
        marker = tokenizer.encode("\x00Q\x00", add_special_tokens=False)
        # find marker inside suffix
        for i in range(len(suffix) - len(marker) + 1):
            if suffix[i : i + len(marker)] == marker:
                _SUFFIX_TEMPLATE["pre"] = suffix[:i]
                _SUFFIX_TEMPLATE["post"] = suffix[i + len(marker):]
                break
        assert "pre" in _SUFFIX_TEMPLATE, "marker not found in rendered suffix"
    return (_SUFFIX_TEMPLATE["pre"]
            + tokenizer.encode(probe_text, add_special_tokens=False)
            + _SUFFIX_TEMPLATE["post"])


def answer(model, tokenizer, cache, suffix, eos_ids):
    if len(suffix) > 1:
        extend_cache(model, cache, suffix[:-1])
    logits = first_step_logits(model, cache, suffix[-1])
    toks, _ = greedy_generate(model, cache, logits, PROBE_MAX_TOKENS, eos_ids)
    return tokenizer.decode(toks).strip()


def run_conv(model, tokenizer, conv, donor_summary, items, hp0_check=False):
    msgs = conv["messages"]
    tail_start_msg = conv["sections"]["middle_end_msg"]
    eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
    rope_base = model.args.rope_theta

    summary = generate_summary(model, tokenizer, msgs,
                               request=SUMMARY_REQUEST_BRIEF)
    conv_ids = canonical_ids(tokenizer, msgs)
    b_msgs = build_b_messages(msgs, summary["text"], tail_start_msg)
    b_ids = canonical_ids(tokenizer, b_msgs)

    if hp0_check:
        hp0 = arm_h_pack_snapshot(summary, rope_base,
                                  pack_offset=summary["s_start"])
        hg = arm_c_snapshot(summary, summary["conv_end"], True)
        d = max(max_abs_diff(a[0], b[0]) + max_abs_diff(a[1], b[1])
                for a, b in zip(hp0, hg))
        print(f"    HP-0 (pack at original offset == H-gap arrays): "
              f"max|diff| = {d}")
        assert d == 0.0, "HP-0 identity failed"

    a_snap = snapshot_cache(prefill(model, conv_ids)[0])
    b_snap = snapshot_cache(prefill(model, b_ids)[0])
    bmp_ids = bmin_pack_ids(summary, conv_ids)
    bmp_snap = snapshot_cache(prefill(model, bmp_ids)[0])
    hp_snap = arm_h_pack_snapshot(summary, rope_base)
    wrong_snap = None
    if donor_summary is not None:
        # donor S entries packed after THIS conversation's sinks
        donor_pack = arm_h_pack_snapshot(donor_summary, rope_base)
        wrong_snap = []
        for (dk, dv, doff), (k, v, off) in zip(donor_pack, a_snap):
            from arms import N_SINK
            pk = mx.concatenate([k[..., :N_SINK, :], dk[..., N_SINK:, :]], axis=2)
            pv = mx.concatenate([v[..., :N_SINK, :], dv[..., N_SINK:, :]], axis=2)
            wrong_snap.append((pk, pv, doff))

    out = {"summary_text": summary["text"],
           "summary_tokens": len(summary["gen_ids"]), "arms": {}}
    arm_defs = [
        ("A", lambda: rebuild_cache(a_snap), "render", msgs),
        ("B", lambda: rebuild_cache(b_snap), "render", b_msgs),
        ("B-min-pack", lambda: rebuild_cache(bmp_snap), "packed", None),
        ("H-pack", lambda: rebuild_cache(hp_snap), "packed", None),
        ("H-gap", lambda: gapped_cache_from(
            arm_c_snapshot(summary, summary["conv_end"], True)), "render", msgs),
    ]
    if wrong_snap is not None:
        arm_defs.append(
            ("H-pack-wrongS", lambda: rebuild_cache(wrong_snap), "packed", None))

    for name, mk, mode, rmsgs in arm_defs:
        t0 = time.time()
        answers = {}
        for item in items:
            if mode == "render":
                sfx = probe_suffix_ids(tokenizer, rmsgs, item["probe"])
            else:
                sfx = packed_probe_suffix(tokenizer, item["probe"])
            answers[item["id"]] = answer(model, tokenizer, mk(), sfx, eos_ids)
        out["arms"][name] = answers
        print(f"    {name}: {len(answers)} probes ({time.time()-t0:.0f}s)")
    return out


def main():
    model, tokenizer = load(MODEL)
    outdir = Path(f"results/phase2_{TAG}")
    outdir.mkdir(parents=True, exist_ok=True)
    convs = {p.stem: json.load(open(p))
             for p in sorted(Path("data/synthetic").glob("c*.json"))}
    decoys = json.load(open("data/decoy_probes.json"))
    ids = sorted(convs)
    only = set(sys.argv[1:])
    donor_cache = {}

    for i, cid in enumerate(ids):
        if only and cid not in only:
            continue
        outfile = outdir / f"{cid}.json"
        if outfile.exists():
            print(f"{cid}: done, skipping")
            continue
        conv = convs[cid]
        items = [
            {"id": p["id"], "probe": p["probe"], "kind": p["category"]}
            for p in conv["plants"]
            if p["category"] in ("evicted_fact", "referent", "sense")
        ] + [
            {"id": d["id"], "probe": d["probe"], "kind": "decoy"}
            for d in decoys if d["conv"] == cid
        ]
        donor_id = ids[(i + 1) % len(ids)]
        if donor_id not in donor_cache:
            donor_cache.clear()  # keep at most one donor summary in memory
            donor_cache[donor_id] = generate_summary(
                model, tokenizer, convs[donor_id]["messages"],
                request=SUMMARY_REQUEST_BRIEF)
        print(f"== {cid} (donor {donor_id})")
        t0 = time.time()
        res = run_conv(model, tokenizer, conv, donor_cache[donor_id], items,
                       hp0_check=(i == 0))
        json.dump({"id": cid, "model": MODEL, **res},
                  open(outfile, "w"), indent=1, ensure_ascii=False)
        print(f"== {cid} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
