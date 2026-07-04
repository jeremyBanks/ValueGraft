"""Main experiment driver: run all arms on composed conversations.

Per conversation, two evaluation modes:

CONT — arms built over msgs[:-1]; the held-out final assistant message is
       teacher-forced under each arm; primary metric = mean per-token logprob.
PROBE — arms built over the full conversation; each plant's probe question is
        appended as a user turn; greedy answer generated and recorded.

Each mode generates its own in-context summary (shared by B/C/E within the
mode). Arms are processed sequentially to bound memory. Results are written
to results/raw/<conv_id>.json; scoring is a separate pass (score.py).
"""

import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from arms import (
    arm_e_inter_build,
    arm_e_snapshot,
    arm_c_snapshot,
    build_alignment,
    build_b_messages,
    canonical_ids,
    gapped_cache_from,
    generate_summary,
    message_token_starts,
    render,
)
from kvlib import (
    batched_teacher_forced,
    extend_cache,
    first_step_logits,
    greedy_generate,
    prefill,
    rebuild_cache,
    snapshot_cache,
)

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
E_POST_ALPHAS = [0.25, 0.5, 0.75, 1.0]
E_INTER_ALPHAS = [0.5, 1.0]
PROBE_MAX_TOKENS = 160


def clear(*objs):
    for o in objs:
        del o
    gc.collect()
    mx.clear_cache()


class ArmSet:
    """Builds arm caches lazily for a given context (msgs) and tail split."""

    def __init__(self, model, tokenizer, msgs, tail_start_msg):
        self.model, self.tok = model, tokenizer
        self.msgs = msgs
        self.tail_start_msg = tail_start_msg
        self.ids = canonical_ids(tokenizer, msgs)
        starts = message_token_starts(tokenizer, self.ids, len(msgs))
        self.tail_start_tok = starts[tail_start_msg]
        self.special_ids = set(tokenizer.all_special_ids)
        self.summary = generate_summary(model, tokenizer, msgs)
        self.b_msgs = build_b_messages(msgs, self.summary["text"], tail_start_msg)
        self.b_ids = canonical_ids(tokenizer, self.b_msgs)
        b_starts = message_token_starts(tokenizer, self.b_ids, len(self.b_msgs))
        self.regions = [
            ((b_starts[2], len(self.b_ids)),
             (self.tail_start_tok, self.summary["conv_end"])),
            ((b_starts[1], b_starts[2]),
             (self.summary["s_start"], self.summary["s_end"])),
        ]
        self.pairs = build_alignment(
            self.b_ids, self.summary["old_ids"], self.special_ids, self.regions
        )
        self._b_snap = None
        self.stats = {
            "n_tokens": len(self.ids),
            "tail_start_tok": self.tail_start_tok,
            "tail_frac": 1 - self.tail_start_tok / len(self.ids),
            "summary_tokens": len(self.summary["gen_ids"]),
            "aligned_pairs": len(self.pairs),
            "summary_text": self.summary["text"],
        }

    def b_snap(self):
        if self._b_snap is None:
            cache, _ = prefill(self.model, self.b_ids)
            self._b_snap = snapshot_cache(cache)
        return self._b_snap

    def variants(self, e_post_alphas=E_POST_ALPHAS, e_inter_alphas=E_INTER_ALPHAS):
        """Yield (name, make_cache, ids_of_context). make_cache() returns a
        fresh cache list holding exactly the arm's context (no gen prompt)."""
        yield "A", (lambda: rebuild_cache(self._a_snap)), self.ids
        yield "B", (lambda: rebuild_cache(self.b_snap())), self.b_ids
        yield "C", (lambda: gapped_cache_from(
            arm_c_snapshot(self.summary, self.tail_start_tok, True))), self.ids
        yield "D", (lambda: gapped_cache_from(
            arm_c_snapshot(self.summary, self.tail_start_tok, False))), self.ids
        for a in e_post_alphas:
            snap = arm_e_snapshot(
                self.b_snap(), self.summary["snapshot"], self.pairs, a
            )
            yield f"E-post-a{a}", (lambda s=snap: rebuild_cache(s)), self.b_ids
        for a in e_inter_alphas:
            snap = arm_e_inter_build(
                self.model, self.b_ids, self.summary["snapshot"], self.pairs, a
            )
            yield f"E-inter-a{a}", (lambda s=snap: rebuild_cache(s)), self.b_ids

    def prepare_a(self):
        cache, _ = prefill(self.model, self.ids)
        self._a_snap = snapshot_cache(cache)


def continue_from(model, tokenizer, make_cache, ctx_ids, suffix_ids):
    """Extend an arm cache with suffix (all but last token), return cache +
    first-step logits (1-token decode protocol)."""
    cache = make_cache()
    assert len(suffix_ids) >= 1
    if len(suffix_ids) > 1:
        extend_cache(model, cache, suffix_ids[:-1])
    logits = first_step_logits(model, cache, suffix_ids[-1])
    return cache, logits


def run_cont_mode(model, tokenizer, msgs, tail_start_msg, cont_text):
    """Teacher-forced scoring of the held-out continuation under each arm."""
    ctx = msgs
    aset = ArmSet(model, tokenizer, ctx, tail_start_msg)
    aset.prepare_a()
    # generation-prompt suffix: render(ctx, True) extends canonical(ctx)
    gp = render(tokenizer, ctx, True)
    assert gp[: len(aset.ids)] == aset.ids
    gp_suffix_full = gp[len(aset.ids):]
    gp_b = render(tokenizer, aset.b_msgs, True)
    assert gp_b[: len(aset.b_ids)] == aset.b_ids
    gp_suffix_b = gp_b[len(aset.b_ids):]
    assert gp_suffix_full == gp_suffix_b

    cont_ids = tokenizer.encode(cont_text, add_special_tokens=False)
    out = {"stats": aset.stats, "arms": {}}
    for name, mk, ctx_ids in aset.variants():
        t0 = time.time()
        cache = mk()
        feed = gp_suffix_full + cont_ids[:-1]
        lps = batched_teacher_forced(model, cache, feed, cont_ids)
        out["arms"][name] = {
            "mean_logprob": sum(lps) / len(lps),
            "logprobs": lps,
        }
        print(f"    CONT {name}: {out['arms'][name]['mean_logprob']:.4f} "
              f"({time.time()-t0:.0f}s)")
        clear(cache)
    out["n_cont_tokens"] = len(cont_ids)
    return out


def run_probe_mode(model, tokenizer, msgs, tail_start_msg, plants,
                   arm_filter=None):
    """Greedy probe answers under each arm."""
    aset = ArmSet(model, tokenizer, msgs, tail_start_msg)
    aset.prepare_a()
    eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
    out = {"stats": aset.stats, "arms": {}}
    for name, mk, ctx_ids in aset.variants():
        if arm_filter and name not in arm_filter:
            continue
        t0 = time.time()
        answers = {}
        for plant in plants:
            probe_msgs_suffix = [{"role": "user", "content": plant["probe"]}]
            # canonical ctx + probe turn + gen prompt
            if ctx_ids is aset.ids:
                full = render(tokenizer, msgs + probe_msgs_suffix, True)
            else:
                full = render(tokenizer, aset.b_msgs + probe_msgs_suffix, True)
            assert full[: len(ctx_ids)] == ctx_ids
            cache, logits = continue_from(
                model, tokenizer, mk, ctx_ids, full[len(ctx_ids):]
            )
            toks, _ = greedy_generate(
                model, cache, logits, PROBE_MAX_TOKENS, eos_ids
            )
            answers[plant["id"]] = tokenizer.decode(toks).strip()
            clear(cache)
        out["arms"][name] = answers
        print(f"    PROBE {name}: {len(answers)} probes ({time.time()-t0:.0f}s)")
    return out


def run_conversation(model, tokenizer, conv_path, outdir):
    conv = json.load(open(conv_path))
    cid = conv["id"]
    outfile = outdir / f"{cid}.json"
    if outfile.exists():
        print(f"{cid}: done already, skipping")
        return
    print(f"== {cid} ({conv['sections']['total_tokens']} tokens)")
    msgs = conv["messages"]
    tail_start_msg = conv["sections"]["middle_end_msg"]

    cont_text = msgs[-1]["content"]
    t0 = time.time()
    cont_res = run_cont_mode(model, tokenizer, msgs[:-1], tail_start_msg, cont_text)
    probe_res = run_probe_mode(model, tokenizer, msgs, tail_start_msg, conv["plants"])
    result = {
        "id": cid,
        "model": MODEL,
        "cont": cont_res,
        "probes": probe_res,
        "wall_seconds": time.time() - t0,
    }
    json.dump(result, open(outfile, "w"), indent=1, ensure_ascii=False)
    print(f"== {cid} done in {result['wall_seconds']:.0f}s")


def main():
    model, tokenizer = load(MODEL)
    outdir = Path("results/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path("data/synthetic").glob("c*.json"))
    only = set(sys.argv[1:])
    for p in paths:
        if only and p.stem not in only:
            continue
        run_conversation(model, tokenizer, p, outdir)


if __name__ == "__main__":
    main()
