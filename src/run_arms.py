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

import os

MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
E_POST_ALPHAS = [float(x) for x in os.environ.get("SC_E_POST", "0.25,0.5,0.75,1.0").split(",") if x]
E_INTER_ALPHAS = [float(x) for x in os.environ.get("SC_E_INTER", "0.5,1.0").split(",") if x]
OUTDIR = os.environ.get("SC_OUTDIR", "results/raw")
PROBE_MAX_TOKENS = 160


def clear(*objs):
    for o in objs:
        del o
    gc.collect()
    mx.clear_cache()


class ArmSet:
    """Builds arm caches lazily for a given context (msgs) and tail split."""

    def __init__(self, model, tokenizer, msgs, tail_start_msg,
                 summary_request=None):
        self.model, self.tok = model, tokenizer
        self.msgs = msgs
        self.tail_start_msg = tail_start_msg
        self.ids = canonical_ids(tokenizer, msgs)
        starts = message_token_starts(tokenizer, self.ids, len(msgs))
        if tail_start_msg is None:  # natural mode: boundary nearest 75%
            from arms import pick_tail_start
            tail_start_msg, _ = pick_tail_start(tokenizer, msgs, self.ids, 1)
            self.tail_start_msg = tail_start_msg
        self.tail_start_tok = starts[tail_start_msg]
        self.special_ids = set(tokenizer.all_special_ids)
        self.summary = generate_summary(model, tokenizer, msgs,
                                        request=summary_request)
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
        self._bmin_snap = None
        self.bmin_msgs = self.b_msgs[:2]  # system + context-note only
        self.bmin_ids = canonical_ids(tokenizer, self.bmin_msgs)
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
        """Yield (name, make_cache, ids_of_context, msgs_for_render).
        make_cache() returns a fresh cache list holding exactly the arm's
        context (no gen prompt)."""
        yield "A", (lambda: rebuild_cache(self._a_snap)), self.ids, self.msgs
        yield "B", (lambda: rebuild_cache(self.b_snap())), self.b_ids, self.b_msgs
        yield "C", (lambda: gapped_cache_from(
            arm_c_snapshot(self.summary, self.tail_start_tok, True))), \
            self.ids, self.msgs
        yield "D", (lambda: gapped_cache_from(
            arm_c_snapshot(self.summary, self.tail_start_tok, False))), \
            self.ids, self.msgs
        # Arm H (SelfGist, gap variant): sinks + S's in-context entries only —
        # arm C with an empty tail. Positions/probes continue after S.
        yield "H-gap", (lambda: gapped_cache_from(
            arm_c_snapshot(self.summary, self.summary["conv_end"], True))), \
            self.ids, self.msgs
        # B-min: matched control for H — system + identical summary text,
        # freshly encoded, no tail.
        yield "B-min", (lambda: rebuild_cache(self.bmin_snap())), \
            self.bmin_ids, self.bmin_msgs
        for a in e_post_alphas:
            snap = arm_e_snapshot(
                self.b_snap(), self.summary["snapshot"], self.pairs, a
            )
            yield f"E-post-a{a}", (lambda s=snap: rebuild_cache(s)), \
                self.b_ids, self.b_msgs
        for a in e_inter_alphas:
            snap = arm_e_inter_build(
                self.model, self.b_ids, self.summary["snapshot"], self.pairs, a
            )
            yield f"E-inter-a{a}", (lambda s=snap: rebuild_cache(s)), \
                self.b_ids, self.b_msgs

    def bmin_snap(self):
        if self._bmin_snap is None:
            cache, _ = prefill(self.model, self.bmin_ids)
            self._bmin_snap = snapshot_cache(cache)
        return self._bmin_snap

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

    cont_ids = tokenizer.encode(cont_text, add_special_tokens=False)
    out = {"stats": aset.stats, "arms": {}}
    for name, mk, ctx_ids, render_msgs in aset.variants():
        t0 = time.time()
        gp = render(tokenizer, render_msgs, True)
        if gp[: len(ctx_ids)] != ctx_ids:
            print(f"    CONT {name}: SKIP (gen-prompt render not a prefix "
                  f"extension of canonical context)")
            out["arms"][name] = {"error": "render_prefix_mismatch"}
            continue
        gp_suffix = gp[len(ctx_ids):]
        cache = mk()
        feed = gp_suffix + cont_ids[:-1]
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
    for name, mk, ctx_ids, render_msgs in aset.variants():
        if arm_filter and name not in arm_filter:
            continue
        t0 = time.time()
        answers = {}
        for plant in plants:
            probe_msgs_suffix = [{"role": "user", "content": plant["probe"]}]
            full = render(tokenizer, render_msgs + probe_msgs_suffix, True)
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


def run_natural(model, tokenizer, conv_path, outdir):
    """Natural mode: hold out the last `holdout_msgs` messages; teacher-force
    the entire rendered holdout suffix (message frames included) under each
    arm. Tail = message boundary nearest 75% of the truncated context."""
    from arms import pick_tail_start

    conv = json.load(open(conv_path))
    cid = conv["id"]
    outfile = outdir / f"{cid}.json"
    if outfile.exists():
        print(f"{cid}: done already, skipping")
        return
    msgs = conv["messages"]
    ctx = msgs[: len(msgs) - conv["holdout_msgs"]]
    assert ctx[-1]["role"] == "assistant"

    full_canon = canonical_ids(tokenizer, msgs)
    t0 = time.time()
    aset = ArmSet(model, tokenizer, ctx, None)
    # holdout suffix in canonical rendering
    assert full_canon[: len(aset.ids)] == aset.ids
    suffix = full_canon[len(aset.ids):]
    print(f"== {cid}: {len(full_canon)} tokens, holdout {len(suffix)}")
    aset.prepare_a()

    out = {"stats": aset.stats, "arms": {}}
    for name, mk, ctx_ids, render_msgs in aset.variants():
        t1 = time.time()
        cache = mk()
        # Score suffix[1:]; suffix[0] is the deterministic <|im_start|> frame
        # token, skipping it avoids needing the last context token's logits.
        feed, targets = suffix[:-1], suffix[1:]
        lps = batched_teacher_forced(model, cache, feed, targets)
        out["arms"][name] = {"mean_logprob": sum(lps) / len(lps),
                             "logprobs": lps}
        print(f"    NAT {name}: {out['arms'][name]['mean_logprob']:.4f} "
              f"({time.time()-t1:.0f}s)")
        clear(cache)
    json.dump({"id": cid, "model": MODEL, "cont": out,
               "wall_seconds": time.time() - t0},
              open(outfile, "w"), indent=1, ensure_ascii=False)
    print(f"== {cid} done in {time.time()-t0:.0f}s")


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
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for p in sorted(Path("data/synthetic").glob("c*.json")):
        if only and p.stem not in only:
            continue
        run_conversation(model, tokenizer, p, outdir)
    for p in sorted(Path("data/natural").glob("n*.json")):
        if only and p.stem not in only:
            continue
        run_natural(model, tokenizer, p, outdir)


if __name__ == "__main__":
    main()
