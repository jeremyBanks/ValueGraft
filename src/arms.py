"""Experimental arms: context builders and cache builders.

Token indices refer to the canonical non-final rendering of a message list
(every assistant message rendered as non-final, no <think> blocks), computed
by rendering with a trailing dummy user turn and cutting it off. Appending a
real user turn (probe, summary request) extends this rendering exactly.

  sinks   = first N_SINK tokens
  tail    = tokens from tail_start_tok (a message boundary) to conv end
  evicted = [N_SINK, tail_start_tok)

Arms:
  A  full context (oracle)
  B  system + summary-as-context-note + verbatim tail messages, re-prefilled
  C  gapped cache: sinks + tail original entries + S generation-time entries,
     original positions kept, position counter continues from S's end
  D  ablation: C without the S entries
  E  Arm B context prefilled fresh, then V <- (1-a)*V_fresh + a*V_old at
     exact-twin positions (difflib matching blocks >= MIN_BLOCK tokens)
"""

import difflib
import sys

import mlx.core as mx

sys.path.insert(0, "src")
from kvlib import (
    GappedKVCache,
    greedy_generate,
    prefill,
    rebuild_cache,
    snapshot_cache,
)
from mlx_lm.models.cache import KVCache

N_SINK = 4
MIN_BLOCK = 8  # minimum matching-block length for E alignment

SUMMARY_REQUEST = (
    "Please write a thorough context note summarizing our conversation so far, "
    "for someone who will continue this conversation without seeing it. Cover: "
    "decisions made and what was chosen over what; open threads and next steps; "
    "definitions, names, and terms we introduced and what they mean; constraints "
    "and preferences either of us stated; approaches or options we tried and "
    "ruled out, and why. Be redundant and specific; use retrieval-friendly "
    "wording. Write it as flowing prose or bullet points, roughly 300-500 words. "
    "Do not add commentary before or after the note itself."
)

# Summary-shadow manipulation (brief §6 plant 6, via scarcity): a terse
# summary starves the text channel so probe recovery must come from retained
# activations. Used by the brief-summary condition.
SUMMARY_REQUEST_BRIEF = (
    "Please write a very brief context note (3-5 sentences, no lists) giving "
    "only the big picture of our conversation so far: what the project is and "
    "roughly where we are. Do not include specific decisions, names, numbers, "
    "or details. No commentary before or after."
)


def render(tokenizer, msgs, gen_prompt):
    return tokenizer.apply_chat_template(
        msgs, add_generation_prompt=gen_prompt, tokenize=True
    )


def im_start_id(tokenizer):
    ids = tokenizer.encode("<|im_start|>")
    assert len(ids) == 1
    return ids[0]


def canonical_ids(tokenizer, msgs):
    with_dummy = render(tokenizer, msgs + [{"role": "user", "content": "x"}], False)
    ims = im_start_id(tokenizer)
    starts = [i for i, t in enumerate(with_dummy) if t == ims]
    assert len(starts) == len(msgs) + 1
    return with_dummy[: starts[len(msgs)]]


def message_token_starts(tokenizer, ids, n_msgs):
    ims = im_start_id(tokenizer)
    starts = [i for i, t in enumerate(ids) if t == ims]
    assert len(starts) == n_msgs, f"{len(starts)} im_starts for {n_msgs} msgs"
    return starts


def pick_tail_start(tokenizer, msgs, ids, min_msg, frac=0.75):
    """(msg_idx, tok_idx) of the message boundary closest to frac of tokens."""
    starts = message_token_starts(tokenizer, ids, len(msgs))
    target = frac * len(ids)
    best = min(range(min_msg, len(msgs)), key=lambda i: abs(starts[i] - target))
    return best, starts[best]


# ---------------------------------------------------------------- summary

def generate_summary(model, tokenizer, msgs, max_tokens=900, request=None):
    """Generate S greedily in-context at the end of `msgs`.

    Returns text, S's old-context token span [s_start, s_end) (positions whose
    generation-time K/V entries exist in the snapshot), the full old-context
    ids (conversation + request + gen prompt + S tokens), and the snapshot.
    """
    conv_ids = canonical_ids(tokenizer, msgs)
    req_ids = render(
        tokenizer,
        msgs + [{"role": "user", "content": request or SUMMARY_REQUEST}],
        True,
    )
    assert req_ids[: len(conv_ids)] == conv_ids
    cache, logits = prefill(model, req_ids)
    eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
    toks, _ = greedy_generate(model, cache, logits, max_tokens, eos_ids)
    # cache holds entries for every generated token except the last one fed
    # nowhere (eos or cap); drop it from the S span uniformly.
    gen_ids = toks[:-1]
    return {
        "text": tokenizer.decode(gen_ids).strip(),
        "gen_ids": gen_ids,
        "conv_end": len(conv_ids),
        "s_start": len(req_ids),
        "s_end": len(req_ids) + len(gen_ids),
        "old_ids": req_ids + gen_ids,
        "snapshot": snapshot_cache(cache),
    }


# ---------------------------------------------------------------- contexts

def build_b_messages(msgs, summary_text, tail_start_msg):
    note = (
        "[Context note] Earlier parts of this conversation were compacted. "
        "Summary of what came before:\n\n" + summary_text
    )
    return [
        msgs[0],
        {"role": "assistant", "content": note},
        *msgs[tail_start_msg:],
    ]


# ---------------------------------------------------------------- arm caches
# Each builder returns (snapshot_or_cache, ids). Snapshots let a single build
# serve many probes (rebuild per probe).

def arm_a_build(model, tokenizer, msgs):
    ids = canonical_ids(tokenizer, msgs)
    cache, _ = prefill(model, ids)
    return snapshot_cache(cache), ids


def arm_b_build(model, tokenizer, b_msgs):
    ids = canonical_ids(tokenizer, b_msgs)
    cache, _ = prefill(model, ids)
    return snapshot_cache(cache), ids


def arm_c_snapshot(summary, tail_start_tok, include_summary=True):
    """Retained-entry arrays for the gapped cache; rebuild per probe with
    gapped_cache_from()."""
    snap = summary["snapshot"]
    conv_end = summary["conv_end"]
    s0, s1 = summary["s_start"], summary["s_end"]
    out = []
    for k, v, off in snap:
        pk = [k[..., :N_SINK, :], k[..., tail_start_tok:conv_end, :]]
        pv = [v[..., :N_SINK, :], v[..., tail_start_tok:conv_end, :]]
        if include_summary:
            pk.append(k[..., s0:s1, :])
            pv.append(v[..., s0:s1, :])
        # position counter continues from the last retained entry's position
        # (s1 with summary) rather than the raw snapshot offset, so the next
        # token lands adjacent to S (or to the tail for arm D).
        out.append((mx.concatenate(pk, axis=2), mx.concatenate(pv, axis=2),
                    s1 if include_summary else conv_end))
    return out


def gapped_cache_from(c_snap):
    return [GappedKVCache(mx.array(k), mx.array(v), off) for k, v, off in c_snap]


# ---------------------------------------------------------------- alignment

def build_alignment(b_ids, old_ids, special_ids, regions):
    """Exact-twin (new_pos, old_pos) pairs via per-region difflib matching.

    regions: list of ((new_lo, new_hi), (old_lo, old_hi)) span pairs to align
    independently. Regions must be matched separately because SequenceMatcher
    blocks are monotone in both sequences, and Arm B reorders S relative to
    the tail (S before tail in the new context, after it in the old one) —
    global matching silently drops one of the two regions.

    Blocks shorter than MIN_BLOCK are ignored (spurious matches of common
    tokens). Sink positions and special tokens are excluded."""
    pairs = []
    for (nlo, nhi), (olo, ohi) in regions:
        sm = difflib.SequenceMatcher(
            a=b_ids[nlo:nhi], b=old_ids[olo:ohi], autojunk=False
        )
        for blk in sm.get_matching_blocks():
            if blk.size < MIN_BLOCK:
                continue
            for i in range(blk.size):
                npos, opos = nlo + blk.a + i, olo + blk.b + i
                if npos < N_SINK or b_ids[npos] in special_ids:
                    continue
                pairs.append((npos, opos))
    return pairs


class SwapKVCache:
    """E-inter: blends old V into incoming values AS THEY ARE WRITTEN during
    prefill, so layer l's attention output (and hence every upper layer's
    fresh K/V) is computed over old payloads. One instance per layer.

    old_v: this layer's old-context V array. pairs: (new_pos, old_pos)."""

    def __init__(self, old_v, pairs, alpha):
        self.inner = KVCache()
        self.old_v = old_v
        self.pairs = pairs
        self.alpha = alpha

    @property
    def offset(self):
        return self.inner.offset

    @property
    def state(self):
        return self.inner.state

    def size(self):
        return self.inner.size()

    def update_and_fetch(self, keys, values):
        prev = self.inner.offset
        L = values.shape[2]
        in_window = [(n, o) for n, o in self.pairs if prev <= n < prev + L]
        if in_window and self.alpha > 0.0:
            nidx = mx.array([n - prev for n, _ in in_window])
            oidx = mx.array([o for _, o in in_window])
            old = self.old_v[..., oidx, :].astype(mx.float32)
            fresh = values[..., nidx, :].astype(mx.float32)
            values = mx.array(values)
            values[..., nidx, :] = (
                (1 - self.alpha) * fresh + self.alpha * old
            ).astype(values.dtype)
        return self.inner.update_and_fetch(keys, values)


def arm_e_inter_build(model, b_ids, old_snap, pairs, alpha):
    """Prefill Arm B's context with SwapKVCache per layer (E-inter)."""
    from kvlib import extend_cache

    cache = [
        SwapKVCache(old_snap[li][1], pairs, alpha)
        for li in range(len(model.layers))
    ]
    extend_cache(model, cache, b_ids)
    return [
        (c.inner.state[0], c.inner.state[1], c.inner.offset) for c in cache
    ]


def arm_e_snapshot(b_snap, old_snap, pairs, alpha, layer_set=None):
    """E-post: blend old V into a fresh Arm B snapshot at aligned positions."""
    new_idx = mx.array([n for n, _ in pairs])
    old_idx = mx.array([o for _, o in pairs])
    out = []
    for li, (k, v, off) in enumerate(b_snap):
        if (layer_set is not None and li not in layer_set) or alpha == 0.0:
            out.append((k, v, off))
            continue
        v_old = old_snap[li][1][..., old_idx, :].astype(mx.float32)
        v_fresh = v[..., new_idx, :].astype(mx.float32)
        blended = ((1 - alpha) * v_fresh + alpha * v_old).astype(v.dtype)
        v2 = mx.array(v)
        v2[..., new_idx, :] = blended
        out.append((k, v2, off))
    mx.eval([x for k, v, _ in out for x in (k, v)])
    return out
