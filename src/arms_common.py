"""Runtime-agnostic logic shared by the MLX (arms.py) and HF transformers
(arms_hf.py) stacks: chat-template token handling, message boundaries,
B-context construction, exact-twin alignment, summary request texts.
NO mlx/torch imports. Copied from arms.py (which remains authoritative for
the MLX stack); keep in sync if either changes.
"""

import difflib

N_SINK = 4
MIN_BLOCK = 8

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

SUMMARY_REQUEST_BRIEF = (
    "Please write a very brief context note (3-5 sentences, no lists) giving "
    "only the big picture of our conversation so far: what the project is and "
    "roughly where we are. Do not include specific decisions, names, numbers, "
    "or details. No commentary before or after."
)


def render(tokenizer, msgs, gen_prompt):
    """Token ids for msgs via chat template (works for both stacks' tokenizers
    when the template returns token ids; HF fast tokenizers: use text round
    trip)."""
    out = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=gen_prompt, tokenize=True
    )
    if hasattr(out, "ids"):  # tokenizers.Encoding
        return list(out.ids)
    if out and not isinstance(out[0], int):  # HF sometimes nests
        return list(out[0])
    return list(out)


def render_hf(tokenizer, msgs, gen_prompt):
    """HF-safe render via text (avoids Encoding-object pitfalls)."""
    text = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=gen_prompt, tokenize=False
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def im_start_id(tokenizer):
    ids = tokenizer.encode("<|im_start|>", add_special_tokens=False) \
        if hasattr(tokenizer, "encode") else tokenizer.encode("<|im_start|>")
    assert len(ids) == 1
    return ids[0]


def canonical_ids(tokenizer, msgs, renderer=None):
    """Canonical non-final rendering (dummy-user trick; see DECISIONS.md)."""
    r = renderer or render
    with_dummy = r(tokenizer, msgs + [{"role": "user", "content": "x"}], False)
    ims = im_start_id(tokenizer)
    starts = [i for i, t in enumerate(with_dummy) if t == ims]
    assert len(starts) == len(msgs) + 1
    return with_dummy[: starts[len(msgs)]]


def message_token_starts(tokenizer, ids, n_msgs):
    ims = im_start_id(tokenizer)
    starts = [i for i, t in enumerate(ids) if t == ims]
    assert len(starts) == n_msgs, f"{len(starts)} im_starts for {n_msgs} msgs"
    return starts


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


def bmin_pack_ids(summary, conv_ids):
    return list(conv_ids[:N_SINK]) + list(summary["gen_ids"])


def build_alignment(b_ids, old_ids, special_ids, regions):
    """Exact-twin (new_pos, old_pos) pairs via per-region difflib matching.
    See arms.py docstring for the region-crossing rationale."""
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
