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


def _find_exact_subblock(hay, needle):
    """Offset of the first exact contiguous occurrence of `needle` in `hay`,
    or None. O(len(hay)*len(needle)) worst case but the blocks here are the
    same text on both sides so the match is found immediately."""
    m = len(needle)
    if m == 0:
        return 0
    if m > len(hay):
        return None
    first = needle[0]
    last_start = len(hay) - m
    for start in range(last_start + 1):
        if hay[start] != first:
            continue
        if hay[start:start + m] == needle:
            return start
    return None


def build_alignment_difflib(b_ids, old_ids, special_ids, regions):
    """Exact-twin (new_pos, old_pos) pairs via per-region difflib matching.
    See arms.py docstring for the region-crossing rationale.

    LEGACY (pre 2026-07-07): kept for reference / equivalence checks. The active
    build_alignment is build_alignment_direct (verified equivalent 2026-07-07)."""
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


def build_alignment_direct(b_ids, old_ids, special_ids, regions):
    """Exact-twin (new_pos, old_pos) pairs via a direct 1:1 span map.

    Same signature/return as build_alignment_difflib. For each region
    ((nlo,nhi),(olo,ohi)) the OLD span old_ids[olo:ohi] is the summary/tail
    token block (the "needle"); it appears verbatim inside the NEW region
    b_ids[nlo:nhi] (the "haystack"), possibly offset by a message-wrapper
    prefix (e.g. the "[Context note] ..." preamble in front of the summary).
    We locate that exact contiguous block and map 1:1 by offset:
        (nlo + found_offset + i, olo + i)  for i in range(len(old_block))
    skipping sink positions (< N_SINK) and special tokens, exactly as the
    difflib version did.

    Because the two sides carry the SAME text, the correct alignment is a
    direct positional map; difflib was overkill and silently dropped any run
    shorter than MIN_BLOCK on a tokenization divergence. Here a divergence
    (old block not found verbatim in the new region) is a LOUD failure.

    Introduced 2026-07-07 as the replacement for the difflib matcher; verified
    pair-for-pair equivalent on the synthetic corpus + fixed summaries."""
    pairs = []
    for (nlo, nhi), (olo, ohi) in regions:
        old_block = old_ids[olo:ohi]
        hay = b_ids[nlo:nhi]
        off = _find_exact_subblock(hay, old_block)
        if off is None:
            raise ValueError(
                "build_alignment_direct: old span old_ids[%d:%d] (len %d) not "
                "found as an exact contiguous subsequence of new region "
                "b_ids[%d:%d] (len %d) -- tokenization diverged between the "
                "write-time and compacted renderings of the same text."
                % (olo, ohi, len(old_block), nlo, nhi, len(hay))
            )
        for i in range(len(old_block)):
            npos, opos = nlo + off + i, olo + i
            if npos < N_SINK or b_ids[npos] in special_ids:
                continue
            pairs.append((npos, opos))
    return pairs


# Active alignment: direct 1:1 exact-span map. Method changed from difflib
# SequenceMatcher to direct exact-subblock search on 2026-07-07; equivalence
# verified pair-for-pair (12/12 synthetic convs + fixed_summaries, Qwen tok).
# build_alignment_difflib retained above for reference.
def build_alignment(b_ids, old_ids, special_ids, regions):
    return build_alignment_direct(b_ids, old_ids, special_ids, regions)


# ---- template adapter: message boundaries for non-Qwen templates ----

def message_token_starts_prefix(tokenizer, msgs, renderer):
    """Template-agnostic message boundaries via incremental prefix rendering.
    Valid when the template renders msgs[:i] as an exact prefix of
    msgs[:i+1] (true for Mistral [INST] templates; NOT for Qwen-2507, which
    injects <think> into final assistant messages — use im_start scan there).
    Verified per-model by the ladder before use."""
    starts = []
    prev = []
    for i in range(len(msgs)):
        cur = renderer(tokenizer, msgs[: i + 1], False)
        assert cur[: len(prev)] == prev, f"template not prefix-stable at {i}"
        starts.append(len(prev))
        prev = cur
    return starts


def detect_template_family(tokenizer):
    name = getattr(tokenizer, "name_or_path", "").lower()
    if "gemma" in name:
        return "gemma"
    probe = renderer_probe = [{"role": "user", "content": "x"}]
    text = tokenizer.apply_chat_template(probe, add_generation_prompt=True,
                                         tokenize=False)
    if "<|im_start|>" in text:
        return "qwen"
    if "[INST]" in text:
        return "mistral"
    return "unknown"


def template_ops(tokenizer, renderer=None):
    """Family-dispatched (canonical_fn, starts_fn, prep_msgs_fn).

    qwen   : dummy-user canonicalization + <|im_start|> scanning; msgs as-is.
    mistral: plain render is canonical (no think blocks); prefix-diff
             boundaries; system message folded into the first user turn
             (Mistral's template relocates system text into the FINAL [INST],
             destroying prefix stability — verified 2026-07-05).
    """
    r = renderer or render
    fam = detect_template_family(tokenizer)
    if fam == "qwen":
        return (lambda m: canonical_ids(tokenizer, m, renderer=r),
                lambda ids, n: message_token_starts(tokenizer, ids, n),
                lambda m: m,
                fam)
    if fam == "mistral":
        def prep(msgs):
            if msgs and msgs[0]["role"] == "system":
                sys_txt = msgs[0]["content"]
                rest = [dict(x) for x in msgs[1:]]
                assert rest and rest[0]["role"] == "user"
                rest[0]["content"] = sys_txt + "\n\n" + rest[0]["content"]
                return rest
            return list(msgs)
        def canon(msgs):
            return r(tokenizer, msgs, False)
        def starts(_ids, n_msgs_ctx):
            raise NotImplementedError("use starts_from_msgs for mistral")
        return canon, starts, prep, fam
    raise ValueError(f"unsupported template family: {fam}")


def canonical_ids_any(tokenizer, msgs, renderer):
    """Family-aware canonical ids: qwen needs the dummy-user trick; gemma and
    mistral templates are prefix-stable so plain rendering IS canonical."""
    fam = detect_template_family(tokenizer)
    if fam == "qwen":
        return canonical_ids(tokenizer, msgs, renderer=renderer)
    return renderer(tokenizer, msgs, False)


def message_starts_any(tokenizer, msgs, ids, renderer):
    """Family-aware message boundary positions."""
    fam = detect_template_family(tokenizer)
    if fam == "qwen":
        return message_token_starts(tokenizer, ids, len(msgs))
    return message_token_starts_prefix(tokenizer, msgs, renderer)


def merge_consecutive_roles(msgs):
    out = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"],
                       "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(dict(m))
    return out


def build_b_messages_gemma(msgs, summary_text, tail_start_msg):
    """Alternation-safe B context for gemma-family templates:
    [system?, user(note), assistant(ack), *tail(starting at a USER msg)].
    Summary msg index = 1 (after system fold this is still msgs[1] in our
    list), ack = 2, tail starts at index 3."""
    note = (
        "[Context note] Earlier parts of this conversation were compacted. "
        "Summary of what came before:\n\n" + summary_text
    )
    return [
        msgs[0],
        {"role": "user", "content": note},
        {"role": "assistant",
         "content": "Understood - I have that context and will continue."},
        *msgs[tail_start_msg:],
    ]


SUMMARY_REQUEST_PROD = (
    "Context is about to be condensed. Write a context summary for an AI "
    "coding agent that will continue this work seeing ONLY this summary "
    "plus the most recent messages. Preserve, specifically and concretely: "
    "(1) the TASK: what issue/goal is being worked on, quoting key "
    "requirements verbatim where stated; (2) STATE: which files and "
    "functions have been examined or modified (exact paths), and what was "
    "found or changed; (3) DECISIONS and findings so far, including "
    "approaches ruled out and errors encountered; (4) NEXT STEPS: what "
    "remains to be done, including any verification or testing steps "
    "already planned. Be specific with names, paths, and numbers — this "
    "summary is the agent's only memory of everything before the recent "
    "tail. Aim for 300-500 words. No commentary before or after."
)
