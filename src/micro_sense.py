"""Micro-experiment: does a transplanted VALUE vector carry word sense?

Isolates H2 with no summaries, no compaction pipeline. For each item:

  OLD context: disambiguating sentence(s) fixing the MINORITY sense of an
               ambiguous word + a neutral carrier sentence using the word.
  NEW context: the same carrier sentence with NO disambiguation.

The carrier sentence is token-identical in both contexts. Arms:

  oracle   : OLD context (upper bound — disambiguation present in text)
  fresh    : NEW context, untouched (lower bound — model's default sense)
  V-swap   : NEW context, values at the carrier-sentence positions replaced
             by their OLD-context twins (alpha sweep; keys stay fresh)
  KV-swap  : same but keys AND values replaced (position-compatible because
             the carrier is placed at the SAME absolute positions by padding)

Readout: forced-choice logprob difference. We append a question with two
lettered options and measure logprob(minority letter) - logprob(majority
letter) at the answer position. Positive = model resolved the minority sense.

Items are constructed so the minority sense is strongly counter-prior, so
`fresh` should pick the majority sense and `oracle` the minority one.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from kvlib import first_step_logits, prefill, rebuild_cache, snapshot_cache

import os
MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")

# Each item: disambig (fixes minority sense), pad (neutral filler so carrier
# lands at identical absolute positions in both contexts), carrier (uses the
# word, sense-ambiguous on its face), question, minority/majority options.
ITEMS = [
    {
        "id": "bank",
        "disambig": "Field notes, day 12. Reminder to self: whenever I write 'the bank' in these notes I always mean the river bank by the willow trees, never the financial branch in town.",
        "carrier": "In the afternoon I went down to the bank again and waited for almost an hour.",
        "question": "In the last sentence, what does 'the bank' most likely refer to?",
        "minority": "the edge of a river",
        "majority": "a financial institution",
    },
    {
        "id": "java",
        "disambig": "Trip journal. Note: when I mention Java in this journal I'm talking about the Indonesian island itself, not coffee and not the programming language.",
        "carrier": "I spent all of Tuesday dealing with Java, which honestly exhausted me.",
        "question": "In the last sentence, what does 'Java' most likely refer to?",
        "minority": "the Indonesian island",
        "majority": "the programming language",
    },
    {
        "id": "mercury",
        "disambig": "Lab notebook. Convention for these pages: 'Mercury' here always names our team's internal database-migration project, never the planet and never the chemical element.",
        "carrier": "Mercury gave us trouble all week and we still have no clean fix.",
        "question": "In the last sentence, what does 'Mercury' most likely refer to?",
        "minority": "an internal software project",
        "majority": "the planet or the chemical element",
    },
    {
        "id": "pitch",
        "disambig": "Groundskeeper's log. To be clear about my shorthand: 'the pitch' in this log always means the cricket field's playing surface, not a sales presentation.",
        "carrier": "We worked on the pitch until it was nearly dark outside.",
        "question": "In the last sentence, what does 'the pitch' most likely refer to?",
        "minority": "a sports playing surface",
        "majority": "a sales presentation",
    },
    {
        "id": "crane",
        "disambig": "Birdwatcher's diary. A note on my wording: in this diary 'the crane' always refers to the sandhill crane that nests by the pond, not construction equipment.",
        "carrier": "The crane was gone before sunrise, which surprised everyone at the site.",
        "question": "In the last sentence, what does 'the crane' most likely refer to?",
        "minority": "a large bird",
        "majority": "a piece of construction machinery",
    },
    {
        "id": "port",
        "disambig": "Sommelier's tasting book. Convention: when I write 'the port' in these entries I always mean the fortified wine from Douro, never a harbor and never a network port.",
        "carrier": "The port was disappointing that evening, though nobody said so aloud.",
        "question": "In the last sentence, what does 'the port' most likely refer to?",
        "minority": "a fortified wine",
        "majority": "a harbor for ships",
    },
]

ALPHAS = [0.25, 0.5, 0.75, 1.0]


def build_context(tokenizer, item, with_disambig):
    """User message: [disambig or neutral pad] + carrier. Padding keeps the
    carrier at identical absolute positions in both contexts."""
    dis_ids = tokenizer.encode(item["disambig"] + "\n\n", add_special_tokens=False)
    pad_text = "Notebook. " + "The days pass one after another, quietly. " * 40
    pad_ids = tokenizer.encode(pad_text, add_special_tokens=False)[: len(dis_ids)]
    assert len(pad_ids) == len(dis_ids), "pad shorter than disambig"
    car_ids = tokenizer.encode(item["carrier"], add_special_tokens=False)

    prefix = build_chat_prefix(tokenizer)
    head = dis_ids if with_disambig else pad_ids
    ids = prefix + head + car_ids
    car_start = len(prefix) + len(head)
    return ids, car_start, len(car_ids)


_PREFIX_CACHE = {}


def build_chat_prefix(tokenizer):
    """Token ids for system + start of a user message."""
    if "p" not in _PREFIX_CACHE:
        msgs = [{"role": "system", "content": "You are a careful reader."},
                {"role": "user", "content": "\x00MARK\x00"}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            tokenize=False)
        prefix_text = ids.split("\x00MARK\x00")[0]
        _PREFIX_CACHE["p"] = tokenizer.encode(prefix_text, add_special_tokens=False)
    return _PREFIX_CACHE["p"]


def question_suffix(tokenizer, item, order_flip):
    a, b = (item["minority"], item["majority"])
    if order_flip:
        a, b = b, a
    text = (
        f"\n\n{item['question']}\n(A) {a}\n(B) {b}\n"
        "Answer with just the letter."
    )
    msgs_suffix = text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tokenizer.encode(msgs_suffix, add_special_tokens=False)
    minority_letter = "B" if order_flip else "A"
    majority_letter = "A" if order_flip else "B"
    return ids, minority_letter, majority_letter


def letter_margin(model, cache, suffix_ids, tokenizer, min_l, maj_l):
    logits = None
    ids = mx.array(suffix_ids)[None]
    logits = model(ids, cache=cache)[:, -1, :]
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tmin = tokenizer.encode(min_l, add_special_tokens=False)
    tmaj = tokenizer.encode(maj_l, add_special_tokens=False)
    assert len(tmin) == 1 and len(tmaj) == 1
    return (lp[0, tmin[0]] - lp[0, tmaj[0]]).item()


def swap_snapshot(new_snap, old_snap, start, length, alpha, swap_keys):
    out = []
    for (kn, vn, off), (ko, vo, _) in zip(new_snap, old_snap):
        v2 = mx.array(vn)
        old = vo[..., start : start + length, :].astype(mx.float32)
        new = vn[..., start : start + length, :].astype(mx.float32)
        v2[..., start : start + length, :] = (
            (1 - alpha) * new + alpha * old
        ).astype(vn.dtype)
        k2 = kn
        if swap_keys:
            k2 = mx.array(kn)
            oldk = ko[..., start : start + length, :].astype(mx.float32)
            newk = kn[..., start : start + length, :].astype(mx.float32)
            k2[..., start : start + length, :] = (
                (1 - alpha) * newk + alpha * oldk
            ).astype(kn.dtype)
        out.append((k2, v2, off))
    return out


def main():
    model, tokenizer = load(MODEL)
    results = []
    for item in ITEMS:
        old_ids, car_start, car_len = build_context(tokenizer, item, True)
        new_ids, car_start2, car_len2 = build_context(tokenizer, item, False)
        assert (car_start, car_len) == (car_start2, car_len2)
        assert old_ids[car_start:] == new_ids[car_start:]

        oc, _ = prefill(model, old_ids)
        old_snap = snapshot_cache(oc)
        nc, _ = prefill(model, new_ids)
        new_snap = snapshot_cache(nc)

        row = {"id": item["id"]}
        for flip in (False, True):
            sfx, min_l, maj_l = question_suffix(tokenizer, item, flip)
            tag = "flip" if flip else "std"
            row[f"oracle_{tag}"] = letter_margin(
                model, rebuild_cache(old_snap), sfx, tokenizer, min_l, maj_l)
            row[f"fresh_{tag}"] = letter_margin(
                model, rebuild_cache(new_snap), sfx, tokenizer, min_l, maj_l)
            for a in ALPHAS:
                sw = swap_snapshot(new_snap, old_snap, car_start, car_len, a, False)
                row[f"vswap{a}_{tag}"] = letter_margin(
                    model, rebuild_cache(sw), sfx, tokenizer, min_l, maj_l)
            sw = swap_snapshot(new_snap, old_snap, car_start, car_len, 1.0, True)
            row[f"kvswap1.0_{tag}"] = letter_margin(
                model, rebuild_cache(sw), sfx, tokenizer, min_l, maj_l)
        results.append(row)
        print(json.dumps(row, indent=None))

    Path("results").mkdir(exist_ok=True)
    json.dump(results, open("results/micro_sense.json", "w"), indent=1)
    keys = [k for k in results[0] if k != "id"]
    print("\nmean margins (minority - majority logprob; >0 means disambiguated):")
    for k in keys:
        vals = [r[k] for r in results]
        print(f"  {k:16s} {sum(vals)/len(vals):+.3f}")


if __name__ == "__main__":
    main()
