"""RETIRED: generate frozen stochastic carrier attempts for local N48 v1.

V1 terminated pre-treatment at 0/48 accepted rank-1 attempts. This file remains
only to reproduce the archived failure; v2 uses fixed external carriers and
must not invoke this runner. This phase is outcome blind: it never constructs a graft, appends a focal
probe, or scores a target. Each attempt is written atomically before the next
attempt begins. Semantic carrier review remains independent and external.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, "src")
from arms import render  # noqa: E402
from kvlib import prefill  # noqa: E402
from powered_v13_recipe import canonical_json_bytes  # noqa: E402
from powered_v13_schema import CARRIER_REQUEST  # noqa: E402
from provenance import (  # noqa: E402
    build_manifest,
    capture_mlx_provenance,
    write_run_manifest,
)


DESIGN_ID = "coherent-state-local-mlx-n48-v1"
MODEL_DEFAULT = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
TEMPERATURE = 0.7
TOP_P = 0.95
CONTENT_CAP = 80
WORD_BOUNDS = (40, 60)
TOKEN_BOUNDS = (40, 80)


def attempt_seed(candidate_id: str, render_index: int,
                 attempt_index: int) -> tuple[int, str]:
    raw = canonical_json_bytes(
        [DESIGN_ID, candidate_id, f"r{render_index}", attempt_index])
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1), digest.hex()


def _normal(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _bounded_contains(text: str, surface: str) -> bool:
    text, surface = _normal(text), _normal(surface)
    start = 0
    while True:
        index = text.find(surface, start)
        if index < 0:
            return False
        end = index + len(surface)
        left = index == 0 or not text[index - 1].isalnum()
        right = end == len(text) or not text[end].isalnum()
        if left and right:
            return True
        start = index + 1


def _forbidden_forms(fixture: dict) -> list[str]:
    inventory = fixture["carrier_forbidden_inventory"]
    forms = {str(value) for value in inventory["exact_forbidden_forms"]}
    forms.add(str(inventory["nonfocal"]["fact_name"]))
    for numeric in inventory["rule_semantics"].get("numeric_values", []):
        forms.add(str(numeric))
    for value in inventory["rule_semantics"].get("time_forms", []):
        forms.add(str(value))
    return sorted((value for value in forms if value),
                  key=lambda value: (-len(value), value.casefold()))


def mechanical_review(tokenizer, fixture: dict, text: str,
                      ids: list[int], termination: int | None,
                      hit_cap: bool) -> dict:
    reasons = []
    words = len(text.split())
    if not (WORD_BOUNDS[0] <= words <= WORD_BOUNDS[1]):
        reasons.append(f"WORD_COUNT_{words}_OUTSIDE_40_60")
    if not (TOKEN_BOUNDS[0] <= len(ids) <= TOKEN_BOUNDS[1]):
        reasons.append(f"TOKEN_COUNT_{len(ids)}_OUTSIDE_40_80")
    if hit_cap:
        reasons.append("CONTENT_CAP_WITHOUT_EOS")
    if termination != tokenizer.eos_token_id:
        reasons.append("NO_NORMAL_EOS")
    if any(value in set(tokenizer.all_special_ids) for value in ids):
        reasons.append("EMBEDDED_SPECIAL_TOKEN")
    if any(character.isdigit() for character in text):
        reasons.append("UNICODE_DIGIT")
    decoded = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != text:
        reasons.append("DECODE_TEXT_MISMATCH")
    if tokenizer.encode(text, add_special_tokens=False) != ids:
        reasons.append("ENCODE_DECODE_ID_MISMATCH")
    leaked = [surface for surface in _forbidden_forms(fixture)
              if _bounded_contains(text, surface)]
    if leaked:
        reasons.append("FORBIDDEN_SURFACE:" + "|".join(leaked))
    return {
        "status": ("MECHANICAL_PASS_SEMANTIC_REVIEW_PENDING"
                   if not reasons else "MECHANICAL_FAIL"),
        "reasons": reasons,
        "word_count": words,
        "content_token_count": len(ids),
        "terminated_with_eos": termination == tokenizer.eos_token_id,
        "termination_token_id": termination,
        "hit_content_cap": hit_cap,
        "forbidden_form_count": len(_forbidden_forms(fixture)),
    }


def generate_attempt(model, tokenizer, fixture: dict, render_index: int,
                     attempt_index: int) -> dict:
    candidate_id = fixture["stable_candidate_id"]
    seed, digest = attempt_seed(candidate_id, render_index, attempt_index)
    history = fixture["variants"]["C"]["messages"]
    middle = fixture["middle_end_msg"]
    prefix_messages = history[:middle] + [
        {"role": "user", "content": CARRIER_REQUEST}]
    prefix_ids = render(tokenizer, prefix_messages, True)
    cache, logits = prefill(model, prefix_ids)
    sampler = make_sampler(temp=TEMPERATURE, top_p=TOP_P, top_k=0)
    mx.random.seed(seed)
    eos = tokenizer.eos_token_id
    special = set(tokenizer.all_special_ids)
    content_ids: list[int] = []
    token_logprobs: list[float] = []
    termination = None
    hit_cap = False
    started = time.time()
    for call_index in range(CONTENT_CAP + 1):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token = sampler(logprobs)
        token_id = int(token.item())
        if token_id == eos:
            termination = token_id
            break
        if call_index == CONTENT_CAP:
            hit_cap = True
            break
        content_ids.append(token_id)
        token_logprobs.append(float(logprobs[0, token_id].item()))
        logits = model(token[None], cache=cache)[:, -1, :]
        mx.eval(logits)
    text = tokenizer.decode(
        content_ids, skip_special_tokens=False,
        clean_up_tokenization_spaces=False)
    review = mechanical_review(
        tokenizer, fixture, text, content_ids, termination, hit_cap)
    return {
        "schema": "coherent_state_local_n48_v1_carrier_attempt_v1",
        "design_id": DESIGN_ID,
        "candidate_id": candidate_id,
        "stratum": fixture["stratum_id"],
        "rank": fixture["materialization_authorization"]["permutation_rank"],
        "fixture_literal_history_pair_sha256":
            fixture["literal_history_pair_sha256"],
        "render_id": f"r{render_index}",
        "attempt_index": attempt_index,
        "attempt_digest_sha256": digest,
        "seed_63bit": seed,
        "sampler": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": None,
            "content_token_cap": CONTENT_CAP,
            "call_width": 1,
        },
        "prompt": CARRIER_REQUEST,
        "complete_generation_prefix_messages": prefix_messages,
        "prefix_token_ids": prefix_ids,
        "prefix_token_ids_sha256": hashlib.sha256(
            canonical_json_bytes(prefix_ids)).hexdigest(),
        "content_text": text,
        "content_token_ids": content_ids,
        "content_token_logprobs": token_logprobs,
        "termination_token_id": termination,
        "hit_content_cap": hit_cap,
        "contains_special_id": any(value in special for value in content_ids),
        "mechanical_review": review,
        "semantic_review": "PENDING_INDEPENDENT",
        "wall_seconds": time.time() - started,
    }


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--attempt", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--render", type=int, choices=(1, 2))
    parser.add_argument("--candidate-root", type=Path, default=Path(
        "data/coherent_state_local_n48_v1/candidates"))
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = capture_mlx_provenance(model, args.model)
    manifest = build_manifest(
        model_provenance=provenance,
        dtype_env=None,
        harness="src/local_n48_carriers.py",
        intervention={"arm": None, "graft_type": None, "alpha": None,
                      "alignment": None,
                      "note": "outcome-blind carrier generation only"},
        metric={"definition": "none; no probe or target is constructed",
                "is_proxy": False, "resolve_rate_measured": False},
        condition={"summary_kind": "target-neutral stochastic carrier",
                   "summary_request_sha256": hashlib.sha256(
                       CARRIER_REQUEST.encode()).hexdigest(),
                   "summary_source": "subject-generated under full C history"},
        corpus={"name": "local N48 frozen ranked pool", "split": "phase-a",
                "n": 0, "instance_ids": None},
        extra={"design_id": DESIGN_ID, "treatment_outcomes_computed": False},
    )
    manifest_path = args.out / "manifest.json"
    if not manifest_path.exists():
        write_run_manifest(args.out, manifest)

    paths = sorted(args.candidate_root.glob("*/*.json"))
    selected = []
    for path in paths:
        fixture = json.loads(path.read_text())
        rank = fixture["materialization_authorization"]["permutation_rank"]
        if rank <= args.max_rank:
            selected.append((path, fixture))
    print(json.dumps({
        "model": args.model,
        "candidate_count": len(selected),
        "attempt": args.attempt,
        "renders": [args.render] if args.render else [1, 2],
        "output": str(args.out),
    }), flush=True)

    completed = 0
    mechanical_pass = 0
    for path, fixture in selected:
        for render_index in ([args.render] if args.render else [1, 2]):
            output = (args.out / "carriers" / fixture["stratum_id"] /
                      (f"rank-{fixture['materialization_authorization']['permutation_rank']:02d}_"
                       f"{fixture['stable_candidate_id']}_r{render_index}_"
                       f"a{args.attempt}.json"))
            if output.exists():
                continue
            result = generate_attempt(
                model, tokenizer, fixture, render_index, args.attempt)
            result["fixture_path"] = str(path)
            result["_manifest"] = manifest
            _write_exclusive(output, result)
            completed += 1
            mechanical_pass += (
                result["mechanical_review"]["status"].startswith(
                    "MECHANICAL_PASS"))
            print(json.dumps({
                "candidate": fixture["stable_candidate_id"][:12],
                "stratum": fixture["stratum_id"],
                "rank": result["rank"],
                "render": result["render_id"],
                "status": result["mechanical_review"]["status"],
                "words": result["mechanical_review"]["word_count"],
                "tokens": len(result["content_token_ids"]),
                "seconds": round(result["wall_seconds"], 3),
            }), flush=True)
    print(json.dumps({"completed": completed,
                      "mechanical_pass": mechanical_pass}), flush=True)


if __name__ == "__main__":
    main()
