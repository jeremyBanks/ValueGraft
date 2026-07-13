"""Applied MLX value-graft identity gate on the literal long canary.

This is deliberately small: it generates the canary's normal summary, builds
fresh compaction B, then routes B's own value rows through the exact
``arm_e_snapshot`` assignment path at alpha=1. The resulting cache and held-out
continuation log-probabilities must be bit-exact to untouched B.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from arms import arm_e_snapshot, canonical_ids, render  # noqa: E402
from kvlib import batched_teacher_forced, rebuild_cache  # noqa: E402
from provenance import (  # noqa: E402
    build_manifest,
    capture_mlx_provenance,
    stamp,
    write_run_manifest,
)
from run_arms import ArmSet  # noqa: E402


MODEL = os.environ.get(
    "SC_MODEL", "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit")
OUTDIR = Path(os.environ.get(
    "SC_OUTDIR", "results/coherent_state_local_mlx_l5_identity"))
FIXTURE = Path(os.environ.get("SC_FIXTURE", "data/synthetic/c01.json"))


def _max_abs(left, right) -> float:
    return float(mx.max(mx.abs(left.astype(mx.float32) -
                               right.astype(mx.float32))).item())


def _score(model, tokenizer, snapshot, context_ids, context_messages,
           continuation_ids) -> list[float]:
    generated_prompt = render(tokenizer, context_messages, True)
    if generated_prompt[:len(context_ids)] != context_ids:
        raise RuntimeError("generation-prompt render is not a prefix extension")
    feed = generated_prompt[len(context_ids):] + continuation_ids[:-1]
    return batched_teacher_forced(
        model, rebuild_cache(snapshot), feed, continuation_ids)


def main() -> None:
    started = time.time()
    if OUTDIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTDIR}")
    OUTDIR.mkdir(parents=True)

    model, tokenizer = load(MODEL)
    fixture = json.loads(FIXTURE.read_text())
    messages = fixture["messages"]
    context = messages[:-1]
    tail_start = fixture["sections"]["middle_end_msg"]
    continuation_ids = tokenizer.encode(
        messages[-1]["content"], add_special_tokens=False)

    armset = ArmSet(model, tokenizer, context, tail_start)
    fresh = armset.b_snap()
    identity_pairs = [(new_position, new_position)
                      for new_position, _ in armset.pairs]
    sham = arm_e_snapshot(fresh, fresh, identity_pairs, 1.0)

    layer_checks = []
    for layer, ((fresh_k, fresh_v, fresh_offset),
                (sham_k, sham_v, sham_offset)) in enumerate(zip(fresh, sham)):
        layer_checks.append({
            "layer": layer,
            "offset_equal": fresh_offset == sham_offset,
            "k_max_abs": _max_abs(fresh_k, sham_k),
            "v_max_abs": _max_abs(fresh_v, sham_v),
        })

    b_logprobs = _score(
        model, tokenizer, fresh, armset.b_ids, armset.b_msgs,
        continuation_ids)
    sham_logprobs = _score(
        model, tokenizer, sham, armset.b_ids, armset.b_msgs,
        continuation_ids)
    cache_exact = all(
        item["offset_equal"] and item["k_max_abs"] == 0.0 and
        item["v_max_abs"] == 0.0 for item in layer_checks)
    score_exact = b_logprobs == sham_logprobs

    provenance = capture_mlx_provenance(model, MODEL)
    manifest = build_manifest(
        model_provenance=provenance,
        dtype_env=None,
        harness="src/l5_mlx_graft_identity.py",
        intervention={
            "arm": "B-sham",
            "graft_type": "value",
            "alpha": 1.0,
            "alignment": "identity at every applied destination row",
            "source": "fresh B values",
        },
        metric={
            "definition": "bit-exact cache and held-out continuation identity",
            "is_proxy": False,
            "resolve_rate_measured": False,
        },
        condition={
            "summary_kind": "prod",
            "summary_source": "self-gen greedy",
            "summary_request_sha256": None,
        },
        corpus={
            "name": "technical canary",
            "split": "N=0",
            "n": 0,
            "instance_ids": [fixture["id"]],
        },
    )
    result = {
        "schema": "coherent_state_local_mlx_l5_graft_identity_v1",
        "status": "PASS" if cache_exact and score_exact else "FAIL",
        "fixture": str(FIXTURE),
        "fixture_id": fixture["id"],
        "semantic_n": 0,
        "aligned_rows": len(identity_pairs),
        "summary_ids": armset.summary["gen_ids"],
        "summary_text": armset.summary["text"],
        "fresh_mean_logprob": sum(b_logprobs) / len(b_logprobs),
        "sham_mean_logprob": sum(sham_logprobs) / len(sham_logprobs),
        "logprobs_bit_exact": score_exact,
        "cache_bit_exact": cache_exact,
        "layer_checks": layer_checks,
        "wall_seconds": time.time() - started,
    }
    stamp(result, manifest)
    write_run_manifest(OUTDIR, manifest)
    (OUTDIR / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "aligned_rows": result["aligned_rows"],
        "fresh_mean_logprob": result["fresh_mean_logprob"],
        "sham_mean_logprob": result["sham_mean_logprob"],
        "wall_seconds": result["wall_seconds"],
        "output": str(OUTDIR),
    }, indent=2), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
