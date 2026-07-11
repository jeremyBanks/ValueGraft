#!/usr/bin/env python3
"""Bounded c10 query-schedule origin diagnostic.

This is an external, non-authorizing diagnostic.  It deliberately lives outside
the frozen ``scripts/*coherent*.py`` apparatus glob and does not modify any v10
artifact.  It asks whether the first 23 cache rows depend on query-call shape or
on the content of causally masked future tokens.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


SCHEMA = 1
DIAGNOSTIC_ID = "coherent-state-c10-schedule-origin-v1"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TOKENIZER_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TOKENIZER_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
CASE_PATH = "data/synthetic/c10.json"
REPLACEMENT_CASE_PATH = "data/synthetic/c02.json"
PREFIX_LENGTH = 4096
FIRST_BLOCK_LENGTH = 23
EXPECTED_FULL_PREFIX_LENGTH = 8430
EXPECTED_FULL_PREFIX_SHA256 = (
    "4cecf2f426bb80e2d0339e31ec6d1a4c14ed97ab72a57854484ce8183190368d")
EXPECTED_FIRST4096_SHA256 = (
    "d13c3330ffd77dcf44152f8ad049962a174dc042019e194c95cd000013580e70")
EXPECTED_FIRST23_SHA256 = (
    "5247681f21fefc41e688a2502bde1e523c0147032bb1f3846bd1d83c435d52de")
EXPECTED_REPLACEMENT_FULL_PREFIX_LENGTH = 8385
EXPECTED_REPLACEMENT_FULL_PREFIX_SHA256 = (
    "50f2938159c611e64b6ad1e0194e5368e3b22d55b8309bebf3303d3041628475")
EXPECTED_REPLACEMENT_FIRST4096_SHA256 = (
    "cedfaf53152dc40fd8f5cd6d275a02cf2e86682d9143d4d8262c687d7e491042")
EXPECTED_LAYERS = 28
EXPECTED_KV_HEADS = 8
EXPECTED_HEAD_DIM = 128
EXPECTED_TORCH_THREADS = 4
EXPECTED_TORCH_INTEROP_THREADS = 10
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MAX_OUTPUT_BYTES = 4_000_000


class DiagnosticError(RuntimeError):
    """A fail-closed diagnostic precondition or control failed."""

    def __init__(self, message: str, *, evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.evidence = evidence or {}


class NondeterminismError(DiagnosticError):
    """A repeated identical branch did not reproduce bit-for-bit."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def _sha256_ints(values: Iterable[int]) -> str:
    return _sha256_json([int(value) for value in values])


def _payload_sha256(doc: dict[str, Any]) -> str:
    return _sha256_json({
        key: value for key, value in doc.items() if key != "payload_sha256"
    })


def seal_payload(doc: dict[str, Any]) -> dict[str, Any]:
    sealed = {key: value for key, value in doc.items()
              if key != "payload_sha256"}
    sealed["payload_sha256"] = _payload_sha256(sealed)
    return sealed


def verify_sealed_payload(doc: dict[str, Any]) -> None:
    if doc.get("payload_sha256") != _payload_sha256(doc):
        raise DiagnosticError("diagnostic payload hash differs")


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True,
        text=not binary, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace") if binary else proc.stderr
        raise DiagnosticError(
            f"git {' '.join(args)} failed: {stderr.strip()[:300]}")
    return proc.stdout


def _require_committed_bytes(repo: Path, relative: str, commit: str) -> bytes:
    path = (repo / relative).resolve()
    try:
        canonical = path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise DiagnosticError(f"source is outside repository: {relative}") from exc
    if canonical != relative or not path.is_file():
        raise DiagnosticError(f"source path is absent or noncanonical: {relative}")
    committed = bytes(_git(repo, "show", f"{commit}:{relative}", binary=True))
    observed = path.read_bytes()
    if committed != observed:
        raise DiagnosticError(f"source differs from committed bytes: {relative}")
    return observed


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8)
    return raw.numpy().tobytes()


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": _sha256_bytes(_tensor_bytes(tensor)),
    }


def tensors_bit_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape and
        left.dtype == right.dtype and
        torch.equal(
            left.detach().cpu().contiguous().view(torch.uint8),
            right.detach().cpu().contiguous().view(torch.uint8),
        )
    )


Snapshot = list[tuple[torch.Tensor, torch.Tensor]]


def snapshot_record(snapshot: Snapshot) -> dict[str, Any]:
    rows = []
    for layer, (key, value) in enumerate(snapshot):
        rows.append({
            "layer": layer,
            "key": tensor_record(key),
            "value": tensor_record(value),
        })
    return {
        "layer_count": len(rows),
        "layers": rows,
        "aggregate_sha256": _sha256_json(rows),
    }


def _validate_snapshot_shape(snapshot: Snapshot, length: int, label: str) -> None:
    if len(snapshot) != EXPECTED_LAYERS:
        raise DiagnosticError(
            f"{label} layer count {len(snapshot)} != {EXPECTED_LAYERS}")
    expected = [1, EXPECTED_KV_HEADS, length, EXPECTED_HEAD_DIM]
    for layer, (key, value) in enumerate(snapshot):
        for kind, tensor in (("K", key), ("V", value)):
            if list(tensor.shape) != expected or tensor.dtype != torch.bfloat16:
                raise DiagnosticError(
                    f"{label} layer {layer} {kind} shape/dtype differs: "
                    f"{list(tensor.shape)}/{tensor.dtype}")


def compare_repeat_snapshots(
        left: Snapshot, right: Snapshot,
        left_logits: torch.Tensor, right_logits: torch.Tensor,
        label: str) -> dict[str, Any]:
    if len(left) != len(right):
        raise DiagnosticError(f"{label} repeat layer coverage differs")
    per_layer = []
    cache_exact = True
    for layer, ((left_k, left_v), (right_k, right_v)) in enumerate(
            zip(left, right)):
        k_exact = tensors_bit_equal(left_k, right_k)
        v_exact = tensors_bit_equal(left_v, right_v)
        cache_exact &= k_exact and v_exact
        per_layer.append({
            "layer": layer,
            "k_bit_exact": k_exact,
            "v_bit_exact": v_exact,
            "k_max_abs": float((left_k.float() - right_k.float()).abs().max()),
            "v_max_abs": float((left_v.float() - right_v.float()).abs().max()),
        })
    logits_exact = tensors_bit_equal(left_logits, right_logits)
    result = {
        "bit_exact": cache_exact,
        "cache_bit_exact": cache_exact,
        "logits_bit_exact": logits_exact,
        "logits_max_abs": float(
            (left_logits.float() - right_logits.float()).abs().max()),
        "per_layer": per_layer,
    }
    if not cache_exact:
        raise NondeterminismError(
            f"{label} repeats are not bit-identical", evidence=result)
    return result


def _row_maxima(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    if left.shape != right.shape or left.ndim != 4:
        raise DiagnosticError("row comparison tensor shapes differ")
    values = (left.float() - right.float()).abs()
    return [float(value) for value in values.amax(dim=(0, 1, 3)).tolist()]


def _row_bit_exact(left: torch.Tensor, right: torch.Tensor) -> list[bool]:
    if left.shape != right.shape or left.ndim != 4 or left.dtype != right.dtype:
        raise DiagnosticError("row bit-comparison tensor metadata differs")
    left_bits = left.detach().cpu().contiguous().view(torch.uint8)
    right_bits = right.detach().cpu().contiguous().view(torch.uint8)
    different = left_bits != right_bits
    return [bool(value) for value in (~different.any(dim=(0, 1, 3))).tolist()]


def compare_region(
        left: Snapshot, right: Snapshot, *, left_start: int,
        right_start: int, width: int, include_per_row: bool,
        label: str) -> dict[str, Any]:
    if width < 1 or len(left) != len(right):
        raise DiagnosticError(f"{label} comparison coverage is invalid")
    for side, snapshot, start in (
            ("left", left, left_start), ("right", right, right_start)):
        if start < 0:
            raise DiagnosticError(f"{label} {side} start is negative")
        for layer, (key, value) in enumerate(snapshot):
            if (key.ndim != 4 or value.ndim != 4 or
                    key.shape[2] < start + width or
                    value.shape[2] < start + width):
                raise DiagnosticError(
                    f"{label} {side} layer {layer} does not cover the "
                    f"requested rows")
    per_layer = []
    divergences: list[tuple[int, int, str]] = []
    aggregate_k_rows = [0.0] * width
    aggregate_v_rows = [0.0] * width
    for layer, ((left_k, left_v), (right_k, right_v)) in enumerate(
            zip(left, right)):
        lk = left_k[..., left_start:left_start + width, :]
        lv = left_v[..., left_start:left_start + width, :]
        rk = right_k[..., right_start:right_start + width, :]
        rv = right_v[..., right_start:right_start + width, :]
        k_rows = _row_maxima(lk, rk)
        v_rows = _row_maxima(lv, rv)
        k_bits = _row_bit_exact(lk, rk)
        v_bits = _row_bit_exact(lv, rv)
        row_records = []
        for offset, (k_max, v_max, k_exact, v_exact) in enumerate(zip(
                k_rows, v_rows, k_bits, v_bits)):
            aggregate_k_rows[offset] = max(aggregate_k_rows[offset], k_max)
            aggregate_v_rows[offset] = max(aggregate_v_rows[offset], v_max)
            row = left_start + offset
            if not k_exact:
                divergences.append((layer, row, "K"))
            if not v_exact:
                divergences.append((layer, row, "V"))
            if include_per_row:
                row_records.append({
                    "left_row": row,
                    "right_row": right_start + offset,
                    "k_bit_exact": k_exact,
                    "v_bit_exact": v_exact,
                    "k_max_abs": k_max,
                    "v_max_abs": v_max,
                })
        record = {
            "layer": layer,
            "k_bit_exact": all(k_bits),
            "v_bit_exact": all(v_bits),
            "k_max_abs": max(k_rows),
            "v_max_abs": max(v_rows),
        }
        if include_per_row:
            record["per_row"] = row_records
        per_layer.append(record)
    first_layer_row = min(divergences, default=None)
    first_row_layer = min(
        ((row, layer, kind) for layer, row, kind in divergences),
        default=None)
    per_row_aggregate = []
    for offset, (k_max, v_max) in enumerate(zip(
            aggregate_k_rows, aggregate_v_rows)):
        row_divergences = [
            item for item in divergences if item[1] == left_start + offset]
        per_row_aggregate.append({
            "left_row": left_start + offset,
            "right_row": right_start + offset,
            "k_bit_exact": not any(item[2] == "K" for item in row_divergences),
            "v_bit_exact": not any(item[2] == "V" for item in row_divergences),
            "k_max_abs": k_max,
            "v_max_abs": v_max,
        })
    return {
        "label": label,
        "left_start": left_start,
        "right_start": right_start,
        "width": width,
        "bit_exact": not divergences,
        "k_max_abs": max(row["k_max_abs"] for row in per_layer),
        "v_max_abs": max(row["v_max_abs"] for row in per_layer),
        "first_divergence_layer_row_kind": (
            list(first_layer_row) if first_layer_row else None),
        "first_divergence_row_layer_kind": (
            list(first_row_layer) if first_row_layer else None),
        "per_layer": per_layer,
        "per_row_aggregate_across_layers": per_row_aggregate,
    }


def classify_three_way(
        a_vs_b: dict[str, Any], b_vs_c_first: dict[str, Any],
        b_vs_c_after: dict[str, Any]) -> dict[str, Any]:
    if b_vs_c_after.get("bit_exact") is not False:
        raise DiagnosticError(
            "B-C positive control did not differ after row 22")
    first = a_vs_b.get("first_divergence_layer_row_kind")
    if a_vs_b.get("bit_exact") is not True and (
            not isinstance(first, list) or len(first) != 3):
        raise DiagnosticError("A-B divergence lacks an exact coordinate")
    if isinstance(first, list) and first[0] == 0:
        return {
            "outcome_class": "CONSTRUCTION_DIVERGENCE",
            "interpretation": (
                "Layer-0 K/V changed despite identical first-23 tokens and "
                "positions; token/position/cache construction must be audited."),
            "fallback_required": False,
        }
    if b_vs_c_first.get("bit_exact") is not True:
        return {
            "outcome_class": "FUTURE_TOKEN_INFLUENCE",
            "interpretation": (
                "B and C have identical shape and identical rows 0-22, but "
                "future-token content changed those early cache rows."),
            "fallback_required": False,
        }
    if a_vs_b.get("bit_exact") is True:
        return {
            "outcome_class": "NO_FIRST_BOUNDARY_DIVERGENCE",
            "interpretation": (
                "The persisted c10 divergence begins after the first 23 rows; "
                "run the prospectively declared full-prefix boundary scan."),
            "fallback_required": True,
        }
    return {
        "outcome_class": "QUERY_SHAPE_ROUNDING",
        "interpretation": (
            "B-C is bit-exact on causally protected rows while A-B first "
            "diverges after layer 0; query-call shape changes deterministic "
            "bf16 execution without future-token influence."),
        "fallback_required": False,
    }


def _load_inputs(repo: Path, tokenizer, commit: str) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "src"))
    from arms_common import SUMMARY_REQUEST
    from coherent_state_cases import correct_source_messages, validate_native_conversation
    from coherent_state_tokens import generation_prefix_ids

    case_raw = _require_committed_bytes(repo, CASE_PATH, commit)
    case = json.loads(case_raw)
    validate_native_conversation(case)
    if str(case.get("id")) != "c10":
        raise DiagnosticError("frozen case path does not contain c10")
    prefix = generation_prefix_ids(
        tokenizer, correct_source_messages(case, SUMMARY_REQUEST))
    if len(prefix) != EXPECTED_FULL_PREFIX_LENGTH:
        raise DiagnosticError(
            f"c10 prefix length {len(prefix)} != {EXPECTED_FULL_PREFIX_LENGTH}")
    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        raise DiagnosticError("production tokenizer im_start marker is not singular")
    starts = [index for index, token in enumerate(prefix)
              if int(token) == int(marker_ids[0])]
    if len(starts) < 2 or starts[1] != FIRST_BLOCK_LENGTH:
        raise DiagnosticError(
            f"c10 exact first block is not {FIRST_BLOCK_LENGTH}: {starts[:3]}")

    replacement_raw = _require_committed_bytes(
        repo, REPLACEMENT_CASE_PATH, commit)
    replacement_case = json.loads(replacement_raw)
    validate_native_conversation(replacement_case)
    if str(replacement_case.get("id")) != "c02":
        raise DiagnosticError("replacement case path does not contain c02")
    replacement_prefix = generation_prefix_ids(
        tokenizer,
        correct_source_messages(replacement_case, SUMMARY_REQUEST))
    replacement_hashes = {
        "full": _sha256_ints(replacement_prefix),
        "first4096": _sha256_ints(replacement_prefix[:PREFIX_LENGTH]),
    }
    if (len(replacement_prefix) != EXPECTED_REPLACEMENT_FULL_PREFIX_LENGTH or
            replacement_hashes != {
                "full": EXPECTED_REPLACEMENT_FULL_PREFIX_SHA256,
                "first4096": EXPECTED_REPLACEMENT_FIRST4096_SHA256,
            }):
        raise DiagnosticError(
            "c02 replacement prefix length or token hashes differ")
    original = [int(value) for value in prefix[:PREFIX_LENGTH]]
    observed_hashes = {
        "full": _sha256_ints(prefix),
        "first4096": _sha256_ints(original),
        "first23": _sha256_ints(original[:FIRST_BLOCK_LENGTH]),
    }
    expected_hashes = {
        "full": EXPECTED_FULL_PREFIX_SHA256,
        "first4096": EXPECTED_FIRST4096_SHA256,
        "first23": EXPECTED_FIRST23_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise DiagnosticError(
            f"c10 token hashes differ: observed={observed_hashes}")
    branch_c = (
        original[:FIRST_BLOCK_LENGTH] +
        [int(value) for value in replacement_prefix[
            FIRST_BLOCK_LENGTH:PREFIX_LENGTH]]
    )
    changed = [index for index, pair in enumerate(zip(original, branch_c))
               if pair[0] != pair[1]]
    if (not changed or min(changed) < FIRST_BLOCK_LENGTH or
            max(changed) >= PREFIX_LENGTH):
        raise DiagnosticError(
            "branch C did not change only causally future rows 23:4096")
    positions = list(range(PREFIX_LENGTH))
    return {
        "branch_tokens": {
            "A": original[:FIRST_BLOCK_LENGTH],
            "B": original,
            "C": branch_c,
        },
        "record": {
            "case_path": CASE_PATH,
            "case_raw_sha256": _sha256_bytes(case_raw),
            "case_canonical_sha256": _sha256_json(case),
            "full_prefix_length": len(prefix),
            "full_prefix_sha256": observed_hashes["full"],
            "first4096_sha256": observed_hashes["first4096"],
            "first23_sha256": observed_hashes["first23"],
            "position_ids_sha256": _sha256_ints(positions),
            "cache_positions_sha256": _sha256_ints(positions),
            "first_block_length": FIRST_BLOCK_LENGTH,
            "prefix_length": PREFIX_LENGTH,
            "replacement_case_path": REPLACEMENT_CASE_PATH,
            "replacement_case_raw_sha256": _sha256_bytes(replacement_raw),
            "replacement_case_canonical_sha256": _sha256_json(
                replacement_case),
            "replacement_full_prefix_length": len(replacement_prefix),
            "replacement_full_prefix_sha256": replacement_hashes["full"],
            "replacement_first4096_sha256": replacement_hashes["first4096"],
            "branch_c_sha256": _sha256_ints(branch_c),
            "changed_position_count": len(changed),
            "changed_positions_sha256": _sha256_ints(changed),
            "only_rows_23_to_4095_may_differ": True,
        },
    }


def _snapshot_cache(cache) -> Snapshot:
    if not hasattr(cache, "layers"):
        raise DiagnosticError("transformers cache lacks v5 layers")
    snapshot = []
    for layer in cache.layers:
        snapshot.append((layer.keys.detach().clone(), layer.values.detach().clone()))
    return snapshot


def _run_branch(model, token_ids: list[int]) \
        -> tuple[Snapshot, torch.Tensor]:
    ids = torch.tensor([token_ids], dtype=torch.long, device="cpu")
    positions = torch.arange(len(token_ids), dtype=torch.long)[None]
    cache_positions = torch.arange(len(token_ids), dtype=torch.long)
    with torch.no_grad():
        output = model(
            input_ids=ids,
            position_ids=positions,
            cache_position=cache_positions,
            past_key_values=None,
            use_cache=True,
            logits_to_keep=1,
        )
    snapshot = _snapshot_cache(output.past_key_values)
    _validate_snapshot_shape(snapshot, len(token_ids), "branch")
    return snapshot, output.logits.detach().clone()


def _run_repeated_branch(model, name: str, token_ids: list[int]) \
        -> tuple[Snapshot, dict[str, Any]]:
    first, first_logits = _run_branch(model, token_ids)
    first_record = {
        "snapshot": snapshot_record(first),
        "last_logits": tensor_record(first_logits),
    }
    second, second_logits = _run_branch(model, token_ids)
    second_record = {
        "snapshot": snapshot_record(second),
        "last_logits": tensor_record(second_logits),
    }
    branch_record = {
        "name": name,
        "token_count": len(token_ids),
        "token_sha256": _sha256_ints(token_ids),
        "position_ids_sha256": _sha256_ints(range(len(token_ids))),
        "cache_positions_sha256": _sha256_ints(range(len(token_ids))),
        "repeat_1": first_record,
        "repeat_2": second_record,
    }
    try:
        branch_record["repeat_identity"] = compare_repeat_snapshots(
            first, second, first_logits, second_logits, f"branch {name}")
    except NondeterminismError as exc:
        branch_record["repeat_identity"] = exc.evidence
        raise NondeterminismError(
            str(exc), evidence={"failed_branch": branch_record}) from exc
    return first, branch_record


def _threading_environment() -> dict[str, Any]:
    return {
        "environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()),
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_parallel_info": torch.__config__.parallel_info(),
    }


def _configure_threads() -> None:
    changed = {key: os.environ.get(key) for key in THREAD_ENV_KEYS
               if os.environ.get(key) not in (None, "")}
    if changed:
        raise DiagnosticError(
            f"threading environment differs from ladder launch: {changed}")
    torch.set_num_threads(EXPECTED_TORCH_THREADS)
    torch.set_num_interop_threads(EXPECTED_TORCH_INTEROP_THREADS)
    if (torch.get_num_threads() != EXPECTED_TORCH_THREADS or
            torch.get_num_interop_threads() != EXPECTED_TORCH_INTEROP_THREADS):
        raise DiagnosticError("torch thread pinning did not take effect")


def _load_subject():
    model_snapshot = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    tokenizer_snapshot = Path(snapshot_download(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True))
    if model_snapshot.name != MODEL_REVISION:
        raise DiagnosticError(
            f"resolved model revision {model_snapshot.name} != {MODEL_REVISION}")
    if tokenizer_snapshot.name != TOKENIZER_REVISION:
        raise DiagnosticError(
            "resolved production tokenizer revision differs")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    revisions = {
        "model_requested": MODEL_REVISION,
        "model_resolved": getattr(model.config, "_commit_hash", None),
        "model_snapshot_commit": model_snapshot.name,
        "model_config_sha256": _sha256_json(model.config.to_dict()),
        "tokenizer_requested": TOKENIZER_REVISION,
        "tokenizer_snapshot_commit": tokenizer_snapshot.name,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_sha256": _sha256_json(tokenizer.get_vocab()),
        "tokenizer_chat_template_sha256": _sha256_bytes(
            str(tokenizer.chat_template).encode()),
        "tokenizer_special_tokens_sha256": _sha256_json(
            tokenizer.special_tokens_map),
    }
    if revisions["model_resolved"] != MODEL_REVISION:
        raise DiagnosticError("loaded model config revision differs")
    if model.device.type != "cpu":
        raise DiagnosticError(f"diagnostic subject device is {model.device}, not CPU")
    parameter_dtypes = sorted({str(value.dtype) for value in model.parameters()})
    if parameter_dtypes != ["torch.bfloat16"]:
        raise DiagnosticError(
            f"diagnostic parameter dtypes differ: {parameter_dtypes}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from coherent_state_runtime import eager_backend_fingerprint
    backend = eager_backend_fingerprint(model)
    if (backend.get("requested_implementation") != "eager" or
            backend.get("expected_layer_count") != EXPECTED_LAYERS or
            [row.get("layer_index") for row in backend.get("layers", [])] !=
            list(range(EXPECTED_LAYERS)) or
            any(row.get("resolved_implementation") != "eager"
                for row in backend.get("layers", []))):
        raise DiagnosticError("diagnostic eager backend attestation differs")
    return model, tokenizer, revisions, backend


def run_diagnostic(repo: Path, *, commit: str) -> dict[str, Any]:
    _configure_threads()
    model, tokenizer, revisions, backend = _load_subject()
    try:
        inputs = _load_inputs(repo, tokenizer, commit)
    except DiagnosticError as exc:
        evidence = dict(exc.evidence)
        evidence.update({
            "subject_revisions": revisions,
            "backend_fingerprint": backend,
            "threading": _threading_environment(),
            "input_sources": {
                "case_path": CASE_PATH,
                "replacement_case_path": REPLACEMENT_CASE_PATH,
            },
        })
        raise type(exc)(str(exc), evidence=evidence) from exc
    snapshots: dict[str, Snapshot] = {}
    branch_records = {}
    try:
        for name in ("A", "B", "C"):
            snapshot, record = _run_repeated_branch(
                model, name, inputs["branch_tokens"][name])
            snapshots[name] = snapshot
            branch_records[name] = record
        a_vs_b = compare_region(
            snapshots["A"], snapshots["B"], left_start=0, right_start=0,
            width=FIRST_BLOCK_LENGTH, include_per_row=True,
            label="A_vs_B_first23")
        b_vs_c_first = compare_region(
            snapshots["B"], snapshots["C"], left_start=0, right_start=0,
            width=FIRST_BLOCK_LENGTH, include_per_row=True,
            label="B_vs_C_first23")
        b_vs_c_after = compare_region(
            snapshots["B"], snapshots["C"],
            left_start=FIRST_BLOCK_LENGTH, right_start=FIRST_BLOCK_LENGTH,
            width=PREFIX_LENGTH - FIRST_BLOCK_LENGTH, include_per_row=False,
            label="B_vs_C_after_row22_positive_control")
        decision = classify_three_way(a_vs_b, b_vs_c_first, b_vs_c_after)
    except DiagnosticError as exc:
        evidence = dict(exc.evidence)
        evidence.update({
            "completed_branches": branch_records,
            "subject_revisions": revisions,
            "backend_fingerprint": backend,
            "inputs": inputs["record"],
            "threading": _threading_environment(),
        })
        raise type(exc)(str(exc), evidence=evidence) from exc
    fallback = {
        "trigger": (
            "A_vs_B_first23.bit_exact == true AND "
            "B_vs_C_first23.bit_exact == true"),
        "status": "REQUIRED" if decision["fallback_required"] else "NOT_TRIGGERED",
        "question": "Locate the first full-prefix boundary-dependent cache row.",
        "exact_full_prefix_length": EXPECTED_FULL_PREFIX_LENGTH,
        "exact_full_prefix_sha256": inputs["record"]["full_prefix_sha256"],
        "ordinary_partition": [4096, 4096, 238],
        "first_boundary_isolated_partition": [23, 4073, 4096, 238],
        "requirements": {
            "each_branch_repeats": 2,
            "within_branch_complete_snapshot_bit_identity": True,
            "compare_complete_KV_all_28_layers_all_8430_rows": True,
            "record_first_divergence_layer_row_kind": True,
            "same_model_revision_dtype_backend_device_threads": True,
            "non_authorizing_new_unique_sealed_artifact": True,
        },
    }
    return {
        "schema": SCHEMA,
        "diagnostic_id": DIAGNOSTIC_ID,
        "status": "COMPLETE",
        "authorizing": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "repository": {"commit": commit, "branch": "trunk"},
        "subject": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "torch.bfloat16",
            "device": "cpu",
            "attention_backend": "eager",
            "revisions": revisions,
            "backend_fingerprint": backend,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "threading": _threading_environment(),
        },
        "inputs": inputs["record"],
        "branches": branch_records,
        "comparisons": {
            "A_vs_B_first23": a_vs_b,
            "B_vs_C_first23": b_vs_c_first,
            "B_vs_C_after_row22_positive_control": b_vs_c_after,
        },
        "decision": decision,
        "predeclared_fallback": fallback,
    }


def _atomic_write_new(path: Path, doc: dict[str, Any]) -> None:
    if path.exists():
        raise DiagnosticError(f"refusing to overwrite diagnostic: {path}")
    sealed = seal_payload(doc)
    raw = _canonical(sealed) + b"\n"
    if len(raw) >= MAX_OUTPUT_BYTES:
        raise DiagnosticError(
            f"diagnostic output is not commit-safe: {len(raw)} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise DiagnosticError(f"diagnostic temporary path exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    reread = json.loads(path.read_text())
    verify_sealed_payload(reread)
    if reread != sealed:
        raise DiagnosticError("diagnostic read-back differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/c10_schedule_origin"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    branch = str(_git(repo, "symbolic-ref", "--short", "HEAD")).strip()
    commit = str(_git(repo, "rev-parse", "HEAD")).strip()
    if branch != "trunk" or len(commit) != 40:
        raise SystemExit("diagnostic must run from an exact trunk commit")
    script_relative = Path(__file__).resolve().relative_to(repo).as_posix()
    script_raw = _require_committed_bytes(repo, script_relative, commit)
    output_dir = (repo / args.output_dir).resolve()
    try:
        output_dir_relative = output_dir.relative_to(repo).as_posix()
    except ValueError as exc:
        raise SystemExit("diagnostic output directory must be inside repository") from exc
    if output_dir_relative != "results/c10_schedule_origin":
        raise SystemExit(
            "diagnostic output directory must be results/c10_schedule_origin")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_dir / (
        f"c10_schedule_origin_Qwen3-0.6B_{stamp}.json")
    print(
        f"RUN c10_schedule_origin model={MODEL_ID}@{MODEL_REVISION} "
        f"-> {output.relative_to(repo)}", flush=True)
    try:
        result = run_diagnostic(repo, commit=commit)
        result["repository"].update({
            "diagnostic_script_path": script_relative,
            "diagnostic_script_raw_sha256": _sha256_bytes(script_raw),
        })
        _atomic_write_new(output, result)
        print(
            f"COMPLETE outcome={result['decision']['outcome_class']} "
            f"output={output.relative_to(repo)}", flush=True)
        return 0
    except Exception as exc:
        outcome_class = (
            "NONDETERMINISTIC"
            if isinstance(exc, NondeterminismError)
            else "INVALID" if isinstance(exc, DiagnosticError)
            else "ERROR"
        )
        failure = {
            "schema": SCHEMA,
            "diagnostic_id": DIAGNOSTIC_ID,
            "status": "FAIL",
            "authorizing": False,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "repository": {"commit": commit, "branch": branch},
            "subject": {
                "model": MODEL_ID, "revision": MODEL_REVISION,
                "dtype": "torch.bfloat16", "device": "cpu",
                "attention_backend": "eager",
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
            "decision": {"outcome_class": outcome_class},
            "failure_evidence": getattr(exc, "evidence", {}),
            "diagnostic_script_path": script_relative,
            "diagnostic_script_raw_sha256": _sha256_bytes(script_raw),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "threading": _threading_environment(),
            },
        }
        _atomic_write_new(output, failure)
        print(
            f"FAIL {type(exc).__name__}: {exc} "
            f"output={output.relative_to(repo)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
