"""Pure deterministic controls for the v12 coherent-state canary.

This module constructs one preregistered nonsemantic value-row perturbation at
a time.  It owns no cache and retains no state between calls.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import torch


PLACEBO_SEED = "coherent-state-v12-placebo-20260711"
_CASE_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class CanaryControlError(ValueError):
    """The requested deterministic control violated its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryControlError(message)


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _rademacher(
    count: int, *, case_id: str, layer_index: int, row_index: int, attempt: int,
) -> tuple[torch.Tensor, str]:
    seed_material = (
        f"{PLACEBO_SEED}\0case={case_id}\0layer={layer_index}"
        f"\0row={row_index}\0attempt={attempt}"
    )
    signs: list[float] = []
    block = 0
    while len(signs) < count:
        digest = hashlib.sha256(
            seed_material.encode("utf-8") + b"\0block=" + str(block).encode("ascii")
        ).digest()
        for byte in digest:
            for bit in range(8):
                signs.append(1.0 if ((byte >> bit) & 1) else -1.0)
                if len(signs) == count:
                    break
            if len(signs) == count:
                break
        block += 1
    return torch.tensor(signs, dtype=torch.float64), seed_material


def _cosine(dot: float, left_norm: float, right_norm: float) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def norm_matched_value_placebo_row(
    fresh: torch.Tensor,
    correct: torch.Tensor,
    wrong: torch.Tensor,
    *,
    case_id: str,
    layer_index: int,
    row_index: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``fresh + u`` with ``u`` orthogonal and norm-matched to C-W.

    Projection and scaling occur deterministically in float64 on CPU.  The
    returned tensor uses ``fresh``'s dtype and device.  Diagnostics distinguish
    the exact pre-cast construction from the perturbation actually applied
    after rounding to the source dtype.
    """
    _require(bool(_CASE_ID.fullmatch(case_id)), "invalid case_id")
    _require(isinstance(layer_index, int) and layer_index >= 0,
             "layer_index must be a nonnegative integer")
    _require(isinstance(row_index, int) and row_index >= 0,
             "row_index must be a nonnegative integer")
    _require(isinstance(fresh, torch.Tensor), "fresh must be a tensor")
    _require(isinstance(correct, torch.Tensor), "correct must be a tensor")
    _require(isinstance(wrong, torch.Tensor), "wrong must be a tensor")
    _require(fresh.shape == correct.shape == wrong.shape and fresh.numel() > 0,
             "fresh/correct/wrong row shapes must match and be nonempty")
    _require(fresh.dtype == correct.dtype == wrong.dtype,
             "fresh/correct/wrong row dtypes must match")
    _require(fresh.dtype in (torch.float16, torch.bfloat16, torch.float32,
                             torch.float64),
             "value rows must use a floating dtype")
    _require(torch.isfinite(fresh).all().item(), "fresh row is nonfinite")
    _require(torch.isfinite(correct).all().item(), "correct row is nonfinite")
    _require(torch.isfinite(wrong).all().item(), "wrong row is nonfinite")

    shape = list(fresh.shape)
    fresh64 = fresh.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    correct64 = correct.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    wrong64 = wrong.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    delta = correct64 - wrong64
    delta_norm = float(torch.linalg.vector_norm(delta).item())

    if delta_norm == 0.0:
        result = fresh.detach().clone()
        diagnostics = {
            "schema": "coherent_canary_v12_value_placebo_row_v1",
            "case_id": case_id,
            "layer_index": layer_index,
            "row_index": row_index,
            "shape": shape,
            "dtype": str(fresh.dtype),
            "seed_base": PLACEBO_SEED,
            "seed_material": None,
            "projection_attempt": None,
            "target_delta_l2": 0.0,
            "pre_cast_u_l2": 0.0,
            "pre_cast_dot_with_delta": 0.0,
            "pre_cast_cosine_with_delta": 0.0,
            "applied_delta_l2": 0.0,
            "applied_dot_with_delta": 0.0,
            "applied_cosine_with_delta": 0.0,
            "applied_relative_norm_error": 0.0,
            "fresh_sha256": _tensor_sha256(fresh),
            "correct_sha256": _tensor_sha256(correct),
            "wrong_sha256": _tensor_sha256(wrong),
            "placebo_sha256": _tensor_sha256(result),
            "zero_semantic_delta": True,
        }
        return result, diagnostics

    delta_sq = float(torch.dot(delta, delta).item())
    projected = None
    seed_material = None
    attempt = 0
    while attempt < 1024:
        random, candidate_seed = _rademacher(
            delta.numel(), case_id=case_id, layer_index=layer_index,
            row_index=row_index, attempt=attempt,
        )
        candidate = random - (torch.dot(random, delta) / delta_sq) * delta
        candidate_norm = float(torch.linalg.vector_norm(candidate).item())
        if math.isfinite(candidate_norm) and candidate_norm > 0.0:
            projected = candidate * (delta_norm / candidate_norm)
            seed_material = candidate_seed
            break
        attempt += 1
    _require(projected is not None and seed_material is not None,
             "failed to construct a nonzero orthogonal direction")

    pre_norm = float(torch.linalg.vector_norm(projected).item())
    pre_dot = float(torch.dot(projected, delta).item())
    result_cpu = (fresh64 + projected).reshape(shape).to(dtype=fresh.dtype)
    applied = result_cpu.to(dtype=torch.float64).reshape(-1) - fresh64
    applied_norm = float(torch.linalg.vector_norm(applied).item())
    applied_dot = float(torch.dot(applied, delta).item())
    result = result_cpu.to(device=fresh.device)

    diagnostics = {
        "schema": "coherent_canary_v12_value_placebo_row_v1",
        "case_id": case_id,
        "layer_index": layer_index,
        "row_index": row_index,
        "shape": shape,
        "dtype": str(fresh.dtype),
        "seed_base": PLACEBO_SEED,
        "seed_material": seed_material,
        "projection_attempt": attempt,
        "target_delta_l2": delta_norm,
        "pre_cast_u_l2": pre_norm,
        "pre_cast_dot_with_delta": pre_dot,
        "pre_cast_cosine_with_delta": _cosine(pre_dot, pre_norm, delta_norm),
        "applied_delta_l2": applied_norm,
        "applied_dot_with_delta": applied_dot,
        "applied_cosine_with_delta": _cosine(
            applied_dot, applied_norm, delta_norm),
        "applied_relative_norm_error": abs(applied_norm - delta_norm) / delta_norm,
        "fresh_sha256": _tensor_sha256(fresh),
        "correct_sha256": _tensor_sha256(correct),
        "wrong_sha256": _tensor_sha256(wrong),
        "placebo_sha256": _tensor_sha256(result),
        "zero_semantic_delta": False,
    }
    return result, diagnostics
