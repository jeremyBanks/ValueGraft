#!/usr/bin/env python3
"""Losslessly reconstruct e01 and decompose focal phrase contrasts by token."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_PACKAGE = Path(
    "results/coherent_canary_v12_treatment_package/"
    "coherent-canary-v12-treatment-e01_exact-subject_20260712T040237879370Z"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reconstruct(package_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = package_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "lossless-json-gzip-base64-package/v1":
        raise ValueError("unexpected package schema")

    compressed_parts: list[bytes] = []
    for expected_index, descriptor in enumerate(manifest["chunks"], start=1):
        if descriptor["index"] != expected_index:
            raise ValueError("chunk indices are not contiguous")
        chunk_path = package_dir / descriptor["name"]
        chunk_bytes = chunk_path.read_bytes()
        if len(chunk_bytes) != descriptor["file_size_bytes"]:
            raise ValueError(f"chunk size differs: {chunk_path}")
        if sha256(chunk_bytes) != descriptor["file_sha256"]:
            raise ValueError(f"chunk hash differs: {chunk_path}")
        chunk = json.loads(chunk_bytes)
        if chunk.get("payload_encoding") != "base64":
            raise ValueError("unexpected chunk payload encoding")
        segment = base64.b64decode(chunk["payload"], validate=True)
        if len(segment) != descriptor["compressed_segment_size_bytes"]:
            raise ValueError("compressed segment size differs")
        if sha256(segment) != descriptor["compressed_segment_sha256"]:
            raise ValueError("compressed segment hash differs")
        if chunk["compressed_offset_bytes"] != sum(
                len(part) for part in compressed_parts):
            raise ValueError("compressed segment offset differs")
        compressed_parts.append(segment)

    compressed = b"".join(compressed_parts)
    compression = manifest["compression"]
    if len(compressed) != compression["compressed_size_bytes"]:
        raise ValueError("compressed package size differs")
    if sha256(compressed) != compression["compressed_sha256"]:
        raise ValueError("compressed package hash differs")
    raw = gzip.decompress(compressed)
    original = manifest["original"]
    if len(raw) != original["size_bytes"]:
        raise ValueError("raw artifact size differs")
    if sha256(raw) != original["sha256"]:
        raise ValueError("raw artifact hash differs")
    return manifest, json.loads(raw)


def token_differences(left: dict[str, Any], right: dict[str, Any],
                      target: str) -> list[float]:
    left_values = left[target]["token_logprobs"]
    right_values = right[target]["token_logprobs"]
    if len(left_values) != len(right_values):
        raise ValueError(f"{target} token lengths differ")
    return [float(a - b) for a, b in zip(left_values, right_values)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def decompose(raw: dict[str, Any]) -> dict[str, Any]:
    treatment = raw["treatment"]
    primary = [arm for arm in treatment["arms"]
               if arm["arm_kind"] == "primary"]
    arms = {(arm["schedule"], arm["region"], arm["cell"]): arm
            for arm in primary}
    requested = [
        ("N", "R2_boundary", "full_KV", "CC", "WW"),
        ("N", "R2_boundary", "value_only", "FC", "FW"),
        ("P", "R2_boundary", "full_KV", "CC", "WW"),
        ("P", "R2_boundary", "value_only", "FC", "FW"),
    ]
    decompositions: list[dict[str, Any]] = []
    for schedule, region, family, correct_cell, wrong_cell in requested:
        left = arms[(schedule, region, correct_cell)]["scores"]["focal"]
        right = arms[(schedule, region, wrong_cell)]["scores"]["focal"]
        correct = token_differences(left, right, "correct")
        counter = token_differences(left, right, "counterfactual")
        if len(correct) != len(counter):
            raise ValueError("correct/counterfactual token lengths differ")
        semantic = [a - b for a, b in zip(correct, counter)]
        decompositions.append({
            "schedule": schedule,
            "region": region,
            "family": family,
            "contrast": f"{correct_cell}-{wrong_cell}",
            "correct_target_text": left["correct_text"],
            "correct_target_token_ids": left["correct"]["target_token_ids"],
            "counterfactual_target_text": left["counterfactual_text"],
            "counterfactual_target_token_ids":
                left["counterfactual"]["target_token_ids"],
            "correct_target_token_logprob_differences": correct,
            "counterfactual_target_token_logprob_differences": counter,
            "semantic_margin_token_differences": semantic,
            "correct_target_mean_difference": mean(correct),
            "counterfactual_target_mean_difference": mean(counter),
            "semantic_margin_mean_difference": mean(semantic),
            "first_token_semantic_direction_positive": semantic[0] > 0,
            "all_token_semantic_directions_positive":
                all(value > 0 for value in semantic),
        })

    focal_generations = sorted({
        arm["scores"]["focal"]["generation"]["decoded_content"]
        for arm in primary
    })
    nonfocal_generations = sorted({
        arm["scores"]["nonfocal"]["generation"]["decoded_content"]
        for arm in primary
    })
    return {
        "schema": "coherent-canary-v12-e01-token-decomposition/v1",
        "case_id": treatment["case_id"],
        "design_id": treatment["design_id"],
        "subject": {
            "model": raw["runtime_fingerprint"]["requested_model"],
            "revision": raw["runtime_fingerprint"]["requested_revision"],
            "dtype": raw["runtime_fingerprint"]["dtype"],
            "attention_backend":
                raw["runtime_fingerprint"]["attention_backend"],
        },
        "primary_arm_count": len(primary),
        "available_placebo_control_count":
            treatment["available_placebo_control_count"],
        "unique_primary_focal_generations": focal_generations,
        "unique_primary_nonfocal_generations": nonfocal_generations,
        "decompositions": decompositions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest, raw = reconstruct(args.package_dir)
    if args.output is not None:
        print(
            "RUN coherent-canary-v12-e01-token-decomposition "
            f"model={raw['runtime_fingerprint']['requested_model']} "
            f"-> {args.output}", file=sys.stderr
        )
    result = decompose(raw)
    result["source"] = {
        "package_dir": str(args.package_dir),
        "package_manifest_sha256":
            sha256((args.package_dir / "manifest.json").read_bytes()),
        "raw_artifact_sha256": manifest["original"]["sha256"],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
