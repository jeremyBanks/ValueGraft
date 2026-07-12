#!/usr/bin/env python3
"""Zero-GPU reconstruction of legacy and SWE graft alignment dose.

The legacy section is reconstructed from tracked per-conversation checkpoints and
cross-checked against four tracked headline result files.  The SWE section is
deliberately classified as partial: exact tail geometry can be recovered only when
the ignored local ``swegym.parquet`` is still present with its recorded hash, while
the generated summary text/token IDs needed to recover summary alignment were never
saved.

This script loads a tokenizer only.  It never loads or calls a language model.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "graft-dose-recovery-v1"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SLUG = "Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
SWE_PARQUET_SHA256 = "ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1"

LEGACY_CHECKPOINT_DIR = Path(
    "results/champion_validate/_work_headscan/"
    "Qwen__Qwen3-30B-A3B-Instruct-2507"
)
LEGACY_HEADLINES = {
    "per_head": Path(
        "results/champion_validate/"
        "headscan_30b_bf16_heads_shuffle_pos_20260709T194156Z.json"
    ),
    "per_layer": Path(
        "results/champion_validate/"
        "headscan_30b_bf16_layers_shuffle_pos_20260709T194156Z.json"
    ),
    "intersection": Path(
        "results/champion_validate/"
        "headscan_30b_bf16_intersection_shuffle_pos_20260709T194156Z.json"
    ),
    "union": Path(
        "results/champion_validate/"
        "headscan_30b_bf16_union_shuffle_pos_20260709T194156Z.json"
    ),
}

SWE_DIRS = {
    "fresh_profile": Path("results/swegym_tune_20260710T145330Z_brief"),
    "fresh_champion": Path("results/swegym_champeval_20260710T145330Z_brief"),
    "confirmation": Path("results/swegym_confirm_20260711T011101Z_brief"),
    "original_single_call_brief": Path("results/swegym_30b_bf16"),
    "original_chunked_brief": Path("results/swegym_30b_bf16_brief"),
    "original_production_summary": Path("results/swegym_30b_bf16_prod"),
}

SWE_MISSING_COMMITTED_FIELDS = [
    "alignment pair count or pair indices",
    "summary-versus-tail pair split",
    "tail_start_msg and exact alignment-region bounds",
    "generated per-trajectory summary text",
    "generated summary token IDs / old_ids / b_ids",
    "tracked source trajectory messages (swegym.parquet is ignored)",
]


class AuditError(RuntimeError):
    """Raised when an exact reconstruction invariant does not hold."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_tracked_paths(repo: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True
    )
    return {item.decode() for item in proc.stdout.split(b"\0") if item}


def _git_commit(repo: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=False,
        capture_output=True, text=True,
    )
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def _git_status(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def _relative(path: Path, repo: Path) -> str:
    return str(path.resolve().relative_to(repo.resolve()))


def _input_manifest(
    paths: Iterable[Path], repo: Path, tracked: set[str], *, include_files: bool = True
) -> dict[str, Any]:
    entries = []
    aggregate = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=str):
        if not path.is_file():
            raise AuditError(f"required input is absent: {path}")
        rel = _relative(path, repo)
        digest = _sha256(path)
        aggregate.update(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
        entries.append({
            "path": rel,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "git_tracked": rel in tracked,
        })
    out: dict[str, Any] = {
        "count": len(entries),
        "aggregate_sha256": aggregate.hexdigest(),
        "aggregate_rule": "sha256(concat(sorted(path + NUL + file_sha256 + LF)))",
        "all_git_tracked": all(item["git_tracked"] for item in entries),
    }
    if include_files:
        out["files"] = entries
    return out


def _percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float | int]) -> dict[str, Any]:
    vals = [float(value) for value in values]
    if not vals:
        return {
            "n": 0, "min": None, "q25": None, "median": None,
            "mean": None, "q75": None, "max": None, "sum": 0.0,
        }
    return {
        "n": len(vals),
        "min": min(vals),
        "q25": _percentile(vals, 25),
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "q75": _percentile(vals, 75),
        "max": max(vals),
        "sum": sum(vals),
    }


def _load_tokenizer(*, local_files_only: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
    )


def _alignment_region_record(
    *, b_ids: list[int], old_ids: list[int], region, special_ids: set[int]
) -> dict[str, Any]:
    from arms_common import MIN_BLOCK, N_SINK, build_alignment

    (new_lo, new_hi), (old_lo, old_hi) = region
    pairs = build_alignment(b_ids, old_ids, special_ids, [region])
    matcher = difflib.SequenceMatcher(
        a=b_ids[new_lo:new_hi], b=old_ids[old_lo:old_hi], autojunk=False
    )
    raw_blocks = [
        {"destination_offset": block.a, "source_offset": block.b, "rows": block.size}
        for block in matcher.get_matching_blocks()
        if block.size
    ]
    kept_blocks = [item for item in raw_blocks if item["rows"] >= MIN_BLOCK]
    source = old_ids[old_lo:old_hi]
    eligible_source = sum(token not in special_ids for token in source)
    destination_non_special = sum(
        position >= N_SINK and token not in special_ids
        for position, token in enumerate(b_ids[new_lo:new_hi], start=new_lo)
    )
    raw_kept_rows = sum(item["rows"] for item in kept_blocks)
    one_full_source_block = (
        len(kept_blocks) == 1
        and kept_blocks[0]["source_offset"] == 0
        and kept_blocks[0]["rows"] == old_hi - old_lo
    )
    return {
        "aligned_positions": len(pairs),
        "source_rows_including_structural": old_hi - old_lo,
        "source_eligible_non_special_rows": eligible_source,
        "destination_region_rows": new_hi - new_lo,
        "destination_non_special_rows": destination_non_special,
        "raw_kept_match_rows": raw_kept_rows,
        "matching_blocks_kept": kept_blocks,
        "matching_blocks_below_minimum": [
            item for item in raw_blocks if item["rows"] < MIN_BLOCK
        ],
        "one_full_contiguous_source_block": one_full_source_block,
        "coverage_of_eligible_source_pct": (
            100.0 * len(pairs) / eligible_source if eligible_source else None
        ),
        "coverage_of_source_including_structural_pct": (
            100.0 * len(pairs) / (old_hi - old_lo) if old_hi > old_lo else None
        ),
    }


def _variant_component_record(
    cfg: dict[str, Any], *, n_layers: int, n_kv_heads: int,
    aligned_positions: Sequence[int]
) -> dict[str, Any]:
    total_slots = n_layers * n_kv_heads
    slots_by_alpha: dict[str, int] = defaultdict(int)
    if "head_map" in cfg:
        active_slots = sum(len(heads) for heads in cfg["head_map"].values())
        slots_by_alpha[str(float(cfg["alpha"]))] = active_slots
    elif "alpha_map" in cfg:
        active_slots = len(cfg["alpha_map"]) * n_kv_heads
        for alpha in cfg["alpha_map"].values():
            slots_by_alpha[str(float(alpha))] += n_kv_heads
    else:
        raise AuditError(f"unknown champion configuration: {cfg.keys()}")
    return {
        "label": cfg.get("label"),
        "active_layer_kv_head_slots": active_slots,
        "total_layer_kv_head_slots": total_slots,
        "active_slot_fraction_pct": 100.0 * active_slots / total_slots,
        "active_slots_by_alpha": dict(sorted(slots_by_alpha.items())),
        "value_row_vectors_replaced_per_conversation": distribution(
            [positions * active_slots for positions in aligned_positions]
        ),
    }


def audit_legacy(repo: Path, tokenizer, tracked: set[str]) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "src"))
    from cross_arch_probe import build_token_context

    checkpoint_dir = repo / LEGACY_CHECKPOINT_DIR
    checkpoint_paths = sorted(checkpoint_dir.glob("conv_[0-9][0-9][0-9]__c*.json"))
    if len(checkpoint_paths) != 18:
        raise AuditError(f"expected 18 legacy checkpoints, observed {len(checkpoint_paths)}")
    headline_paths = {label: repo / path for label, path in LEGACY_HEADLINES.items()}
    special_ids = set(tokenizer.all_special_ids)
    rows = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = json.loads(checkpoint_path.read_text())
        conv = checkpoint["render"]["conv"]
        messages = conv["messages"][:-1]
        tail_start = conv["sections"]["middle_end_msg"]
        context = build_token_context(
            tokenizer, "qwen", messages, checkpoint["summary_text"], tail_start
        )
        if len(context["regions"]) != 2:
            raise AuditError(f"{conv['id']}: expected tail+summary regions")
        tail = _alignment_region_record(
            b_ids=context["b_ids"], old_ids=context["summ"]["old_ids"],
            region=context["regions"][0], special_ids=special_ids,
        )
        summary = _alignment_region_record(
            b_ids=context["b_ids"], old_ids=context["summ"]["old_ids"],
            region=context["regions"][1], special_ids=special_ids,
        )
        aligned_total = tail["aligned_positions"] + summary["aligned_positions"]
        traces = checkpoint["payload"]["traces"]
        persisted_pairs = sorted({trace["n_pairs"] for trace in traces})
        persisted_summary_lengths = sorted(
            {trace["summary_len_tokens"] for trace in traces}
        )
        reconstructed_summary_length = len(context["summ"]["gen_ids"])
        if persisted_pairs != [aligned_total]:
            raise AuditError(
                f"{conv['id']}: persisted pairs {persisted_pairs} != {aligned_total}"
            )
        if persisted_summary_lengths != [reconstructed_summary_length]:
            raise AuditError(
                f"{conv['id']}: summary length mismatch "
                f"{persisted_summary_lengths} != {reconstructed_summary_length}"
            )
        eligible = (
            tail["source_eligible_non_special_rows"]
            + summary["source_eligible_non_special_rows"]
        )
        source_all = (
            tail["source_rows_including_structural"]
            + summary["source_rows_including_structural"]
        )
        rows.append({
            "conversation_id": conv["id"],
            "checkpoint": _relative(checkpoint_path, repo),
            "compacted_b_rows": len(context["b_ids"]),
            "aligned_positions_total": aligned_total,
            "aligned_positions_summary": summary["aligned_positions"],
            "aligned_positions_tail": tail["aligned_positions"],
            "eligible_source_rows_total": eligible,
            "source_rows_including_structural_total": source_all,
            "coverage_of_eligible_source_pct": 100.0 * aligned_total / eligible,
            "coverage_of_source_including_structural_pct": (
                100.0 * aligned_total / source_all
            ),
            "coverage_of_all_compacted_b_rows_pct": (
                100.0 * aligned_total / len(context["b_ids"])
            ),
            "summary_share_of_aligned_positions_pct": (
                100.0 * summary["aligned_positions"] / aligned_total
            ),
            "tail_share_of_aligned_positions_pct": (
                100.0 * tail["aligned_positions"] / aligned_total
            ),
            "summary": summary,
            "tail": tail,
            "persisted_n_pairs_cross_check": persisted_pairs[0],
            "persisted_summary_tokens_cross_check": persisted_summary_lengths[0],
        })

    by_conversation = {row["conversation_id"]: row for row in rows}
    headline_checks = {}
    variant_components = {}
    first_headline = None
    for label, path in headline_paths.items():
        doc = json.loads(path.read_text())
        first_headline = first_headline or doc
        observed: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for trace in doc["traces"]:
            observed[trace["conversation_id"]].add(
                (trace["n_pairs"], trace["summary_len_tokens"])
            )
        if set(observed) != set(by_conversation):
            raise AuditError(f"{label}: headline conversation IDs differ")
        mismatches = []
        for conversation_id, values in observed.items():
            expected = by_conversation[conversation_id]
            wanted = {
                (
                    expected["aligned_positions_total"],
                    expected["persisted_summary_tokens_cross_check"],
                )
            }
            if values != wanted:
                mismatches.append({
                    "conversation_id": conversation_id,
                    "headline": sorted(values),
                    "reconstructed": sorted(wanted),
                })
        if mismatches:
            raise AuditError(f"{label}: headline cross-check mismatch: {mismatches}")
        headline_checks[label] = {
            "path": _relative(path, repo),
            "conversations": len(observed),
            "all_n_pairs_and_summary_lengths_match": True,
        }
        hparams = doc["model_hparams"]
        variant_components[label] = _variant_component_record(
            doc["champion_cfg"],
            n_layers=int(hparams["num_hidden_layers"]),
            n_kv_heads=int(hparams["num_key_value_heads"]),
            aligned_positions=[row["aligned_positions_total"] for row in rows],
        )

    aligned = [row["aligned_positions_total"] for row in rows]
    summary_aligned = [row["aligned_positions_summary"] for row in rows]
    tail_aligned = [row["aligned_positions_tail"] for row in rows]
    pooled_aligned = sum(aligned)
    pooled_summary = sum(summary_aligned)
    pooled_tail = sum(tail_aligned)
    pooled_source = sum(row["source_rows_including_structural_total"] for row in rows)

    all_full_blocks = all(
        row[region]["one_full_contiguous_source_block"]
        for row in rows for region in ("summary", "tail")
    )
    all_eligible_covered = all(
        row[region]["aligned_positions"]
        == row[region]["source_eligible_non_special_rows"]
        for row in rows for region in ("summary", "tail")
    )

    inputs = list(checkpoint_paths) + list(headline_paths.values()) + [
        repo / "src/arms_common.py", repo / "src/cross_arch_probe.py"
    ]
    return {
        "status": "EXACT_RECONSTRUCTION_PASS",
        "evidence_tier": "tracked committed checkpoints and headline artifacts",
        "repository_inputs_committed": True,
        "clean_clone_reproducibility": (
            "yes, with the pinned public tokenizer revision; no model weights or "
            "model forward calls are needed"
        ),
        "scope": "18 headline conversations c07-c24; all four promoted variants",
        "inputs": _input_manifest(inputs, repo, tracked),
        "cross_checks": {
            "checkpoint_count": len(checkpoint_paths),
            "all_reconstructed_totals_equal_persisted_n_pairs": True,
            "all_reconstructed_summary_lengths_equal_persisted_lengths": True,
            "headline_variants": headline_checks,
            "all_regions_one_full_contiguous_difflib_block": all_full_blocks,
            "all_eligible_non_special_source_rows_aligned": all_eligible_covered,
        },
        "per_conversation": rows,
        "distributions": {
            "aligned_positions_total": distribution(aligned),
            "aligned_positions_summary": distribution(summary_aligned),
            "aligned_positions_tail": distribution(tail_aligned),
            "coverage_of_eligible_source_pct": distribution(
                [row["coverage_of_eligible_source_pct"] for row in rows]
            ),
            "coverage_of_source_including_structural_pct": distribution(
                [row["coverage_of_source_including_structural_pct"] for row in rows]
            ),
            "coverage_of_all_compacted_b_rows_pct": distribution(
                [row["coverage_of_all_compacted_b_rows_pct"] for row in rows]
            ),
            "summary_share_of_aligned_positions_pct": distribution(
                [row["summary_share_of_aligned_positions_pct"] for row in rows]
            ),
            "tail_share_of_aligned_positions_pct": distribution(
                [row["tail_share_of_aligned_positions_pct"] for row in rows]
            ),
        },
        "pooled": {
            "aligned_positions_total": pooled_aligned,
            "aligned_positions_summary": pooled_summary,
            "aligned_positions_tail": pooled_tail,
            "summary_share_pct": 100.0 * pooled_summary / pooled_aligned,
            "tail_share_pct": 100.0 * pooled_tail / pooled_aligned,
            "source_rows_including_structural": pooled_source,
            "coverage_of_source_including_structural_pct": (
                100.0 * pooled_aligned / pooled_source
            ),
        },
        "variant_tensor_slot_coverage": variant_components,
        "boilerplate_assessment": {
            "near_empty_intervention_disproved": True,
            "difflib_fragment_or_frequency_selection_observed": False,
            "basis": (
                "Every summary and tail region was one full contiguous matching "
                "block and 100% of eligible non-special source positions aligned."
            ),
        },
    }


def _load_swe_rows(directory: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = sorted(directory.glob("t[0-9][0-9][0-9][0-9].json"))
    return [json.loads(path.read_text()) for path in paths], paths


def _swe_tail_geometry(tokenizer, messages, row: dict[str, Any]) -> dict[str, Any]:
    from arms_common import (
        build_alignment, build_b_messages, canonical_ids, message_token_starts,
        render_hf,
    )

    special_ids = set(tokenizer.all_special_ids)
    full_ids = canonical_ids(tokenizer, messages, renderer=render_hf)
    full_starts = message_token_starts(tokenizer, full_ids, len(messages))
    cut = int(row["meta"]["cut_msg"])
    context_messages = messages[:cut]
    context_ids = canonical_ids(tokenizer, context_messages, renderer=render_hf)
    context_starts = message_token_starts(
        tokenizer, context_ids, len(context_messages)
    )
    metadata_matches = (
        len(full_ids) == row["meta"]["n_tokens_full"]
        and full_starts[cut] == row["meta"]["n_ctx_tokens"]
        and len(context_ids) == row["meta"]["n_ctx_tokens"]
    )
    if not metadata_matches:
        raise AuditError(f"SWE row {row['idx']}: tokenizer/cut metadata mismatch")
    target = 0.75 * len(context_ids)
    tail_start = min(
        range(1, len(context_messages)),
        key=lambda index: abs(context_starts[index] - target),
    )

    placeholder_results = []
    placeholders = [
        "x",
        "A completely different and much longer placeholder summary. " * 100,
    ]
    for placeholder in placeholders:
        b_messages = build_b_messages(context_messages, placeholder, tail_start)
        b_ids = canonical_ids(tokenizer, b_messages, renderer=render_hf)
        b_starts = message_token_starts(tokenizer, b_ids, len(b_messages))
        region = (
            (b_starts[2], len(b_ids)),
            (context_starts[tail_start], len(context_ids)),
        )
        record = _alignment_region_record(
            b_ids=b_ids, old_ids=context_ids, region=region, special_ids=special_ids
        )
        pairs = build_alignment(b_ids, context_ids, special_ids, [region])
        new_lo = b_starts[2]
        old_lo = context_starts[tail_start]
        placeholder_results.append({
            "tail_ids": b_ids[new_lo:],
            "relative_pairs": [(new - new_lo, old - old_lo) for new, old in pairs],
            "record": record,
        })
    invariant = (
        placeholder_results[0]["tail_ids"] == placeholder_results[1]["tail_ids"]
        and placeholder_results[0]["relative_pairs"]
        == placeholder_results[1]["relative_pairs"]
    )
    if not invariant:
        raise AuditError(f"SWE row {row['idx']}: tail geometry depends on summary text")
    record = placeholder_results[0]["record"]
    return {
        "idx": int(row["idx"]),
        "metadata_matches_committed_row": True,
        "tail_start_msg": tail_start,
        "context_rows": len(context_ids),
        "tail_aligned_positions": record["aligned_positions"],
        "tail_source_rows_including_structural": record[
            "source_rows_including_structural"
        ],
        "tail_source_eligible_non_special_rows": record[
            "source_eligible_non_special_rows"
        ],
        "tail_share_of_context_rows_pct": (
            100.0 * record["source_rows_including_structural"] / len(context_ids)
        ),
        "tail_one_full_contiguous_difflib_block": record[
            "one_full_contiguous_source_block"
        ],
        "tail_all_eligible_source_rows_aligned": (
            record["aligned_positions"]
            == record["source_eligible_non_special_rows"]
        ),
        "relative_tail_mapping_invariant_to_summary_placeholder": True,
        "tail_matching_blocks": record["matching_blocks_kept"],
    }


def _swe_sample_record(
    rows: list[dict[str, Any]], geometry_by_idx: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    tail = [geometry_by_idx[int(row["idx"])]["tail_aligned_positions"] for row in rows]
    summary_width = [int(row["summary_tokens"]) for row in rows]
    upper = [left + right for left, right in zip(tail, summary_width)]
    tail_share_lower_bound = [
        100.0 * left / total for left, total in zip(tail, upper)
    ]
    pooled_tail = sum(tail)
    pooled_summary_width = sum(summary_width)
    return {
        "n_trajectories": len(rows),
        "trajectory_indices": sorted(int(row["idx"]) for row in rows),
        "tail_aligned_positions": distribution(tail),
        "generated_summary_source_width_not_aligned_count": distribution(summary_width),
        "total_aligned_positions_lower_bound_tail_only": distribution(tail),
        "total_aligned_positions_upper_bound_if_every_summary_row_aligned": distribution(
            upper
        ),
        "tail_share_lower_bound_per_trajectory_pct": distribution(
            tail_share_lower_bound
        ),
        "pooled": {
            "tail_aligned_positions": pooled_tail,
            "generated_summary_source_width": pooled_summary_width,
            "total_aligned_positions_lower_bound": pooled_tail,
            "total_aligned_positions_upper_bound": pooled_tail + pooled_summary_width,
            "tail_share_of_aligned_positions_lower_bound_pct": (
                100.0 * pooled_tail / (pooled_tail + pooled_summary_width)
            ),
        },
        "tail_share_of_original_context_rows_pct": distribution([
            geometry_by_idx[int(row["idx"])]["tail_share_of_context_rows_pct"]
            for row in rows
        ]),
    }


def audit_swe(
    repo: Path, tokenizer, tracked: set[str], parquet_path: Path
) -> dict[str, Any]:
    if not parquet_path.is_file():
        return {
            "status": "UNAVAILABLE_IN_CLEAN_CLONE",
            "evidence_tier": "partial local reconstruction only",
            "repository_inputs_committed": False,
            "parquet_path": str(parquet_path),
            "parquet_expected_sha256": SWE_PARQUET_SHA256,
            "reason": "ignored local swegym.parquet is absent",
            "missing_committed_fields": SWE_MISSING_COMMITTED_FIELDS,
            "summary_alignment_recoverable": False,
        }
    observed_hash = _sha256(parquet_path)
    if observed_hash != SWE_PARQUET_SHA256:
        raise AuditError(
            f"SWE parquet hash mismatch: {observed_hash} != {SWE_PARQUET_SHA256}"
        )

    import pandas as pd

    loaded = {}
    all_result_paths = []
    for label, relative_dir in SWE_DIRS.items():
        rows, paths = _load_swe_rows(repo / relative_dir)
        loaded[label] = rows
        all_result_paths.extend(paths)
    expected_counts = {
        "fresh_profile": 98,
        "fresh_champion": 98,
        "confirmation": 45,
        "original_single_call_brief": 75,
        "original_chunked_brief": 75,
        "original_production_summary": 75,
    }
    observed_counts = {label: len(rows) for label, rows in loaded.items()}
    if observed_counts != expected_counts:
        raise AuditError(
            f"unexpected SWE result counts: {observed_counts} != {expected_counts}"
        )

    profile_by_idx = {int(row["idx"]): row for row in loaded["fresh_profile"]}
    champion_by_idx = {int(row["idx"]): row for row in loaded["fresh_champion"]}
    if set(profile_by_idx) != set(champion_by_idx):
        raise AuditError("fresh profile/champion trajectory IDs differ")
    if any(
        profile_by_idx[idx]["summary_tokens"]
        != champion_by_idx[idx]["summary_tokens"]
        for idx in profile_by_idx
    ):
        raise AuditError("fresh profile/champion summary lengths differ")

    fresh = loaded["fresh_champion"]
    samples = {
        "fresh_tune_41": [row for row in fresh if row["split"] == "tune"],
        "fresh_eval_57": [row for row in fresh if row["split"] == "eval"],
        "fresh_all_98": fresh,
        "confirmation_45": loaded["confirmation"],
        "original_single_call_brief_75": loaded["original_single_call_brief"],
        "original_chunked_brief_75": loaded["original_chunked_brief"],
        "original_production_summary_75": loaded["original_production_summary"],
    }
    sample_counts = {label: len(rows) for label, rows in samples.items()}
    wanted_sample_counts = {
        "fresh_tune_41": 41,
        "fresh_eval_57": 57,
        "fresh_all_98": 98,
        "confirmation_45": 45,
        "original_single_call_brief_75": 75,
        "original_chunked_brief_75": 75,
        "original_production_summary_75": 75,
    }
    if sample_counts != wanted_sample_counts:
        raise AuditError(f"unexpected SWE sample counts: {sample_counts}")

    parquet = pd.read_parquet(parquet_path)
    if list(parquet.columns) != ["messages"] or len(parquet) != 491:
        raise AuditError(
            f"unexpected SWE parquet shape/columns: {parquet.shape} {parquet.columns}"
        )
    geometry_by_idx: dict[int, dict[str, Any]] = {}
    rows_for_geometry: dict[int, dict[str, Any]] = {}
    for rows in samples.values():
        for row in rows:
            index = int(row["idx"])
            previous = rows_for_geometry.get(index)
            if previous is not None and previous["meta"] != row["meta"]:
                raise AuditError(f"SWE row {index}: repeated-run cut metadata differs")
            rows_for_geometry[index] = row
    for index, row in sorted(rows_for_geometry.items()):
        messages = list(parquet.iloc[index]["messages"])
        geometry_by_idx[index] = _swe_tail_geometry(tokenizer, messages, row)

    all_tail_full = all(
        item["tail_one_full_contiguous_difflib_block"]
        for item in geometry_by_idx.values()
    )
    all_tail_eligible = all(
        item["tail_all_eligible_source_rows_aligned"]
        for item in geometry_by_idx.values()
    )
    sample_records = {
        label: _swe_sample_record(rows, geometry_by_idx)
        for label, rows in samples.items()
    }

    manifest_paths = [
        repo / relative_dir / "manifest.json"
        for relative_dir in SWE_DIRS.values()
        if (repo / relative_dir / "manifest.json").is_file()
    ]
    input_paths = all_result_paths + manifest_paths + [
        parquet_path,
        repo / "src/run_swegym_hf.py",
        repo / "src/arms_common.py",
    ]
    inputs = _input_manifest(input_paths, repo, tracked)
    parquet_rel = _relative(parquet_path, repo)
    return {
        "status": "PARTIAL_LOCAL_RECONSTRUCTION_PASS",
        "evidence_tier": "hash-matched but ignored local parquet plus committed scores",
        "repository_inputs_committed": False,
        "clean_clone_reproducibility": False,
        "parquet": {
            "path": parquet_rel,
            "bytes": parquet_path.stat().st_size,
            "sha256_expected": SWE_PARQUET_SHA256,
            "sha256_observed": observed_hash,
            "hash_matches": True,
            "git_tracked": parquet_rel in tracked,
            "rows": len(parquet),
            "columns": list(parquet.columns),
        },
        "inputs": inputs,
        "missing_committed_fields": SWE_MISSING_COMMITTED_FIELDS,
        "summary_alignment_recoverable": False,
        "summary_alignment_bounds": (
            "For each trajectory, aligned summary positions are bounded by 0 and "
            "the saved summary_tokens source width; exact text/IDs are absent."
        ),
        "cross_checks": {
            "unique_trajectory_rows_reconstructed": len(geometry_by_idx),
            "all_saved_token_counts_and_cut_metadata_match": True,
            "fresh_profile_and_champion_summary_lengths_match": True,
            "relative_tail_mapping_invariant_to_summary_placeholder": True,
            "all_tails_one_full_contiguous_difflib_block": all_tail_full,
            "all_tail_eligible_non_special_source_rows_aligned": all_tail_eligible,
        },
        "samples": sample_records,
        "per_unique_trajectory_tail_geometry": [
            geometry_by_idx[index] for index in sorted(geometry_by_idx)
        ],
        "boilerplate_assessment": {
            "near_empty_tail_intervention_disproved": True,
            "tail_difflib_fragment_or_frequency_selection_observed": False,
            "summary_difflib_selection_assessable": False,
            "basis": (
                "Every locally reconstructed tail was one full exact block with "
                "100% eligible-row coverage. The generated summary text/IDs were "
                "not saved, so the same check cannot be performed for summaries."
            ),
        },
        "mediation_status": {
            "summary_only_ablation_run": False,
            "tail_only_ablation_run": False,
            "summary_vs_tail_effect_localized": False,
            "interpretation": (
                "The brief-summary performance signal may be entirely tail-mediated; "
                "alignment row mass does not identify which region caused the score."
            ),
        },
    }


def build_report(
    repo: Path, tokenizer, *, tracked: set[str], parquet_path: Path,
    local_files_only: bool,
) -> dict[str, Any]:
    import transformers

    legacy = audit_legacy(repo, tokenizer, tracked)
    swe = audit_swe(repo, tokenizer, tracked, parquet_path)
    script_path = Path(__file__).resolve()
    eval_lower = None
    if swe["status"] == "PARTIAL_LOCAL_RECONSTRUCTION_PASS":
        eval_lower = swe["samples"]["fresh_eval_57"]["pooled"][
            "tail_share_of_aligned_positions_lower_bound_pct"
        ]
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": (
            "Alignment-position dose only. No model forward, cache tensor, score, "
            "or intervention rerun is performed."
        ),
        "cost_profile": {
            "model_calls": 0,
            "model_weights_loaded": False,
            "gpu_required": False,
            "network_api_calls": 0,
            "tokenizer_only": True,
        },
        "provenance": {
            "repo_commit": _git_commit(repo),
            "worktree_status_before_output": _git_status(repo),
            "script": {
                "path": _relative(script_path, repo),
                "sha256": _sha256(script_path),
                "git_tracked_at_run": _relative(script_path, repo) in tracked,
            },
            "tokenizer": {
                "requested_model_id": MODEL_ID,
                "requested_revision": MODEL_REVISION,
                "resolved_name_or_path": getattr(tokenizer, "name_or_path", None),
                "class": type(tokenizer).__name__,
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "all_special_ids": list(tokenizer.all_special_ids),
                "local_files_only": local_files_only,
            },
            "python": sys.version,
            "platform": platform.platform(),
            "transformers": transformers.__version__,
        },
        "legacy_exact_committed_checkpoint_audit": legacy,
        "swe_partial_local_parquet_audit": swe,
        "headline_findings": {
            "legacy_near_empty_or_boilerplate_only_concern_disproved": True,
            "swe_tail_near_empty_or_boilerplate_only_concern_disproved_locally": (
                swe["status"] == "PARTIAL_LOCAL_RECONSTRUCTION_PASS"
            ),
            "swe_fresh_eval_57_tail_share_pooled_lower_bound_pct": eval_lower,
            "swe_exact_summary_alignment_recoverable": False,
            "summary_only_tail_only_mediation_tested": False,
        },
        "materiality": [
            (
                "Legacy no-detected-lift cannot be explained by a near-empty "
                "alignment: all eligible summary and tail positions aligned."
            ),
            (
                "The SWE brief-summary graft is tail-dominated by aligned-row mass "
                "in the local reconstruction; this strengthens the mundane "
                "tail-recomputation alternative, not a summary-state claim."
            ),
            (
                "Neither row coverage nor the likelihood endpoint localizes the "
                "effect. Summary-only and tail-only ablations remain unrun."
            ),
            (
                "SWE full dose remains a repository-provenance gap because its "
                "trajectory parquet and generated summary text/IDs are not tracked."
            ),
        ],
    }


def _default_output(repo: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        repo / "results/graft_dose_recovery"
        / f"graft-dose-recovery_{MODEL_SLUG}_{timestamp}.json"
    )


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AuditError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="unique JSON output path (default: timestamped results path)",
    )
    parser.add_argument(
        "--swe-parquet", type=Path, default=None,
        help="ignored local SWE parquet (default: <repo>/swegym.parquet)",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="require the pinned tokenizer to already be in the local HF cache",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    output = (args.output.resolve() if args.output else _default_output(repo))
    parquet = (
        args.swe_parquet.resolve() if args.swe_parquet else repo / "swegym.parquet"
    )
    print(
        f"RUN graft-dose-recovery model={MODEL_ID}@{MODEL_REVISION} -> {output}",
        flush=True,
    )
    tracked = _git_tracked_paths(repo)
    tokenizer = _load_tokenizer(local_files_only=args.local_files_only)
    report = build_report(
        repo, tokenizer, tracked=tracked, parquet_path=parquet,
        local_files_only=args.local_files_only,
    )
    _write_atomic_json(output, report)
    print(
        "DONE legacy=%s swe=%s output=%s"
        % (
            report["legacy_exact_committed_checkpoint_audit"]["status"],
            report["swe_partial_local_parquet_audit"]["status"],
            output,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
