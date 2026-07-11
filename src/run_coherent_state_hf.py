"""Production driver for the preregistered coherent-summary-state experiment."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from analyze_coherent_state import analyze, load_checkpoints
from arms_common import SUMMARY_REQUEST
from coherent_state_calibration import run_calibration
from coherent_state_cases import (
    FROZEN_ORDER,
    WRONG_DONOR,
    correct_source_messages,
    frozen_scenarios,
    load_and_validate_targets,
    load_scenarios,
    select_primary_plants,
)
from coherent_state_hf import (
    CoherentStateError,
    replace_summary_rows,
    row_hashes,
    sha256_ids,
)
from coherent_state_runtime import (
    GAPPED_ARM_NAMES,
    append_gapped_post_summary,
    build_gapped_fresh_boundary,
    capture_forced_summary,
    capture_forced_prefix_ids,
    capture_generated_summary,
    complete_assistant_context,
    gapped_arm_boundary,
    score_arm,
    score_target,
    validate_position_schedule,
    validate_generated_replay,
)
from coherent_state_tokens import matched_wrong_prefix_ids
from coherent_state_store import (
    ArtifactError,
    atomic_write_json,
    checkpoint_path,
    promote_checkpoint,
    read_checkpoint,
    validate_scored_checkpoint,
)
from cross_arch_probe import native_render_specs, trim_capped_reply
from l_coherent_state_hf import run_loaded_gapped_gates


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STRUCTURAL_SEED = 20_260_711
PLACEBO_SEED = 20_260_711
ARTIFACT_SCHEMA = 2
DESIGN_ID = "coherent-state-gapped-v1"
AMENDMENT_ID = "COHERENT-STATE-PREREGISTRATION-AMENDMENT-1"
AMENDMENT_PATH = Path("COHERENT-STATE-PREREGISTRATION-AMENDMENT-1.md")
EXPECTED_GEOMETRY = {
    "layers": 48, "attention_heads": 32, "kv_heads": 4,
    "head_dim": 128, "rope_theta": 10_000_000,
}
MAX_REPLY_TOKENS = 320
MAX_SUMMARY_TOKENS = 900
IDENTITY_TOLERANCE = 1e-4
PLACEBO_MOMENT_TOLERANCE = 0.02
PLACEBO_QUANTIZATION_TOLERANCE = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_provenance(run_dir: Path) -> dict:
    commit = git_value("rev-parse", "HEAD")
    # --untracked-files=all is load-bearing: default porcelain collapses a new
    # untracked results tree to its parent directory, which would make a valid
    # leaf run look dirty outside itself on invocation 2 of the resume probe.
    dirty_rows = [x for x in git_value(
        "status", "--porcelain", "--untracked-files=all").splitlines() if x]
    repo = Path(git_value("rev-parse", "--show-toplevel")).resolve()
    allowed = run_dir.resolve()
    disallowed = []
    for row in dirty_rows:
        raw = row[3:].split(" -> ")[-1]
        path = (repo / raw).resolve()
        if path != allowed and allowed not in path.parents:
            disallowed.append(row)
    if disallowed:
        raise CoherentStateError(
            f"production code worktree is dirty outside run directory: {disallowed}")
    return {
        "code_commit": commit, "worktree_dirty_rows": dirty_rows,
        "only_run_directory_dirty": bool(dirty_rows),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def model_geometry(config) -> dict:
    cfg = getattr(config, "text_config", config)
    rp = getattr(cfg, "rope_parameters", None) or {}
    theta = getattr(cfg, "rope_theta", None) or rp.get("rope_theta")
    return {
        "layers": int(cfg.num_hidden_layers),
        "attention_heads": int(cfg.num_attention_heads),
        "kv_heads": int(cfg.num_key_value_heads),
        "head_dim": int(getattr(cfg, "head_dim",
                                cfg.hidden_size // cfg.num_attention_heads)),
        "rope_theta": int(theta),
    }


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def prepare_subject_metadata():
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    resolved = getattr(config, "_commit_hash", None)
    if resolved != REVISION:
        raise CoherentStateError(f"resolved revision {resolved} != {REVISION}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    metadata = {
        "resolved_model_revision": resolved,
        "config_sha256": sha256_json(config.to_dict()),
        "tokenizer_revision_requested": REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_sha256": sha256_json(tokenizer.get_vocab()),
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode()).hexdigest(),
        "special_tokens_map_sha256": sha256_json(tokenizer.special_tokens_map),
    }
    return config, tokenizer, metadata


def load_subject(config, tokenizer):
    if not torch.cuda.is_available():
        raise CoherentStateError("paid production driver requires CUDA")
    geometry = model_geometry(config)
    if geometry != EXPECTED_GEOMETRY:
        raise CoherentStateError(
            f"model geometry {geometry} != {EXPECTED_GEOMETRY}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, device_map=None)
    model.to("cuda")
    model.eval()
    floating = {p.dtype for p in model.parameters() if p.is_floating_point()}
    devices = {p.device.type for p in model.parameters()}
    if floating != {torch.bfloat16}:
        raise CoherentStateError(f"live parameter dtypes are {floating}, not bf16")
    if devices != {"cuda"}:
        raise CoherentStateError(f"live parameter devices are {devices}, not CUDA")
    return model, tokenizer, geometry


def _serializable_source(capture) -> dict:
    return {
        "source_kind": capture.source_kind,
        "prefix_token_ids": capture.prefix_ids,
        "prefix_sha256": capture.prefix_sha256,
        "prefix_token_count": len(capture.prefix_ids),
        "summary_token_ids": capture.summary_ids,
        "summary_token_sha256": sha256_ids(capture.summary_ids),
        "summary_start": capture.summary_start,
        "summary_end": capture.summary_end,
        "prefix_position_ids": list(range(len(capture.prefix_ids))),
        "summary_position_ids": list(range(
            capture.summary_start, capture.summary_end)),
        "physical_cache_position_ids": list(range(
            len(capture.prefix_ids) + len(capture.summary_ids))),
        "summary_row_hashes": capture.row_hashes,
        "trace": capture.trace,
    }


def _tensor_snapshots_equal(a, b) -> bool:
    return len(a) == len(b) and all(
        torch.equal(ka, kb) and torch.equal(va, vb)
        for (ka, va), (kb, vb) in zip(a, b))


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _static_design_self_check() -> None:
    """Keep retired packed-position machinery unreachable from production."""
    tree = ast.parse(Path(__file__).read_text())
    banned = {"move" + "_key_rows", "run_loaded" + "_kernel_gates"}
    reached = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in banned:
            reached.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in banned:
            reached.add(node.attr)
    if reached:
        raise CoherentStateError(
            f"retired packed-position helper reachable in driver: {sorted(reached)}")


def _snapshot_storage_lengths(snapshot) -> list[int]:
    return [int(k.shape[-2]) for k, _ in snapshot]


def _snapshot_hashes(snapshot) -> list[dict]:
    return row_hashes(snapshot)


def _assert_boundary_intervention(arm: str, fresh, branch, layout,
                                  correct_rows, wrong_rows) -> dict:
    """Prove a branch is still pre-tail and changes only its declared rows."""
    s0, s1 = layout.physical_summary_start, layout.physical_summary_end
    expected_len = s1
    fresh_lengths = _snapshot_storage_lengths(fresh)
    branch_lengths = _snapshot_storage_lengths(branch)
    if (any(n != expected_len for n in fresh_lengths) or
            branch_lengths != fresh_lengths):
        raise CoherentStateError(
            f"{arm}: branch is not an exact pre-tail summary boundary")
    summary_expectation = {
        "G_correct": (correct_rows, True, True),
        "G_wrong": (wrong_rows, True, True),
        "G_Vcorrect": (correct_rows, False, True),
        "G_Kcorrect": (correct_rows, True, False),
    }.get(arm)
    for layer, ((kf, vf), (kb, vb)) in enumerate(zip(fresh, branch)):
        for label, base, candidate in (("K", kf, kb), ("V", vf, vb)):
            if (not torch.equal(base[..., :s0, :], candidate[..., :s0, :]) or
                    not torch.equal(base[..., s1:, :], candidate[..., s1:, :])):
                raise CoherentStateError(
                    f"{arm}: {label} changed outside summary at layer {layer}")
        if arm == "G_fresh":
            if not torch.equal(kf, kb) or not torch.equal(vf, vb):
                raise CoherentStateError("G_fresh boundary is not bit-identical")
        elif summary_expectation is not None:
            rows, use_k, use_v = summary_expectation
            kr, vr = rows[layer]
            expected_k = kr.to(kb.device) if use_k else kf[..., s0:s1, :]
            expected_v = vr.to(vb.device) if use_v else vf[..., s0:s1, :]
            if (not torch.equal(expected_k, kb[..., s0:s1, :]) or
                    not torch.equal(expected_v, vb[..., s0:s1, :])):
                raise CoherentStateError(
                    f"{arm}: declared summary insertion mismatch at layer {layer}")
    return {
        "arm": arm,
        "pre_tail_storage_lengths": branch_lengths,
        "pre_tail_row_hashes": _snapshot_hashes(branch),
        "non_summary_rows_bit_exact": True,
        "declared_summary_intervention_exact": arm != "G_delta",
    }


class Runner:
    def __init__(self, args, model, tokenizer, scenarios, targets,
                 fingerprint, provenance, geometry):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.scenarios = scenarios
        self.by_id = {s["id"]: s for s in scenarios}
        self.targets = targets
        self.fingerprint = fingerprint
        self.provenance = provenance
        self.geometry = geometry
        self.run_dir = args.run_dir

    def ckpath(self, position: int, cid: str) -> Path:
        return checkpoint_path(self.run_dir, position, cid)

    def score_gapped_bounded(self, arm, builder, layout, plants,
                             fresh_boundary, correct_rows, wrong_rows):
        """Intervene at summary boundary, append tail, score, then discard."""
        rows = []
        saved_diagnostics = None
        reference_pre_hashes = None
        reference_post_hashes = None
        branch_audit = None
        for plant in plants:
            target = self.targets[plant["id"]]
            boundary, diag = builder()
            if saved_diagnostics is None:
                saved_diagnostics = diag
            current_audit = _assert_boundary_intervention(
                arm, fresh_boundary, boundary, layout, correct_rows, wrong_rows)
            pre_hashes = current_audit["pre_tail_row_hashes"]
            if reference_pre_hashes is None:
                reference_pre_hashes = pre_hashes
                branch_audit = current_audit
            elif pre_hashes != reference_pre_hashes:
                raise CoherentStateError(
                    f"{arm}: repeated branch construction changed pre-tail hashes")
            snap = append_gapped_post_summary(self.model, boundary, layout)
            del boundary
            if any(n != len(layout.context_ids)
                   for n in _snapshot_storage_lengths(snap)):
                raise CoherentStateError(
                    f"{arm}: post-tail cache length does not match context")
            post_hashes = _snapshot_hashes(snap)
            if reference_post_hashes is None:
                reference_post_hashes = post_hashes
            elif post_hashes != reference_post_hashes:
                raise CoherentStateError(
                    f"{arm}: repeated tail recomputation changed hashes")
            correct = score_target(
                self.model, self.tokenizer, snap, layout.messages,
                layout.context_ids, plant["probe"], target["correct"],
                consume_snapshot=True,
                logical_context_end=layout.logical_next_position)
            boundary, _ = builder()
            current_audit = _assert_boundary_intervention(
                arm, fresh_boundary, boundary, layout, correct_rows, wrong_rows)
            if current_audit["pre_tail_row_hashes"] != reference_pre_hashes:
                raise CoherentStateError(
                    f"{arm}: correct/counterfactual branches differ before tail")
            snap = append_gapped_post_summary(self.model, boundary, layout)
            del boundary
            if _snapshot_hashes(snap) != reference_post_hashes:
                raise CoherentStateError(
                    f"{arm}: correct/counterfactual tail recomputation differs")
            counterfactual = score_target(
                self.model, self.tokenizer, snap, layout.messages,
                layout.context_ids, plant["probe"], target["counterfactual"],
                consume_snapshot=True,
                logical_context_end=layout.logical_next_position)
            rows.append({
                "plant_id": plant["id"], "category": plant["category"],
                "probe": plant["probe"], "correct": correct,
                "counterfactual": counterfactual,
                "margin": (correct["mean_logprob"] -
                           counterfactual["mean_logprob"]),
            })
        branch_audit = {
            **(branch_audit or {}),
            "post_tail_storage_lengths": [len(layout.context_ids)] * len(fresh_boundary),
            "post_tail_row_hashes": reference_post_hashes,
            "repeated_branch_hashes_exact": True,
            "tail_recomputed_from_boundary": True,
        }
        return ({"plants": rows,
                 "conversation_margin": sum(x["margin"] for x in rows) / len(rows)},
                saved_diagnostics or [], branch_audit)

    def load_rendered(self, position: int) -> dict:
        cid = FROZEN_ORDER[position - 1]
        doc = read_checkpoint(self.ckpath(position, cid), self.fingerprint)
        if doc is None:
            raise ArtifactError(f"missing rendered checkpoint for {cid}")
        if doc.get("stage") == "void":
            raise ArtifactError(f"render checkpoint is void: {cid}")
        return doc

    def ensure_render_positions(self, positions: set[int]) -> None:
        missing = []
        for idx in sorted(positions):
            cid = FROZEN_ORDER[idx]
            if read_checkpoint(self.ckpath(idx + 1, cid), self.fingerprint) is None:
                missing.append(idx)
        if not missing:
            print(f"RENDER_RESUME positions={[x + 1 for x in sorted(positions)]}",
                  flush=True)
            return
        print(f"PHASE RENDER missing_positions={[x + 1 for x in missing]}", flush=True)

        def on_rendered(idx, conv, _plants, reply_records):
            cid = FROZEN_ORDER[idx]
            if conv.get("id") != cid:
                raise ArtifactError(f"render order mismatch: {conv.get('id')} != {cid}")
            assistants = [m["content"] for m in conv.get("messages", [])
                          if m.get("role") == "assistant"]
            if len(assistants) != len(reply_records):
                raise ArtifactError(f"{cid}: reply-record coverage mismatch")
            for turn, (canonical, record) in enumerate(
                    zip(assistants, reply_records), 1):
                ids = record.get("token_ids")
                raw = record.get("raw_text")
                if not isinstance(ids, list) or len(ids) != record.get("n_tokens"):
                    raise ArtifactError(f"{cid}: turn {turn} raw token IDs absent")
                if self.tokenizer.decode(ids).strip() != raw:
                    raise ArtifactError(f"{cid}: turn {turn} raw render mismatch")
                raw_hash = hashlib.sha256(json.dumps(
                    ids, separators=(",", ":")).encode()).hexdigest()
                if raw_hash != record.get("raw_token_ids_sha256"):
                    raise ArtifactError(f"{cid}: turn {turn} raw token hash mismatch")
                expected = trim_capped_reply(raw) if record.get("hit_token_cap") else raw
                if canonical != expected or record.get("canonical_text") != expected:
                    raise ArtifactError(f"{cid}: turn {turn} canonical trim mismatch")
                block = record.get("canonical_block_ids")
                block_hash = hashlib.sha256(json.dumps(
                    block, separators=(",", ":")).encode()).hexdigest() \
                    if isinstance(block, list) else None
                if block_hash != record.get("canonical_block_ids_sha256"):
                    raise ArtifactError(f"{cid}: turn {turn} canonical ID hash mismatch")
                if bool(record.get("ended_on_eos")) == bool(
                        record.get("hit_token_cap")):
                    raise ArtifactError(f"{cid}: turn {turn} termination flags conflict")
            path = self.ckpath(idx + 1, cid)
            prior = read_checkpoint(path, self.fingerprint)
            if prior is not None:
                if (prior.get("conversation") != conv or
                        prior.get("reply_records") != reply_records):
                    raise ArtifactError(
                        f"refusing to overwrite changed render: {path}")
            else:
                atomic_write_json(path, {
                    "schema": ARTIFACT_SCHEMA,
                    "design_id": DESIGN_ID,
                    "amendment_id": AMENDMENT_ID,
                    "stage": "rendered", "status": "rendered",
                    "fingerprint": self.fingerprint,
                    "order_position": idx + 1,
                    "conversation_id": cid,
                    "conversation": conv,
                    "reply_records": reply_records,
                })
            print(f"CHECKPOINT_RENDERED position={idx + 1} id={cid}", flush=True)

        native_render_specs(
            self.model, self.tokenizer, "qwen", self.scenarios, 12,
            max_reply_tokens=MAX_REPLY_TOKENS, temp=0.0,
            seed_base=STRUCTURAL_SEED, conv_start=0, batched_render=False,
            native_batch=1, only_positions=set(missing),
            on_conv_rendered=on_rendered)

    def process_conversation(self, position: int) -> dict:
        cid = FROZEN_ORDER[position - 1]
        path = self.ckpath(position, cid)
        existing = self.load_rendered(position)
        if existing.get("stage") == "scored":
            validate_scored_checkpoint(existing)
            marker = self.run_dir / "resume_probe.json"
            if position == 1 and marker.exists():
                marker_doc = json.loads(marker.read_text())
                if marker_doc.get("schema") != ARTIFACT_SCHEMA:
                    raise ArtifactError("resume probe schema mismatch")
                expected = marker_doc["checkpoint_sha256"]
                observed = sha256_file(path)
                if observed != expected:
                    raise ArtifactError(
                        f"resume changed first scored checkpoint: {observed} != {expected}")
                if marker_doc.get("resume_probe_verified") is not True:
                    marker_doc = {
                        **marker_doc,
                        "status": "VERIFIED",
                        "resume_probe_verified": True,
                        "verified_at": utc_now(),
                    }
                    atomic_write_json(marker, marker_doc)
                elif marker_doc.get("status") != "VERIFIED":
                    raise ArtifactError("immutable resume verification is inconsistent")
                print(f"RESUME_PROBE_PASS id={cid} sha256={observed}", flush=True)
            print(f"CHECKPOINT_SCORED_REUSE position={position} id={cid}", flush=True)
            return existing
        conv = existing["conversation"]
        donor_id = WRONG_DONOR[cid]
        donor_pos = FROZEN_ORDER.index(donor_id) + 1
        donor = self.load_rendered(donor_pos)["conversation"]
        scenario = self.by_id[cid]
        plants = select_primary_plants(scenario)
        started = time.monotonic()
        print(f"PHASE CAPTURE position={position} id={cid} donor={donor_id}", flush=True)

        try:
            correct_messages = correct_source_messages(conv, SUMMARY_REQUEST)
            generated = capture_generated_summary(
                self.model, self.tokenizer, correct_messages,
                max_tokens=MAX_SUMMARY_TOKENS)
            replay = capture_forced_summary(
                self.model, self.tokenizer, correct_messages,
                generated.summary_ids, source_kind="correct_stepwise_replay")
            identity = validate_generated_replay(
                generated, replay, IDENTITY_TOLERANCE)
            if generated.source_kind != "generated_incremental":
                raise CoherentStateError("correct source is not generated_incremental")
            if not (generated.summary_start == replay.summary_start and
                    generated.summary_end == replay.summary_end and
                    generated.summary_ids == replay.summary_ids):
                raise CoherentStateError(
                    "generated/replay summary span or IDs differ")
            replay.cache = None

            prior_summary = existing.get("summary")
            if prior_summary is not None:
                if (prior_summary.get("token_ids") != generated.summary_ids or
                        prior_summary.get("actual_row_hashes") != generated.row_hashes):
                    raise CoherentStateError(
                        "resumed generated summary differs from captured artifact")
            existing = promote_checkpoint(path, existing, {
                "summary": {
                    "text": generated.summary_text,
                    "token_ids": generated.summary_ids,
                    "token_sha256": sha256_ids(generated.summary_ids),
                    "request": SUMMARY_REQUEST,
                    "request_sha256": hashlib.sha256(
                        SUMMARY_REQUEST.encode()).hexdigest(),
                    "actual_row_hashes": generated.row_hashes,
                },
                "sources": {
                    "correct_actual": _serializable_source(generated),
                    "correct_replay": _serializable_source(replay),
                    "generated_replay_identity": identity,
                },
                "capture_progress": {
                    "generated_replay_persisted": True,
                },
            }, "captured")
            print(f"CHECKPOINT_CAPTURED position={position} id={cid}", flush=True)

            generated.cache = None
            replay.cache = None
            _release_cuda()

            matched_wrong = matched_wrong_prefix_ids(
                self.tokenizer, conv, donor, SUMMARY_REQUEST)
            if matched_wrong.correct_ids != generated.prefix_ids:
                raise CoherentStateError(
                    "matched wrong-history baseline is not the correct prefix")
            wrong = capture_forced_prefix_ids(
                self.model, self.tokenizer, matched_wrong.wrong_ids,
                generated.summary_ids,
                source_kind="wrong_history_exact_length_counterfactual",
                summary_start=generated.summary_start)
            if (wrong.summary_ids != generated.summary_ids or
                    wrong.summary_start != generated.summary_start or
                    wrong.summary_end != generated.summary_end):
                raise CoherentStateError("wrong-source summary span or IDs differ")
            changed_positions = [
                i for i, (a, b) in enumerate(zip(
                    matched_wrong.correct_ids, matched_wrong.wrong_ids)) if a != b]
            declared_content = set(matched_wrong.content_positions)
            if not changed_positions:
                raise CoherentStateError("wrong-history construction changed no token")
            if any(i not in declared_content for i in changed_positions):
                raise CoherentStateError(
                    "wrong-history changed an undeclared content position")
            if any(matched_wrong.correct_ids[i] != matched_wrong.wrong_ids[i]
                   for i in matched_wrong.structural_positions):
                raise CoherentStateError("wrong-history changed structural tokens")
            special_ids = set(int(x) for x in self.tokenizer.all_special_ids)
            replacement_ids = [
                token for replacement in matched_wrong.replacements
                for token in replacement.replacement_ids]
            if not replacement_ids or any(
                    int(token) in special_ids for token in replacement_ids):
                raise CoherentStateError(
                    "wrong-history replacement is empty or contains special tokens")
            wrong_record = {
                **_serializable_source(wrong),
                "decoded_prefix_diagnostic": self.tokenizer.decode(wrong.prefix_ids),
            }
            wrong_construction = {
                **asdict(matched_wrong),
                "correct_prefix_sha256": sha256_ids(matched_wrong.correct_ids),
                "wrong_prefix_sha256": sha256_ids(matched_wrong.wrong_ids),
                "changed_positions": changed_positions,
                "changed_position_count": len(changed_positions),
                "changed_subset_of_declared_content": True,
                "structural_positions_exact": True,
                "replacement_special_token_count": 0,
            }
            existing = promote_checkpoint(path, existing, {
                "sources": {
                    "wrong": wrong_record,
                    "wrong_exact_length_construction": wrong_construction,
                },
                "capture_progress": {
                    "wrong_source_and_mapping_persisted": True,
                },
            }, "captured")
            wrong.cache = None
            _release_cuda()

            layout, fresh_trace, fresh_boundary, fresh_rows = \
                build_gapped_fresh_boundary(
                self.model, self.tokenizer, conv, generated.summary_text,
                generated.summary_ids, SUMMARY_REQUEST, generated.prefix_ids)
            if not (generated.summary_start == wrong.summary_start ==
                    layout.source_summary_start):
                raise CoherentStateError(
                    "correct/wrong/fresh summary logical starts differ")
            if (len(generated.prefix_ids) != len(wrong.prefix_ids) or
                    len(layout.prefix_position_ids) != len(layout.prefix_ids)):
                raise CoherentStateError("gapped prefix coverage mismatch")
            fresh_start = int(fresh_trace.start_position)
            fresh_end = int(fresh_trace.end_position)
            expected_end = generated.summary_start + len(generated.summary_ids)
            if not (fresh_start == generated.summary_start == replay.summary_start ==
                    wrong.summary_start == layout.source_summary_start):
                raise CoherentStateError("one or more summary starts differ")
            if not (fresh_end == generated.summary_end == replay.summary_end ==
                    wrong.summary_end == expected_end):
                raise CoherentStateError("one or more summary ends differ")
            if (layout.summary_position_ids != list(range(
                    generated.summary_start, generated.summary_end)) or
                    layout.physical_summary_end - layout.physical_summary_start !=
                    len(generated.summary_ids)):
                raise CoherentStateError("destination summary schedule differs")
            request_suffix = layout.prefix_ids[layout.system_end:]
            if (generated.prefix_ids[:layout.system_end] !=
                    layout.prefix_ids[:layout.system_end] or
                    generated.prefix_ids[layout.request_logical_start:] != request_suffix or
                    layout.prefix_position_ids[-1] != generated.summary_start - 1 or
                    layout.request_logical_start < layout.system_end):
                raise CoherentStateError(
                    "gapped system or request/header island gate failed")
            if layout.context_ids[:layout.physical_summary_start] != layout.prefix_ids:
                raise CoherentStateError("destination prefix IDs changed")
            if (layout.context_ids[layout.physical_summary_start:
                                   layout.physical_summary_end] != generated.summary_ids):
                raise CoherentStateError("destination summary IDs changed")
            source_schedule = validate_position_schedule(
                range(generated.summary_start, generated.summary_end),
                range(generated.summary_start, generated.summary_end),
                physical_start=generated.summary_start)
            prefix_schedule = validate_position_schedule(
                layout.prefix_position_ids, range(len(layout.prefix_ids)),
                physical_start=0)
            summary_schedule = validate_position_schedule(
                layout.summary_position_ids,
                range(layout.physical_summary_start, layout.physical_summary_end),
                physical_start=layout.physical_summary_start)
            post_schedule = validate_position_schedule(
                layout.post_summary_position_ids,
                range(layout.physical_summary_end, len(layout.context_ids)),
                physical_start=layout.physical_summary_end)

            # Self-replacement is an exact tensor no-op and may touch only the
            # declared physical summary span. This is checked before any outcome.
            self_replaced = replace_summary_rows(
                fresh_boundary, fresh_rows, layout.physical_summary_start,
                use_keys=True, use_values=True)
            if not _tensor_snapshots_equal(fresh_boundary, self_replaced):
                raise CoherentStateError("fresh self-replacement is not bit-identical")
            del self_replaced

            def inserted_exact(source_rows, branch) -> bool:
                start = layout.physical_summary_start
                return all(
                    torch.equal(ks.to(kb.device),
                                kb[..., start:start + ks.shape[-2], :]) and
                    torch.equal(vs.to(vb.device),
                                vb[..., start:start + vs.shape[-2], :])
                    for (ks, vs), (kb, vb) in zip(source_rows, branch))

            correct_boundary, _ = gapped_arm_boundary(
                "G_correct", fresh_boundary, generated.rows, wrong.rows,
                layout.physical_summary_start, PLACEBO_SEED + position)
            wrong_boundary, _ = gapped_arm_boundary(
                "G_wrong", fresh_boundary, generated.rows, wrong.rows,
                layout.physical_summary_start, PLACEBO_SEED + position)
            if not inserted_exact(generated.rows, correct_boundary):
                raise CoherentStateError("correct gapped K/V insertion is not exact")
            if not inserted_exact(wrong.rows, wrong_boundary):
                raise CoherentStateError("wrong gapped K/V insertion is not exact")
            del correct_boundary, wrong_boundary
            _release_cuda()

            wrong_nll = -sum(wrong.trace["token_logprobs"]) / len(
                wrong.trace["token_logprobs"])
            layout_record = {
                "messages": layout.messages,
                "prefix_token_ids": layout.prefix_ids,
                "context_token_ids": layout.context_ids,
                "context_sha256": sha256_ids(layout.context_ids),
                "context_position_ids": layout.context_position_ids,
                "prefix_position_ids": layout.prefix_position_ids,
                "summary_position_ids": layout.summary_position_ids,
                "post_summary_position_ids": layout.post_summary_position_ids,
                "physical_summary_start": layout.physical_summary_start,
                "physical_summary_end": layout.physical_summary_end,
                "logical_summary_start": layout.source_summary_start,
                "logical_summary_end": expected_end,
                "logical_next_position": layout.logical_next_position,
                "system_end": layout.system_end,
                "request_logical_start": layout.request_logical_start,
                "post_summary_token_ids": layout.post_summary_ids,
                "cache_position_ids": list(range(len(layout.context_ids))),
                "position_policy": "gapped_same_source_summary_position",
                "position_schedules": {
                    "natural_source_summary": source_schedule,
                    "gapped_prefix": prefix_schedule,
                    "gapped_summary": summary_schedule,
                    "gapped_post_summary": post_schedule,
                },
            }
            pre_score_gates = {
                "generated_replay_identity": identity,
                "all_summary_starts_equal": True,
                "all_summary_ends_equal": True,
                "fresh_self_replacement_exact": True,
                "correct_insertion_exact": True,
                "wrong_insertion_exact": True,
                "correct_wrong_prefix_length_equal": True,
                "wrong_changed_subset_declared_content": True,
                "wrong_changed_at_least_one_token": True,
                "structural_request_header_tail_positions_exact": True,
                "replacement_special_tokens_excluded": True,
                "request_header_suffix_exact": True,
                "system_island_exact": True,
                "logical_islands_nonoverlapping": True,
                "summary_positions_equal": True,
                "cache_position_contiguous": True,
                "exact_summary_span": True,
                "confirmatory_key_rotation_calls": 0,
                "no_semantic_outcome_scored_before_gate": True,
                "pre_score_pass": True,
            }
            existing = promote_checkpoint(path, existing, {
                "sources": {
                    "fresh": {
                        "source_kind": "gapped_fresh_stepwise_forced",
                        "prefix_token_ids": layout.prefix_ids,
                        "prefix_sha256": sha256_ids(layout.prefix_ids),
                        "prefix_position_ids": layout.prefix_position_ids,
                        "physical_summary_start": layout.physical_summary_start,
                        "physical_summary_end": layout.physical_summary_end,
                        "summary_start": layout.source_summary_start,
                        "summary_end": expected_end,
                        "summary_token_ids": generated.summary_ids,
                        "summary_row_hashes": row_hashes(fresh_rows),
                        "trace": asdict(fresh_trace),
                    },
                    "wrong_summary_mean_nll": wrong_nll,
                },
                "destination": layout_record,
                "gates": pre_score_gates,
                "capture_progress": {
                    "layout_and_pre_score_gates_persisted": True,
                },
            }, "captured")
            print(f"CHECKPOINT_GATED position={position} id={cid}", flush=True)

            # All first-case structure/position gates have passed. Reconstruct the
            # full-history reference now; no downstream outcome was scored earlier.
            a_capture = capture_forced_summary(
                self.model, self.tokenizer, correct_messages,
                generated.summary_ids,
                source_kind="a_full_forced_replay_reconstruction")
            a_messages, a_ids, a_snapshot = complete_assistant_context(
                self.model, self.tokenizer, correct_messages, a_capture)
            a_source_record = _serializable_source(a_capture)
            a_capture.cache = None
            a_score = score_arm(
                self.model, self.tokenizer, a_snapshot, a_messages, a_ids,
                plants, self.targets)
            del a_snapshot
            _release_cuda()

            arm_scores = {"A_full": a_score}
            arm_diagnostics = {}
            branch_audits = {}
            outcomes = {"A_full": a_score["conversation_margin"]}
            existing = promote_checkpoint(path, existing, {
                "sources": {"a_full_forced_replay_reconstruction": a_source_record},
                "arm_scores": {"A_full": a_score},
                "conversation_outcomes": {"A_full": outcomes["A_full"]},
                "scoring_progress": {"A_full": "persisted"},
            }, "captured")

            self_score, _, self_branch_audit = self.score_gapped_bounded(
                "G_fresh",
                lambda: (replace_summary_rows(
                    fresh_boundary, fresh_rows,
                    layout.physical_summary_start,
                    use_keys=True, use_values=True), []),
                layout, plants, fresh_boundary, generated.rows, wrong.rows)

            noop_max = None
            for arm in GAPPED_ARM_NAMES[1:]:
                score, diag, branch_audit = self.score_gapped_bounded(
                    arm,
                    lambda arm=arm: gapped_arm_boundary(
                        arm, fresh_boundary, generated.rows, wrong.rows,
                        layout.physical_summary_start, PLACEBO_SEED + position),
                    layout, plants, fresh_boundary, generated.rows, wrong.rows)
                arm_scores[arm] = score
                arm_diagnostics[arm] = diag
                branch_audits[arm] = branch_audit
                outcomes[arm] = score["conversation_margin"]
                if arm == "G_fresh":
                    def token_lps(s):
                        return [lp for row in s["plants"]
                                for side in ("correct", "counterfactual")
                                for lp in row[side]["token_logprobs"]]
                    f_lp, self_lp = token_lps(score), token_lps(self_score)
                    noop_max = max(abs(a - b) for a, b in zip(f_lp, self_lp))
                    if noop_max > IDENTITY_TOLERANCE:
                        raise CoherentStateError(
                            f"tokenwise fresh self-replacement diff {noop_max}")
                    if branch_audit != self_branch_audit:
                        raise CoherentStateError(
                            "fresh direct and self-replaced branch audits differ")
                if arm == "G_delta":
                    if not diag or any(x["fixed_points"] for x in diag):
                        raise CoherentStateError("placebo derangement has fixed points")
                    if max(x["max_multiset_diff"] for x in diag) != 0:
                        raise CoherentStateError("placebo intended delta multiset changed")
                    if max(max(x["mean_diff"], x["covariance_diff"])
                           for x in diag) > PLACEBO_MOMENT_TOLERANCE:
                        raise CoherentStateError("placebo moment diagnostic exceeds tolerance")
                    if max(max(x["applied_delta_max_abs_error"],
                               x["applied_multiset_diff"],
                               x["applied_mean_diff"],
                               x["applied_covariance_diff"])
                           for x in diag) > PLACEBO_QUANTIZATION_TOLERANCE:
                        raise CoherentStateError(
                            "placebo applied-delta quantization exceeds tolerance")
                existing = promote_checkpoint(path, existing, {
                    "arm_scores": {arm: score},
                    "arm_diagnostics": {arm: diag},
                    "branch_audits": {arm: branch_audit},
                    "conversation_outcomes": {arm: outcomes[arm]},
                    "scoring_progress": {arm: "persisted"},
                }, "captured")
                print(
                    f"CHECKPOINT_ARM position={position} id={cid} arm={arm}",
                    flush=True)
                _release_cuda()

            calibration = run_calibration(self.model, self.tokenizer, cid)
            additions = {
                "arm_scores": arm_scores,
                "target_provenance": {
                    plant["id"]: self.targets[plant["id"]]
                    for plant in plants
                },
                "arm_diagnostics": arm_diagnostics,
                "branch_audits": branch_audits,
                "conversation_outcomes": outcomes,
                "calibration": calibration,
                "calibration_outcomes": calibration["outcomes"],
                "gates": {
                    "technical_pass": True,
                    "failures": [],
                    "self_transplant_tokenwise_max_abs": noop_max,
                    "post_summary_recomputed_per_arm": True,
                    "repeated_branch_hashes_exact": True,
                },
                "scoring_progress": {"calibration": "persisted", "complete": True},
                "runtime": {
                    **self.provenance,
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at": utc_now(),
                },
            }
            scored = promote_checkpoint(path, existing, additions, "scored")
            existing = scored
            validate_scored_checkpoint(scored)
            del fresh_boundary
            _release_cuda()
            print(
                f"CHECKPOINT_SCORED position={position} id={cid} "
                f"GF={outcomes['G_correct'] - outcomes['G_fresh']:+.6f} "
                f"GW={outcomes['G_correct'] - outcomes['G_wrong']:+.6f}",
                flush=True)
            return scored
        except Exception as exc:
            failure = {
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "failed_at": utc_now(),
            }
            current = read_checkpoint(path, self.fingerprint)
            if current is not None:
                preserved = dict(current)
                prior_failures = (preserved.get("gates") or {}).get("failures", [])
                preserved["stage"] = "void"
                preserved["status"] = "void"
                preserved["failure"] = failure
                preserved["gates"] = {**(preserved.get("gates") or {}),
                                      "technical_pass": False,
                                      "failures": prior_failures + [failure]}
                preserved["runtime"] = {**(preserved.get("runtime") or {}),
                                        **self.provenance,
                                        "elapsed_seconds": time.monotonic() - started}
                atomic_write_json(path, preserved)
            else:
                atomic_write_json(path.with_suffix(".failure.json"), failure)
            raise

    def run_stage(self, limit: int) -> dict:
        for position in range(1, limit + 1):
            cid = FROZEN_ORDER[position - 1]
            donor_position = FROZEN_ORDER.index(WRONG_DONOR[cid]) + 1
            self.ensure_render_positions({position - 1, donor_position - 1})
            self.process_conversation(position)
            if (position == 1 and self.args.resume_probe_stop_after_one and
                    not (self.run_dir / "resume_probe.json").exists()):
                path = self.ckpath(1, FROZEN_ORDER[0])
                atomic_write_json(self.run_dir / "resume_probe.json", {
                    "schema": ARTIFACT_SCHEMA,
                    "design_id": DESIGN_ID,
                    "amendment_id": AMENDMENT_ID,
                    "status": "INTENTIONAL_RESTART_REQUIRED",
                    "resume_probe_verified": False,
                    "conversation_id": FROZEN_ORDER[0],
                    "checkpoint_sha256": sha256_file(path),
                    "created_at": utc_now(),
                })
                print("INTENTIONAL_RESUME_PROBE_STOP after=1", flush=True)
                raise IntentionalResumeProbe()
        docs = [d for d in load_checkpoints(self.run_dir)
                if int(d["order_position"]) <= limit]
        stats = analyze(docs)
        out = self.run_dir / f"analysis_n{len(docs):02d}.json"
        atomic_write_json(out, stats)
        print(f"STAGE_ANALYSIS n={len(docs)} decision={stats['serial_decision']}",
              flush=True)
        return stats


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios.json"))
    ap.add_argument("--targets", type=Path,
                    default=Path("data/coherent_state_targets.json"))
    ap.add_argument("--resume-probe-stop-after-one", action="store_true",
                    help="operational gate: exit 75 after first scored checkpoint")
    return ap.parse_args()


class IntentionalResumeProbe(RuntimeError):
    pass


def main():
    args = parse_args()
    print(f"RUN coherent_state model={MODEL}@{REVISION} -> {args.run_dir}",
          flush=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "manifest.json"
    gate_path = args.run_dir / "production_kernel_gate.json"
    invocation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    gate_attempt_path = args.run_dir / f"production_kernel_gate_{invocation_id}.json"
    phase = "INVOCATION_START"
    fingerprint = None
    provenance = None
    geometry = None
    started_at = utc_now()
    production_gate = None

    def write_manifest(status: str, current_phase: str, **extra) -> None:
        prior_doc = (json.loads(manifest_path.read_text())
                     if manifest_path.exists() else {})
        doc = {
            **prior_doc,
            "schema": ARTIFACT_SCHEMA,
            "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID,
            "status": status,
            "phase": current_phase,
            "started_at": prior_doc.get("started_at", started_at),
            "updated_at": utc_now(),
            "resume_probe_verified": bool(
                (json.loads((args.run_dir / "resume_probe.json").read_text())
                 if (args.run_dir / "resume_probe.json").exists() else {})
                .get("resume_probe_verified", False)),
            **extra,
        }
        atomic_write_json(manifest_path, doc)

    try:
        _static_design_self_check()
        if not AMENDMENT_PATH.exists():
            raise CoherentStateError(f"missing frozen amendment: {AMENDMENT_PATH}")
        phase = "PROVENANCE"
        provenance = runtime_provenance(args.run_dir)
        config, tokenizer, subject_metadata = prepare_subject_metadata()
        provenance["subject"] = subject_metadata
        scenario_map = load_scenarios(args.scenarios)
        scenarios = frozen_scenarios(scenario_map)
        targets = load_and_validate_targets(args.targets, scenario_map)
        fingerprint = {
            "schema": ARTIFACT_SCHEMA,
            "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID,
            "amendment_sha256": sha256_file(AMENDMENT_PATH),
            "model": MODEL, "revision": REVISION,
            "code_commit": provenance["code_commit"],
            "scenario_sha256": sha256_file(args.scenarios),
            "targets_sha256": sha256_file(args.targets),
            "summary_request_sha256": hashlib.sha256(
                SUMMARY_REQUEST.encode()).hexdigest(),
            "frozen_order": list(FROZEN_ORDER),
            "wrong_donors": WRONG_DONOR,
            "structural_seed": STRUCTURAL_SEED,
            "placebo_seed": PLACEBO_SEED,
            "max_reply_tokens": MAX_REPLY_TOKENS,
            "max_summary_tokens": MAX_SUMMARY_TOKENS,
            "identity_tolerance": IDENTITY_TOLERANCE,
            "placebo_moment_tolerance": PLACEBO_MOMENT_TOLERANCE,
            "placebo_quantization_tolerance": PLACEBO_QUANTIZATION_TOLERANCE,
            "subject_metadata": subject_metadata,
        }
        prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if prior is not None and prior.get("fingerprint") != fingerprint:
            raise ArtifactError("run manifest fingerprint mismatch")
        started_at = prior.get("started_at") if prior else started_at
        phase = "SETUP"
        write_manifest(
            "SETUP", phase,
            resumed_at=utc_now() if prior else None,
            fingerprint=fingerprint, provenance=provenance,
            production_kernel_gate_path=gate_path.name)
        print("PHASE SETUP", flush=True)
        phase = "MODEL_LOADING"
        model, tokenizer, geometry = load_subject(config, tokenizer)
        phase = "MODEL_READY"
        write_manifest(
            "RUNNING", phase, fingerprint=fingerprint,
            provenance=provenance, geometry=geometry,
            production_kernel_gate_path=gate_path.name)
        print(f"MODEL_READY revision={REVISION} dtype=bf16 geometry={geometry}",
              flush=True)
        phase = "PRODUCTION_GATE"
        gate_sink = {}
        try:
            production_gate = run_loaded_gapped_gates(
                model, tokenizer,
                identity_tolerance=IDENTITY_TOLERANCE,
                placebo_quantization_tolerance=PLACEBO_QUANTIZATION_TOLERANCE,
                placebo_moment_tolerance=PLACEBO_MOMENT_TOLERANCE,
                diagnostic_sink=gate_sink)
        except Exception as gate_exc:
            production_gate = {
                **gate_sink,
                "passes": False,
                "failure": {
                    "error_type": type(gate_exc).__name__,
                    "error": str(gate_exc),
                    "traceback": traceback.format_exc(),
                },
            }
        gate_status = "PASS" if production_gate.get("passes") is True else "FAIL"
        gate_doc = {
            "schema": ARTIFACT_SCHEMA,
            "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID,
            "status": gate_status, "completed_at": utc_now(),
            "model": MODEL, "revision": REVISION,
            "dtype": "torch.bfloat16", "geometry": geometry,
            "gates": production_gate,
        }
        if gate_status == "FAIL":
            gate_doc["error"] = (
                production_gate.get("failure") or
                production_gate.get("failures") or
                "production gapped gate returned passes=false")
        atomic_write_json(gate_attempt_path, gate_doc)
        atomic_write_json(gate_path, gate_doc)
        if gate_status != "PASS":
            raise CoherentStateError(
                "production gapped gate failed; complete evidence was persisted")
        print("PRODUCTION_GAPPED_GATE_PASS", flush=True)
        phase = "SEMANTIC_RUN"
        write_manifest(
            "RUNNING", phase, fingerprint=fingerprint,
            provenance=provenance, geometry=geometry,
            production_kernel_gate_path=gate_path.name,
            production_kernel_gate_attempt_path=gate_attempt_path.name,
            production_kernel_gate_status=gate_status)
        runner = Runner(args, model, tokenizer, scenarios, targets, fingerprint,
                        provenance, geometry)
        stats6 = runner.run_stage(6)
        if stats6["serial_decision"] == "EXTEND_TO_12":
            stats = runner.run_stage(12)
        else:
            stats = stats6
        marker_path = args.run_dir / "resume_probe.json"
        marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
        if marker.get("resume_probe_verified") is not True:
            raise ArtifactError("forced-restart resume probe was not verified")
        phase = "COMPLETE"
        write_manifest(
            "COMPLETE", phase, completed_at=utc_now(),
            fingerprint=fingerprint, provenance=provenance, geometry=geometry,
            production_kernel_gate_path=gate_path.name,
            production_kernel_gate_attempt_path=gate_attempt_path.name,
            production_kernel_gate_status="PASS",
            production_kernel_gate=production_gate,
            n_conversations=stats["n_conversations"],
            serial_decision=stats["serial_decision"],
            interpretation=stats["interpretation"])
        print(
            f"COHERENCE_RUN_DONE n={stats['n_conversations']} "
            f"decision={stats['serial_decision']}", flush=True)
        return 0
    except IntentionalResumeProbe:
        phase = "FORCED_RESTART"
        write_manifest(
            "RESUME_REQUIRED", phase, fingerprint=fingerprint,
            provenance=provenance, geometry=geometry,
            production_kernel_gate_path=gate_path.name,
            production_kernel_gate_attempt_path=(
                gate_attempt_path.name if gate_attempt_path.exists() else None),
            production_kernel_gate_status=(
                "PASS" if production_gate and production_gate.get("passes") else None))
        return 75
    except Exception as exc:
        failure = {"schema": ARTIFACT_SCHEMA,
                   "design_id": DESIGN_ID, "amendment_id": AMENDMENT_ID,
                   "status": "ERROR", "phase": phase, "failed_at": utc_now(),
                   "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc()}
        failure_path = args.run_dir / "failure.json"
        atomic_write_json(failure_path, failure)
        if phase in {"MODEL_READY", "PRODUCTION_GATE"} and not gate_path.exists():
            atomic_write_json(gate_path, {
                "schema": ARTIFACT_SCHEMA,
                "design_id": DESIGN_ID,
                "amendment_id": AMENDMENT_ID,
                "status": "FAIL", "completed_at": utc_now(),
                "model": MODEL, "revision": REVISION,
                "dtype": "torch.bfloat16", "geometry": geometry,
                "gates": {"passes": False, "failure": failure},
            })
        if gate_path.exists() and not gate_attempt_path.exists():
            atomic_write_json(
                gate_attempt_path, json.loads(gate_path.read_text()))
        void_refs = sorted(
            p.name for p in args.run_dir.glob("conv_*.json")
            if json.loads(p.read_text()).get("stage") == "void")
        gate_doc = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        if gate_doc.get("status") == "PASS" and not void_refs:
            run_void = args.run_dir / "conv_00_run_failure.json"
            atomic_write_json(run_void, {
                "schema": ARTIFACT_SCHEMA,
                "design_id": DESIGN_ID,
                "amendment_id": AMENDMENT_ID,
                "stage": "void", "status": "void",
                "fingerprint": fingerprint,
                "order_position": 0,
                "conversation_id": "run-level-failure",
                "failure": failure,
            })
            void_refs = [run_void.name]
        failure_manifest = {
            "failure_path": failure_path.name,
            "production_kernel_gate_path": (
                gate_path.name if gate_path.exists() else None),
            "production_kernel_gate_attempt_path": (
                gate_attempt_path.name if gate_attempt_path.exists() else None),
            "void_checkpoint_paths": void_refs,
        }
        if fingerprint is not None:
            failure_manifest["fingerprint"] = fingerprint
        if provenance is not None:
            failure_manifest["provenance"] = provenance
        if geometry is not None:
            failure_manifest["geometry"] = geometry
        write_manifest("ERROR", phase, **failure_manifest)
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
