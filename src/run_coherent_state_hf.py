"""Production driver for the preregistered coherent-summary-state experiment."""

from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
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
    validate_native_conversation,
)
from coherent_state_hf import (
    CoherentStateError,
    replace_summary_rows,
    row_hashes,
    sha256_ids,
)
from coherent_state_integrity import (
    AMENDMENT_ID as INTEGRITY_AMENDMENT_ID,
    DESIGN_ID as INTEGRITY_DESIGN_ID,
    IntegrityError,
    INDEX_NAME,
    RECEIPT_NAME,
    apparatus_inventory,
    verify_prior_technical_authorization,
    write_sealed_payload,
    write_terminal_envelope,
)
from coherent_state_runtime import (
    GAPPED_ARM_NAMES,
    append_gapped_post_summary,
    build_gapped_fresh_boundary,
    capture_forced_summary,
    capture_forced_prefix_ids,
    capture_generated_summary,
    complete_assistant_context,
    eager_backend_fingerprint,
    gapped_arm_boundary,
    measure_gapped_destination_schedule,
    measure_generated_replay,
    score_arm,
    score_target,
    validate_position_schedule,
)
from coherent_state_tokens import (
    gapped_destination_layout,
    generation_prefix_ids,
    matched_wrong_prefix_ids,
)
from coherent_state_store import (
    ArtifactError,
    atomic_write_json,
    checkpoint_path,
    promote_checkpoint,
    read_checkpoint,
    validate_scored_checkpoint,
)
from cross_arch_probe import native_render_specs, trim_capped_reply
from l_coherent_state_hf import (
    run_exact_render_schedule_fixture,
    run_loaded_gapped_gates,
    v8_gate_schema,
)


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STRUCTURAL_SEED = 20_260_711
ARTIFACT_SCHEMA = 2
DESIGN_ID = "coherent-state-gapped-v8"
AMENDMENT_ID = "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7-8"
AMENDMENT_PATHS = (
    Path("COHERENT-STATE-PREREGISTRATION-AMENDMENT-1.md"),
    Path("COHERENT-STATE-PREREGISTRATION-AMENDMENT-2.md"),
    Path("COHERENT-STATE-PREREGISTRATION-AMENDMENT-3.md"),
    Path("COHERENT-STATE-PREREGISTRATION-AMENDMENT-4.md"),
    Path("COHERENT-STATE-PREREGISTRATION-AMENDMENT-5.md"),
    Path("COHERENT-STATE-PREREGISTRATION-AMENDMENT-6.md"),
    Path("COHERENT-STATE-PREREGISTRATION-AMENDMENT-7.md"),
    Path("COHERENT-STATE-PREREGISTRATION-AMENDMENT-8.md"),
)
ATTENTION_BACKEND = "eager"
EXPECTED_GEOMETRY = {
    "layers": 48, "attention_heads": 32, "kv_heads": 4,
    "head_dim": 128, "rope_theta": 10_000_000,
}
MAX_REPLY_TOKENS = 320
MAX_SUMMARY_TOKENS = 900
IDENTITY_TOLERANCE = 1e-4
ZERO_GAP_TOLERANCE = 5e-4
MAX_TECHNICAL_LOGICAL_POSITION = 9_509

if DESIGN_ID != INTEGRITY_DESIGN_ID or AMENDMENT_ID != INTEGRITY_AMENDMENT_ID:
    raise RuntimeError("driver/integrity v8 identities disagree")


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


def load_external_donors(donor_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    donors = {}
    provenance = {}
    donor_ids = list(WRONG_DONOR.values())
    if len(set(donor_ids)) != len(donor_ids):
        raise CoherentStateError("external wrong-history donors are reused")
    if set(donor_ids).intersection(FROZEN_ORDER):
        raise CoherentStateError("external donor is also a scored target")
    for donor_id in donor_ids:
        path = donor_dir / f"{donor_id}.json"
        if not path.is_file():
            raise CoherentStateError(f"external donor file absent: {path}")
        donor = json.loads(path.read_text())
        if donor.get("id") != donor_id:
            raise CoherentStateError(
                f"external donor ID mismatch: {path} has {donor.get('id')}")
        validate_native_conversation(donor)
        author = (donor.get("meta") or {}).get("author")
        if not isinstance(author, str) or not author.strip():
            raise CoherentStateError(f"external donor lacks author: {path}")
        donors[donor_id] = donor
        provenance[donor_id] = {
            "donor_id": donor_id,
            "path": str(path),
            "sha256": sha256_file(path),
            "recorded_author": author,
            "subject_native": False,
        }
    return donors, provenance


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
        "execution_packages": {
            name: importlib.metadata.version(name) for name in (
                "accelerate", "huggingface_hub", "safetensors",
                "sentencepiece", "torch", "transformers")
        },
    }


def model_geometry(config) -> dict:
    cfg = getattr(config, "text_config", config)
    rp = getattr(cfg, "rope_parameters", None) or {}
    theta = getattr(cfg, "rope_theta", None) or rp.get("rope_theta")
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    return {
        "layers": int(cfg.num_hidden_layers),
        "attention_heads": int(cfg.num_attention_heads),
        "kv_heads": int(cfg.num_key_value_heads),
        "head_dim": int(head_dim),
        "rope_theta": int(theta),
    }


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode()).hexdigest()


class DurableDiagnosticSink(dict):
    """A mapping that atomically checkpoints each top-level gate update."""

    def __init__(self, persist):
        super().__init__()
        self._persist = persist

    def _changed(self) -> None:
        self._persist(dict(self))

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self._changed()

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        self._changed()

    def clear(self) -> None:
        super().clear()
        self._changed()

    def pop(self, key, default=None):
        out = super().pop(key, default)
        self._changed()
        return out

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._changed()


def terminalize_running_gate_attempt(
        attempt_path: Path, gate_path: Path, failure: dict, **fields) -> bool:
    """Turn an already-opened unique gate sink into a durable terminal FAIL."""
    if not attempt_path.exists():
        return False
    attempt = json.loads(attempt_path.read_text())
    if attempt.get("status") != "RUNNING":
        return False
    failed = {
        **attempt,
        **fields,
        "status": "FAIL",
        "completed_at": utc_now(),
        "gates": {
            **(attempt.get("gates") or {}),
            "passes": False,
            "failure": failure,
        },
        "error": failure,
    }
    atomic_write_json(attempt_path, failed)
    atomic_write_json(gate_path, failed)
    return True


def prepare_subject_metadata():
    config = AutoConfig.from_pretrained(
        MODEL, revision=REVISION, attn_implementation=ATTENTION_BACKEND)
    resolved = getattr(config, "_commit_hash", None)
    if resolved != REVISION:
        raise CoherentStateError(f"resolved revision {resolved} != {REVISION}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    metadata = {
        "resolved_model_revision": resolved,
        "attention_backend_requested": ATTENTION_BACKEND,
        "config_sha256": sha256_json(config.to_dict()),
        "tokenizer_revision_requested": REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_sha256": sha256_json(tokenizer.get_vocab()),
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode()).hexdigest(),
        "special_tokens_map_sha256": sha256_json(tokenizer.special_tokens_map),
    }
    return config, tokenizer, metadata


def load_subject(config, tokenizer, *, backend_progress=None):
    if not torch.cuda.is_available():
        raise CoherentStateError("paid production driver requires CUDA")
    geometry = model_geometry(config)
    if geometry != EXPECTED_GEOMETRY:
        raise CoherentStateError(
            f"model geometry {geometry} != {EXPECTED_GEOMETRY}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, config=config,
        attn_implementation=ATTENTION_BACKEND,
        dtype=torch.bfloat16, device_map=None)
    model.to("cuda")
    model.eval()
    floating = {p.dtype for p in model.parameters() if p.is_floating_point()}
    devices = {p.device.type for p in model.parameters()}
    if floating != {torch.bfloat16}:
        raise CoherentStateError(f"live parameter dtypes are {floating}, not bf16")
    if devices != {"cuda"}:
        raise CoherentStateError(f"live parameter devices are {devices}, not CUDA")
    if getattr(model.config, "_commit_hash", None) != REVISION:
        raise CoherentStateError(
            f"live model revision {getattr(model.config, '_commit_hash', None)} "
            f"!= {REVISION}")
    backend = (eager_backend_fingerprint(model)
               if backend_progress is None else
               eager_backend_fingerprint(model, progress=backend_progress))
    return model, tokenizer, geometry, backend


def model_context_limit(model) -> int:
    cfg = getattr(model.config, "text_config", model.config)
    limit = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    if limit <= MAX_TECHNICAL_LOGICAL_POSITION:
        raise CoherentStateError(
            f"model context limit {limit} does not cover frozen technical "
            f"position {MAX_TECHNICAL_LOGICAL_POSITION}")
    return limit


def assert_technical_gate_has_no_semantic_scores(gates: dict) -> None:
    """Fail closed if the technical-only gate leaks a treatment outcome."""
    forbidden = {
        "arm_scores", "conversation_outcomes", "calibration_outcomes",
        "technical_margins_not_semantic_outcomes",
        "semantic_outcomes", "target_scores",
    }

    def walk(value, path="gates"):
        if isinstance(value, dict):
            hit = forbidden.intersection(value)
            if hit:
                raise CoherentStateError(
                    f"technical gate contains forbidden semantic score field "
                    f"{path}.{sorted(hit)[0]}")
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(gates)


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


def terminalize_partial_checkpoints(run_dir: Path, failure: dict) -> list[str]:
    """Preserve partial artifacts but make failure harvest terminal/validatable."""
    void_refs = []
    for path in sorted(run_dir.glob("conv_*.json")):
        doc = json.loads(path.read_text())
        stage = doc.get("stage")
        if stage in {"rendered", "captured"}:
            prior_failures = (doc.get("gates") or {}).get("failures", [])
            doc["stage"] = "void"
            doc["status"] = "void"
            doc["failure"] = doc.get("failure") or failure
            doc["gates"] = {
                **(doc.get("gates") or {}),
                "technical_pass": False,
                "failures": prior_failures + [failure],
            }
            atomic_write_json(path, doc)
            stage = "void"
        if stage == "void":
            void_refs.append(path.name)
    return void_refs


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


def _snapshot_span_hashes(snapshot, start: int, end: int) -> list[dict]:
    if not 0 <= start < end:
        raise CoherentStateError(f"invalid snapshot hash span [{start}, {end})")
    return row_hashes([
        (keys[..., start:end, :], values[..., start:end, :])
        for keys, values in snapshot
    ])


def _declared_summary_hashes(arm: str, fresh_hashes: list[dict],
                             correct_hashes: list[dict],
                             wrong_hashes: list[dict]) -> tuple[list[dict], str, str]:
    if not (len(fresh_hashes) == len(correct_hashes) == len(wrong_hashes)):
        raise CoherentStateError("summary lineage layer-count mismatch")
    sources = {
        "G_fresh": (fresh_hashes, "fresh", "fresh"),
        "G_correct": (correct_hashes, "correct_actual", "correct_actual"),
        "G_wrong": (wrong_hashes, "wrong_history", "wrong_history"),
    }
    if arm in sources:
        return sources[arm]
    if arm == "G_Vcorrect":
        return ([
            {"layer": fresh["layer"], "k_sha256": fresh["k_sha256"],
             "v_sha256": correct["v_sha256"]}
            for fresh, correct in zip(fresh_hashes, correct_hashes)
        ], "fresh", "correct_actual")
    if arm == "G_Kcorrect":
        return ([
            {"layer": fresh["layer"], "k_sha256": correct["k_sha256"],
             "v_sha256": fresh["v_sha256"]}
            for fresh, correct in zip(fresh_hashes, correct_hashes)
        ], "correct_actual", "fresh")
    raise CoherentStateError(f"unknown gapped lineage arm {arm}")


def _strict_generated_replay_witness(generated, replay, numerical: dict) -> dict:
    """Bind the replay waiver to exact K/V bytes, not only a tolerance."""
    actual_hashes = generated.row_hashes
    replay_hashes = replay.row_hashes
    hashes_exact = actual_hashes == replay_hashes
    numerical_passes = numerical.get("passes") is True
    return {
        **numerical,
        "numerical_tolerance_passes": numerical_passes,
        "actual_summary_row_hashes": actual_hashes,
        "replay_summary_row_hashes": replay_hashes,
        "summary_row_hashes_bit_exact": hashes_exact,
        "raw_tensor_archive_waived_by_exact_replay": hashes_exact,
        "passes": numerical_passes and hashes_exact,
    }


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
    fresh_summary_hashes = _snapshot_span_hashes(fresh, s0, s1)
    correct_hashes = _snapshot_hashes(correct_rows)
    wrong_hashes = _snapshot_hashes(wrong_rows)
    declared_hashes, declared_k_source, declared_v_source = \
        _declared_summary_hashes(
            arm, fresh_summary_hashes, correct_hashes, wrong_hashes)
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
    inserted_hashes = _snapshot_span_hashes(branch, s0, s1)
    if inserted_hashes != declared_hashes:
        raise CoherentStateError(
            f"{arm}: inserted summary hashes differ from declared lineage")
    fresh_prefix_hashes = _snapshot_span_hashes(fresh, 0, s0)
    branch_prefix_hashes = _snapshot_span_hashes(branch, 0, s0)
    if fresh_prefix_hashes != branch_prefix_hashes:
        raise CoherentStateError(f"{arm}: pre-summary hash lineage differs")
    return {
        "arm": arm,
        "pre_tail_storage_lengths": branch_lengths,
        "pre_tail_row_hashes": _snapshot_hashes(branch),
        "fresh_summary_row_hashes": fresh_summary_hashes,
        "inserted_summary_row_hashes": inserted_hashes,
        "declared_source_summary_row_hashes": declared_hashes,
        "declared_k_source": declared_k_source,
        "declared_v_source": declared_v_source,
        "fresh_before_summary_row_hashes": fresh_prefix_hashes,
        "branch_before_summary_row_hashes": branch_prefix_hashes,
        "non_summary_rows_bit_exact": True,
        "declared_summary_intervention_exact": True,
        "summary_hash_lineage_exact": True,
    }


class Runner:
    def __init__(self, args, model, tokenizer, scenarios, targets,
                 fingerprint, provenance, geometry, donors, donor_provenance):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.scenarios = scenarios
        self.by_id = {s["id"]: s for s in scenarios}
        self.targets = targets
        self.fingerprint = fingerprint
        self.provenance = provenance
        self.geometry = geometry
        self.donors = donors
        self.donor_provenance = donor_provenance
        self.run_dir = args.run_dir
        self.context_limit = model_context_limit(model)

    def _restore_generated_summary(self, existing: dict,
                                   correct_messages: list[dict]):
        """Rebuild live tensors from a durable generation without regenerating text."""
        summary = existing.get("summary") or {}
        actual = (existing.get("sources") or {}).get("correct_actual") or {}
        ids = summary.get("token_ids")
        if not isinstance(ids, list) or not ids:
            raise ArtifactError("durable generated summary has no token IDs")
        if summary.get("token_sha256") != sha256_ids(ids):
            raise ArtifactError("durable generated summary token hash mismatch")
        if self.tokenizer.decode(ids) != summary.get("text"):
            raise ArtifactError("durable generated summary text/ID mismatch")
        restored = capture_forced_summary(
            self.model, self.tokenizer, correct_messages, ids,
            source_kind="generated_incremental_exact_reconstruction")
        if (restored.prefix_ids != actual.get("prefix_token_ids") or
                restored.summary_start != actual.get("summary_start") or
                restored.summary_end != actual.get("summary_end") or
                restored.row_hashes != actual.get("summary_row_hashes")):
            raise ArtifactError(
                "durable generated source does not reconstruct bit-exactly")
        prior_trace = actual.get("trace") or {}
        if restored.trace.get("token_ids") != prior_trace.get("token_ids"):
            raise ArtifactError("durable generated trace token IDs changed")
        old_lp = prior_trace.get("token_logprobs") or []
        new_lp = restored.trace.get("token_logprobs") or []
        if len(old_lp) != len(new_lp) or max(
                (abs(float(a) - float(b)) for a, b in zip(old_lp, new_lp)),
                default=float("inf")) > IDENTITY_TOLERANCE:
            raise ArtifactError("durable generated trace log-probabilities changed")
        restored.summary_text = summary["text"]
        return restored

    def _persist_generated_summary(self, path: Path, existing: dict,
                                   generated) -> dict:
        """Durably save all generated evidence before independent validation."""
        return promote_checkpoint(path, existing, {
            "summary": {
                "text": generated.summary_text,
                "token_ids": generated.summary_ids,
                "token_sha256": sha256_ids(generated.summary_ids),
                "request": SUMMARY_REQUEST,
                "request_sha256": hashlib.sha256(
                    SUMMARY_REQUEST.encode()).hexdigest(),
                "actual_row_hashes": generated.row_hashes,
                "generation_trace": generated.trace,
            },
            "sources": {
                "correct_actual": {
                    **_serializable_source(generated),
                    "capture_materialization": "live_incremental_generation_rows",
                    "raw_tensor_archived": False,
                    "exact_replay_waiver_required_before_scoring": True,
                },
            },
            "capture_progress": {
                "generated_source_persisted": True,
                "generated_replay_pending_at_durable_save": True,
            },
        }, "captured")

    def _record_scoring_source_materialization(
            self, path: Path, kind: str) -> dict:
        """Append one of two bounded source-materialization facts durably."""
        allowed = {
            "live_incremental_generation_rows",
            "bit_exact_stepwise_resume_reconstruction",
        }
        if kind not in allowed:
            raise ArtifactError(f"invalid scoring source materialization: {kind}")
        current = json.loads(path.read_text())
        sources = dict(current.get("sources") or {})
        used = sources.get("scoring_source_materializations", [])
        if (not isinstance(used, list) or not set(used).issubset(allowed) or
                len(used) > len(allowed)):
            raise ArtifactError("scoring source materialization record is malformed")
        sources["scoring_source_materializations"] = sorted(set(used) | {kind})
        # This singular field is deliberately replaced before any scoring on a
        # resumed attempt. The history above remains additive; this value names
        # the materialization that supplies every branch in the current attempt.
        sources["scoring_source_materialization_used"] = kind
        current["sources"] = sources
        atomic_write_json(path, current)
        return current

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
        donor = self.donors[donor_id]
        scenario = self.by_id[cid]
        plants = select_primary_plants(scenario)
        started = time.monotonic()
        print(f"PHASE CAPTURE position={position} id={cid} donor={donor_id}", flush=True)

        try:
            correct_messages = correct_source_messages(conv, SUMMARY_REQUEST)
            prefix_ids = generation_prefix_ids(self.tokenizer, correct_messages)
            if len(prefix_ids) + MAX_SUMMARY_TOKENS > self.context_limit:
                raise CoherentStateError(
                    f"{cid}: prefix plus frozen summary cap exceeds context "
                    f"{len(prefix_ids)}+{MAX_SUMMARY_TOKENS}>{self.context_limit}")

            def schedule_progress(evidence: dict) -> None:
                current = json.loads(path.read_text())
                atomic_write_json(path, {
                    **current,
                    "pre_score_schedule_equivalence": evidence,
                    "capture_progress": {
                        **(current.get("capture_progress") or {}),
                        "actual_render_schedule_status": evidence.get("status"),
                        "actual_render_schedule_semantic_scoring_performed": False,
                    },
                })

            actual_schedule = run_exact_render_schedule_fixture(
                self.model, self.tokenizer, conv,
                tolerance=ZERO_GAP_TOLERANCE, progress=schedule_progress)
            if (actual_schedule.get("passes") is not True or
                    actual_schedule.get("complete_prefix_token_ids") != prefix_ids or
                    actual_schedule.get("semantic_scoring_performed") is not False):
                raise CoherentStateError(
                    f"{cid}: actual rendered prefix schedule gate failed")
            existing = json.loads(path.read_text())
            _release_cuda()
            durable_generation = bool(
                (existing.get("capture_progress") or {}).get(
                    "generated_source_persisted"))
            if durable_generation:
                generated = self._restore_generated_summary(
                    existing, correct_messages)
                scoring_materialization = \
                    "bit_exact_stepwise_resume_reconstruction"
                print(
                    f"SUMMARY_RESUME position={position} id={cid} "
                    f"tokens={len(generated.summary_ids)}",
                    flush=True)
            else:
                generated = capture_generated_summary(
                    self.model, self.tokenizer, correct_messages,
                    max_tokens=MAX_SUMMARY_TOKENS)
                if generated.source_kind != "generated_incremental":
                    raise CoherentStateError(
                        "correct source is not generated_incremental")
                # This is the expensive, irreplaceable generation artifact. Save
                # it atomically before running even the independent replay gate.
                existing = self._persist_generated_summary(
                    path, existing, generated)
                scoring_materialization = "live_incremental_generation_rows"
                print(
                    f"SUMMARY_DURABLE position={position} id={cid} "
                    f"tokens={len(generated.summary_ids)}",
                    flush=True)

            # Exercise the exact compacted destination at this render's real
            # length and logical gap, through every saved summary token, before
            # constructing a wrong source, arm, target, or semantic outcome.
            declared_layout = gapped_destination_layout(
                self.tokenizer, conv, generated.summary_text,
                generated.summary_ids, SUMMARY_REQUEST,
                generated.prefix_ids)

            def destination_schedule_progress(evidence: dict) -> None:
                current = json.loads(path.read_text())
                atomic_write_json(path, {
                    **current,
                    "pre_score_destination_schedule_equivalence": evidence,
                    "capture_progress": {
                        **(current.get("capture_progress") or {}),
                        "actual_destination_schedule_status":
                            evidence.get("status"),
                        "actual_destination_schedule_semantic_scoring_performed":
                            False,
                    },
                })

            destination_schedule = measure_gapped_destination_schedule(
                self.model, declared_layout, generated.summary_ids,
                conversation_id=cid,
                tolerance=ZERO_GAP_TOLERANCE,
                progress=destination_schedule_progress)
            if (destination_schedule.get("passes") is not True or
                    destination_schedule.get("prefix_token_ids") !=
                    declared_layout.prefix_ids or
                    destination_schedule.get("summary_token_ids") !=
                    generated.summary_ids or
                    destination_schedule.get(
                        "semantic_scoring_performed") is not False):
                raise CoherentStateError(
                    f"{cid}: actual gapped destination schedule gate failed")
            _release_cuda()

            replay = capture_forced_summary(
                self.model, self.tokenizer, correct_messages,
                generated.summary_ids, source_kind="correct_stepwise_replay")
            identity = _strict_generated_replay_witness(
                generated, replay, measure_generated_replay(
                    generated, replay, IDENTITY_TOLERANCE))
            identity["scoring_source_materialization_used"] = \
                scoring_materialization
            if not (generated.summary_start == replay.summary_start and
                    generated.summary_end == replay.summary_end and
                    generated.summary_ids == replay.summary_ids):
                raise CoherentStateError(
                    "generated/replay summary span or IDs differ")
            replay.cache = None

            existing = promote_checkpoint(path, existing, {
                "sources": {
                    "correct_replay": _serializable_source(replay),
                    "generated_replay_identity": identity,
                },
                "capture_progress": {
                    "generated_replay_validated": identity["passes"],
                },
            }, "captured")
            if not identity["passes"]:
                raise CoherentStateError(
                    "generated/replay strict identity failed: "
                    f"numerical={identity['numerical_tolerance_passes']} "
                    f"row_hashes={identity['summary_row_hashes_bit_exact']}")
            existing = self._record_scoring_source_materialization(
                path, scoring_materialization)
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
                "external_donor": self.donor_provenance[donor_id],
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

            reconstructed_layout = gapped_destination_layout(
                self.tokenizer, conv, generated.summary_text,
                generated.summary_ids, SUMMARY_REQUEST,
                generated.prefix_ids)
            if reconstructed_layout != declared_layout:
                raise CoherentStateError(
                    "destination layout changed after its schedule gate")
            if declared_layout.logical_next_position > self.context_limit:
                raise CoherentStateError(
                    f"{cid}: logical destination end "
                    f"{declared_layout.logical_next_position} exceeds context "
                    f"{self.context_limit}")
            layout, fresh_trace, fresh_boundary, fresh_rows = \
                build_gapped_fresh_boundary(
                self.model, self.tokenizer, conv, generated.summary_text,
                generated.summary_ids, SUMMARY_REQUEST, generated.prefix_ids)
            if not (generated.summary_start == wrong.summary_start ==
                    layout.source_summary_start):
                raise CoherentStateError(
                    "correct/wrong/fresh summary logical starts differ")
            if layout != declared_layout:
                raise CoherentStateError(
                    "destination layout changed between context gate and execution")
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
                layout.physical_summary_start, 0)
            wrong_boundary, _ = gapped_arm_boundary(
                "G_wrong", fresh_boundary, generated.rows, wrong.rows,
                layout.physical_summary_start, 0)
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
                        layout.physical_summary_start, 0),
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
            self.ensure_render_positions({position - 1})
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
    ap.add_argument("--donor-dir", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--resume-probe-stop-after-one", action="store_true",
                    help="operational gate: exit 75 after first scored checkpoint")
    ap.add_argument(
        "--technical-only", action="store_true",
        help=("deprecated explicit spelling of the default technical-only "
              "mode; retained for launch-script clarity"))
    ap.add_argument(
        "--semantic-authorization", type=Path,
        help=("explicit prior committed v8 technical PASS directory; enables "
              "the otherwise unreachable separate semantic process"))
    ap.add_argument(
        "--technical-result-commit",
        help="named trunk commit containing the prior technical directory")
    args = ap.parse_args()
    if args.semantic_authorization is None and args.technical_result_commit:
        ap.error("--technical-result-commit requires --semantic-authorization")
    if args.semantic_authorization is not None and not args.technical_result_commit:
        ap.error("--semantic-authorization requires --technical-result-commit")
    if args.semantic_authorization is not None and args.technical_only:
        ap.error("semantic authorization and --technical-only are mutually exclusive")
    # Technical-only is the fail-closed default.  No absent/false flag can
    # expose rendering or scoring.
    args.technical_only = args.semantic_authorization is None
    return args


class IntentionalResumeProbe(RuntimeError):
    pass


def _repo_relative(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _input_inventory(repo: Path, args, donor_provenance: dict) -> dict:
    paths = {args.scenarios.resolve(), args.targets.resolve()}
    paths.update((args.donor_dir / f"{cid}.json").resolve()
                 for cid in set(FROZEN_ORDER).union(WRONG_DONOR.values()))
    rows = []
    for path in sorted(paths):
        if not path.is_file():
            raise CoherentStateError(f"frozen input absent: {path}")
        rows.append({
            "path": _repo_relative(repo, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "files": rows,
        "aggregate_sha256": sha256_json(rows),
        "external_donor_provenance": donor_provenance,
    }


def _build_static_fingerprint(args, provenance, subject_metadata,
                              donor_provenance, apparatus) -> dict:
    repo = Path(git_value("rev-parse", "--show-toplevel")).resolve()
    return {
        "schema": ARTIFACT_SCHEMA,
        "design_id": DESIGN_ID,
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": {
            path.name: sha256_file(path) for path in AMENDMENT_PATHS},
        "model": MODEL,
        "revision": REVISION,
        "dtype": "torch.bfloat16",
        "code_commit": provenance["code_commit"],
        "runtime_environment": {
            **{key: provenance.get(key) for key in (
                "python", "platform", "torch", "transformers", "cuda", "gpu")},
            "execution_packages": provenance.get("execution_packages"),
        },
        "apparatus_inventory": apparatus,
        "input_inventory": _input_inventory(repo, args, donor_provenance),
        "scenario_sha256": sha256_file(args.scenarios),
        "targets_sha256": sha256_file(args.targets),
        "summary_request_sha256": hashlib.sha256(
            SUMMARY_REQUEST.encode()).hexdigest(),
        "frozen_order": list(FROZEN_ORDER),
        "wrong_donors": WRONG_DONOR,
        "structural_seed": STRUCTURAL_SEED,
        "max_reply_tokens": MAX_REPLY_TOKENS,
        "max_summary_tokens": MAX_SUMMARY_TOKENS,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "zero_gap_tolerance": ZERO_GAP_TOLERANCE,
        "attention_backend": ATTENTION_BACKEND,
        "max_technical_logical_position": MAX_TECHNICAL_LOGICAL_POSITION,
        "arms": list(GAPPED_ARM_NAMES),
        "subject_metadata": subject_metadata,
    }


def _binding_static(value: dict) -> dict:
    # The technical launch commit necessarily precedes the commit that adds its
    # result.  Code identity is bound by the apparatus inventory and ancestry;
    # every other static field is exact.
    return {key: item for key, item in value.items() if key != "code_commit"}


def _write_identical_terminal_gate(unique_path: Path, canonical_path: Path,
                                   doc: dict) -> dict:
    sealed = write_sealed_payload(unique_path, doc)
    write_sealed_payload(canonical_path, sealed)
    if (unique_path.stat().st_size >= MAX_TERMINAL_PAYLOAD_BYTES or
            canonical_path.stat().st_size >= MAX_TERMINAL_PAYLOAD_BYTES):
        raise IntegrityError("terminal gate payload is not below 4 MiB")
    if unique_path.read_bytes() != canonical_path.read_bytes():
        raise IntegrityError("unique and canonical gate bytes differ")
    return sealed


HEAVY_GATE_STAGES = (
    "committed_case_schedule_fixtures",
    "external_donor_construction",
)
MAX_TERMINAL_PAYLOAD_BYTES = 4 * 1024 * 1024


def _terminalize_gate_lifecycle(gates: dict, failure: dict | None = None) -> dict:
    """Close every declared lifecycle state before a terminal gate is sealed."""
    closed = copy.deepcopy(gates)
    for name in closed.get("stage_order", []):
        stage = closed.get(name)
        if not isinstance(stage, dict):
            continue
        status = stage.get("status")
        if status in {"PENDING", "RUNNING"}:
            stage["status"] = (
                "ERROR" if status == "RUNNING" else "SKIPPED_DEPENDENCY")
            stage["passes"] = False
            stage["completed_at"] = utc_now()
            stage["failure_evidence"] = {
                "failed_prerequisite": (
                    name if status == "RUNNING" else
                    "driver_or_prior_gate_failure"),
                "reason": ((failure or {}).get("error") or
                           "attempt terminated before this stage closed"),
                "prior_status": status,
            }
    closed["passes"] = bool(
        failure is None and
        closed.get("stage_order") and all(
            isinstance(closed.get(name), dict) and
            closed[name].get("status") == "PASS"
            for name in closed["stage_order"]))
    closed["status"] = "PASS" if closed["passes"] else "FAIL"
    if failure is not None:
        closed["failure"] = failure
    return closed


def _externalize_heavy_gate_stages(
        run_dir: Path, gates: dict, terminal_status: str) -> tuple[dict, list[str]]:
    """Move complete heavy lifecycle stages into sealed, indexed sidecars."""
    compact = copy.deepcopy(gates)
    refs = {}
    paths = []
    for stage_name in HEAVY_GATE_STAGES:
        if stage_name not in compact:
            raise IntegrityError(f"predeclared heavy stage absent: {stage_name}")
        lifecycle = compact.pop(stage_name)
        relative = f"technical_stage_{stage_name}.json"
        path = run_dir / relative
        sidecar = write_sealed_payload(path, {
            "schema": ARTIFACT_SCHEMA,
            "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID,
            "status": terminal_status,
            "kind": "technical_gate_stage_sidecar",
            "stage_name": stage_name,
            "lifecycle": lifecycle,
        })
        byte_count = path.stat().st_size
        if byte_count >= MAX_TERMINAL_PAYLOAD_BYTES:
            raise IntegrityError(
                f"{relative} is {byte_count} bytes, not below 4 MiB")
        refs[stage_name] = {
            "path": relative,
            "byte_count": byte_count,
            "raw_file_sha256": sha256_file(path),
            "payload_sha256": sidecar["payload_sha256"],
        }
        paths.append(relative)
    compact["stage_refs"] = refs
    return compact, paths


def _seal_semantic_terminal(run_dir: Path, terminal_status: str) -> None:
    """Seal and index every terminal semantic JSON artifact in-place."""
    if ((run_dir / INDEX_NAME).exists() or (run_dir / RECEIPT_NAME).exists()):
        raise IntegrityError("semantic terminal envelope already exists")
    paths = sorted(
        path for path in run_dir.rglob("*.json")
        if path.name not in {INDEX_NAME, RECEIPT_NAME})
    if not paths:
        raise IntegrityError("semantic terminal has no JSON payloads")
    relatives = []
    for path in paths:
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict):
            raise IntegrityError(f"semantic terminal payload is not object: {path}")
        doc = {
            "schema": doc.get("schema", ARTIFACT_SCHEMA),
            "design_id": doc.get("design_id", DESIGN_ID),
            "amendment_id": doc.get("amendment_id", AMENDMENT_ID),
            **doc,
        }
        if (doc["schema"] != ARTIFACT_SCHEMA or doc["design_id"] != DESIGN_ID or
                doc["amendment_id"] != AMENDMENT_ID):
            raise IntegrityError(f"semantic payload identity differs: {path}")
        write_sealed_payload(path, doc)
        if path.stat().st_size >= MAX_TERMINAL_PAYLOAD_BYTES:
            raise IntegrityError(f"semantic payload is not below 4 MiB: {path}")
        relatives.append(path.relative_to(run_dir).as_posix())
    write_terminal_envelope(
        run_dir, relatives, terminal_status=terminal_status)


def _technical_main(args) -> int:
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if list(run_dir.glob("*.json")) or list(run_dir.glob("conv_*.json")):
        raise ArtifactError("technical attempt requires a fresh unique run directory")
    manifest_path = run_dir / "manifest.json"
    gate_path = run_dir / "production_kernel_gate.json"
    invocation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    gate_attempt_path = run_dir / f"production_kernel_gate_{invocation_id}.json"
    started_at = utc_now()
    phase = "INVOCATION_START"
    provenance = None
    static_fingerprint = None
    fingerprint = None
    apparatus = None
    geometry = None
    backend_fingerprint = None
    context_limit = None
    gate_schema = v8_gate_schema(
        identity_tolerance=IDENTITY_TOLERANCE,
        zero_gap_tolerance=ZERO_GAP_TOLERANCE,
        case_dir=args.donor_dir, donor_dir=args.donor_dir)

    def persist_running(gates: dict, **extra) -> None:
        prior = json.loads(gate_attempt_path.read_text()) \
            if gate_attempt_path.exists() else {}
        document = {
            **prior,
            "schema": ARTIFACT_SCHEMA,
            "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID,
            "status": "RUNNING",
            "started_at": prior.get("started_at", started_at),
            "updated_at": utc_now(),
            "model": MODEL,
            "revision": REVISION,
            "dtype": "torch.bfloat16",
            "attention_backend_requested": ATTENTION_BACKEND,
            "technical_only": True,
            "fingerprint_static": static_fingerprint,
            "apparatus_inventory": apparatus,
            "gates": gates,
            **extra,
        }
        atomic_write_json(gate_attempt_path, document)
        if json.loads(gate_attempt_path.read_text()) != document:
            raise ArtifactError("durable gate progress read-back mismatch")

    gate_sink = DurableDiagnosticSink(lambda value: persist_running(value))
    dict.update(gate_sink, gate_schema)
    # Exhaustive lifecycle schema exists durably before any model load.
    persist_running(dict(gate_sink))
    atomic_write_json(manifest_path, {
        "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
        "amendment_id": AMENDMENT_ID, "status": "SETUP",
        "phase": phase, "started_at": started_at, "technical_only": True,
        "production_kernel_gate_path": gate_path.name,
        "production_kernel_gate_attempt_path": gate_attempt_path.name,
    })

    try:
        phase = "STATIC_PROVENANCE"
        static_stage = dict(gate_sink["static_provenance"])
        static_stage.update({
            "status": "RUNNING", "started_at": utc_now(),
            "observed_coverage": 0,
        })
        gate_sink["static_provenance"] = static_stage
        _static_design_self_check()
        missing = [path for path in AMENDMENT_PATHS if not path.is_file()]
        if missing:
            raise CoherentStateError(f"missing frozen amendments: {missing}")
        repo = Path(git_value("rev-parse", "--show-toplevel")).resolve()
        provenance = runtime_provenance(run_dir)
        apparatus = apparatus_inventory(repo)
        config, tokenizer, subject_metadata = prepare_subject_metadata()
        scenario_map = load_scenarios(args.scenarios)
        frozen_scenarios(scenario_map)
        load_and_validate_targets(args.targets, scenario_map)
        _, donor_provenance = load_external_donors(args.donor_dir)
        static_fingerprint = _build_static_fingerprint(
            args, provenance, subject_metadata, donor_provenance, apparatus)
        static_stage.update({
            "status": "PASS", "passes": True,
            "observed_coverage": 1, "completed_at": utc_now(),
            "failure_evidence": None,
            "raw": {
                "fingerprint_static_sha256": sha256_json(static_fingerprint),
                "apparatus_inventory_sha256": sha256_json(apparatus),
                "input_inventory_sha256": sha256_json(
                    static_fingerprint["input_inventory"]),
                "code_commit": static_fingerprint["code_commit"],
                "model": MODEL, "revision": REVISION,
                "dtype": "torch.bfloat16",
                "attention_backend": ATTENTION_BACKEND,
                "technical_only": True,
            },
        })
        gate_sink["static_provenance"] = static_stage
        persist_running(dict(gate_sink), fingerprint_static=static_fingerprint)
        atomic_write_json(manifest_path, {
            "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID, "status": "SETUP",
            "phase": phase, "started_at": started_at, "updated_at": utc_now(),
            "technical_only": True, "fingerprint_static": static_fingerprint,
            "apparatus_inventory": apparatus, "provenance": provenance,
            "production_kernel_gate_path": gate_path.name,
            "production_kernel_gate_attempt_path": gate_attempt_path.name,
        })

        phase = "MODEL_LOADING"

        def backend_progress(partial: dict) -> None:
            stage = dict(gate_sink["attention_backend"])
            stage.update({
                "status": "RUNNING",
                "started_at": stage.get("started_at") or utc_now(),
                "observed_coverage": len(partial.get("layers", [])),
                "raw": partial,
            })
            gate_sink["attention_backend"] = stage

        model, tokenizer, geometry, backend_fingerprint = load_subject(
            config, tokenizer, backend_progress=backend_progress)
        context_limit = model_context_limit(model)
        subject_metadata = {
            **subject_metadata,
            "attention_backend_resolved": ATTENTION_BACKEND,
            "attention_backend_fingerprint": backend_fingerprint,
            "context_limit": context_limit,
        }
        provenance["subject"] = subject_metadata
        fingerprint = {
            **static_fingerprint,
            "subject_metadata": subject_metadata,
            "attention_backend_fingerprint": backend_fingerprint,
            "context_limit": context_limit,
        }
        persist_running(
            dict(gate_sink), fingerprint=fingerprint, geometry=geometry,
            context_limit=context_limit,
            attention_backend_fingerprint=backend_fingerprint)
        print(f"MODEL_READY revision={REVISION} dtype=bf16 "
              f"attention_backend={ATTENTION_BACKEND} "
              f"backend_sha256={backend_fingerprint['sha256']} "
              f"context_limit={context_limit} geometry={geometry}", flush=True)

        phase = "PRODUCTION_GATE"
        try:
            production_gate = run_loaded_gapped_gates(
                model, tokenizer,
                identity_tolerance=IDENTITY_TOLERANCE,
                zero_gap_tolerance=ZERO_GAP_TOLERANCE,
                diagnostic_sink=gate_sink,
                case_dir=args.donor_dir,
                donor_dir=args.donor_dir)
        except Exception as gate_exc:
            production_gate = {
                **gate_sink, "passes": False,
                "failure": {
                    "error_type": type(gate_exc).__name__,
                    "error": str(gate_exc), "traceback": traceback.format_exc(),
                },
            }
        assert_technical_gate_has_no_semantic_scores(production_gate)
        persisted_gate = json.loads(gate_attempt_path.read_text()).get("gates")
        if persisted_gate != dict(production_gate):
            raise ArtifactError(
                "terminal gate differs from durably persisted measurements")
        backend_gate = production_gate.get("attention_backend") or {}
        if (production_gate.get("passes") is not True or
                backend_gate.get("passes") is not True or
                backend_gate.get("fingerprint") != backend_fingerprint):
            raise CoherentStateError(
                "production technical gate failed or backend attestation changed")
        if list(run_dir.glob("conv_*.json")):
            raise ArtifactError("technical process produced a conversation checkpoint")

        completed_at = utc_now()
        production_gate = _terminalize_gate_lifecycle(production_gate)
        production_gate, sidecar_paths = _externalize_heavy_gate_stages(
            run_dir, production_gate, "PASS")
        gate_doc = _write_identical_terminal_gate(
            gate_attempt_path, gate_path, {
                "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
                "amendment_id": AMENDMENT_ID, "status": "PASS",
                "completed_at": completed_at, "model": MODEL,
                "revision": REVISION, "dtype": "torch.bfloat16",
                "geometry": geometry, "attention_backend": ATTENTION_BACKEND,
                "attention_backend_fingerprint": backend_fingerprint,
                "context_limit": context_limit, "fingerprint": fingerprint,
                "fingerprint_static": static_fingerprint,
                "apparatus_inventory": apparatus, "technical_only": True,
                "gates": production_gate,
            })
        manifest = write_sealed_payload(manifest_path, {
            "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID, "status": "TECHNICAL_PASS",
            "phase": "TECHNICAL_COMPLETE", "started_at": started_at,
            "completed_at": completed_at, "technical_only": True,
            "model": MODEL, "revision": REVISION, "dtype": "torch.bfloat16",
            "attention_backend": ATTENTION_BACKEND,
            "attention_backend_fingerprint": backend_fingerprint,
            "geometry": geometry, "context_limit": context_limit,
            "fingerprint_static": static_fingerprint,
            "fingerprint": fingerprint, "apparatus_inventory": apparatus,
            "provenance": provenance,
            "production_kernel_gate_path": gate_path.name,
            "production_kernel_gate_attempt_path": gate_attempt_path.name,
            "production_kernel_gate_payload_sha256": gate_doc["payload_sha256"],
        })
        write_terminal_envelope(
            run_dir,
            [gate_attempt_path.name, gate_path.name, manifest_path.name,
             *sidecar_paths],
            terminal_status="PASS")
        print("COHERENCE_TECHNICAL_RECEIPT_DURABLE status=PASS", flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID, "status": "ERROR", "phase": phase,
            "failed_at": utc_now(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        }
        try:
            if ((run_dir / INDEX_NAME).exists() or
                    (run_dir / RECEIPT_NAME).exists()):
                raise IntegrityError(
                    "terminal envelope already started; refusing to mutate payloads")
            write_sealed_payload(run_dir / "failure.json", failure)
            failed_gates = _terminalize_gate_lifecycle(
                dict(gate_sink), failure)
            failed_gates, sidecar_paths = _externalize_heavy_gate_stages(
                run_dir, failed_gates, "FAIL")
            _write_identical_terminal_gate(
                gate_attempt_path, gate_path, {
                    "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
                    "amendment_id": AMENDMENT_ID, "status": "FAIL",
                    "completed_at": utc_now(), "model": MODEL,
                    "revision": REVISION, "dtype": "torch.bfloat16",
                    "geometry": geometry, "attention_backend": ATTENTION_BACKEND,
                    "attention_backend_fingerprint": backend_fingerprint,
                    "context_limit": context_limit, "fingerprint": fingerprint,
                    "fingerprint_static": static_fingerprint,
                    "apparatus_inventory": apparatus, "technical_only": True,
                    "gates": failed_gates, "error": failure,
                })
            write_sealed_payload(manifest_path, {
                "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
                "amendment_id": AMENDMENT_ID, "status": "ERROR",
                "phase": phase, "started_at": started_at,
                "completed_at": utc_now(), "technical_only": True,
                "model": MODEL, "revision": REVISION, "dtype": "torch.bfloat16",
                "fingerprint_static": static_fingerprint,
                "fingerprint": fingerprint, "apparatus_inventory": apparatus,
                "provenance": provenance, "geometry": geometry,
                "attention_backend": ATTENTION_BACKEND,
                "attention_backend_fingerprint": backend_fingerprint,
                "context_limit": context_limit, "failure_path": "failure.json",
                "production_kernel_gate_path": gate_path.name,
                "production_kernel_gate_attempt_path": gate_attempt_path.name,
            })
            write_terminal_envelope(
                run_dir,
                [gate_attempt_path.name, gate_path.name, manifest_path.name,
                 "failure.json", *sidecar_paths], terminal_status="FAIL")
            print("COHERENCE_TECHNICAL_RECEIPT_DURABLE status=FAIL", flush=True)
        except Exception as terminal_exc:
            print(f"FATAL terminalization failed: {terminal_exc}",
                  file=sys.stderr, flush=True)
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


def _semantic_main(args) -> int:
    """Separate process; unreachable without an explicit committed attestation."""
    run_dir = args.run_dir
    authorization_dir = args.semantic_authorization
    resolved_run = run_dir.resolve()
    resolved_authorization = authorization_dir.resolve()
    if (resolved_run == resolved_authorization or
            resolved_authorization in resolved_run.parents or
            resolved_run in resolved_authorization.parents):
        raise IntegrityError(
            "semantic run and technical authorization directories overlap")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    started_at = utc_now()
    phase = "SEMANTIC_AUTHORIZATION"
    fingerprint = None
    provenance = None
    geometry = None
    try:
        repo = Path(git_value("rev-parse", "--show-toplevel")).resolve()
        launch_commit = git_value("rev-parse", "HEAD")
        current_apparatus = apparatus_inventory(repo)
        authorization = verify_prior_technical_authorization(
            repo, args.semantic_authorization, args.technical_result_commit,
            launch_commit, current_apparatus)
        provenance = runtime_provenance(run_dir)
        config, tokenizer, subject_metadata = prepare_subject_metadata()
        scenario_map = load_scenarios(args.scenarios)
        scenarios = frozen_scenarios(scenario_map)
        targets = load_and_validate_targets(args.targets, scenario_map)
        donors, donor_provenance = load_external_donors(args.donor_dir)
        static_fingerprint = _build_static_fingerprint(
            args, provenance, subject_metadata, donor_provenance,
            current_apparatus)
        if _binding_static(static_fingerprint) != _binding_static(
                authorization.static_fingerprint):
            raise IntegrityError(
                "semantic static/data fingerprint differs from technical PASS")
        atomic_write_json(manifest_path, {
            "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID,
            "status": "STATIC_AUTHORIZATION_VERIFIED_MODEL_PENDING",
            "phase": phase, "started_at": started_at, "technical_only": False,
            "semantic_authorization": asdict(authorization),
            "authorization_checks": {
                "terminal_payloads_exact": True,
                "committed_directory_bytes_exact": True,
                "result_commit_is_ancestor": True,
                "result_commit_on_trunk": True,
                "harvest_attestation_committed_exact": True,
                "independent_harvest_revalidation_passed": True,
                "apparatus_inventory_exact": True,
                "static_data_fingerprint_exact": True,
                "backend_attestation_exact": False,
            },
            "fingerprint_static": static_fingerprint,
            "apparatus_inventory": current_apparatus,
        })

        phase = "MODEL_LOADING"
        model, tokenizer, geometry, backend_fingerprint = load_subject(
            config, tokenizer)
        context_limit = model_context_limit(model)
        prior_gate = json.loads(
            (args.semantic_authorization / "production_kernel_gate.json").read_text())
        if backend_fingerprint != prior_gate.get("attention_backend_fingerprint"):
            raise IntegrityError(
                "recomputed semantic backend differs from technical PASS")
        subject_metadata = {
            **subject_metadata,
            "attention_backend_resolved": ATTENTION_BACKEND,
            "attention_backend_fingerprint": backend_fingerprint,
            "context_limit": context_limit,
        }
        provenance["subject"] = subject_metadata
        fingerprint = {
            **static_fingerprint,
            "subject_metadata": subject_metadata,
            "attention_backend_fingerprint": backend_fingerprint,
            "context_limit": context_limit,
            "semantic_authorization": {
                "technical_result_commit": authorization.result_commit,
                "technical_run_dir": authorization.run_dir,
                "gate_payload_sha256": authorization.gate_payload_sha256,
                "raw_sha256": authorization.raw_sha256,
                "harvest_path": authorization.harvest["path"],
                "harvest_raw_sha256": authorization.harvest["raw_sha256"],
                "harvest_payload_sha256": authorization.harvest["payload_sha256"],
                "apparatus_aggregate_sha256": current_apparatus[
                    "aggregate_sha256"],
                "backend_exact": True,
                "static_fingerprint_exact": True,
            },
        }
        atomic_write_json(manifest_path, {
            "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID, "status": "RUNNING",
            "phase": "SEMANTIC_RUN", "started_at": started_at,
            "updated_at": utc_now(), "technical_only": False,
            "fingerprint_static": static_fingerprint,
            "fingerprint": fingerprint, "provenance": provenance,
            "geometry": geometry, "context_limit": context_limit,
            "attention_backend": ATTENTION_BACKEND,
            "attention_backend_fingerprint": backend_fingerprint,
            "semantic_authorization": asdict(authorization),
            "authorization_checks": {
                "terminal_payloads_exact": True,
                "committed_directory_bytes_exact": True,
                "result_commit_is_ancestor": True,
                "result_commit_on_trunk": True,
                "harvest_attestation_committed_exact": True,
                "independent_harvest_revalidation_passed": True,
                "apparatus_inventory_exact": True,
                "static_data_fingerprint_exact": True,
                "backend_attestation_exact": True,
            },
        })
        phase = "SEMANTIC_RUN"
        runner = Runner(
            args, model, tokenizer, scenarios, targets, fingerprint,
            provenance, geometry, donors, donor_provenance)
        stats6 = runner.run_stage(6)
        stats = runner.run_stage(12) \
            if stats6["serial_decision"] == "EXTEND_TO_12" else stats6
        marker_path = run_dir / "resume_probe.json"
        marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
        if marker.get("resume_probe_verified") is not True:
            raise ArtifactError("forced-restart resume probe was not verified")
        atomic_write_json(manifest_path, {
            **json.loads(manifest_path.read_text()),
            "status": "COMPLETE", "phase": "COMPLETE",
            "completed_at": utc_now(),
            "resume_probe_verified": True,
            "n_conversations": stats["n_conversations"],
            "serial_decision": stats["serial_decision"],
            "interpretation": stats["interpretation"],
        })
        _seal_semantic_terminal(run_dir, "PASS")
        print(f"COHERENCE_RUN_DONE n={stats['n_conversations']} "
              f"decision={stats['serial_decision']}", flush=True)
        return 0
    except IntentionalResumeProbe:
        manifest = json.loads(manifest_path.read_text())
        atomic_write_json(manifest_path, {
            **manifest, "status": "RESUME_REQUIRED", "phase": "FORCED_RESTART",
            "updated_at": utc_now(),
        })
        return 75
    except Exception as exc:
        failure = {
            "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
            "amendment_id": AMENDMENT_ID, "status": "ERROR", "phase": phase,
            "failed_at": utc_now(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        }
        if ((run_dir / INDEX_NAME).exists() or
                (run_dir / RECEIPT_NAME).exists()):
            print("FATAL semantic terminal envelope already started; "
                  "payloads left immutable", file=sys.stderr, flush=True)
        else:
            write_sealed_payload(run_dir / "failure.json", failure)
            voids = terminalize_partial_checkpoints(run_dir, failure)
            prior = (json.loads(manifest_path.read_text())
                     if manifest_path.exists() else {})
            atomic_write_json(manifest_path, {
                **prior, "schema": ARTIFACT_SCHEMA, "design_id": DESIGN_ID,
                "amendment_id": AMENDMENT_ID, "status": "ERROR", "phase": phase,
                "updated_at": utc_now(), "technical_only": False,
                "void_checkpoint_paths": voids,
            })
            try:
                _seal_semantic_terminal(run_dir, "FAIL")
            except Exception as terminal_exc:
                print(f"FATAL semantic terminalization failed: {terminal_exc}",
                      file=sys.stderr, flush=True)
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


def main():
    args = parse_args()
    mode = "semantic" if args.semantic_authorization is not None else "technical-only"
    print(f"RUN coherent_state model={MODEL}@{REVISION} mode={mode} -> "
          f"{args.run_dir}", flush=True)
    if args.semantic_authorization is not None:
        return _semantic_main(args)
    return _technical_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
