"""Production driver for the preregistered coherent-summary-state experiment."""

from __future__ import annotations

import argparse
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
from arms_hf import rope_base
from coherent_state_calibration import run_calibration
from coherent_state_cases import (
    FROZEN_ORDER,
    WRONG_DONOR,
    correct_source_messages,
    frozen_scenarios,
    load_and_validate_targets,
    load_scenarios,
    select_primary_plants,
    wrong_source_messages,
)
from coherent_state_hf import (
    CoherentStateError,
    compare_rows,
    move_key_rows,
    replace_summary_rows,
    row_hashes,
    sha256_ids,
)
from coherent_state_runtime import (
    ARM_NAMES,
    arm_snapshot,
    build_fresh_destination,
    capture_forced_summary,
    capture_generated_summary,
    complete_assistant_context,
    score_arm,
    score_target,
    validate_generated_replay,
)
from coherent_state_store import (
    ArtifactError,
    atomic_write_json,
    checkpoint_path,
    promote_checkpoint,
    read_checkpoint,
    save_render,
    validate_scored_checkpoint,
)
from cross_arch_probe import native_render_specs
from l_coherent_state_hf import run_loaded_kernel_gates


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STRUCTURAL_SEED = 20_260_711
PLACEBO_SEED = 20_260_711
EXPECTED_GEOMETRY = {
    "layers": 48, "attention_heads": 32, "kv_heads": 4,
    "head_dim": 128, "rope_theta": 10_000_000,
}
MAX_REPLY_TOKENS = 320
MAX_SUMMARY_TOKENS = 900
IDENTITY_TOLERANCE = 1e-4
ROTATION_TOLERANCE = 0.02
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
    dirty_rows = [x for x in git_value("status", "--porcelain").splitlines() if x]
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
        "prefix_sha256": capture.prefix_sha256,
        "prefix_token_count": len(capture.prefix_ids),
        "summary_start": capture.summary_start,
        "summary_end": capture.summary_end,
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
        self.rope_theta = float(geometry["rope_theta"])

    def ckpath(self, position: int, cid: str) -> Path:
        return checkpoint_path(self.run_dir, position, cid)

    def score_bounded(self, builder, layout, plants):
        """Score one fresh branch per target; never retain a full arm snapshot."""
        rows = []
        saved_diagnostics = None
        for plant in plants:
            target = self.targets[plant["id"]]
            snap, diag = builder()
            if saved_diagnostics is None:
                saved_diagnostics = diag
            correct = score_target(
                self.model, self.tokenizer, snap, layout.messages,
                layout.context_ids, plant["probe"], target["correct"],
                consume_snapshot=True)
            snap, _ = builder()
            counterfactual = score_target(
                self.model, self.tokenizer, snap, layout.messages,
                layout.context_ids, plant["probe"], target["counterfactual"],
                consume_snapshot=True)
            rows.append({
                "plant_id": plant["id"], "category": plant["category"],
                "probe": plant["probe"], "correct": correct,
                "counterfactual": counterfactual,
                "margin": (correct["mean_logprob"] -
                           counterfactual["mean_logprob"]),
            })
        return ({"plants": rows,
                 "conversation_margin": sum(x["margin"] for x in rows) / len(rows)},
                saved_diagnostics or [])

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
            save_render(
                self.ckpath(idx + 1, cid), fingerprint=self.fingerprint,
                order_position=idx + 1, conversation=conv,
                reply_records=reply_records)
            if conv.get("meta", {}).get("truncated_replies", 0):
                current = read_checkpoint(
                    self.ckpath(idx + 1, cid), self.fingerprint)
                promote_checkpoint(self.ckpath(idx + 1, cid), current, {
                    "gates": {"technical_pass": False, "failures": [{
                        "type": "reply_cap",
                        "count": conv["meta"]["truncated_replies"],
                    }]}
                }, "void")
                raise CoherentStateError(
                    f"{cid} hit the {MAX_REPLY_TOKENS}-token native reply cap")
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
                expected = json.loads(marker.read_text())["checkpoint_sha256"]
                observed = sha256_file(path)
                if observed != expected:
                    raise ArtifactError(
                        f"resume changed first scored checkpoint: {observed} != {expected}")
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
            replay.cache = None

            prior_summary = existing.get("summary")
            if prior_summary is not None:
                if (prior_summary.get("token_ids") != generated.summary_ids or
                        prior_summary.get("actual_row_hashes") != generated.row_hashes):
                    raise CoherentStateError(
                        "resumed generated summary differs from captured artifact")
            elif existing.get("stage") == "rendered":
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
                }, "captured")
                print(f"CHECKPOINT_CAPTURED position={position} id={cid}", flush=True)

            # Full-history competence/headroom reference, scored before the full
            # correct cache is released.  Summary EOS/wrapper is appended exactly.
            a_messages, a_ids, a_snapshot = complete_assistant_context(
                self.model, self.tokenizer, correct_messages, generated)
            generated.cache = None
            a_score = score_arm(
                self.model, self.tokenizer, a_snapshot, a_messages, a_ids,
                plants, self.targets)
            del a_snapshot
            _release_cuda()

            layout, fresh_trace, fresh_snapshot, fresh_rows = build_fresh_destination(
                self.model, self.tokenizer, conv, generated.summary_text,
                generated.summary_ids, SUMMARY_REQUEST)
            wrong_messages = wrong_source_messages(
                conv, donor, SUMMARY_REQUEST)
            wrong = capture_forced_summary(
                self.model, self.tokenizer, wrong_messages,
                generated.summary_ids, source_kind="wrong_history_counterfactual")
            wrong.cache = None

            # Self-replacement is an exact tensor no-op and may touch only the
            # declared summary span.  This is checked before any semantic arm.
            self_replaced = replace_summary_rows(
                fresh_snapshot, fresh_rows, layout.summary_start,
                use_keys=True, use_values=True)
            if not _tensor_snapshots_equal(fresh_snapshot, self_replaced):
                raise CoherentStateError("fresh self-replacement is not bit-identical")
            del self_replaced
            _release_cuda()
            self_score, _ = self.score_bounded(
                lambda: (replace_summary_rows(
                    fresh_snapshot, fresh_rows, layout.summary_start,
                    use_keys=True, use_values=True), []),
                layout, plants)

            # Production position-movement diagnostic.  The ladder freezes the
            # tolerance passed on the command line before the paid run.
            correct_delta = layout.summary_start - generated.summary_start
            wrong_delta = layout.summary_start - wrong.summary_start
            moved = move_key_rows(generated.rows, correct_delta, self.rope_theta)
            roundtrip = move_key_rows(moved, -correct_delta, self.rope_theta)
            rotation_rows = compare_rows(generated.rows, roundtrip)
            rotation_max = max(x["k_max_abs"] for x in rotation_rows)
            zero_rows = compare_rows(
                generated.rows,
                move_key_rows(generated.rows, 0, self.rope_theta))
            zero_max = max(x["k_max_abs"] for x in zero_rows)
            if zero_max != 0:
                raise CoherentStateError(f"zero key rotation changed K by {zero_max}")
            if rotation_max > ROTATION_TOLERANCE:
                raise CoherentStateError(
                    f"key rotation roundtrip {rotation_max} exceeds "
                    f"{ROTATION_TOLERANCE}")
            del moved, roundtrip

            arm_scores = {"A_full": a_score}
            arm_diagnostics = {}
            outcomes = {"A_full": a_score["conversation_margin"]}
            for arm in ARM_NAMES[1:]:
                score, diag = self.score_bounded(
                    lambda arm=arm: arm_snapshot(
                        arm, fresh_snapshot, generated.rows, wrong.rows,
                        layout.summary_start, correct_delta, wrong_delta,
                        self.rope_theta, PLACEBO_SEED + position),
                    layout, plants)
                arm_scores[arm] = score
                arm_diagnostics[arm] = diag
                outcomes[arm] = score["conversation_margin"]
                if arm == "F_fresh":
                    def token_lps(s):
                        return [lp for row in s["plants"]
                                for side in ("correct", "counterfactual")
                                for lp in row[side]["token_logprobs"]]
                    f_lp, self_lp = token_lps(score), token_lps(self_score)
                    noop_max = max(abs(a - b) for a, b in zip(f_lp, self_lp))
                    if noop_max > IDENTITY_TOLERANCE:
                        raise CoherentStateError(
                            f"tokenwise fresh self-replacement diff {noop_max}")
                if arm == "D_delta":
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
                _release_cuda()

            calibration = run_calibration(
                self.model, self.tokenizer, cid, self.rope_theta)
            wrong_nll = -sum(wrong.trace["token_logprobs"]) / len(
                wrong.trace["token_logprobs"])
            additions = {
                "sources": {
                    **existing["sources"],
                    "fresh": {
                        "source_kind": "fresh_stepwise_forced",
                        "prefix_sha256": sha256_ids(layout.prefix_ids),
                        "summary_start": layout.summary_start,
                        "summary_end": layout.summary_end,
                        "summary_row_hashes": row_hashes(fresh_rows),
                        "trace": asdict(fresh_trace),
                    },
                    "wrong": _serializable_source(wrong),
                    "wrong_summary_mean_nll": wrong_nll,
                },
                "destination": {
                    "context_token_ids": layout.context_ids,
                    "context_sha256": sha256_ids(layout.context_ids),
                    "summary_start": layout.summary_start,
                    "summary_end": layout.summary_end,
                    "post_summary_token_ids": layout.post_summary_ids,
                    "correct_key_delta": correct_delta,
                    "wrong_key_delta": wrong_delta,
                },
                "arm_scores": arm_scores,
                "arm_diagnostics": arm_diagnostics,
                "conversation_outcomes": outcomes,
                "calibration": calibration,
                "calibration_outcomes": calibration["outcomes"],
                "gates": {
                    "technical_pass": True, "failures": [],
                    "generated_replay_identity": identity,
                    "fresh_self_replacement_exact": True,
                    "fresh_self_replacement_tokenwise_max_abs": noop_max,
                    "alpha_zero_tokenwise_max_abs": noop_max,
                    "irrelevant_history_no_state_change_max_abs": noop_max,
                    "key_rotation_zero_max_abs": zero_max,
                    "rotation_roundtrip_max_abs": rotation_max,
                    "rotation_tolerance": ROTATION_TOLERANCE,
                    "exact_summary_span": True,
                },
                "runtime": {
                    **self.provenance,
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at": utc_now(),
                },
            }
            scored = promote_checkpoint(path, existing, additions, "scored")
            existing = scored
            validate_scored_checkpoint(scored)
            del fresh_snapshot
            _release_cuda()
            print(
                f"CHECKPOINT_SCORED position={position} id={cid} "
                f"CF={outcomes['C_coherent'] - outcomes['F_fresh']:+.6f} "
                f"CW={outcomes['C_coherent'] - outcomes['W_wrong']:+.6f}",
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
                    "status": "INTENTIONAL_RESTART_REQUIRED",
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
    try:
        provenance = runtime_provenance(args.run_dir)
        config, tokenizer, subject_metadata = prepare_subject_metadata()
        provenance["subject"] = subject_metadata
        scenario_map = load_scenarios(args.scenarios)
        scenarios = frozen_scenarios(scenario_map)
        targets = load_and_validate_targets(args.targets, scenario_map)
        fingerprint = {
            "schema": 1, "model": MODEL, "revision": REVISION,
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
            "rotation_tolerance": ROTATION_TOLERANCE,
            "subject_metadata": subject_metadata,
        }
        prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if prior is not None and prior.get("fingerprint") != fingerprint:
            raise ArtifactError("run manifest fingerprint mismatch")
        started_at = prior.get("started_at") if prior else utc_now()
        atomic_write_json(manifest_path, {
            "status": "SETUP", "started_at": started_at,
            "resumed_at": utc_now() if prior else None,
            "fingerprint": fingerprint, "provenance": provenance,
        })
        print("PHASE SETUP", flush=True)
        model, tokenizer, geometry = load_subject(config, tokenizer)
        print(f"MODEL_READY revision={REVISION} dtype=bf16 geometry={geometry}",
              flush=True)
        production_gate = run_loaded_kernel_gates(
            model, tokenizer, identity_tolerance=IDENTITY_TOLERANCE,
            rotation_tolerance=ROTATION_TOLERANCE,
            placebo_quantization_tolerance=PLACEBO_QUANTIZATION_TOLERANCE,
            placebo_moment_tolerance=PLACEBO_MOMENT_TOLERANCE)
        atomic_write_json(args.run_dir / "production_kernel_gate.json", {
            "status": "PASS", "completed_at": utc_now(),
            "model": MODEL, "revision": REVISION,
            "dtype": "torch.bfloat16", "geometry": geometry,
            "gates": production_gate,
        })
        print(
            "PRODUCTION_KERNEL_GATE_PASS "
            f"nativeK={production_gate['native_shift_k_max_abs']} "
            f"nativeV={production_gate['native_shift_v_max_abs']}", flush=True)
        runner = Runner(args, model, tokenizer, scenarios, targets, fingerprint,
                        provenance, geometry)
        stats6 = runner.run_stage(6)
        if stats6["serial_decision"] == "EXTEND_TO_12":
            stats = runner.run_stage(12)
        else:
            stats = stats6
        atomic_write_json(manifest_path, {
            "status": "COMPLETE", "started_at": started_at,
            "completed_at": utc_now(), "fingerprint": fingerprint,
            "provenance": provenance, "geometry": geometry,
            "production_kernel_gate": production_gate,
            "n_conversations": stats["n_conversations"],
            "serial_decision": stats["serial_decision"],
            "interpretation": stats["interpretation"],
        })
        print(
            f"COHERENCE_RUN_DONE n={stats['n_conversations']} "
            f"decision={stats['serial_decision']}", flush=True)
        return 0
    except IntentionalResumeProbe:
        return 75
    except Exception as exc:
        failure = {"status": "ERROR", "failed_at": utc_now(),
                   "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc()}
        atomic_write_json(args.run_dir / "failure.json", failure)
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
