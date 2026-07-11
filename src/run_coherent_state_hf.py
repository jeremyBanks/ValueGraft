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


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STRUCTURAL_SEED = 20_260_711
PLACEBO_SEED = 20_260_711
EXPECTED_GEOMETRY = {
    "layers": 48, "attention_heads": 32, "kv_heads": 4,
    "head_dim": 128, "rope_theta": 10_000_000,
}


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


def runtime_provenance() -> dict:
    commit = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))
    if dirty:
        raise CoherentStateError("production run requires a clean worktree")
    return {
        "code_commit": commit, "worktree_dirty": dirty,
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


def load_subject():
    if not torch.cuda.is_available():
        raise CoherentStateError("paid production driver requires CUDA")
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    resolved = getattr(config, "_commit_hash", None)
    if resolved != REVISION:
        raise CoherentStateError(f"resolved revision {resolved} != {REVISION}")
    geometry = model_geometry(config)
    if geometry != EXPECTED_GEOMETRY:
        raise CoherentStateError(
            f"model geometry {geometry} != {EXPECTED_GEOMETRY}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
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

    def load_rendered(self, position: int) -> dict:
        cid = FROZEN_ORDER[position - 1]
        doc = read_checkpoint(self.ckpath(position, cid), self.fingerprint)
        if doc is None:
            raise ArtifactError(f"missing rendered checkpoint for {cid}")
        return doc

    def ensure_renders(self, limit: int) -> None:
        missing = []
        for idx, cid in enumerate(FROZEN_ORDER[:limit]):
            if read_checkpoint(self.ckpath(idx + 1, cid), self.fingerprint) is None:
                missing.append(idx)
        if not missing:
            print(f"RENDER_RESUME n={limit} all checkpoints present", flush=True)
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
            print(f"CHECKPOINT_RENDERED position={idx + 1} id={cid}", flush=True)

        native_render_specs(
            self.model, self.tokenizer, "qwen", self.scenarios, 12,
            max_reply_tokens=self.args.max_reply_tokens, temp=0.0,
            seed_base=STRUCTURAL_SEED, conv_start=0, batched_render=False,
            native_batch=1, only_positions=set(missing),
            on_conv_rendered=on_rendered)

    def process_conversation(self, position: int) -> dict:
        cid = FROZEN_ORDER[position - 1]
        path = self.ckpath(position, cid)
        existing = self.load_rendered(position)
        if existing.get("stage") == "scored":
            validate_scored_checkpoint(existing)
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
                max_tokens=self.args.max_summary_tokens)
            replay = capture_forced_summary(
                self.model, self.tokenizer, correct_messages,
                generated.summary_ids, source_kind="correct_stepwise_replay")
            identity = validate_generated_replay(
                generated, replay, self.args.identity_tolerance)
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

            # Production position-movement diagnostic.  The ladder freezes the
            # tolerance passed on the command line before the paid run.
            correct_delta = layout.summary_start - generated.summary_start
            wrong_delta = layout.summary_start - wrong.summary_start
            moved = move_key_rows(generated.rows, correct_delta, self.rope_theta)
            roundtrip = move_key_rows(moved, -correct_delta, self.rope_theta)
            rotation_rows = compare_rows(generated.rows, roundtrip)
            rotation_max = max(x["k_max_abs"] for x in rotation_rows)
            if rotation_max > self.args.rotation_tolerance:
                raise CoherentStateError(
                    f"key rotation roundtrip {rotation_max} exceeds "
                    f"{self.args.rotation_tolerance}")
            del moved, roundtrip

            arm_scores = {"A_full": a_score}
            arm_diagnostics = {}
            outcomes = {"A_full": a_score["conversation_margin"]}
            for arm in ARM_NAMES[1:]:
                snap, diag = arm_snapshot(
                    arm, fresh_snapshot, generated.rows, wrong.rows,
                    layout.summary_start, correct_delta, wrong_delta,
                    self.rope_theta, PLACEBO_SEED + position)
                score = score_arm(
                    self.model, self.tokenizer, snap, layout.messages,
                    layout.context_ids, plants, self.targets)
                arm_scores[arm] = score
                arm_diagnostics[arm] = diag
                outcomes[arm] = score["conversation_margin"]
                del snap
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
                    "rotation_roundtrip_max_abs": rotation_max,
                    "rotation_tolerance": self.args.rotation_tolerance,
                    "exact_summary_span": True,
                },
                "runtime": {
                    **self.provenance,
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at": utc_now(),
                },
            }
            scored = promote_checkpoint(path, existing, additions, "scored")
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
            try:
                promote_checkpoint(path, existing, {
                    "gates": {"technical_pass": False,
                              "failures": [failure]},
                    "runtime": {**self.provenance,
                                "elapsed_seconds": time.monotonic() - started},
                }, "void")
            except Exception:
                atomic_write_json(path.with_suffix(".failure.json"), failure)
            raise

    def run_stage(self, limit: int) -> dict:
        self.ensure_renders(limit)
        for position in range(1, limit + 1):
            self.process_conversation(position)
        docs = load_checkpoints(self.run_dir)
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
    ap.add_argument("--max-reply-tokens", type=int, default=320)
    ap.add_argument("--max-summary-tokens", type=int, default=900)
    ap.add_argument("--identity-tolerance", type=float, default=1e-4)
    ap.add_argument("--rotation-tolerance", type=float, required=True,
                    help="frozen by the production ladder before launch")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"RUN coherent_state model={MODEL}@{REVISION} -> {args.run_dir}",
          flush=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "manifest.json"
    try:
        provenance = runtime_provenance()
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
            "max_reply_tokens": args.max_reply_tokens,
            "max_summary_tokens": args.max_summary_tokens,
            "identity_tolerance": args.identity_tolerance,
            "rotation_tolerance": args.rotation_tolerance,
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
        model, tokenizer, geometry = load_subject()
        print(f"MODEL_READY revision={REVISION} dtype=bf16 geometry={geometry}",
              flush=True)
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
            "n_conversations": stats["n_conversations"],
            "serial_decision": stats["serial_decision"],
            "interpretation": stats["interpretation"],
        })
        print(
            f"COHERENCE_RUN_DONE n={stats['n_conversations']} "
            f"decision={stats['serial_decision']}", flush=True)
        return 0
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
