import json
import hashlib

import pytest

from coherent_state_store import (
    ArtifactError,
    checkpoint_path,
    promote_checkpoint,
    read_checkpoint,
    save_render,
    validate_scored_checkpoint,
)


def backend_attestation():
    def config(scope):
        return {
            "scope": scope, "config_class": "FixtureConfig",
            "_attn_implementation": "eager",
            "_attn_implementation_internal": "eager",
            "resolved_implementation": "eager",
        }
    payload = {
        "requested_implementation": "eager",
        "model_config": config("model_config"),
        "text_config": config("text_config"),
        "text_config_is_model_config": False,
        "expected_layer_count": 48,
        "layers": [{
            "layer_index": i, "module_name": f"model.layers.{i}.self_attn",
            "module_class": "FixtureAttention",
            "module_config_class": "FixtureConfig",
            "module_config__attn_implementation": "eager",
            "module_config__attn_implementation_internal": "eager",
            "resolved_implementation": "eager",
        } for i in range(48)],
    }
    payload["sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def test_render_checkpoint_is_atomic_and_resume_stable(tmp_path):
    fp = {"model": "m", "revision": "r"}
    conv = {"id": "c10", "messages": [{"role": "system", "content": "x"}]}
    path = checkpoint_path(tmp_path, 1, "c10")
    first = save_render(path, fingerprint=fp, order_position=1,
                        conversation=conv, reply_records=[])
    assert read_checkpoint(path, fp) == first
    assert save_render(path, fingerprint=fp, order_position=1,
                       conversation=conv, reply_records=[]) == first
    assert not list(tmp_path.glob(".*.tmp"))


def test_resume_rejects_changed_render_or_fingerprint(tmp_path):
    fp = {"model": "m"}
    path = checkpoint_path(tmp_path, 1, "c10")
    conv = {"id": "c10", "messages": []}
    save_render(path, fingerprint=fp, order_position=1,
                conversation=conv, reply_records=[])
    with pytest.raises(ArtifactError, match="fingerprint"):
        read_checkpoint(path, {"model": "different"})
    with pytest.raises(ArtifactError, match="changed render"):
        save_render(path, fingerprint=fp, order_position=1,
                    conversation={"id": "c10", "messages": [1]},
                    reply_records=[])


def test_stage_promotion_is_additive_and_no_regression(tmp_path):
    fp = {"model": "m"}
    path = checkpoint_path(tmp_path, 1, "c10")
    rendered = save_render(
        path, fingerprint=fp, order_position=1,
        conversation={"id": "c10", "messages": []}, reply_records=[])
    captured = promote_checkpoint(path, rendered, {"summary": {"ids": [1]}},
                                  "captured")
    assert read_checkpoint(path, fp, "captured")["summary"]["ids"] == [1]
    with pytest.raises(ArtifactError, match="regression"):
        promote_checkpoint(path, captured, {}, "rendered")
    with pytest.raises(ArtifactError, match="overwrite"):
        promote_checkpoint(path, captured, {"summary": {"ids": [2]}},
                           "scored")


def test_scored_validation_rejects_packed_or_nonfinite_artifacts():
    arms = {
        "A_full": 1.0, "G_fresh": 0.0, "G_correct": 0.2,
        "G_wrong": 0.1, "G_Vcorrect": 0.1, "G_Kcorrect": 0.1,
    }
    doc = {
        "schema": 2,
        "amendment_id": "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7-8-9",
        "design_id": "coherent-state-gapped-v9",
        "fingerprint": {
            "attention_backend": "eager",
            "attention_backend_fingerprint": backend_attestation(),
        },
        "conversation": {}, "summary": {}, "sources": {},
        "destination": {
            "position_policy": "gapped_same_source_summary_position"},
        "arm_scores": {key: {} for key in arms},
        "conversation_outcomes": dict(arms),
        "calibration_outcomes": {
            "G_fresh": 0.0, "G_correct": 1.0, "G_wrong": 0.0},
        "gates": {"technical_pass": True}, "runtime": {},
        "pre_score_schedule_equivalence": {
            "status": "PASS", "passes": True,
            "semantic_scoring_performed": False},
    }
    validate_scored_checkpoint(doc)
    doc["conversation_outcomes"]["F_fresh"] = 0.0
    with pytest.raises(ArtifactError, match="unknown arms"):
        validate_scored_checkpoint(doc)
    del doc["conversation_outcomes"]["F_fresh"]
    doc["conversation_outcomes"]["G_correct"] = float("nan")
    with pytest.raises(ArtifactError, match="non-finite"):
        validate_scored_checkpoint(doc)


def test_scored_validation_rejects_wrong_backend():
    arms = {
        "A_full": 1.0, "G_fresh": 0.0, "G_correct": 0.2,
        "G_wrong": 0.1, "G_Vcorrect": 0.1, "G_Kcorrect": 0.1,
    }
    doc = {
        "schema": 2,
        "amendment_id": "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7-8-9",
        "design_id": "coherent-state-gapped-v9",
        "fingerprint": {
            "attention_backend": "sdpa",
            "attention_backend_fingerprint": backend_attestation(),
        },
        "conversation": {}, "summary": {}, "sources": {},
        "destination": {"position_policy": "gapped_same_source_summary_position"},
        "arm_scores": {key: {} for key in arms},
        "conversation_outcomes": arms,
        "calibration_outcomes": {
            "G_fresh": 0.0, "G_correct": 1.0, "G_wrong": 0.0},
        "gates": {"technical_pass": True}, "runtime": {},
        "pre_score_schedule_equivalence": {
            "status": "PASS", "passes": True,
            "semantic_scoring_performed": False},
    }
    with pytest.raises(ArtifactError, match="eager attention"):
        validate_scored_checkpoint(doc)
