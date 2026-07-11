from types import SimpleNamespace
import inspect
import json

import pytest
import torch

import l_coherent_state_hf as ladder

from l_coherent_state_hf import (
    FROZEN_CASE_CONTINUATION_POSITIONS,
    LadderDurableDiagnosticSink,
    MAX_TECHNICAL_LOGICAL_POSITION,
    V5_GATE_STAGE_ORDER,
    _begin_stage,
    _require_summary_boundary,
    _scalar_metric_max,
    _validate_exact_length_wrong,
    _verify_intervention,
    v5_gate_schema,
    write_sharded_ladder_result,
)


def _snapshot(offset=0.0):
    key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2) + offset
    value = key + 100
    return [(key, value)]


def test_summary_boundary_rejects_precomputed_tail():
    layout = SimpleNamespace(physical_summary_end=4)
    _require_summary_boundary(_snapshot(), layout)
    tailed = [(torch.zeros(1, 1, 5, 2), torch.zeros(1, 1, 5, 2))]
    with pytest.raises(RuntimeError, match="fork exactly at summary boundary"):
        _require_summary_boundary(tailed, layout)


def test_exact_insert_preserves_every_non_summary_row():
    fresh = _snapshot()
    source = [(torch.full((1, 1, 2, 2), 9.0),
               torch.full((1, 1, 2, 2), 19.0))]
    treated = [(fresh[0][0].clone(), fresh[0][1].clone())]
    treated[0][0][..., 1:3, :] = source[0][0]
    treated[0][1][..., 1:3, :] = source[0][1]
    _verify_intervention(
        fresh, treated, source, 1, use_keys=True, use_values=True)

    treated[0][1][..., 3, :] += 1
    with pytest.raises(RuntimeError, match="non-summary rows changed"):
        _verify_intervention(
            fresh, treated, source, 1, use_keys=True, use_values=True)


def test_exact_length_wrong_rejects_structure_and_special_content():
    correct = [10, 11, 12, 13]
    wrong = [10, 21, 22, 13]
    _validate_exact_length_wrong(correct, wrong, [0, 3], [1, 2], [99])

    with pytest.raises(RuntimeError, match="altered structure"):
        _validate_exact_length_wrong(
            correct, [14, 21, 22, 13], [0, 3], [1, 2], [99])
    with pytest.raises(RuntimeError, match="special token"):
        _validate_exact_length_wrong(
            correct, [10, 99, 22, 13], [0, 3], [1, 2], [99])


def test_retired_diagnostic_keeps_wrong_sign_failure_injection():
    source = inspect.getsource(ladder._run_loaded_gapped_gates_v4_legacy)
    assert "wrong_sign_failure_injection_detected" in source
    assert "wrong_sign_shift_k_max_abs" in source


def test_v8_gate_schema_is_exhaustive_ordered_and_pending():
    schema = v5_gate_schema()
    assert schema["design_id"] == "coherent-state-gapped-v8"
    assert schema["max_technical_logical_position"] == 9509
    assert MAX_TECHNICAL_LOGICAL_POSITION == 9509
    assert schema["stage_order"] == list(V5_GATE_STAGE_ORDER)
    assert schema["stage_order"][0] == "static_provenance"
    assert all(schema[name]["status"] == "PENDING"
               for name in V5_GATE_STAGE_ORDER)
    assert schema["committed_case_schedule_fixtures"]["expected_coverage"] == 12
    assert schema["external_donor_construction"]["expected_coverage"] == 12
    assert list(FROZEN_CASE_CONTINUATION_POSITIONS) == [
        "c10", "c02", "c01", "c04", "c07", "c11",
        "c05", "c09", "c06", "c12", "c08", "c03",
    ]
    assert max(FROZEN_CASE_CONTINUATION_POSITIONS.values()) == 9509


def test_backend_stage_resumes_durable_model_load_progress_only():
    schema = v5_gate_schema(expected_attention_layers=2)
    schema["attention_backend"].update({
        "status": "RUNNING", "started_at": "already-durable",
        "observed_coverage": 1})
    resumed = _begin_stage(
        schema, "attention_backend", resume_running=True)
    assert resumed["status"] == "RUNNING"
    assert resumed["started_at"] == "already-durable"
    schema["synthetic_schedule_fixtures"]["status"] = "RUNNING"
    with pytest.raises(RuntimeError, match="did not begin from PENDING"):
        _begin_stage(schema, "synthetic_schedule_fixtures")


def test_cuda_oom_paths_are_explicitly_rethrown_before_later_model_work():
    synthetic = inspect.getsource(ladder.run_frozen_schedule_fixtures)
    committed = inspect.getsource(ladder.run_committed_case_schedule_fixtures)
    for source in (synthetic, committed):
        assert "_unsafe_model_exception(exc)" in source
        assert "raise unsafe_exc" in source or "raise\n" in source


def test_v5_gate_never_clears_caller_sink_and_persists_raw_replay_first():
    source = inspect.getsource(ladder.run_loaded_gapped_gates)
    assert "sink.clear" not in source
    assert "persist before verdict validation" in source
    assert "SKIPPED_DEPENDENCY" in inspect.getsource(ladder._skip_stage)


def test_ladder_writer_externalizes_gate_stages_and_preserves_failure(tmp_path):
    gate = v5_gate_schema(expected_attention_layers=2)
    gate.update({"status": "FAIL", "passes": False,
                 "failures": ["generated_replay_identity"]})
    result = {
        "schema": 2,
        "amendment_id": gate["amendment_id"],
        "design_id": gate["design_id"],
        "status": "FAIL",
        "diagnostics": {"loaded_gapped_production_gate": gate},
    }
    output = tmp_path / "coherent_ladder_unique.json"
    manifest = write_sharded_ladder_result(output, result)
    on_disk = json.loads(output.read_text())
    external = on_disk["diagnostics"]["loaded_gapped_production_gate"]
    assert external["externalized"] is True
    assert external["status"] == "FAIL"
    assert set(external["stage_refs"]) == set(ladder.V5_GATE_STAGE_ORDER)
    for ref in external["stage_refs"].values():
        path = tmp_path / ref["path"]
        assert path.stat().st_size == ref["byte_count"] < 4_000_000
    assert manifest["payload_sha256"] == on_disk["payload_sha256"]
    with pytest.raises(RuntimeError, match="overwrite"):
        write_sharded_ladder_result(output, result)


def test_ladder_durable_sink_persists_running_stage_then_terminalizes(tmp_path):
    output = tmp_path / "coherent_ladder_durable.json"
    sink = LadderDurableDiagnosticSink(output)
    schema = v5_gate_schema(expected_attention_layers=2)
    for key, value in schema.items():
        sink[key] = value
    attention_path = tmp_path / (
        "coherent_ladder_durable__stage_attention_backend.json")
    pending = json.loads(attention_path.read_text())
    assert pending["stage"]["status"] == "PENDING"
    sink["attention_backend"] = {
        **sink["attention_backend"], "status": "PASS", "passes": True}
    terminal = json.loads(attention_path.read_text())
    assert terminal["stage"]["status"] == "PASS"
    gate = dict(sink)
    gate.update({"status": "FAIL", "passes": False,
                 "failures": ["generated_replay_identity"]})
    result = {
        "schema": 2, "amendment_id": schema["amendment_id"],
        "design_id": schema["design_id"], "status": "FAIL",
        "diagnostics": {"loaded_gapped_production_gate": gate},
    }
    manifest = write_sharded_ladder_result(output, result)
    assert manifest["status"] == "FAIL"
    with pytest.raises(RuntimeError, match="resume/overwrite"):
        LadderDurableDiagnosticSink(output)


def test_model_identity_aggregates_ignore_per_layer_raw_payloads():
    rebuild_raw = {
        "logits_max_abs": 0.01,
        "k_max_abs": 0.02,
        "v_max_abs": 0.03,
        "per_layer": [{"layer": 0, "k_max_abs": 0.02,
                       "v_max_abs": 0.03}],
    }
    assert _scalar_metric_max(
        rebuild_raw,
        ("logits_max_abs", "k_max_abs", "v_max_abs")) == 0.03
    future_raw = {
        "earlier_logits_max_abs": 0.04,
        "earlier_cache_max_abs": 0.05,
        "per_layer": [{"layer": 0, "k_max_abs": 0.01,
                       "v_max_abs": 0.05}],
    }
    assert _scalar_metric_max(
        future_raw,
        ("earlier_logits_max_abs", "earlier_cache_max_abs")) == 0.05
    source = inspect.getsource(ladder.run_loaded_gapped_gates)
    assert "max(raw.values())" not in source


def test_model_identity_aggregate_rejects_missing_or_nonfinite_scalar():
    with pytest.raises(RuntimeError, match="missing or non-finite"):
        _scalar_metric_max(
            {"logits_max_abs": 0.0, "per_layer": []},
            ("logits_max_abs", "k_max_abs"))
    with pytest.raises(RuntimeError, match="missing or non-finite"):
        _scalar_metric_max(
            {"earlier_logits_max_abs": float("nan")},
            ("earlier_logits_max_abs",))


def test_exact_render_schedule_gate_uses_actual_prefix_before_semantic_scoring(
        monkeypatch):
    monkeypatch.setattr(ladder, "_case_schedule_layout", lambda *_: {
        "correct_prefix_ids": [1, 2, 3, 4],
        "fresh_prefix_ids": [1, 4],
        "ordinary_resolved_call_widths": [4],
        "message_block_resolved_call_widths": [1, 2, 1],
        "system_equal": True, "request_header_equal": True,
        "blocks_nonempty": True, "blocks_ordered_nonoverlapping": True,
        "blocks_cover_prefix": True,
    })
    monkeypatch.setattr(ladder, "_compare_schedules", lambda *_args, **_kwargs: {
        "status": "PASS", "passes": True, "tolerance": 5e-4,
        "base_measurement_complete": True,
        "continuation_measurement_complete": True,
        "per_layer": [], "continuation_per_layer": [],
        "cache_k_max_abs": 0.0, "cache_v_max_abs": 0.0,
        "last_logits_max_abs": 0.0, "selected_margin_abs_shift": 0.0,
        "continuation_logits_max_abs": 0.0,
        "continuation_k_max_abs": 0.0, "continuation_v_max_abs": 0.0,
        "observed_aggregate": 0.0,
    })
    observed = ladder.run_exact_render_schedule_fixture(
        object(), object(), {"id": "c10"})
    assert observed["status"] == "PASS"
    assert observed["semantic_scoring_performed"] is False
    assert observed["complete_prefix_token_ids"] == [1, 2, 3, 4]


def test_exact_render_schedule_persists_exception_before_propagating(monkeypatch):
    monkeypatch.setattr(ladder, "_case_schedule_layout", lambda *_: {
        "correct_prefix_ids": [1, 2], "fresh_prefix_ids": [1],
        "ordinary_resolved_call_widths": [2],
        "message_block_resolved_call_widths": [1, 1],
        "system_equal": True, "request_header_equal": True,
        "blocks_nonempty": True, "blocks_ordered_nonoverlapping": True,
        "blocks_cover_prefix": True,
    })
    monkeypatch.setattr(
        ladder, "_compare_schedules",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            torch.OutOfMemoryError("injected")))
    observed = []
    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        ladder.run_exact_render_schedule_fixture(
            object(), object(), {"id": "c10"}, progress=observed.append)
    assert observed[-1]["status"] == "ERROR"
    assert observed[-1]["failure_evidence"]["error_type"] == "OutOfMemoryError"
