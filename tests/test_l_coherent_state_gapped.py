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
    _require_summary_boundary,
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


def test_v6_gate_schema_is_exhaustive_ordered_and_pending():
    schema = v5_gate_schema()
    assert schema["design_id"] == "coherent-state-gapped-v6"
    assert schema["max_technical_logical_position"] == 9509
    assert MAX_TECHNICAL_LOGICAL_POSITION == 9509
    assert schema["stage_order"] == list(V5_GATE_STAGE_ORDER)
    assert all(schema[name]["status"] == "PENDING"
               for name in V5_GATE_STAGE_ORDER)
    assert schema["committed_case_schedule_fixtures"]["expected_coverage"] == 12
    assert schema["external_donor_construction"]["expected_coverage"] == 12
    assert list(FROZEN_CASE_CONTINUATION_POSITIONS) == [
        "c10", "c02", "c01", "c04", "c07", "c11",
        "c05", "c09", "c06", "c12", "c08", "c03",
    ]
    assert max(FROZEN_CASE_CONTINUATION_POSITIONS.values()) == 9509


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
