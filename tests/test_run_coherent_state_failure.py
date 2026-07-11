from __future__ import annotations

import json

from run_coherent_state_hf import terminalize_partial_checkpoints


def _write(path, stage):
    path.write_text(json.dumps({
        "schema": 2,
        "stage": stage,
        "status": "scored" if stage == "scored" else stage,
        "conversation": {"id": path.stem},
    }))


def test_failure_terminalizes_partial_checkpoints_without_losing_content(tmp_path):
    rendered = tmp_path / "conv_01_c10.json"
    captured = tmp_path / "conv_02_c02.json"
    scored = tmp_path / "conv_03_c01.json"
    _write(rendered, "rendered")
    _write(captured, "captured")
    _write(scored, "scored")

    failure = {"error_type": "Injected", "error": "case failed"}
    refs = terminalize_partial_checkpoints(tmp_path, failure)

    assert refs == [rendered.name, captured.name]
    for path in (rendered, captured):
        doc = json.loads(path.read_text())
        assert doc["stage"] == doc["status"] == "void"
        assert doc["conversation"] == {"id": path.stem}
        assert doc["failure"] == failure
        assert doc["gates"]["technical_pass"] is False
    assert json.loads(scored.read_text())["stage"] == "scored"
