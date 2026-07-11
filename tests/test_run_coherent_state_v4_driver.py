from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_coherent_state_hf as driver
from coherent_state_hf import CoherentStateError, sha256_ids


def test_durable_diagnostic_sink_persists_every_top_level_mutation():
    snapshots = []
    sink = driver.DurableDiagnosticSink(
        lambda value: snapshots.append(json.loads(json.dumps(value))))
    sink.update({"phase": "RUNNING"})
    sink["fixture"] = {"passes": True}
    sink["passes"] = True
    assert snapshots == [
        {"phase": "RUNNING"},
        {"phase": "RUNNING", "fixture": {"passes": True}},
        {"phase": "RUNNING", "fixture": {"passes": True}, "passes": True},
    ]


def test_open_gate_attempt_terminalizes_failure_atomically(tmp_path):
    attempt = tmp_path / "production_kernel_gate_unique.json"
    canonical = tmp_path / "production_kernel_gate.json"
    attempt.write_text(json.dumps({
        "schema": 2,
        "design_id": driver.DESIGN_ID,
        "amendment_id": driver.AMENDMENT_ID,
        "status": "RUNNING",
        "gates": {"attention_backend": {"status": "RUNNING"}},
    }))
    failure = {"error_type": "Injected", "error": "model load stopped"}
    assert driver.terminalize_running_gate_attempt(
        attempt, canonical, failure, geometry=None) is True
    unique = json.loads(attempt.read_text())
    assert json.loads(canonical.read_text()) == unique
    assert unique["status"] == "FAIL"
    assert unique["gates"]["passes"] is False
    assert unique["gates"]["failure"] == failure
    assert unique["error"] == failure
    assert driver.terminalize_running_gate_attempt(
        attempt, canonical, {"error": "overwrite"}) is False
    assert json.loads(attempt.read_text()) == unique


def test_technical_gate_rejects_semantic_score_fields_recursively():
    driver.assert_technical_gate_has_no_semantic_scores({
        "passes": True,
        "fixed_margin_diagnostic": {"selected_token_margin_shift": 0.0},
    })
    with pytest.raises(CoherentStateError, match="forbidden semantic score"):
        driver.assert_technical_gate_has_no_semantic_scores({
            "passes": True,
            "nested": {"technical_margins_not_semantic_outcomes": {}},
        })


def test_context_limit_covers_long_frozen_fixture_and_rejects_boundary():
    good = SimpleNamespace(config=SimpleNamespace(
        max_position_embeddings=driver.MAX_TECHNICAL_LOGICAL_POSITION + 1))
    assert driver.model_context_limit(good) == \
        driver.MAX_TECHNICAL_LOGICAL_POSITION + 1
    bad = SimpleNamespace(config=SimpleNamespace(
        max_position_embeddings=driver.MAX_TECHNICAL_LOGICAL_POSITION))
    with pytest.raises(CoherentStateError, match="does not cover"):
        driver.model_context_limit(bad)


def test_subject_config_requests_eager_before_model_load(monkeypatch):
    seen = {}
    config = SimpleNamespace(
        _commit_hash=driver.REVISION,
        to_dict=lambda: {"checkpoint": "exact"},
    )
    tokenizer = SimpleNamespace(
        chat_template="template",
        special_tokens_map={"eos_token": "</s>"},
        get_vocab=lambda: {"a": 1},
    )

    def config_load(*args, **kwargs):
        seen["config"] = (args, kwargs)
        return config

    def tokenizer_load(*args, **kwargs):
        seen["tokenizer"] = (args, kwargs)
        return tokenizer

    monkeypatch.setattr(driver.AutoConfig, "from_pretrained", config_load)
    monkeypatch.setattr(driver.AutoTokenizer, "from_pretrained", tokenizer_load)
    observed_config, observed_tokenizer, metadata = \
        driver.prepare_subject_metadata()
    assert observed_config is config and observed_tokenizer is tokenizer
    assert seen["config"][1]["revision"] == driver.REVISION
    assert seen["config"][1]["attn_implementation"] == "eager"
    assert seen["tokenizer"][1]["revision"] == driver.REVISION
    assert metadata["attention_backend_requested"] == "eager"


def test_subject_load_is_exact_bf16_cuda_eager(monkeypatch):
    cfg = SimpleNamespace(
        _commit_hash=driver.REVISION,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        rope_parameters={"rope_theta": 10_000_000},
    )
    param = SimpleNamespace(
        dtype=driver.torch.bfloat16,
        device=SimpleNamespace(type="cuda"),
        is_floating_point=lambda: True,
    )

    class Model:
        config = cfg

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([param])

    seen = {}

    def load(*args, **kwargs):
        seen.update(kwargs)
        return Model()

    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(driver.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(
        driver, "eager_backend_fingerprint", lambda _model: {"sha256": "a" * 64})
    model, tokenizer, geometry, backend = driver.load_subject(cfg, object())
    assert model is not None and tokenizer is not None
    assert geometry == driver.EXPECTED_GEOMETRY
    assert backend["sha256"] == "a" * 64
    assert seen["revision"] == driver.REVISION
    assert seen["config"] is cfg
    assert seen["dtype"] is driver.torch.bfloat16
    assert seen["attn_implementation"] == "eager"


def _generated_capture():
    return SimpleNamespace(
        source_kind="generated_incremental",
        prefix_ids=[1, 2],
        prefix_sha256=sha256_ids([1, 2]),
        summary_ids=[3, 4],
        summary_text="saved summary",
        summary_start=2,
        summary_end=4,
        row_hashes=[{"layer": 0, "k": "k", "v": "v"}],
        trace={"token_ids": [3, 4], "token_logprobs": [-1.0, -2.0]},
    )


def test_generated_summary_is_atomic_durable_and_immutable(tmp_path: Path):
    runner = driver.Runner.__new__(driver.Runner)
    path = tmp_path / "conv_01_c10.json"
    existing = {
        "schema": 2,
        "design_id": driver.DESIGN_ID,
        "amendment_id": driver.AMENDMENT_ID,
        "stage": "rendered",
        "status": "rendered",
        "fingerprint": {"test": True},
    }
    generated = _generated_capture()
    saved = runner._persist_generated_summary(path, existing, generated)
    on_disk = json.loads(path.read_text())
    assert on_disk == saved
    assert on_disk["stage"] == "captured"
    assert on_disk["summary"]["text"] == "saved summary"
    assert on_disk["summary"]["token_ids"] == [3, 4]
    assert on_disk["summary"]["generation_trace"] == generated.trace
    assert on_disk["sources"]["correct_actual"]["summary_row_hashes"] == \
        generated.row_hashes
    assert on_disk["capture_progress"]["generated_source_persisted"] is True
    changed = _generated_capture()
    changed.summary_text = "changed"
    with pytest.raises(driver.ArtifactError, match="overwrite field"):
        runner._persist_generated_summary(path, saved, changed)


def test_durable_summary_resumes_by_exact_reconstruction_not_generation(
        monkeypatch, tmp_path):
    runner = driver.Runner.__new__(driver.Runner)
    runner.tokenizer = SimpleNamespace(decode=lambda ids: "saved summary")
    path = tmp_path / "conv_01_c10.json"
    saved = runner._persist_generated_summary(path, {
        "schema": 2, "stage": "rendered", "status": "rendered",
    }, _generated_capture())
    calls = []

    def forced(_model, _tokenizer, messages, ids, *, source_kind):
        calls.append((messages, ids, source_kind))
        return _generated_capture()

    monkeypatch.setattr(driver, "capture_forced_summary", forced)
    runner.model = object()
    restored = runner._restore_generated_summary(
        saved, [{"role": "user", "content": "summarize"}])
    assert restored.summary_ids == [3, 4]
    assert calls == [
        ([{"role": "user", "content": "summarize"}], [3, 4],
         "generated_incremental_exact_reconstruction")]


def test_technical_only_flag_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        driver.sys, "argv",
        ["run_coherent_state_hf.py", "--run-dir", str(tmp_path),
         "--technical-only"])
    args = driver.parse_args()
    assert args.technical_only is True
