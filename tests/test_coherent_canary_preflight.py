from types import SimpleNamespace
from pathlib import Path
import subprocess

import pytest
import torch

from coherent_canary_preflight import (
    CanaryPreflightError, repository_binding, runtime_fingerprint,
)


class FakeAttention(torch.nn.Module):
    def __init__(self, index, config):
        super().__init__()
        self.layer_idx = index
        self.config = config
        self.q_proj = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg = SimpleNamespace(
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8,
            rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
            eos_token_id=9, _attn_implementation="eager",
            _attn_implementation_internal="eager")
        cfg.text_config = cfg
        self.config = cfg
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.layers = torch.nn.ModuleList([FakeAttention(i, cfg) for i in range(2)])


def test_runtime_fingerprint_binds_geometry_dtype_backend_and_eos():
    result = runtime_fingerprint(
        FakeModel(), SimpleNamespace(eos_token_id=9),
        requested_model="fake", requested_revision="abc",
        resolved_snapshot="abc", expected_geometry={
            "layers": 2, "attention_heads": 4, "kv_heads": 2,
            "head_dim": 8, "rope_theta": 10000.0})
    assert result["dtype"] == "torch.bfloat16"
    assert result["eos_ids"] == [9]
    assert [row["layer"] for row in result["attention_layers"]] == [0, 1]


def test_generation_eos_set_may_include_primary_endoftext_member():
    model = FakeModel()
    model.generation_config.eos_token_id = [9, 7]
    result = runtime_fingerprint(
        model, SimpleNamespace(eos_token_id=9),
        requested_model="fake", requested_revision="abc",
        resolved_snapshot="abc", expected_geometry={
            "layers": 2, "attention_heads": 4, "kv_heads": 2,
            "head_dim": 8, "rope_theta": 10000.0})
    assert result["eos_ids"] == [7, 9]
    assert result["eos_sources"]["tokenizer_primary"] == [9]


def test_primary_eos_outside_generation_set_fails():
    model = FakeModel()
    model.generation_config.eos_token_id = [7]
    with pytest.raises(CanaryPreflightError, match="not contained"):
        runtime_fingerprint(
            model, SimpleNamespace(eos_token_id=9),
            requested_model="fake", requested_revision="abc",
            resolved_snapshot="abc", expected_geometry={
                "layers": 2, "attention_heads": 4, "kv_heads": 2,
                "head_dim": 8, "rope_theta": 10000.0})


def test_runtime_fingerprint_rejects_wrong_geometry():
    with pytest.raises(CanaryPreflightError, match="geometry"):
        runtime_fingerprint(
            FakeModel(), SimpleNamespace(eos_token_id=9),
            requested_model="fake", requested_revision="abc",
            resolved_snapshot="abc", expected_geometry={
                "layers": 3, "attention_heads": 4, "kv_heads": 2,
                "head_dim": 8, "rope_theta": 10000.0})


def test_repository_binding_requires_literal_head_bytes(tmp_path: Path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "x@y"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "X"], check=True)
    path = tmp_path / "a.py"
    path.write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "x"],
                   check=True, capture_output=True)
    result = repository_binding(tmp_path, [Path("a.py")])
    assert result["files"][0]["path"] == "a.py"
    path.write_text("x = 2\n")
    with pytest.raises(CanaryPreflightError, match="differs from HEAD"):
        repository_binding(tmp_path, [Path("a.py")])
