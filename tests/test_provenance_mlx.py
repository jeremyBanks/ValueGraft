import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenance import build_manifest, capture_mlx_provenance  # noqa: E402


class _FakeModel:
    def parameters(self):
        return {}


def test_mlx_provenance_reads_literal_snapshot_config(tmp_path):
    config = {
        "model_type": "example_moe",
        "torch_dtype": "bfloat16",
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_hidden_layers": 4,
        "rope_theta": 12345,
        "quantization": {"bits": 4, "group_size": 64},
    }
    config_raw = json.dumps(config).encode()
    (tmp_path / "config.json").write_bytes(config_raw)
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "chat_template.jinja").write_text("template")

    observed = capture_mlx_provenance(_FakeModel(), str(tmp_path))

    assert observed["config_sha256"] == hashlib.sha256(config_raw).hexdigest()
    assert observed["chat_template_sha256"] == hashlib.sha256(
        b"template").hexdigest()
    assert observed["quantization"] == {"bits": 4, "group_size": 64}
    assert observed["architecture"] == "example_moe"
    assert observed["kv_geometry"] == {
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_hidden_layers": 4,
        "rope_theta": 12345,
    }

    manifest = build_manifest(
        model_provenance=observed,
        dtype_env=None,
        intervention={},
        metric={},
        condition={},
        corpus={},
        harness="test",
        repo=str(tmp_path),
    )
    assert manifest["load"]["dtype_matches_request"] is None
    assert manifest["model"]["config_sha256"] == observed["config_sha256"]
