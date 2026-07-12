from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import precision_probe_p01_loader as loader


TINY = loader.P01Topology(
    layers=2,
    experts_per_layer=3,
    hidden_size=4,
    moe_intermediate_size=2,
    attention_heads=2,
    kv_heads=1,
    head_dim=2,
    vocab_size=11,
    rope_theta=10_000.0,
)


class TinyNorm(torch.nn.Module):
    def __init__(self, width: int, *, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(
            width, dtype=torch.bfloat16, device=device))


class TinyAttention(torch.nn.Module):
    def __init__(self, *, device=None):
        super().__init__()
        q_width = TINY.attention_heads * TINY.head_dim
        kv_width = TINY.kv_heads * TINY.head_dim
        self.q_proj = torch.nn.Linear(
            TINY.hidden_size, q_width, bias=False,
            dtype=torch.bfloat16, device=device)
        self.k_proj = torch.nn.Linear(
            TINY.hidden_size, kv_width, bias=False,
            dtype=torch.bfloat16, device=device)
        self.v_proj = torch.nn.Linear(
            TINY.hidden_size, kv_width, bias=False,
            dtype=torch.bfloat16, device=device)
        self.o_proj = torch.nn.Linear(
            q_width, TINY.hidden_size, bias=False,
            dtype=torch.bfloat16, device=device)
        self.q_norm = TinyNorm(TINY.head_dim, device=device)
        self.k_norm = TinyNorm(TINY.head_dim, device=device)


class TinyExpert(torch.nn.Module):
    def __init__(self, *, device=None):
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            TINY.hidden_size, TINY.moe_intermediate_size, bias=False,
            dtype=torch.bfloat16, device=device)
        self.up_proj = torch.nn.Linear(
            TINY.hidden_size, TINY.moe_intermediate_size, bias=False,
            dtype=torch.bfloat16, device=device)
        self.down_proj = torch.nn.Linear(
            TINY.moe_intermediate_size, TINY.hidden_size, bias=False,
            dtype=torch.bfloat16, device=device)


class TinyMoe(torch.nn.Module):
    def __init__(self, *, device=None):
        super().__init__()
        self.gate = torch.nn.Linear(
            TINY.hidden_size, TINY.experts_per_layer, bias=False,
            dtype=torch.bfloat16, device=device)
        self.experts = torch.nn.ModuleList([
            TinyExpert(device=device) for _ in range(TINY.experts_per_layer)
        ])


class TinyLayer(torch.nn.Module):
    def __init__(self, *, device=None):
        super().__init__()
        self.self_attn = TinyAttention(device=device)
        self.mlp = TinyMoe(device=device)
        self.input_layernorm = TinyNorm(TINY.hidden_size, device=device)
        self.post_attention_layernorm = TinyNorm(
            TINY.hidden_size, device=device)


class TinyBody(torch.nn.Module):
    def __init__(self, *, device=None):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            TINY.vocab_size, TINY.hidden_size,
            dtype=torch.bfloat16, device=device)
        self.layers = torch.nn.ModuleList([
            TinyLayer(device=device) for _ in range(TINY.layers)
        ])
        self.norm = TinyNorm(TINY.hidden_size, device=device)


class TinyModel(torch.nn.Module):
    def __init__(self, *, device=None):
        super().__init__()
        self.model = TinyBody(device=device)
        self.lm_head = torch.nn.Linear(
            TINY.hidden_size, TINY.vocab_size, bias=False,
            dtype=torch.bfloat16, device=device)
        self.config = SimpleNamespace(quantization_config=None)

    def forward(self, input_ids=None, past_key_values=None, position_ids=None,
                cache_position=None, logits_to_keep=0, use_cache=None):
        raise AssertionError("static test model must not execute")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        capture_output=True, text=True).stdout.strip()


def _replace_child(root: torch.nn.Module, dotted_name: str,
                   value: torch.nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def test_exact_frozen_topology_arithmetic() -> None:
    topology = loader.EXACT_TOPOLOGY
    assert topology.expert_modules == 18_432
    assert topology.eligible_modules == 18_672
    assert topology.expert_elements == 28_991_029_248
    assert topology.eligible_elements == 29_909_581_824
    assert topology.parameter_elements == 30_532_122_624


def test_import_does_not_bind_bitsandbytes_or_transformers_module() -> None:
    assert "bitsandbytes" not in loader.__dict__
    assert "transformers" not in loader.__dict__


def test_g0_meta_key_and_linear_coverage_passes_on_tiny_model() -> None:
    model = TinyModel(device="meta")
    result = loader.attest_g0_meta_model(
        model, list(model.state_dict()), topology=TINY, require_meta=True)
    assert result["expert_linear_count"] == 2 * 3 * 3
    assert result["eligible_linear_count"] == TINY.eligible_modules
    assert result["eligible_logical_elements"] == TINY.eligible_elements
    assert result["logical_parameter_elements"] == TINY.parameter_elements


def test_g0_fails_on_checkpoint_key_or_expert_coverage_gap() -> None:
    model = TinyModel(device="meta")
    keys = list(model.state_dict())
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="key sets differ"):
        loader.attest_g0_meta_model(
            model, keys[:-1], topology=TINY, require_meta=True)
    model.model.layers[0].mlp.experts[0].gate_proj = torch.nn.Identity()
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="linear module coverage differs"):
        loader.attest_g0_meta_model(
            model, list(model.state_dict()), topology=TINY, require_meta=True)


def test_g0_synthetic_dynamic_cache_api_roundtrip() -> None:
    class FakeDynamicCache:
        def __init__(self, ddp_cache_data=None):
            self.rows = tuple(ddp_cache_data or ())

        def to_legacy_cache(self):
            return self.rows

    result = loader._attest_cache_api(
        TinyModel(device="meta"), topology=TINY,
        dynamic_cache_class=FakeDynamicCache)
    assert result == {
        "forward_parameters": [
            "cache_position", "logits_to_keep", "past_key_values",
            "position_ids", "use_cache",
        ],
        "dynamic_cache_ddp_constructor": True,
        "synthetic_layers": 2,
        "synthetic_shape": [1, 1, 2, 2],
        "synthetic_dtype": "torch.bfloat16",
        "snapshot_rebuild_equal": True,
    }


class FakeParams4bit(torch.nn.Parameter):
    @staticmethod
    def __new__(cls, data: torch.Tensor, state):
        result = torch.Tensor._make_subclass(cls, data, False)
        result.quant_state = state
        result.bnb_quantized = True
        result.quant_type = "nf4"
        result.compress_statistics = True
        return result


class FakeLinear4bit(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, device="cpu"):
        super().__init__(in_features, out_features, bias=False,
                         dtype=torch.bfloat16, device=device)
        state2 = SimpleNamespace(
            absmax=torch.ones(1, device=device),
            code=torch.ones(1, device=device),
        )
        dequantized = torch.zeros(
            out_features, in_features, dtype=torch.bfloat16, device=device)
        state = SimpleNamespace(
            quant_type="nf4", nested=True, state2=state2,
            dtype=torch.bfloat16,
            absmax=torch.ones(1, device=device),
            code=torch.ones(1, device=device),
            offset=torch.ones(1, device=device),
            dequantized=dequantized,
        )
        self.weight = FakeParams4bit(
            torch.zeros(out_features, in_features,
                        dtype=torch.uint8, device=device), state)
        self.compute_dtype = torch.bfloat16


class FakeFunctional:
    @staticmethod
    def dequantize_4bit(data, state):
        return state.dequantized


FAKE_BNB = SimpleNamespace(
    nn=SimpleNamespace(Linear4bit=FakeLinear4bit,
                       Params4bit=FakeParams4bit),
    functional=FakeFunctional,
)


def _tiny_nf4_model() -> TinyModel:
    model = TinyModel(device="cpu")
    model.config.quantization_config = SimpleNamespace(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
        llm_int8_skip_modules=["lm_head"],
        llm_int8_enable_fp32_cpu_offload=False,
    )
    model.is_loaded_in_4bit = True
    model.is_quantized = True
    model.quantization_method = "bitsandbytes"
    model.hf_quantizer = type("Bnb4BitHfQuantizer", (), {})()
    expected = loader._expected_module_names(TINY)["eligible"]
    for name in sorted(expected):
        original = dict(model.named_modules())[name]
        _replace_child(model, name, FakeLinear4bit(
            original.in_features, original.out_features))
    return model


def test_nf4_attestation_checks_all_experts_and_nonzero_sentinels(
        tmp_path: Path) -> None:
    model = _tiny_nf4_model()
    result = loader.attest_nf4_coverage(
        model, snapshot=tmp_path, weight_map={}, topology=TINY,
        expected_device="cpu", bnb_module=FAKE_BNB,
        source_loader=lambda name: torch.ones(
            loader._linear_shape(name.removesuffix(".weight"), TINY),
            dtype=torch.bfloat16))
    assert result["linear4bit_modules"] == TINY.eligible_modules
    assert result["expert_linear4bit_modules"] == TINY.expert_modules
    assert result["expert_logical_coverage"] == {
        "observed": TINY.expert_elements,
        "expected": TINY.expert_elements,
    }
    assert len(result["sentinels"]) == 9
    assert all(row["max_abs_error"] == 1.0 for row in result["sentinels"])


def test_nf4_attestation_rejects_one_ordinary_expert(tmp_path: Path) -> None:
    model = _tiny_nf4_model()
    model.model.layers[1].mlp.experts[2].down_proj = torch.nn.Linear(
        TINY.moe_intermediate_size, TINY.hidden_size, bias=False,
        dtype=torch.bfloat16)
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="Linear4bit module coverage differs"):
        loader.attest_nf4_coverage(
            model, snapshot=tmp_path, weight_map={}, topology=TINY,
            expected_device="cpu", bnb_module=FAKE_BNB,
            source_loader=lambda name: torch.ones(1))


def test_nf4_attestation_rejects_uninitialized_or_single_quant_state(
        tmp_path: Path) -> None:
    model = _tiny_nf4_model()
    bad = model.model.layers[0].mlp.experts[0].gate_proj.weight.quant_state
    bad.nested = False
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="initialization differs"):
        loader.attest_nf4_coverage(
            model, snapshot=tmp_path, weight_map={}, topology=TINY,
            expected_device="cpu", bnb_module=FAKE_BNB,
            source_loader=lambda name: torch.ones(1))


@pytest.mark.parametrize("field", ["offset", "state2_code", "compute_dtype"])
def test_nf4_attestation_binds_real_0492_nested_state_fields(
        tmp_path: Path, field: str) -> None:
    """The fake mirrors Params4bit/QuantState fields in bnb 0.49.2 source."""
    model = _tiny_nf4_model()
    module = model.model.layers[0].mlp.experts[0].gate_proj
    if field == "offset":
        module.weight.quant_state.offset = None
    elif field == "state2_code":
        module.weight.quant_state.state2.code = None
    else:
        module.compute_dtype = torch.float32
    source_by_name = {
        name + ".weight": torch.ones(
            loader._linear_shape(name, TINY), dtype=torch.bfloat16)
        for name in loader._expected_module_names(TINY)["experts"]
    }
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="initialization differs|state placement differs"):
        loader.attest_nf4_coverage(
            model, snapshot=tmp_path, weight_map={}, topology=TINY,
            expected_device="cpu", bnb_module=FAKE_BNB,
            source_loader=source_by_name.__getitem__)


def test_bf16_attestation_accepts_only_complete_unquantized_bf16() -> None:
    model = TinyModel(device="cpu")
    result = loader.attest_bf16_coverage(
        model, topology=TINY, expected_device="cpu")
    assert result["logical_parameter_elements"] == TINY.parameter_elements
    assert result["quantization_modules"] == []
    model.is_loaded_in_4bit = True
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="contains quantization"):
        loader.attest_bf16_coverage(
            model, topology=TINY, expected_device="cpu")


def test_device_attestation_returns_json_safe_hf_device_map() -> None:
    model = TinyModel(device="cpu")
    model.hf_device_map = {"": 0}
    result = loader._module_device_attestation(
        model, expected_device="cpu")
    assert result["hf_device_map"] == {"": 0}
    json.dumps(result, sort_keys=True)


def test_both_transformers_attention_config_fields_must_be_eager() -> None:
    config = SimpleNamespace(
        _attn_implementation="eager",
        _attn_implementation_internal="eager")
    assert loader._attention_backend_attestation(config) == {
        "_attn_implementation": "eager",
        "_attn_implementation_internal": "eager",
    }
    config._attn_implementation_internal = "sdpa"
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="attention backend differs"):
        loader._attention_backend_attestation(config)


def test_eos_fingerprint_uses_exact_authoritative_generation_set() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=151_645),
        generation_config=SimpleNamespace(eos_token_id=[151_645, 151_643]))
    tokenizer = SimpleNamespace(eos_token_id=151_645)
    result = loader._eos_attestation(model, tokenizer)
    assert result["eos_ids"] == [151_643, 151_645]
    model.generation_config.eos_token_id = [151_645]
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="generation EOS metadata differs"):
        loader._eos_attestation(model, tokenizer)


def test_kv_attestation_requires_every_bf16_layer_and_exact_geometry() -> None:
    rows = [
        (torch.ones(1, TINY.kv_heads, 5, TINY.head_dim,
                    dtype=torch.bfloat16),
         torch.zeros(1, TINY.kv_heads, 5, TINY.head_dim,
                     dtype=torch.bfloat16))
        for _ in range(TINY.layers)
    ]
    result = loader.attest_kv_cache(
        rows, token_count=5, topology=TINY, expected_device="cpu")
    assert result["layers"] == TINY.layers
    assert result["expected_shape"] == [1, 1, 5, 2]
    rows[1] = (rows[1][0].float(), rows[1][1])
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="layer 1"):
        loader.attest_kv_cache(
            rows, token_count=5, topology=TINY, expected_device="cpu")


def test_dependency_fingerprint_is_exact_and_does_not_import_bnb(
        monkeypatch: pytest.MonkeyPatch) -> None:
    versions = dict(loader.EXPECTED_DEPENDENCIES)
    versions["torch"] = versions["torch"] + "+cu130"
    monkeypatch.setattr(loader.importlib.metadata, "version",
                        lambda name: versions[name])
    monkeypatch.setattr(loader.platform, "python_version",
                        lambda: loader.EXPECTED_PYTHON)
    result = loader.dependency_fingerprint()
    assert result["normalized_packages"] == loader.EXPECTED_DEPENDENCIES
    versions["transformers"] = "5.0.0"
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="versions differ"):
        loader.dependency_fingerprint()


def test_repository_binding_limits_dirty_check_to_p01_inputs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "branch", "-m", "trunk")
    spec = tmp_path / "spec.md"
    code = tmp_path / "loader.py"
    unrelated = tmp_path / "other.txt"
    spec.write_text("frozen\n")
    code.write_text("x = 1\n")
    unrelated.write_text("initial\n")
    _git(tmp_path, "add", "spec.md", "loader.py", "other.txt")
    _git(tmp_path, "commit", "-m", "initial")
    monkeypatch.setattr(loader, "REPOSITORY_INPUTS", {
        "spec.md": hashlib.sha256(spec.read_bytes()).hexdigest(),
    })
    monkeypatch.setattr(loader, "LOADER_PATH", "loader.py")
    unrelated.write_text("dirty but outside p01 binding\n")
    result = loader.repository_binding(tmp_path)
    assert [row["path"] for row in result["files"]] == [
        "loader.py", "spec.md"]
    code.write_text("x = 2\n")
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="differs from HEAD"):
        loader.repository_binding(tmp_path)


def test_host_fingerprint_enforces_existing_a100_driver_cuda_gate(
        monkeypatch: pytest.MonkeyPatch) -> None:
    props = SimpleNamespace(
        name=loader.EXPECTED_GPU_NAME,
        total_memory=81_920 * 1024 * 1024,
    )
    monkeypatch.setattr(loader.platform, "system", lambda: "Linux")
    monkeypatch.setattr(loader.platform, "platform", lambda: "test-linux")
    monkeypatch.setattr(loader.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(loader.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(loader.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(loader.torch.cuda, "get_device_properties",
                        lambda index: props)
    monkeypatch.setattr(loader.torch.cuda, "get_device_capability",
                        lambda index: (8, 0))
    monkeypatch.setattr(loader.torch.version, "cuda", "13.0")
    monkeypatch.setattr(loader.torch.backends.cudnn, "version", lambda: 999)
    row = ("GPU-1, 580.159.03, NVIDIA A100 80GB PCIe, 81920\n")
    monkeypatch.setattr(loader.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=row, stderr=""))
    result = loader.host_fingerprint()
    assert result["gpu_uuid"] == "GPU-1"
    assert result["driver_version"] == "580.159.03"
    bad = row.replace("580.159.03", "550.90.12")
    monkeypatch.setattr(loader.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=bad, stderr=""))
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="driver is too old"):
        loader.host_fingerprint()


def test_tokenizer_attestation_binds_slow_class_template_and_ids(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identity = [3, 4, 5]

    class Qwen2Tokenizer:
        is_fast = False
        vocab_size = 2
        eos_token_id = 5
        pad_token_id = 4
        all_special_ids = [5, 4, 3]
        chat_template = "template"

        def __init__(self):
            self.name_or_path = str(tmp_path)

        def __len__(self):
            return 6

        def apply_chat_template(self, *args, **kwargs):
            return identity

        def convert_tokens_to_ids(self, token):
            return {"a": 3, "b": 4, "c": 5}[token]

        def get_vocab(self):
            return {"x": 0, "y": 1}

    expected = {
        "class": "Qwen2Tokenizer", "is_fast": False,
        "length": 6, "vocab_size": 2,
        "eos_token_id": 5, "pad_token_id": 4,
        "all_special_ids": [5, 4, 3],
        "special_token_ids": {"a": 3, "b": 4, "c": 5},
        "chat_template_sha256": hashlib.sha256(b"template").hexdigest(),
        "vocab_sha256": hashlib.sha256(json.dumps(
            {"x": 0, "y": 1}, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "identity_prefix_token_count": len(identity),
        "identity_prefix_ids_sha256": hashlib.sha256(b"".join(
            value.to_bytes(8, "little", signed=True)
            for value in identity)).hexdigest(),
    }
    monkeypatch.setattr(loader, "TOKENIZER_ATTESTATION", expected)
    observed, ids = loader._attest_tokenizer(
        Qwen2Tokenizer(), snapshot=tmp_path)
    assert observed == expected
    assert ids == identity


def test_loading_info_is_exact_and_empty() -> None:
    good = {key: [] for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    assert loader._loading_info_attestation(good) == good
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="missing_keys"):
        loader._loading_info_attestation({**good, "missing_keys": ["x"]})
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="fields differ"):
        loader._loading_info_attestation({
            key: value for key, value in good.items()
            if key != "error_msgs"})


def test_prepare_public_interface_and_non_v12_bindings(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = tmp_path / loader.REVISION
    snapshot.mkdir()
    model = object()
    tokenizer = object()
    checkpoint = {
        "checkpoint_keys": {"a"}, "weight_map": {"a": "shard"},
        "checkpoint_key_names_sha256": "key-hash",
        "model_id": loader.MODEL_ID, "revision": loader.REVISION,
    }
    monkeypatch.setattr(loader, "repository_binding", lambda repo: {
        "repository_commit": "a" * 40})
    monkeypatch.setattr(loader, "dependency_fingerprint", lambda: {
        "packages": dict(loader.EXPECTED_DEPENDENCIES)})
    monkeypatch.setattr(loader, "host_fingerprint", lambda: {
        "gpu_uuid": "GPU-1"})
    monkeypatch.setattr(loader, "resolve_pinned_snapshot",
                        lambda allow_download: snapshot)
    monkeypatch.setattr(loader, "checkpoint_attestation",
                        lambda path, require_weights: checkpoint)
    monkeypatch.setattr(loader, "g0_static_meta_compatibility", lambda path: {
        "topology": {"checkpoint_key_names_sha256": "key-hash"}})
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(loader, "_attest_tokenizer",
                        lambda value, snapshot: ({"slow": True}, [1, 2]))
    monkeypatch.setattr(loader, "_load_subject",
                        lambda regime, snapshot: (model, {
                            "missing_keys": [], "unexpected_keys": [],
                            "mismatched_keys": [], "error_msgs": [],
                        }, {"regime": regime}))
    monkeypatch.setattr(loader, "_model_config_attestation",
                        lambda value, snapshot: {"exact": True})
    monkeypatch.setattr(loader, "_eos_attestation",
                        lambda model, tokenizer: {
                            "eos_ids": [151_643, 151_645]})
    monkeypatch.setattr(loader, "_module_device_attestation",
                        lambda value, expected_device: {"device": expected_device})
    monkeypatch.setattr(loader, "attest_bf16_coverage",
                        lambda value: {"regime": "bf16"})
    monkeypatch.setattr(loader, "_real_forward_attestation",
                        lambda value, ids: {"kv": "bf16"})
    result = loader.prepare_precision_subject("bf16", tmp_path, False)
    assert result["model"] is model
    assert result["tokenizer"] is tokenizer
    assert result["runtime_fingerprint"]["regime"] == "bf16"
    assert result["runtime_fingerprint"]["eos_ids"] == [151_643, 151_645]
    assert result["bindings"]["formal_v12_decision_eligible"] is False
    assert result["bindings"]["v12_reentry_authorized"] is False
    assert result["bindings"][
        "component_reuse_does_not_inherit_v12_eligibility"] is True
    # The durable runner writes these mappings verbatim; serialization is part
    # of the loader contract rather than an implicit downstream assumption.
    json.dumps(result["bindings"], sort_keys=True)
    json.dumps(result["runtime_fingerprint"], sort_keys=True)
    with pytest.raises(loader.PrecisionProbeLoaderError,
                       match="unknown p01 regime"):
        loader.prepare_precision_subject("fp16", tmp_path, False)
