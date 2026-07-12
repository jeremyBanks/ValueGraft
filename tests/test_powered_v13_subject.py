from __future__ import annotations

import ast
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import powered_v13_subject as subject


REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/"
    "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots" /
    subject.MODEL_REVISION
)
EMPTY_LOADING_INFO = {
    "missing_keys": [], "unexpected_keys": [],
    "mismatched_keys": [], "error_msgs": [],
}
CONVERSION_RECIPE = [
    {
        "source_patterns": [
            "mlp.experts.*.gate_proj.weight",
            "mlp.experts.*.up_proj.weight",
        ],
        "target_patterns": ["mlp.experts.gate_up_proj"],
        "operations": [
            {"class": "MergeModulelist", "dim": 0},
            {"class": "Concatenate", "dim": 1},
        ],
    },
    {
        "source_patterns": ["mlp.experts.*.down_proj.weight"],
        "target_patterns": ["mlp.experts.down_proj"],
        "operations": [{"class": "MergeModulelist", "dim": 0}],
    },
]


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()


class FakeDevice:
    def __init__(self, value: str):
        self.value = value
        self.type = value.split(":", 1)[0]

    def __str__(self) -> str:
        return self.value


class FakeParameter:
    def __init__(self, shape: list[int], dtype: Any, *, device: str = "cuda:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = FakeDevice(device)
        self.requires_grad = True

    def is_floating_point(self) -> bool:
        return True


class Qwen3MoeAttention:
    def __init__(self, layer: int, config: Any):
        self.layer_idx = layer
        self.config = config
        self.q_proj = object()
        self.k_proj = object()
        self.training = True


class _FakeModelBase:
    def __init__(self, snapshot: Path, torch: Any):
        config = SimpleNamespace(
            architectures=["Qwen3MoeForCausalLM"],
            model_type="qwen3_moe",
            num_hidden_layers=48,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            rope_parameters={"rope_theta": 10_000_000.0},
            hidden_size=2048,
            moe_intermediate_size=768,
            num_experts=128,
            decoder_sparse_step=1,
            tie_word_embeddings=False,
            _attn_implementation="eager",
            _attn_implementation_internal="eager",
            _commit_hash=subject.MODEL_REVISION,
            _name_or_path=str(snapshot),
            quantization_config=None,
            eos_token_id=151645,
        )
        config.text_config = config
        self.config = config
        self.name_or_path = str(snapshot)
        self.generation_config = SimpleNamespace(eos_token_id=[151645, 151643])
        self.training = True
        self.hf_device_map = {"": "cuda:0"}
        embedding_count = 151_669 * 2_048
        self._parameters = [
            ("model.embed_tokens.weight",
             FakeParameter([151_669, 2_048], torch.bfloat16)),
            ("model.synthetic_remainder",
             FakeParameter(
                 [subject._SUBJECT["parameter_count"] - embedding_count],
                 torch.bfloat16,
             )),
        ]
        self._attention = [Qwen3MoeAttention(index, config) for index in range(48)]

    def eval(self):
        self.training = False
        for module in self._attention:
            module.training = False
        return self

    def requires_grad_(self, value: bool):
        for _, parameter in self._parameters:
            parameter.requires_grad = value
        return self

    def named_parameters(self, *, remove_duplicate: bool = True):
        del remove_duplicate
        return list(self._parameters)

    def parameters(self):
        return [parameter for _, parameter in self._parameters]

    def buffers(self):
        return []

    def named_modules(self):
        return [("", self), *[
            (f"model.layers.{index}.self_attn", module)
            for index, module in enumerate(self._attention)
        ]]

    def modules(self):
        return [self, *self._attention]

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self._parameters[0][1])


Qwen3MoeForCausalLM = type(
    "Qwen3MoeForCausalLM", (_FakeModelBase,), {}
)


class FakeCuda:
    def __init__(self):
        self.available = True
        self.count = 1
        self.current = 0
        self.name = "NVIDIA A100 80GB PCIe"
        self.memory = 85_093_777_408
        self.major = 8
        self.minor = 0
        self.empty_cache_calls = 0

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def current_device(self):
        return self.current

    def get_device_name(self, _index):
        return self.name

    def get_device_properties(self, _index):
        return SimpleNamespace(
            total_memory=self.memory, major=self.major, minor=self.minor,
        )

    def empty_cache(self):
        self.empty_cache_calls += 1


class FakeTorch:
    __version__ = "2.12.1+cu130"

    def __init__(self):
        self.bfloat16 = object()
        self.version = SimpleNamespace(cuda="13.0")
        self.cuda = FakeCuda()
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 92000),
        )


class FakeFactual:
    __file__ = str(REPO / "src/coherent_canary_loader.py")
    EXACT_SPEC = object()

    def __init__(self, context):
        self.context = context
        self.model_inventory_calls = 0

    def resolve_pinned_snapshot(self, model_id, revision, *, local_files_only):
        self.context.events.append("resolve")
        self.context.resolve_args = (model_id, revision, local_files_only)
        return self.context.resolved_snapshot

    def validate_snapshot_location(self, snapshot, revision, *, repo_id):
        self.context.events.append("validate_snapshot")
        self.context.validate_args = (snapshot, revision, repo_id)

    def inventory_snapshot(self, snapshot, *, revision, kind, **_kwargs):
        self.context.events.append(f"inventory:{kind}")
        assert snapshot == self.context.resolved_snapshot
        assert revision == subject.MODEL_REVISION
        if kind == "model":
            self.model_inventory_calls += 1
            value = deepcopy(self.context.model_inventory)
            if self.context.corrupt_post_inventory and self.model_inventory_calls == 2:
                value["inventory_sha256"] = "f" * 64
            return value
        return deepcopy(self.context.tokenizer_inventory)

    def expected_loaded_parameter_topology(self, _rows, *, spec):
        assert spec is self.EXACT_SPEC
        rows = [{
            "name": name, "shape": list(parameter.shape), "dtype": "BF16",
        } for name, parameter in self.context.model.named_parameters(
            remove_duplicate=False
        )]
        if self.context.wrong_topology:
            rows[0]["shape"][0] -= 1
        return sorted(rows, key=lambda row: row["name"]), {
            "schema": "coherent_canary_loaded_parameter_topology_v1",
            "representation": "transformers-5-qwen2-style-moe-packed",
        }

    def attest_weight_conversion_recipe(self, _model, *, spec):
        assert spec is self.EXACT_SPEC
        return {
            "representation": "transformers-5-qwen2-style-moe-packed",
            "converters": deepcopy(CONVERSION_RECIPE),
        }

    def attest_moe_content_sentinels(
        self, _model, *, spec, model_snapshot, weight_tensors,
    ):
        assert spec is self.EXACT_SPEC
        assert model_snapshot == self.context.resolved_snapshot
        assert weight_tensors == self.context.model_inventory["weight_tensors"]
        if self.context.corrupt_weight_content:
            raise RuntimeError("packed expert content differs")
        rows = []
        for layer, expert in ((0, 0), (24, 64), (47, 127)):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                rows.append({
                    "layer": layer, "expert": expert,
                    "projection": projection, "dtype": "BF16",
                    "sha256": hashlib.sha256(
                        f"{layer}:{expert}:{projection}".encode()
                    ).hexdigest(),
                })
        return {
            "schema": "coherent_canary_weight_content_sentinels_v1",
            "representation": "transformers-5-qwen2-style-moe-packed",
            "sentinels": rows,
        }


class FakeAutoModel:
    context = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.context.events.append("load:model")
        cls.context.model_load = (path, kwargs)
        if cls.context.load_error:
            raise RuntimeError("synthetic load failure")
        return cls.context.model, deepcopy(cls.context.loading_info)


class FakeAutoTokenizer:
    context = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.context.events.append("load:tokenizer")
        cls.context.tokenizer_load = (path, kwargs)
        return cls.context.tokenizer


def release_evidence() -> dict[str, Any]:
    return {
        "status": "PASS",
        "design_id": subject.DESIGN_ID,
        "stage": "TECHNICAL_CANARY",
        "detached_head": "a" * 40,
        "clean_tree": True,
        "authorization": {
            "authorization_commit": "a" * 40,
            "static_root_commit": "b" * 40,
            "manifest_path": "release/stage-t.json",
            "manifest_sha256": "c" * 64,
            "inventory_sha256": "d" * 64,
        },
        "receipt": {"receipt_sha256": "e" * 64},
        "subject_inventory_paths": sorted(subject._REQUIRED_STAGE_T_PATHS),
    }


def dependencies() -> dict[str, Any]:
    return deepcopy(subject._DEPENDENCIES)


def fake_contract_bundle(context) -> subject._ContractBundle:
    model_files: list[dict[str, Any]] = []
    name_to_shard = [{"name": row["name"], "shard": row["shard"]}
                     for row in context.model_inventory["weight_tensors"]]
    map_hash = hashlib.sha256(canonical(sorted(
        name_to_shard, key=lambda row: row["name"]
    ))).hexdigest()
    exact_entry = {
        "model_id": subject.MODEL_ID,
        "revision": subject.MODEL_REVISION,
        "files": model_files,
        "inventory_sha256": "1" * 64,
        "parameter_count": subject._SUBJECT["parameter_count"],
        "weight_tensor_count": 2,
        "weight_name_to_shard_sha256": map_hash,
    }
    legacy = {
        "subjects": {"exact-subject": exact_entry},
        "protocol_tokenizer": {
            "model_id": subject.MODEL_ID,
            "revision": subject.MODEL_REVISION,
            "files": deepcopy(context.tokenizer_inventory["files"]),
            "inventory_sha256": context.tokenizer_inventory["inventory_sha256"],
        },
    }
    contract = {
        "dependencies": deepcopy(subject._DEPENDENCIES),
        "cuda": deepcopy(subject._CUDA_CONTRACT),
        "subject": {
            **deepcopy(subject._SUBJECT),
            "weight_tensor_count": 2,
            "weight_name_to_shard_sha256": map_hash,
        },
    }
    return subject._ContractBundle(
        subject_contract=contract, legacy_contract=legacy,
        subject_contract_sha256="2" * 64,
        legacy_contract_sha256="3" * 64,
    )


def make_context(tokenizer):
    torch = FakeTorch()
    context = SimpleNamespace(
        events=[], torch=torch, resolved_snapshot=SNAPSHOT,
        release=release_evidence(), dependencies=dependencies(),
        corrupt_post_inventory=False, corrupt_weight_content=False,
        wrong_topology=False, load_error=False,
        loading_info=deepcopy(EMPTY_LOADING_INFO), tokenizer=tokenizer,
        nvidia_output=(
            "GPU-12345678-1234-1234-1234-123456789abc, 580.159.03, "
            "NVIDIA A100 80GB PCIe, 81920\n"
        ),
    )
    context.model = Qwen3MoeForCausalLM(SNAPSHOT, torch)
    context.model_inventory = {
        "kind": "model", "repo_id": subject.MODEL_ID,
        "revision": subject.MODEL_REVISION, "snapshot_path": str(SNAPSHOT),
        "files": [], "inventory_sha256": "1" * 64,
        "weight_tensors": [{
            "name": name, "shard": "model-00001-of-00016.safetensors",
            "dtype": "BF16", "shape": list(parameter.shape),
        } for name, parameter in context.model.named_parameters(
            remove_duplicate=False
        )],
    }
    legacy = json.loads((REPO / subject.LEGACY_MODEL_BITS_PATH).read_bytes())
    token_entry = legacy["protocol_tokenizer"]
    context.tokenizer_inventory = {
        "kind": "protocol_tokenizer", "repo_id": subject.MODEL_ID,
        "revision": subject.MODEL_REVISION, "snapshot_path": str(SNAPSHOT),
        "files": deepcopy(token_entry["files"]),
        "inventory_sha256": token_entry["inventory_sha256"],
    }
    context.contract_bundle = fake_contract_bundle(context)
    factual = FakeFactual(context)
    def runtime_fingerprint(*_args, **_kwargs):
        return {
            "requested_model": subject.MODEL_ID,
            "requested_revision": subject.MODEL_REVISION,
            "resolved_snapshot": subject.MODEL_REVISION,
            "dtype": "torch.bfloat16", "attention_backend": "eager",
            "geometry": deepcopy(subject._SUBJECT["geometry"]),
            "eos_ids": [151643, 151645],
            "torch_version": subject._CUDA_CONTRACT["torch_version"],
            "transformers_version": "5.0.0",
        }

    # Module origin is independently checked against the inventoried preflight
    # file.  The callable remains fake; only this private seam uses it.
    import coherent_canary_preflight  # noqa: F401
    runtime_fingerprint.__module__ = "coherent_canary_preflight"
    context.runtime = subject._Runtime(
        torch=torch, auto_model=FakeAutoModel,
        auto_tokenizer=FakeAutoTokenizer, factual=factual,
        runtime_fingerprint=runtime_fingerprint,
    )
    FakeAutoModel.context = context
    FakeAutoTokenizer.context = context

    def verify_release(_repo, _receipt):
        context.events.append("release")
        return deepcopy(context.release)

    def probe_dependencies(_repo):
        context.events.append("dependencies")
        return deepcopy(context.dependencies)

    def load_runtime():
        context.events.append("runtime")
        return context.runtime

    def nvidia_smi():
        context.events.append("nvidia-smi")
        return context.nvidia_output

    def load_contracts(_repo):
        context.events.append("contracts")
        return context.contract_bundle

    def recheck_release(_repo, release, bundle):
        context.events.append("recheck-release")
        assert release["authorization"]["authorization_commit"] == "a" * 40
        assert bundle is context.contract_bundle
        return {
            "status": "PASS", "detached_head": "a" * 40,
            "clean_tree": True,
            "subject_contract_sha256": bundle.subject_contract_sha256,
            "legacy_model_bits_sha256": bundle.legacy_contract_sha256,
        }

    context.seams = subject._SubjectSeams(
        verify_release=verify_release,
        probe_dependencies=probe_dependencies,
        load_runtime=load_runtime,
        nvidia_smi=nvidia_smi,
        load_contracts=load_contracts,
        recheck_release=recheck_release,
    )
    return context


@pytest.fixture(scope="session")
def exact_tokenizer():
    if not SNAPSHOT.is_dir():
        pytest.skip("exact pinned tokenizer snapshot is absent")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        str(SNAPSHOT), local_files_only=True, trust_remote_code=False,
    )


@pytest.fixture(autouse=True)
def no_leaked_handle():
    subject._ACTIVE_TOKEN = None
    yield
    subject._ACTIVE_TOKEN = None


def open_fake(context):
    handle = subject._open_exact_subject(
        REPO, REPO / "receipts", seams=context.seams,
    )
    # The scaffold must not itself retain an alias that production does not.
    context.model = None
    return handle


def test_contract_is_canonical_and_hash_binds_factual_model_bits():
    bundle = subject._load_contracts(REPO)
    raw = (REPO / subject.SUBJECT_CONTRACT_PATH).read_bytes()
    assert raw == canonical(json.loads(raw)) + b"\n"
    assert bundle.subject_contract["subject"]["revision"] == subject.MODEL_REVISION
    assert bundle.subject_contract["load_policy"] == subject._LOAD_POLICY
    assert bundle.legacy_contract_sha256 == subject._LEGACY_BINDING["file_sha256"]


def test_public_api_has_no_policy_hash_clock_spec_or_callback_parameters():
    assert list(inspect.signature(subject.open_exact_subject).parameters) == [
        "repo", "receipt_directory",
    ]
    assert subject.__all__ == [
        "ExactSubjectHandle", "MODEL_ID", "MODEL_REVISION",
        "PoweredV13SubjectError", "open_exact_subject",
    ]


def test_production_import_graph_has_no_pool_recipe_or_permutation_import():
    tree = ast.parse(inspect.getsource(subject))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(
        name.startswith(stem)
        for stem in subject._FORBIDDEN_IMPORT_STEMS
        for name in imported
    )
    top_level = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        (isinstance(node, ast.Import) and
         any(alias.name in {"torch", "transformers", "coherent_canary_loader"}
             for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and
            node.module in {"torch", "transformers", "coherent_canary_loader"})
        for node in top_level
    )


def test_success_order_exact_local_load_arguments_and_runtime_binding(exact_tokenizer):
    context = make_context(exact_tokenizer)
    with open_fake(context) as handle:
        assert type(handle.model).__name__ == "Qwen3MoeForCausalLM"
        fingerprint = handle.runtime_fingerprint
        assert fingerprint["gate_trace"] == [
            "stage_t_detached_clean_receipt",
            "exact_locked_dependencies",
            "one_real_cuda_gpu",
            "one_live_handle_reserved",
            "exact_revision_resolved",
            "pre_load_bf16_snapshot_and_tokenizer_inventory",
            "resolved_local_snapshot_loaded",
            "post_load_snapshot_reinventory",
            "post_load_stage_t_binding",
            "full_post_load_attestation",
        ]
        assert fingerprint["release_binding"]["authorization_commit"] == "a" * 40
        assert fingerprint["parameters"]["parameter_count"] == 30_532_122_624
        assert fingerprint["tokenizer"]["backend_serialization_sha256"] == \
            subject._TOKENIZER["backend_serialization_sha256"]
    assert context.resolve_args == (
        subject.MODEL_ID, subject.MODEL_REVISION, False,
    )
    assert context.tokenizer_load == (str(SNAPSHOT), {
        "local_files_only": True, "trust_remote_code": False,
    })
    path, kwargs = context.model_load
    assert path == str(SNAPSHOT)
    assert kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": context.torch.bfloat16,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
        "device_map": "cuda:0",
        "output_loading_info": True,
    }
    assert context.events.index("release") < context.events.index("dependencies")
    assert context.events.index("dependencies") < context.events.index("runtime")
    assert context.events.index("nvidia-smi") < context.events.index("resolve")
    assert context.events.index("resolve") < context.events.index("load:model")
    assert context.torch.cuda.empty_cache_calls == 1


def test_wrong_release_stops_before_dependencies_cuda_resolve_or_load(exact_tokenizer):
    context = make_context(exact_tokenizer)
    context.release["status"] = "FAIL"
    with pytest.raises(subject.PoweredV13SubjectError, match="Stage-T checkout"):
        open_fake(context)
    assert context.events == ["release"]


def test_wrong_dependency_stops_before_torch_cuda_resolve_or_load(exact_tokenizer):
    context = make_context(exact_tokenizer)
    context.dependencies["distributions"]["transformers"] = "5.0.1"
    with pytest.raises(subject.PoweredV13SubjectError, match="dependency"):
        open_fake(context)
    assert context.events == ["release", "dependencies"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda c: setattr(c.torch.cuda, "available", False), "unavailable"),
        (lambda c: setattr(c.torch.cuda, "count", 2), "exactly CUDA device 0"),
        (lambda c: setattr(c.torch.cuda, "name", "NVIDIA H100 80GB HBM3"),
         "names differ"),
        (lambda c: setattr(c.torch.cuda, "memory", 1), "memory differs"),
        (lambda c: setattr(c.torch, "__version__", "2.12.1"), "build differs"),
        (lambda c: setattr(c.torch.version, "cuda", "12.8"), "runtime version"),
    ],
)
def test_wrong_cuda_attestation_stops_before_resolve(
    exact_tokenizer, mutation, match,
):
    context = make_context(exact_tokenizer)
    mutation(context)
    with pytest.raises(subject.PoweredV13SubjectError, match=match):
        open_fake(context)
    assert "resolve" not in context.events


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ("not-a-uuid, 580.159.03, NVIDIA A100 80GB PCIe, 81920\n", "UUID"),
        ("GPU-12345678-1234-1234-1234-123456789abc, 550.127.05, "
         "NVIDIA A100 80GB PCIe, 81920\n", "below"),
        ("GPU-12345678-1234-1234-1234-123456789abc, 580.159.03, "
         "NVIDIA H100 80GB HBM3, 81920\n", "class differs"),
        ("GPU-12345678-1234-1234-1234-123456789abc, 580.159.03, "
         "NVIDIA A100 80GB PCIe, 40960\n", "memory differs"),
    ],
)
def test_wrong_nvidia_uuid_driver_class_or_memory_stops_before_resolve(
    exact_tokenizer, row, match,
):
    context = make_context(exact_tokenizer)
    context.nvidia_output = row
    with pytest.raises(subject.PoweredV13SubjectError, match=match):
        open_fake(context)
    assert "resolve" not in context.events


def test_wrong_resolved_revision_stops_before_contract_inventory_or_load(exact_tokenizer):
    context = make_context(exact_tokenizer)
    context.resolved_snapshot = SNAPSHOT.parent / ("0" * 40)
    with pytest.raises(subject.PoweredV13SubjectError, match="revision differs"):
        open_fake(context)
    assert "contracts" not in context.events
    assert "load:model" not in context.events


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda c: setattr(c.model.config, "num_hidden_layers", 47), "config/geometry"),
        (lambda c: setattr(c.model.config, "_attn_implementation", "sdpa"),
         "attention backend"),
        (lambda c: setattr(c.model._parameters[0][1], "dtype", "torch.float16"),
         "dtype differs"),
        (lambda c: setattr(c.model._parameters[0][1], "device", FakeDevice("cpu")),
         "devices differ"),
        (lambda c: setattr(c.model, "hf_device_map", {"": 0}), "device map"),
    ],
)
def test_wrong_geometry_backend_dtype_or_device_fails_post_load(
    exact_tokenizer, mutation, match,
):
    context = make_context(exact_tokenizer)
    mutation(context)
    with pytest.raises(subject.PoweredV13SubjectError, match=match):
        open_fake(context)
    assert context.events.index("release") < context.events.index("load:model")
    assert subject._ACTIVE_TOKEN is None


def test_wrong_model_class_fails_post_load_and_evicts(exact_tokenizer):
    context = make_context(exact_tokenizer)
    WrongModel = type("WrongModel", (Qwen3MoeForCausalLM,), {})
    context.model = WrongModel(SNAPSHOT, context.torch)
    with pytest.raises(subject.PoweredV13SubjectError, match="model class"):
        open_fake(context)
    assert subject._ACTIVE_TOKEN is None


@pytest.mark.parametrize("key", sorted(EMPTY_LOADING_INFO))
def test_any_nonempty_loading_info_fails(exact_tokenizer, key):
    context = make_context(exact_tokenizer)
    context.loading_info[key] = ["corrupt.weight"]
    with pytest.raises(subject.PoweredV13SubjectError, match=key):
        open_fake(context)


def test_tokenizer_drift_fails_post_load(exact_tokenizer):
    context = make_context(exact_tokenizer)
    context.tokenizer = SimpleNamespace(
        name_or_path=str(SNAPSHOT), init_kwargs={}, backend_tokenizer=None,
    )
    with pytest.raises(subject.PoweredV13SubjectError, match="wrapper class"):
        open_fake(context)


def test_tokenizer_template_and_vocabulary_drift_fail(exact_tokenizer):
    template_context = make_context(deepcopy(exact_tokenizer))
    template_context.tokenizer.chat_template = "tampered-template"
    with pytest.raises(subject.PoweredV13SubjectError, match="template/vocab"):
        open_fake(template_context)

    vocab_context = make_context(deepcopy(exact_tokenizer))
    original_get_vocab = vocab_context.tokenizer.get_vocab
    vocab_context.tokenizer.get_vocab = lambda: {
        **original_get_vocab(), "tampered-token": 0,
    }
    with pytest.raises(subject.PoweredV13SubjectError, match="template/vocab"):
        open_fake(vocab_context)


def test_checkpoint_weight_topology_corruption_fails_contract():
    tokenizer_inventory = {
        "kind": "protocol_tokenizer", "repo_id": subject.MODEL_ID,
        "revision": subject.MODEL_REVISION, "snapshot_path": str(SNAPSHOT),
        "files": [], "inventory_sha256": "2" * 64,
    }
    model = {
        "kind": "model", "repo_id": subject.MODEL_ID,
        "revision": subject.MODEL_REVISION, "snapshot_path": str(SNAPSHOT),
        "files": [], "inventory_sha256": "1" * 64,
        "weight_tensors": [{
            "name": "weight", "shard": "model.safetensors",
            "dtype": "F16", "shape": [1],
        }],
    }
    map_sha = hashlib.sha256(canonical([{
        "name": "weight", "shard": "model.safetensors",
    }])).hexdigest()
    bundle = subject._ContractBundle(
        subject_contract={"subject": {
            "weight_tensor_count": 1, "parameter_count": 1,
            "weight_name_to_shard_sha256": map_sha,
        }},
        legacy_contract={
            "subjects": {"exact-subject": {
                "files": [], "inventory_sha256": "1" * 64,
                "weight_tensor_count": 1, "parameter_count": 1,
                "weight_name_to_shard_sha256": map_sha,
            }},
            "protocol_tokenizer": {
                "files": [], "inventory_sha256": "2" * 64,
            },
        },
        subject_contract_sha256="3" * 64,
        legacy_contract_sha256="4" * 64,
    )
    with pytest.raises(subject.PoweredV13SubjectError, match="topology row"):
        subject._verify_inventories(
            model, tokenizer_inventory, snapshot=SNAPSHOT, bundle=bundle,
        )


def test_post_load_snapshot_corruption_and_content_corruption_both_fail(
    exact_tokenizer,
):
    post = make_context(exact_tokenizer)
    post.corrupt_post_inventory = True
    with pytest.raises(subject.PoweredV13SubjectError, match="changed across load"):
        open_fake(post)
    content = make_context(exact_tokenizer)
    content.corrupt_weight_content = True
    with pytest.raises(subject.PoweredV13SubjectError, match="content attestation"):
        open_fake(content)
    assert subject._ACTIVE_TOKEN is None


def test_wrong_transformers5_topology_fails(exact_tokenizer):
    context = make_context(exact_tokenizer)
    context.wrong_topology = True
    with pytest.raises(subject.PoweredV13SubjectError, match="topology differs"):
        open_fake(context)


def test_only_one_live_handle_and_close_allows_reopen(exact_tokenizer):
    first_context = make_context(exact_tokenizer)
    first = open_fake(first_context)
    second_context = make_context(exact_tokenizer)
    with pytest.raises(subject.PoweredV13SubjectError, match="already live"):
        open_fake(second_context)
    assert "resolve" not in second_context.events
    first.close()
    assert first.closed
    with pytest.raises(subject.PoweredV13SubjectError, match="closed"):
        _ = first.model
    third_context = make_context(exact_tokenizer)
    third = open_fake(third_context)
    third.close()


def test_external_model_alias_keeps_one_live_slot_until_actual_eviction(
    exact_tokenizer,
):
    first = open_fake(make_context(exact_tokenizer))
    alias = first.model
    first.close()
    assert subject._ACTIVE_TOKEN is not None
    with pytest.raises(subject.PoweredV13SubjectError, match="already live"):
        open_fake(make_context(exact_tokenizer))
    del alias
    gc.collect()
    assert subject._ACTIVE_TOKEN is None
    recovered = open_fake(make_context(exact_tokenizer))
    recovered.close()


def test_failed_load_evicts_reservation_and_retry_can_open(exact_tokenizer):
    failed = make_context(exact_tokenizer)
    failed.load_error = True
    with pytest.raises(subject.PoweredV13SubjectError, match="load failed"):
        open_fake(failed)
    assert subject._ACTIVE_TOKEN is None
    recovered = make_context(exact_tokenizer)
    handle = open_fake(recovered)
    handle.close()
