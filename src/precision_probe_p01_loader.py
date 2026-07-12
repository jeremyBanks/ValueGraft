"""Fail-closed subject loading for precision probe p01.

This module is deliberately parallel to, and does not call, the frozen v12
subject loader.  It targets the isolated Transformers 4.57.6 runtime frozen in
``PRECISION-PROBE-P01-PREREGISTRATION.md``.  In particular, the NF4 arm is not
accepted merely because Transformers sets ``is_loaded_in_4bit``: every routed
expert projection must be a fully initialized bitsandbytes ``Linear4bit``.

``bitsandbytes`` and ``transformers`` are imported only inside functions that
need them.  Importing this module and running its fake/meta tests therefore does
not require a CUDA host or bitsandbytes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

import torch


PROTOCOL_ID = "precision-probe-p01"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
ARCHITECTURE = "Qwen3MoeForCausalLM"
MODEL_TYPE = "qwen3_moe"
ATTENTION_BACKEND = "eager"

EXPECTED_DEPENDENCIES = {
    "torch": "2.12.1",
    "transformers": "4.57.6",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.2",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
}
EXPECTED_PYTHON = "3.12.11"

EXPECTED_GPU_NAME = "NVIDIA A100 80GB PCIe"
MINIMUM_GPU_MEMORY_MIB = 80_000
MINIMUM_NVIDIA_DRIVER = "580.65.06"
EXPECTED_TORCH_CUDA = "13.0"
EXPECTED_COMPUTE_CAPABILITY = (8, 0)

EXPECTED_CONFIG_SHA256 = \
    "a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438"
EXPECTED_GENERATION_CONFIG_SHA256 = \
    "19d306dd769db12a9d710b44cf7f83b635efbe5166b84fb4358a08fb7d88bb53"
EXPECTED_INDEX_SHA256 = \
    "8dde190b862c7c80ec7403c6495de00c60bbaf246ed479cee4506284989c584c"
EXPECTED_CHECKPOINT_BYTES = 61_064_245_248
EXPECTED_CHECKPOINT_KEYS = 18_867

PREREGISTRATION_SHA256 = \
    "5619c5ced93f2e60564fcc2c98e24fd9bbf687f93b6cab6132f5be2105cda374"
REPOSITORY_INPUTS = {
    "PRECISION-PROBE-P01-PREREGISTRATION.md": PREREGISTRATION_SHA256,
    "data/coherent_canary_v12/revision2/session_d/e01.json":
        "6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090",
    "data/coherent_canary_v12/revision2/session_d/e02.json":
        "0d52caab89a6f7d6877f097fc451b5ac529f010720e8118ad89044a054ca9ff0",
    "data/coherent_canary_v12/revision2/session_e/e03.json":
        "3594202e1ec405b7cb2c0a5f9351d3e3c68b65ba6affb38ba83ce5a507273ed3",
    # These are the four model-facing components explicitly reused unchanged
    # by preregistration section 4.  Binding only the new loader would leave a
    # silent provenance hole because p01's runner calls these functions.
    "src/coherent_canary_tokens.py":
        "265eb70ed8a35da6eda5a754e659c2a2af8d9b759b5d88c20fc4852554229142",
    "src/coherent_canary_runtime.py":
        "a6bb4cdb6a16fb2e366f88a3f7d1eba54d2158c4fb60f9f34ee6b343e0fd012e",
    "src/coherent_canary_case.py":
        "89f519e28293ba1048a3c76209f25bcc7e7fd20c2bd2e7a9b23a1f58fd24d48c",
    "src/coherent_canary_technical.py":
        "c0cacfd0227ee83838df643edf5a5ac8dc8ed5478f871bd704daa36c5193dc6f",
}
LOADER_PATH = "src/precision_probe_p01_loader.py"

TOKENIZER_ATTESTATION = {
    "class": "Qwen2Tokenizer",
    "is_fast": False,
    "length": 151_669,
    "vocab_size": 151_643,
    "eos_token_id": 151_645,
    "pad_token_id": 151_643,
    "all_special_ids": [
        151_645, 151_643, 151_644, 151_646, 151_647, 151_648,
        151_649, 151_650, 151_651, 151_652, 151_653, 151_654,
        151_655, 151_656,
    ],
    "special_token_ids": {
        "<|endoftext|>": 151_643,
        "<|im_start|>": 151_644,
        "<|im_end|>": 151_645,
    },
    "chat_template_sha256":
        "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326",
    "vocab_sha256":
        "f488fa45d324a8bc64c84f0e27b47223872d550f2a6900564f77dfa67ca5ff4d",
    "identity_prefix_token_count": 27,
    "identity_prefix_ids_sha256":
        "44bc1b106347e9008c6d7f1d30ab51166942c419ad66819eae733d2e990c3881",
}


class PrecisionProbeLoaderError(RuntimeError):
    """A p01 model, runtime, host, or immutable binding failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionProbeLoaderError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _names_sha256(names: Sequence[str]) -> str:
    return _sha256_bytes(_canonical(sorted(str(name) for name in names)))


@dataclass(frozen=True)
class P01Topology:
    layers: int = 48
    experts_per_layer: int = 128
    hidden_size: int = 2048
    moe_intermediate_size: int = 768
    attention_heads: int = 32
    kv_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 151_936
    rope_theta: float = 10_000_000.0

    @property
    def expert_modules(self) -> int:
        return self.layers * self.experts_per_layer * 3

    @property
    def attention_modules(self) -> int:
        return self.layers * 4

    @property
    def router_modules(self) -> int:
        return self.layers

    @property
    def eligible_modules(self) -> int:
        return self.expert_modules + self.attention_modules + \
            self.router_modules

    @property
    def expert_elements(self) -> int:
        return (self.layers * self.experts_per_layer * 3 *
                self.hidden_size * self.moe_intermediate_size)

    @property
    def attention_elements(self) -> int:
        q_width = self.attention_heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        return self.layers * self.hidden_size * (2 * q_width + 2 * kv_width)

    @property
    def router_elements(self) -> int:
        return self.layers * self.hidden_size * self.experts_per_layer

    @property
    def eligible_elements(self) -> int:
        return self.expert_elements + self.attention_elements + \
            self.router_elements

    @property
    def parameter_elements(self) -> int:
        # Two untied vocabulary matrices, two hidden RMS norms and two
        # head-dimension Q/K norms per layer, plus the final RMS norm.
        norms = (self.layers * (2 * self.hidden_size + 2 * self.head_dim) +
                 self.hidden_size)
        return self.eligible_elements + 2 * self.vocab_size * \
            self.hidden_size + norms


EXACT_TOPOLOGY = P01Topology()
_require(EXACT_TOPOLOGY.expert_modules == 18_432,
         "internal p01 expert-module arithmetic differs")
_require(EXACT_TOPOLOGY.eligible_modules == 18_672,
         "internal p01 eligible-module arithmetic differs")
_require(EXACT_TOPOLOGY.expert_elements == 28_991_029_248,
         "internal p01 expert-element arithmetic differs")
_require(EXACT_TOPOLOGY.eligible_elements == 29_909_581_824,
         "internal p01 eligible-element arithmetic differs")
_require(EXACT_TOPOLOGY.parameter_elements == 30_532_122_624,
         "internal p01 parameter arithmetic differs")


_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)$")
_ATTENTION_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)$")
_ROUTER_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate$")
_SENTINELS = ((0, 0), (24, 64), (47, 127))


def _expected_module_names(topology: P01Topology) -> dict[str, set[str]]:
    experts = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in range(topology.layers)
        for expert in range(topology.experts_per_layer)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    attention = {
        f"model.layers.{layer}.self_attn.{projection}"
        for layer in range(topology.layers)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    routers = {
        f"model.layers.{layer}.mlp.gate"
        for layer in range(topology.layers)
    }
    return {
        "experts": experts,
        "attention": attention,
        "routers": routers,
        "eligible": experts | attention | routers,
    }


def dependency_fingerprint() -> dict[str, Any]:
    """Attest the isolated stack without importing bitsandbytes."""
    try:
        observed = {
            name: importlib.metadata.version(name)
            for name in EXPECTED_DEPENDENCIES
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise PrecisionProbeLoaderError(
            f"required p01 dependency is absent: {exc.name}") from exc
    normalized = dict(observed)
    normalized["torch"] = observed["torch"].split("+", 1)[0]
    expected_python = platform.python_version()
    _require(normalized == EXPECTED_DEPENDENCIES,
             f"p01 dependency versions differ: {observed}")
    _require(expected_python == EXPECTED_PYTHON,
             f"p01 Python version differs: {expected_python}")
    return {
        "packages": observed,
        "normalized_packages": normalized,
        "python": expected_python,
    }


def _version_tuple(value: str) -> tuple[int, ...]:
    _require(bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value)),
             f"invalid version string: {value!r}")
    return tuple(int(part) for part in value.split("."))


def host_fingerprint() -> dict[str, Any]:
    """Bind the existing exact A100/CUDA/driver admission requirements."""
    _require(platform.system() == "Linux", "p01 production host is not Linux")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "p01 requires exactly one visible CUDA device")
    _require(str(torch.version.cuda) == EXPECTED_TORCH_CUDA,
             f"p01 Torch CUDA runtime differs: {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    query = subprocess.run([
        "nvidia-smi", "--query-gpu=uuid,driver_version,name,memory.total",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=False)
    lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    _require(query.returncode == 0 and len(lines) == 1,
             "p01 nvidia-smi exact-device query failed")
    parts = [part.strip() for part in lines[0].split(",")]
    _require(len(parts) == 4, f"p01 invalid nvidia-smi row: {lines[0]!r}")
    uuid, driver, name, memory_text = parts
    try:
        memory_mib = int(memory_text)
    except ValueError as exc:
        raise PrecisionProbeLoaderError(
            f"p01 invalid GPU memory: {memory_text!r}") from exc
    _require(name == EXPECTED_GPU_NAME and props.name == EXPECTED_GPU_NAME,
             f"p01 GPU name differs: {name!r}/{props.name!r}")
    _require(memory_mib >= MINIMUM_GPU_MEMORY_MIB and
             props.total_memory >= MINIMUM_GPU_MEMORY_MIB * 1024 * 1024,
             f"p01 GPU memory is below 80 GB: {memory_mib}/"
             f"{props.total_memory}")
    _require(_version_tuple(driver) >= _version_tuple(MINIMUM_NVIDIA_DRIVER),
             f"p01 NVIDIA driver is too old: {driver}")
    capability = tuple(int(value) for value in
                       torch.cuda.get_device_capability(0))
    _require(capability == EXPECTED_COMPUTE_CAPABILITY,
             f"p01 compute capability differs: {capability}")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": torch.cuda.device_count(),
        "device_index": 0,
        "gpu_uuid": uuid,
        "gpu_name": name,
        "driver_version": driver,
        "memory_total_mib": memory_mib,
        "torch_device_name": props.name,
        "torch_total_memory_bytes": int(props.total_memory),
        "compute_capability": list(capability),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
    }


def repository_binding(repo: Path) -> dict[str, Any]:
    """Bind only the new protocol, loader, and literal case inputs."""
    repo = repo.resolve(strict=True)
    _require((repo / ".git").exists(), "p01 repo is not a Git worktree")
    branch = subprocess.check_output(
        ["git", "-C", str(repo), "branch", "--show-current"],
        text=True).strip()
    _require(branch == "trunk", f"p01 must run on trunk, observed {branch!r}")
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    expected = dict(REPOSITORY_INPUTS)
    expected[LOADER_PATH] = None
    rows = []
    for relative, frozen_hash in sorted(expected.items()):
        path = repo / relative
        _require(path.is_file() and not path.is_symlink(),
                 f"p01 repository input is absent/non-regular: {relative}")
        worktree = path.read_bytes()
        try:
            committed = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"HEAD:{relative}"])
        except subprocess.CalledProcessError as exc:
            raise PrecisionProbeLoaderError(
                f"p01 repository input is not committed: {relative}") from exc
        _require(committed == worktree,
                 f"p01 repository input differs from HEAD: {relative}")
        digest = _sha256_bytes(worktree)
        _require(frozen_hash is None or digest == frozen_hash,
                 f"p01 frozen input digest differs: {relative}")
        rows.append({"path": relative, "sha256": digest,
                     "size_bytes": len(worktree)})
    return {
        "branch": branch,
        "repository_commit": head,
        "files": rows,
        "inventory_sha256": _sha256_bytes(_canonical(rows)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise PrecisionProbeLoaderError(f"invalid JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _validate_snapshot_path(snapshot: Path) -> Path:
    snapshot = snapshot.resolve(strict=True)
    _require(snapshot.is_dir() and snapshot.name == REVISION and
             snapshot.parent.name == "snapshots" and
             snapshot.parent.parent.name ==
             "models--Qwen--Qwen3-30B-A3B-Instruct-2507",
             f"p01 resolved snapshot is not exact: {snapshot}")
    return snapshot


def resolve_pinned_snapshot(*, allow_download: bool) -> Path:
    from huggingface_hub import snapshot_download
    try:
        value = snapshot_download(
            MODEL_ID, revision=REVISION,
            local_files_only=not allow_download)
    except Exception as exc:
        raise PrecisionProbeLoaderError(
            f"could not resolve exact p01 snapshot: {exc}") from exc
    return _validate_snapshot_path(Path(value))


def checkpoint_attestation(snapshot: Path, *, require_weights: bool) \
        -> dict[str, Any]:
    """Attest exact config/index topology and, for G1, all shard presence."""
    snapshot = _validate_snapshot_path(snapshot)
    config_path = snapshot / "config.json"
    generation_path = snapshot / "generation_config.json"
    index_path = snapshot / "model.safetensors.index.json"
    for path in (config_path, generation_path, index_path):
        _require(path.is_file(), f"p01 snapshot file is absent: {path.name}")
    _require(_sha256_file(config_path) == EXPECTED_CONFIG_SHA256,
             "p01 config digest differs")
    _require(_sha256_file(generation_path) ==
             EXPECTED_GENERATION_CONFIG_SHA256,
             "p01 generation config digest differs")
    _require(_sha256_file(index_path) == EXPECTED_INDEX_SHA256,
             "p01 safetensors index digest differs")

    config = _read_json(config_path)
    expected_config = {
        "architectures": [ARCHITECTURE],
        "model_type": MODEL_TYPE,
        "num_hidden_layers": EXACT_TOPOLOGY.layers,
        "num_attention_heads": EXACT_TOPOLOGY.attention_heads,
        "num_key_value_heads": EXACT_TOPOLOGY.kv_heads,
        "head_dim": EXACT_TOPOLOGY.head_dim,
        "hidden_size": EXACT_TOPOLOGY.hidden_size,
        "moe_intermediate_size": EXACT_TOPOLOGY.moe_intermediate_size,
        "num_experts": EXACT_TOPOLOGY.experts_per_layer,
        "vocab_size": EXACT_TOPOLOGY.vocab_size,
        "rope_theta": EXACT_TOPOLOGY.rope_theta,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "quantization_config": None,
    }
    observed_config = {key: config.get(key) for key in expected_config}
    _require(observed_config == expected_config,
             f"p01 checkpoint config differs: {observed_config}")

    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, Mapping),
             "p01 checkpoint weight_map is absent")
    keys = {str(key) for key in weight_map}
    _require(len(keys) == EXPECTED_CHECKPOINT_KEYS,
             f"p01 checkpoint key count differs: {len(keys)}")
    total_size = index.get("metadata", {}).get("total_size")
    _require(total_size == EXPECTED_CHECKPOINT_BYTES and
             total_size // 2 == EXACT_TOPOLOGY.parameter_elements,
             f"p01 checkpoint byte/parameter count differs: {total_size}")
    shard_names = sorted({str(value) for value in weight_map.values()})
    expected_shards = [
        f"model-{index:05d}-of-00016.safetensors"
        for index in range(1, 17)
    ]
    _require(shard_names == expected_shards,
             f"p01 checkpoint shard set differs: {shard_names}")
    if require_weights:
        missing = [name for name in shard_names
                   if not (snapshot / name).is_file()]
        _require(not missing, f"p01 checkpoint shards are absent: {missing}")
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "generation_config_sha256": EXPECTED_GENERATION_CONFIG_SHA256,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "checkpoint_key_count": len(keys),
        "checkpoint_key_names_sha256": _names_sha256(list(keys)),
        "checkpoint_bytes": total_size,
        "logical_parameter_elements": total_size // 2,
        "shards": shard_names,
        "all_shards_present": all((snapshot / name).is_file()
                                   for name in shard_names),
        "checkpoint_keys": keys,
        "weight_map": {str(key): str(value)
                       for key, value in weight_map.items()},
    }


def _linear_shape(name: str, topology: P01Topology) -> tuple[int, int]:
    match = _EXPERT_RE.fullmatch(name)
    if match:
        projection = match.group(3)
        if projection in {"gate_proj", "up_proj"}:
            return topology.moe_intermediate_size, topology.hidden_size
        return topology.hidden_size, topology.moe_intermediate_size
    match = _ATTENTION_RE.fullmatch(name)
    if match:
        projection = match.group(2)
        q_width = topology.attention_heads * topology.head_dim
        kv_width = topology.kv_heads * topology.head_dim
        if projection == "q_proj":
            return q_width, topology.hidden_size
        if projection in {"k_proj", "v_proj"}:
            return kv_width, topology.hidden_size
        return topology.hidden_size, q_width
    if _ROUTER_RE.fullmatch(name):
        return topology.experts_per_layer, topology.hidden_size
    raise PrecisionProbeLoaderError(f"unknown p01 eligible linear: {name}")


def attest_g0_meta_model(
        model, checkpoint_keys: Sequence[str], *,
        topology: P01Topology = EXACT_TOPOLOGY,
        require_meta: bool = True) -> dict[str, Any]:
    """Verify checkpoint/meta keys and every quantization-eligible linear."""
    named_parameters = list(model.named_parameters(remove_duplicate=False))
    if require_meta:
        _require(named_parameters and
                 all(parameter.device.type == "meta"
                     for _, parameter in named_parameters),
                 "p01 G0 model is not entirely meta")
    meta_keys = set(model.state_dict())
    checkpoint_key_set = set(str(value) for value in checkpoint_keys)
    _require(meta_keys == checkpoint_key_set,
             "p01 G0 checkpoint/meta-model key sets differ")
    expected = _expected_module_names(topology)
    linear_modules = {
        name: module for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    _require(set(linear_modules) == expected["eligible"] | {"lm_head"},
             "p01 G0 linear module coverage differs")
    for name in sorted(expected["eligible"]):
        module = linear_modules[name]
        out_features, in_features = _linear_shape(name, topology)
        _require(module.in_features == in_features and
                 module.out_features == out_features and
                 module.bias is None and
                 list(module.weight.shape) == [out_features, in_features],
                 f"p01 G0 eligible linear shape/bias differs: {name}")
    lm_head = linear_modules["lm_head"]
    _require(lm_head.in_features == topology.hidden_size and
             lm_head.out_features == topology.vocab_size and
             lm_head.bias is None,
             "p01 G0 lm_head shape/bias differs")
    parameter_count = sum(parameter.numel()
                          for _, parameter in named_parameters)
    _require(parameter_count == topology.parameter_elements,
             f"p01 G0 meta parameter count differs: {parameter_count}")
    eligible_elements = sum(
        linear_modules[name].in_features * linear_modules[name].out_features
        for name in expected["eligible"])
    _require(eligible_elements == topology.eligible_elements,
             f"p01 G0 eligible logical coverage differs: {eligible_elements}")
    return {
        "checkpoint_key_count": len(checkpoint_key_set),
        "checkpoint_key_names_sha256": _names_sha256(list(checkpoint_key_set)),
        "meta_key_count": len(meta_keys),
        "meta_key_names_sha256": _names_sha256(list(meta_keys)),
        "expert_linear_count": len(expected["experts"]),
        "attention_linear_count": len(expected["attention"]),
        "router_linear_count": len(expected["routers"]),
        "eligible_linear_count": len(expected["eligible"]),
        "eligible_linear_names_sha256":
            _names_sha256(list(expected["eligible"])),
        "expert_logical_elements": topology.expert_elements,
        "eligible_logical_elements": eligible_elements,
        "logical_parameter_elements": parameter_count,
        "lm_head_excluded": True,
    }


def _attest_cache_api(model, *, topology: P01Topology,
                      dynamic_cache_class=None) -> dict[str, Any]:
    """Static/synthetic check of the exact cache APIs reused by p01."""
    if dynamic_cache_class is None:
        from transformers import DynamicCache as dynamic_cache_class
    forward_parameters = set(inspect.signature(model.forward).parameters)
    required_forward = {
        "past_key_values", "position_ids", "cache_position",
        "logits_to_keep", "use_cache",
    }
    _require(required_forward.issubset(forward_parameters),
             f"p01 G0 model forward cache API differs: {forward_parameters}")
    cache_parameters = set(inspect.signature(dynamic_cache_class).parameters)
    _require("ddp_cache_data" in cache_parameters,
             "p01 G0 DynamicCache lacks ddp_cache_data")
    keys = torch.arange(
        topology.kv_heads * 2 * topology.head_dim,
        dtype=torch.bfloat16).reshape(
            1, topology.kv_heads, 2, topology.head_dim)
    values = keys + torch.tensor(1, dtype=torch.bfloat16)
    cache = dynamic_cache_class(ddp_cache_data=[
        (keys, values), (keys.clone(), values.clone())])
    _require(hasattr(cache, "to_legacy_cache"),
             "p01 G0 DynamicCache lacks legacy extraction")
    legacy = cache.to_legacy_cache()
    _require(len(legacy) == 2 and
             all(torch.equal(row[0], keys) and torch.equal(row[1], values)
                 for row in legacy),
             "p01 G0 DynamicCache snapshot/rebuild content differs")
    rebuilt = dynamic_cache_class(ddp_cache_data=[
        (row[0].clone(), row[1].clone()) for row in legacy])
    roundtrip = rebuilt.to_legacy_cache()
    _require(all(torch.equal(a, c) and torch.equal(b, d)
                 for (a, b), (c, d) in zip(legacy, roundtrip)),
             "p01 G0 DynamicCache roundtrip differs")
    return {
        "forward_parameters": sorted(required_forward),
        "dynamic_cache_ddp_constructor": True,
        "synthetic_layers": 2,
        "synthetic_shape": [1, topology.kv_heads, 2, topology.head_dim],
        "synthetic_dtype": "torch.bfloat16",
        "snapshot_rebuild_equal": True,
    }


def g0_static_meta_compatibility(snapshot: Path) -> dict[str, Any]:
    """Run the zero-GPU full-topology compatibility gate."""
    checkpoint = checkpoint_attestation(snapshot, require_weights=False)
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        attn_implementation=ATTENTION_BACKEND)
    _require(type(config).__name__ == "Qwen3MoeConfig" and
             getattr(config, "_attn_implementation", None) ==
             ATTENTION_BACKEND,
             "p01 G0 config class/attention backend differs")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=False,
            attn_implementation=ATTENTION_BACKEND)
    _require(type(model).__name__ == ARCHITECTURE,
             f"p01 G0 model class differs: {type(model).__name__}")
    topology = attest_g0_meta_model(
        model, checkpoint["checkpoint_keys"], require_meta=True)
    cache_api = _attest_cache_api(model, topology=EXACT_TOPOLOGY)
    return {
        "schema": "precision_probe_p01_g0_v1",
        "passes": True,
        "transformers_version": importlib.metadata.version("transformers"),
        "topology": topology,
        "cache_api": cache_api,
    }


def _attest_tokenizer(tokenizer, *, snapshot: Path) -> tuple[dict[str, Any], list[int]]:
    _require(getattr(tokenizer, "is_fast", None) is False,
             "p01 tokenizer is not the required slow tokenizer")
    name_or_path = getattr(tokenizer, "name_or_path", None)
    _require(isinstance(name_or_path, str) and
             Path(name_or_path).resolve() == snapshot.resolve(),
             "p01 tokenizer path differs from exact snapshot")
    messages = [
        {"role": "system", "content": "Answer plainly."},
        {"role": "user", "content":
         "Write one short neutral sentence acknowledging that a record exists."},
    ]
    identity_ids = [int(value) for value in tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True)]
    observed = {
        "class": type(tokenizer).__name__,
        "is_fast": bool(tokenizer.is_fast),
        "length": len(tokenizer),
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "all_special_ids": [int(value) for value in tokenizer.all_special_ids],
        "special_token_ids": {
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in TOKENIZER_ATTESTATION["special_token_ids"]
        },
        "chat_template_sha256": _sha256_bytes(
            str(tokenizer.chat_template).encode("utf-8")),
        "vocab_sha256": _sha256_bytes(_canonical(tokenizer.get_vocab())),
        "identity_prefix_token_count": len(identity_ids),
        "identity_prefix_ids_sha256": _sha256_bytes(b"".join(
            value.to_bytes(8, "little", signed=True)
            for value in identity_ids)),
    }
    _require(observed == TOKENIZER_ATTESTATION,
             f"p01 tokenizer attestation differs: {observed}")
    return observed, identity_ids


def _loading_info_attestation(loading_info: Mapping[str, Any]) -> dict[str, list]:
    expected = {"missing_keys", "unexpected_keys", "mismatched_keys",
                "error_msgs"}
    _require(isinstance(loading_info, Mapping) and
             set(loading_info) == expected,
             f"p01 loading-info fields differ: {loading_info}")
    result = {}
    for key in sorted(expected):
        value = loading_info[key]
        _require(isinstance(value, (list, tuple, set)) and not value,
                 f"p01 model loading reported {key}: {value}")
        result[key] = []
    return result


def _module_device_attestation(model, *, expected_device: str) -> dict[str, Any]:
    parameters = list(model.named_parameters(remove_duplicate=False))
    _require(parameters and all(parameter.device.type != "meta"
                                for _, parameter in parameters),
             "p01 loaded model contains meta/no parameters")
    parameter_devices = sorted({str(parameter.device)
                                for _, parameter in parameters})
    buffer_devices = sorted({str(buffer.device)
                             for _, buffer in model.named_buffers()})
    _require(parameter_devices == [expected_device] and
             all(device == expected_device for device in buffer_devices),
             f"p01 loaded model device placement differs: "
             f"{parameter_devices}/{buffer_devices}")
    device_map = getattr(model, "hf_device_map", None)
    _require(device_map in (None, {"": 0}, {"": "cuda:0"}, {"": expected_device}),
             f"p01 model offload/device map differs: {device_map}")
    # Normalize before this enters the bindings artifact.  HF currently emits
    # an int or string here, but recording a torch.device would make a future
    # otherwise-valid result non-serializable.
    if device_map is None:
        serializable_device_map = None
    else:
        serializable_device_map = {
            str(key): (int(value) if isinstance(value, int) else str(value))
            for key, value in device_map.items()
        }
    adverse_hooks = []
    for name, module in model.named_modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        execution_device = getattr(hook, "execution_device", None)
        if (bool(getattr(hook, "offload", False)) or
                bool(getattr(hook, "offload_buffers", False)) or
                (execution_device is not None and
                 str(execution_device) not in {"cuda", expected_device})):
            adverse_hooks.append(name)
    _require(not adverse_hooks,
             f"p01 model contains adverse offload hooks: {adverse_hooks}")
    return {
        "parameter_devices": parameter_devices,
        "buffer_devices": buffer_devices,
        "hf_device_map": serializable_device_map,
        "adverse_offload_hooks": [],
    }


def _expected_sentinels(topology: P01Topology) -> tuple[tuple[int, int], ...]:
    if topology == EXACT_TOPOLOGY:
        return _SENTINELS
    return ((0, 0), (topology.layers // 2,
                     topology.experts_per_layer // 2),
            (topology.layers - 1, topology.experts_per_layer - 1))


def _source_tensor_loader(snapshot: Path, weight_map: Mapping[str, str],
                          name: str) -> torch.Tensor:
    from safetensors import safe_open
    shard = weight_map.get(name)
    _require(isinstance(shard, str) and (snapshot / shard).is_file(),
             f"p01 sentinel source is absent: {name}")
    try:
        with safe_open(str(snapshot / shard), framework="pt",
                       device="cpu") as handle:
            _require(name in handle.keys(),
                     f"p01 sentinel key is absent: {name}")
            return handle.get_tensor(name)
    except PrecisionProbeLoaderError:
        raise
    except Exception as exc:
        raise PrecisionProbeLoaderError(
            f"p01 sentinel source could not be read: {name}: {exc}") from exc


def attest_nf4_coverage(
        model, *, snapshot: Path, weight_map: Mapping[str, str],
        topology: P01Topology = EXACT_TOPOLOGY,
        expected_device: str = "cuda:0", bnb_module=None,
        source_loader: Callable[[str], torch.Tensor] | None = None
        ) -> dict[str, Any]:
    """Attest every NF4 module and dequantized cross-checkpoint sentinel."""
    if bnb_module is None:
        import bitsandbytes as bnb_module  # lazy: unavailable in normal Mac env
    linear4bit = bnb_module.nn.Linear4bit
    params4bit = bnb_module.nn.Params4bit
    quantization_config = getattr(model.config, "quantization_config", None)
    quantization_method = getattr(model, "quantization_method", None)
    quantization_method_value = getattr(
        quantization_method, "value", quantization_method)
    global_fields = {
        "is_loaded_in_4bit": bool(
            getattr(model, "is_loaded_in_4bit", False)),
        "is_quantized": bool(getattr(model, "is_quantized", False)),
        "quantization_method": (None if quantization_method_value is None
                                else str(quantization_method_value)),
        "hf_quantizer_class": type(
            getattr(model, "hf_quantizer", None)).__name__,
        "config_load_in_4bit": bool(
            getattr(quantization_config, "load_in_4bit", False)),
        "config_load_in_8bit": bool(
            getattr(quantization_config, "load_in_8bit", False)),
        "config_quant_type": getattr(
            quantization_config, "bnb_4bit_quant_type", None),
        "config_double_quant": bool(getattr(
            quantization_config, "bnb_4bit_use_double_quant", False)),
        "config_compute_dtype": str(getattr(
            quantization_config, "bnb_4bit_compute_dtype", None)),
        "config_quant_storage": str(getattr(
            quantization_config, "bnb_4bit_quant_storage", None)),
        "config_skip_modules": list(getattr(
            quantization_config, "llm_int8_skip_modules", []) or []),
        "config_cpu_offload": bool(getattr(
            quantization_config, "llm_int8_enable_fp32_cpu_offload", False)),
    }
    _require(global_fields == {
        "is_loaded_in_4bit": True,
        "is_quantized": True,
        "quantization_method": "bitsandbytes",
        "hf_quantizer_class": "Bnb4BitHfQuantizer",
        "config_load_in_4bit": True,
        "config_load_in_8bit": False,
        "config_quant_type": "nf4",
        "config_double_quant": True,
        "config_compute_dtype": "torch.bfloat16",
        "config_quant_storage": "torch.uint8",
        "config_skip_modules": ["lm_head"],
        "config_cpu_offload": False,
    }, f"p01 NF4 global quantization configuration differs: {global_fields}")
    expected = _expected_module_names(topology)
    quantized = {
        name: module for name, module in model.named_modules()
        if isinstance(module, linear4bit)
    }
    _require(set(quantized) == expected["eligible"],
             "p01 NF4 Linear4bit module coverage differs")
    expert_names = {name for name in quantized if _EXPERT_RE.fullmatch(name)}
    _require(expert_names == expected["experts"],
             "p01 NF4 expert Linear4bit coverage differs")
    ordinary = {
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and
        not isinstance(module, linear4bit)
    }
    _require(ordinary == {"lm_head"},
             f"p01 NF4 ordinary-linear coverage differs: {ordinary}")

    verified_experts = 0
    verified_all = 0
    for name in sorted(expected["eligible"]):
        module = quantized[name]
        weight = getattr(module, "weight", None)
        state = getattr(weight, "quant_state", None)
        out_features, in_features = _linear_shape(name, topology)
        _require(module.in_features == in_features and
                 module.out_features == out_features and
                 isinstance(weight, params4bit) and
                 weight.device == torch.device(expected_device) and
                 bool(getattr(weight, "bnb_quantized", False)) and
                 getattr(weight, "quant_type", None) == "nf4" and
                 bool(getattr(weight, "compress_statistics", False)) and
                 state is not None and
                 getattr(state, "quant_type", None) == "nf4" and
                 bool(getattr(state, "nested", False)) and
                 getattr(state, "state2", None) is not None and
                 getattr(state, "dtype", None) == torch.bfloat16 and
                 getattr(module, "compute_dtype", None) == torch.bfloat16,
                 f"p01 NF4 weight/state initialization differs: {name}")
        state_tensors = [
            getattr(state, "absmax", None), getattr(state, "code", None),
            getattr(state, "offset", None),
            getattr(state.state2, "absmax", None),
            getattr(state.state2, "code", None),
        ]
        _require(all(isinstance(value, torch.Tensor) and
                     str(value.device) == expected_device
                     for value in state_tensors),
                 f"p01 NF4 quantization-state placement differs: {name}")
        verified_all += 1
        if name in expected["experts"]:
            verified_experts += 1
    _require(verified_all == topology.eligible_modules and
             verified_experts == topology.expert_modules,
             "p01 NF4 verified module counts differ")

    forbidden_expert_parameters = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if ".mlp.experts." not in name:
            continue
        module_name = name.rsplit(".", 1)[0]
        if module_name not in expected["experts"] or \
                not isinstance(parameter, params4bit):
            forbidden_expert_parameters.append(name)
    _require(not forbidden_expert_parameters,
             "p01 NF4 expert subtree contains ordinary/packed parameters: "
             f"{forbidden_expert_parameters[:10]}")

    lm_head = dict(model.named_modules())["lm_head"]
    _require(isinstance(lm_head, torch.nn.Linear) and
             not isinstance(lm_head, linear4bit) and
             lm_head.weight.dtype == torch.bfloat16 and
             str(lm_head.weight.device) == expected_device,
             "p01 NF4 skipped lm_head is not full-precision bf16")
    full_precision = [
        (name, parameter) for name, parameter in
        model.named_parameters(remove_duplicate=False)
        if not isinstance(parameter, params4bit)
    ]
    _require(full_precision and all(parameter.dtype == torch.bfloat16 and
                                    str(parameter.device) == expected_device
                                    for _, parameter in full_precision),
             "p01 NF4 nonquantized parameters are not bf16 on CUDA")

    if source_loader is None:
        source_loader = lambda name: _source_tensor_loader(
            snapshot, weight_map, name)
    sentinel_rows = []
    for layer, expert in _expected_sentinels(topology):
        _require(0 <= layer < topology.layers and
                 0 <= expert < topology.experts_per_layer,
                 "p01 NF4 sentinel index is outside topology")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module_name = (f"model.layers.{layer}.mlp.experts.{expert}."
                           f"{projection}")
            module = quantized[module_name]
            source_name = module_name + ".weight"
            source = source_loader(source_name).detach().to(
                device="cpu", dtype=torch.bfloat16).contiguous()
            dequantized = bnb_module.functional.dequantize_4bit(
                module.weight.data, module.weight.quant_state).detach().to(
                    device="cpu", dtype=torch.bfloat16).contiguous()
            expected_shape = list(_linear_shape(module_name, topology))
            _require(list(source.shape) == expected_shape and
                     list(dequantized.shape) == expected_shape and
                     bool(torch.isfinite(dequantized.float()).all()),
                     f"p01 NF4 dequantized sentinel shape/finite differs: "
                     f"{module_name}")
            error = (dequantized.float() - source.float()).abs()
            max_error = float(error.max().item())
            mean_error = float(error.mean().item())
            _require(math.isfinite(max_error) and max_error > 0.0 and
                     math.isfinite(mean_error) and mean_error > 0.0,
                     f"p01 NF4 sentinel has zero/nonfinite quantization error: "
                     f"{module_name}")
            sentinel_rows.append({
                "layer": layer, "expert": expert,
                "projection": projection, "module": module_name,
                "shape": expected_shape,
                "source_dtype": str(source.dtype),
                "dequantized_dtype": str(dequantized.dtype),
                "source_sha256": _sha256_bytes(
                    source.view(torch.uint8).numpy().tobytes()),
                "dequantized_sha256": _sha256_bytes(
                    dequantized.view(torch.uint8).numpy().tobytes()),
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
            })
    return {
        "regime": "nf4",
        "linear4bit_modules": verified_all,
        "expert_linear4bit_modules": verified_experts,
        "attention_linear4bit_modules": len(expected["attention"]),
        "router_linear4bit_modules": len(expected["routers"]),
        "linear4bit_names_sha256": _names_sha256(list(quantized)),
        "expert_logical_coverage": {
            "observed": topology.expert_elements,
            "expected": topology.expert_elements,
        },
        "total_logical_quantized_coverage": {
            "observed": topology.eligible_elements,
            "expected": topology.eligible_elements,
            "checkpoint_total": topology.parameter_elements,
        },
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "torch.bfloat16",
        "global_quantization_fields": global_fields,
        "ordinary_linears": ["lm_head"],
        "sentinels": sentinel_rows,
    }


def attest_bf16_coverage(
        model, *, topology: P01Topology = EXACT_TOPOLOGY,
        expected_device: str = "cuda:0") -> dict[str, Any]:
    """Reject every quantization trace and attest all bf16 model weights."""
    quantized_names = [
        name for name, module in model.named_modules()
        if any(marker in (type(module).__module__ + "." +
                          type(module).__name__).lower()
               for marker in ("bitsandbytes", "linear4bit", "params4bit",
                              "gptq", "awq"))
    ]
    config_quantization = getattr(model.config, "quantization_config", None)
    fields = {
        "config_quantization": config_quantization,
        "is_quantized": bool(getattr(model, "is_quantized", False)),
        "is_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "is_loaded_in_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
        "hf_quantizer_present": getattr(model, "hf_quantizer", None) is not None,
        "quantization_method": getattr(model, "quantization_method", None),
        "load_in_4bit": bool(getattr(model, "load_in_4bit", False)),
        "load_in_8bit": bool(getattr(model, "load_in_8bit", False)),
    }
    _require(not quantized_names and config_quantization is None and
             not any(fields[key] for key in fields if key !=
                     "config_quantization"),
             f"p01 bf16 model contains quantization: {fields}/{quantized_names}")
    parameters = list(model.named_parameters(remove_duplicate=False))
    _require(parameters and all(parameter.is_floating_point() and
                                parameter.dtype == torch.bfloat16 and
                                str(parameter.device) == expected_device
                                for _, parameter in parameters),
             "p01 bf16 weight dtype/device coverage differs")
    parameter_count = sum(parameter.numel() for _, parameter in parameters)
    _require(parameter_count == topology.parameter_elements,
             f"p01 bf16 logical parameter count differs: {parameter_count}")
    expected = _expected_module_names(topology)
    linears = {
        name: module for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    _require(set(linears) == expected["eligible"] | {"lm_head"},
             "p01 bf16 linear module coverage differs")
    return {
        "regime": "bf16",
        "logical_parameter_elements": parameter_count,
        "floating_parameter_dtype": "torch.bfloat16",
        "linear_modules": len(linears),
        "linear_names_sha256": _names_sha256(list(linears)),
        "quantization_modules": [],
        "quantization_fields": fields,
    }


def attest_kv_cache(cache, *, token_count: int,
                    topology: P01Topology = EXACT_TOPOLOGY,
                    expected_device: str = "cuda:0") -> dict[str, Any]:
    """Attest the real-forward K/V layer count, dtype, device, and shape."""
    if hasattr(cache, "to_legacy_cache"):
        rows = cache.to_legacy_cache()
    elif isinstance(cache, (list, tuple)):
        rows = cache
    else:
        raise PrecisionProbeLoaderError(
            f"p01 unsupported real-forward cache: {type(cache).__name__}")
    _require(len(rows) == topology.layers,
             f"p01 real-forward KV layer count differs: {len(rows)}")
    expected_shape = [1, topology.kv_heads, token_count, topology.head_dim]
    layer_rows = []
    for index, row in enumerate(rows):
        _require(isinstance(row, (list, tuple)) and len(row) == 2,
                 f"p01 cache layer {index} is not a K/V pair")
        key, value = row
        _require(isinstance(key, torch.Tensor) and
                 isinstance(value, torch.Tensor) and
                 list(key.shape) == expected_shape and
                 list(value.shape) == expected_shape and
                 key.dtype == value.dtype == torch.bfloat16 and
                 str(key.device) == str(value.device) == expected_device and
                 bool(torch.isfinite(key.float()).all()) and
                 bool(torch.isfinite(value.float()).all()),
                 f"p01 cache layer {index} shape/dtype/device/finite differs")
        layer_rows.append({
            "layer": index,
            "key_shape": list(key.shape), "value_shape": list(value.shape),
            "key_dtype": str(key.dtype), "value_dtype": str(value.dtype),
            "device": str(key.device),
        })
    return {
        "observed_real_forward": True,
        "layers": len(rows),
        "token_count": token_count,
        "expected_shape": expected_shape,
        "dtype": "torch.bfloat16",
        "device": expected_device,
        "layer_rows": layer_rows,
    }


def _real_forward_attestation(model, identity_ids: Sequence[int]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    input_ids = torch.tensor([list(identity_ids)], dtype=torch.long,
                             device=device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True,
                       logits_to_keep=1, return_dict=True)
    logits = getattr(output, "logits", None)
    _require(isinstance(logits, torch.Tensor) and
             list(logits.shape) == [1, 1, EXACT_TOPOLOGY.vocab_size] and
             str(logits.device) == "cuda:0" and
             bool(torch.isfinite(logits.float()).all()),
             "p01 real-forward logits shape/device/finite differs")
    evidence = attest_kv_cache(
        getattr(output, "past_key_values", None),
        token_count=len(identity_ids))
    evidence.update({
        "input_ids_sha256": _sha256_bytes(b"".join(
            int(value).to_bytes(8, "little", signed=True)
            for value in identity_ids)),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_device": str(logits.device),
    })
    return evidence


def _attention_backend_attestation(config) -> dict[str, str]:
    observed = {
        "_attn_implementation":
            str(getattr(config, "_attn_implementation", None)),
        "_attn_implementation_internal":
            str(getattr(config, "_attn_implementation_internal", None)),
    }
    _require(set(observed.values()) == {ATTENTION_BACKEND},
             f"p01 loaded attention backend differs: {observed}")
    return observed


def _model_config_attestation(model, *, snapshot: Path) -> dict[str, Any]:
    _require(type(model).__name__ == ARCHITECTURE,
             f"p01 loaded model class differs: {type(model).__name__}")
    config = getattr(model, "config", None)
    _require(config is not None and
             getattr(config, "model_type", None) == MODEL_TYPE,
             "p01 loaded model config/model_type differs")
    name_or_path = getattr(config, "_name_or_path", None)
    _require(isinstance(name_or_path, str) and
             Path(name_or_path).resolve() == snapshot.resolve(),
             "p01 loaded model path differs from exact snapshot")
    commit_hash = getattr(config, "_commit_hash", None)
    _require(commit_hash in (None, REVISION),
             f"p01 loaded model commit differs: {commit_hash}")
    observed = {
        "layers": int(getattr(config, "num_hidden_layers", -1)),
        "attention_heads": int(getattr(config, "num_attention_heads", -1)),
        "kv_heads": int(getattr(config, "num_key_value_heads", -1)),
        "head_dim": int(getattr(config, "head_dim", -1)),
        "hidden_size": int(getattr(config, "hidden_size", -1)),
        "experts_per_layer": int(getattr(config, "num_experts", -1)),
        "moe_intermediate_size": int(
            getattr(config, "moe_intermediate_size", -1)),
        "vocab_size": int(getattr(config, "vocab_size", -1)),
        "rope_theta": float(getattr(config, "rope_theta", -1)),
    }
    expected = asdict(EXACT_TOPOLOGY)
    _require(observed == expected,
             f"p01 loaded model geometry differs: {observed}")
    backend_fields = _attention_backend_attestation(config)
    attention_rows = []
    for name, module in model.named_modules():
        if type(module).__name__ != "Qwen3MoeAttention":
            continue
        index = getattr(module, "layer_idx", None)
        _require(isinstance(index, int) and
                 getattr(module.config, "_attn_implementation", None) ==
                 ATTENTION_BACKEND,
                 f"p01 attention layer backend/index differs: {name}")
        attention_rows.append({"layer": index, "name": name,
                               "class": type(module).__name__})
    _require([row["layer"] for row in attention_rows] ==
             list(range(EXACT_TOPOLOGY.layers)),
             "p01 loaded attention layer coverage/order differs")
    return {
        "class": type(model).__name__,
        "model_type": MODEL_TYPE,
        "geometry": observed,
        "attention_backend": ATTENTION_BACKEND,
        "attention_backend_fields": backend_fields,
        "attention_layers": attention_rows,
        "requested_revision": REVISION,
        "loaded_config_commit": commit_hash,
        "snapshot_path": str(snapshot.resolve()),
    }


def _eos_attestation(model, tokenizer) -> dict[str, Any]:
    def as_set(value) -> set[int]:
        if isinstance(value, int):
            return {int(value)}
        if isinstance(value, (list, tuple, set)):
            return {int(item) for item in value}
        return set()

    model_eos = as_set(getattr(model.config, "eos_token_id", None))
    tokenizer_eos = as_set(getattr(tokenizer, "eos_token_id", None))
    generation_eos = as_set(getattr(
        getattr(model, "generation_config", None), "eos_token_id", None))
    _require(model_eos == tokenizer_eos == {151_645},
             f"p01 primary EOS metadata differs: "
             f"{model_eos}/{tokenizer_eos}")
    _require(generation_eos == {151_643, 151_645},
             f"p01 generation EOS metadata differs: {generation_eos}")
    _require(model_eos.issubset(generation_eos),
             "p01 primary EOS is not in generation EOS set")
    return {
        "eos_ids": sorted(generation_eos),
        "model_config": sorted(model_eos),
        "tokenizer_primary": sorted(tokenizer_eos),
        "generation_config": sorted(generation_eos),
        "policy": ("generation_config full set is authoritative; model and "
                   "tokenizer primary EOS must be members"),
    }


def _load_subject(regime: str, *, snapshot: Path):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": ATTENTION_BACKEND,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
        "output_loading_info": True,
    }
    load_binding: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": "torch.bfloat16",
        "attn_implementation": ATTENTION_BACKEND,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
    }
    if regime == "nf4":
        # Importing BitsAndBytesConfig does not import bitsandbytes itself; the
        # latter remains lazy in Transformers and in our post-load attestation.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8,
            llm_int8_skip_modules=["lm_head"],
            llm_int8_enable_fp32_cpu_offload=False,
        )
        kwargs["quantization_config"] = quantization_config
        load_binding["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "torch.bfloat16",
            "bnb_4bit_quant_storage": "torch.uint8",
            "llm_int8_skip_modules": ["lm_head"],
            "llm_int8_enable_fp32_cpu_offload": False,
        }
    try:
        loaded = AutoModelForCausalLM.from_pretrained(str(snapshot), **kwargs)
    except Exception as exc:
        raise PrecisionProbeLoaderError(
            f"p01 {regime} exact model load failed: {exc}") from exc
    _require(isinstance(loaded, tuple) and len(loaded) == 2,
             "p01 Transformers did not return model loading information")
    model, loading_info = loaded
    model.eval()
    model.requires_grad_(False)
    _require(not model.training and
             all(not module.training for module in model.modules()) and
             all(not parameter.requires_grad for parameter in model.parameters()),
             "p01 loaded model is not completely frozen/eval")
    return model, loading_info, load_binding


def prepare_precision_subject(regime: str, repo: Path,
                              allow_download: bool) -> dict[str, Any]:
    """Load and fully attest one p01 regime.

    Parameters are positional by design because the runner-facing interface was
    frozen verbatim.  ``regime`` must be ``"nf4"`` or ``"bf16"``.
    """
    _require(regime in {"nf4", "bf16"},
             f"unknown p01 regime: {regime!r}")
    _require(type(allow_download) is bool,
             "p01 allow_download must be a literal bool")
    repository = repository_binding(Path(repo))
    dependencies = dependency_fingerprint()
    host = host_fingerprint()
    snapshot = resolve_pinned_snapshot(allow_download=allow_download)
    checkpoint = checkpoint_attestation(snapshot, require_weights=True)
    g0 = g0_static_meta_compatibility(snapshot)
    _require(g0["topology"]["checkpoint_key_names_sha256"] ==
             checkpoint["checkpoint_key_names_sha256"],
             "p01 G0/G1 checkpoint-key binding differs")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        use_fast=False)
    tokenizer_attestation, identity_ids = _attest_tokenizer(
        tokenizer, snapshot=snapshot)
    model, loading_info, load_binding = _load_subject(
        regime, snapshot=snapshot)
    loading = _loading_info_attestation(loading_info)
    model_config = _model_config_attestation(model, snapshot=snapshot)
    eos = _eos_attestation(model, tokenizer)
    placement = _module_device_attestation(model, expected_device="cuda:0")
    if regime == "nf4":
        weights = attest_nf4_coverage(
            model, snapshot=snapshot,
            weight_map=checkpoint["weight_map"])
    else:
        weights = attest_bf16_coverage(model)
    kv = _real_forward_attestation(model, identity_ids)

    checkpoint_public = {key: value for key, value in checkpoint.items()
                         if key not in {"checkpoint_keys", "weight_map"}}
    bindings = {
        "schema": "precision_probe_p01_subject_bindings_v1",
        "protocol_id": PROTOCOL_ID,
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "component_reuse_does_not_inherit_v12_eligibility": True,
        "regime": regime,
        "repository": repository,
        "dependencies": dependencies,
        "host": host,
        "checkpoint": checkpoint_public,
        "g0": g0,
        "tokenizer": tokenizer_attestation,
        "load": load_binding,
        "loading_info": loading,
        "model": model_config,
        "eos": eos,
        "placement": placement,
        "weights": weights,
        "kv": kv,
    }
    bindings_sha = _sha256_bytes(_canonical(bindings))
    runtime_fingerprint = {
        "schema": "precision_probe_p01_runtime_fingerprint_v1",
        "protocol_id": PROTOCOL_ID,
        "regime": regime,
        "model_id": MODEL_ID,
        "requested_revision": REVISION,
        "resolved_snapshot": snapshot.name,
        "architecture": ARCHITECTURE,
        "attention_backend": ATTENTION_BACKEND,
        "weight_runtime": ("bitsandbytes-nf4-double-quant-bf16-compute"
                           if regime == "nf4" else "bf16"),
        "kv_dtype": "torch.bfloat16",
        "eos_ids": eos["eos_ids"],
        "geometry": asdict(EXACT_TOPOLOGY),
        "repository_commit": repository["repository_commit"],
        "dependency_versions": dependencies["packages"],
        "gpu_uuid": host["gpu_uuid"],
        "bindings_sha256": bindings_sha,
    }
    runtime_fingerprint["fingerprint_sha256"] = _sha256_bytes(
        _canonical(runtime_fingerprint))
    return {
        "model": model,
        "tokenizer": tokenizer,
        "runtime_fingerprint": runtime_fingerprint,
        "bindings": bindings,
    }
