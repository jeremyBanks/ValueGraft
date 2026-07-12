"""Stage-T-gated loader and attestation for the powered-v13 exact subject.

The public entry point has no caller-controlled model, revision, release policy,
hash, clock, or load callback.  It verifies the fixed Stage-T checkout first,
then the exact environment and one A100, and only then resolves or loads model
bytes.  Historical v12 code supplies low-level snapshot and Transformers-5 MoE
facts; it is never allowed to authorize this v13 subject.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import threading
from typing import Any
import weakref


DESIGN_ID = "coherent-state-powered-successor-v13"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
SUBJECT_CONTRACT_PATH = Path(
    "data/coherent_state_powered_v13/subject-contract-v1.json"
)
LEGACY_MODEL_BITS_PATH = Path(
    "data/coherent_canary_v12/model_snapshot_contract.json"
)
STAGE_T_INVENTORY_CONTRACT_PATH = Path(
    "data/coherent_state_powered_v13/technical-canary-inventory-contract.json"
)

_DEPENDENCIES = {
    "python": "3.12.11",
    "uv": "0.9.18",
    "distributions": {
        "accelerate": "1.14.0",
        "huggingface-hub": "1.22.0",
        "safetensors": "0.8.0",
        "sentencepiece": "0.2.1",
        "tokenizers": "0.22.2",
        "torch": "2.12.1",
        "transformers": "5.0.0",
    },
    "lock_files": {
        "pyproject.toml":
            "b4e465c2714a5399904ed7d81f17d058fced6077a8f3f4de21ef9ee3adafe240",
        "uv.lock":
            "09d18804c909ad24be39766f0bc58cc8e6f6d306e27ef9ea10a441c5f380e3fc",
    },
}
_CUDA_CONTRACT = {
    "gpu_name": "NVIDIA A100 80GB PCIe",
    "memory_mib": 81920,
    "torch_total_memory_bytes": 85093777408,
    "compute_capability": [8, 0],
    "driver_minimum": [580, 65],
    "torch_version": "2.12.1+cu130",
    "torch_cuda_version": "13.0",
    "cudnn_version": 92000,
}
_SUBJECT = {
    "key": "exact-subject",
    "model_id": MODEL_ID,
    "revision": MODEL_REVISION,
    "architecture": "Qwen3MoeForCausalLM",
    "model_type": "qwen3_moe",
    "geometry": {
        "layers": 48,
        "attention_heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "rope_theta": 10_000_000.0,
    },
    "moe": {
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "decoder_sparse_step": 1,
        "tie_word_embeddings": False,
    },
    "parameter_count": 30_532_122_624,
    "weight_tensor_count": 18_867,
    "weight_name_to_shard_sha256":
        "1fb6383028ceb4aa49d9a2f13f3345dbc55781b041580fe891fea7a81e8e89e2",
}
_LOAD_POLICY = {
    "local_files_only": True,
    "trust_remote_code": False,
    "dtype": "bfloat16",
    "attention_backend": "eager",
    "low_cpu_mem_usage": True,
    "device_map": "cuda:0",
    "output_loading_info": True,
}
_FACTUAL_CODE = {
    "src/coherent_canary_loader.py":
        "ddc98ce1dc59a89f41ae4bf1ee40af87f26787e83dc48045c382078eddf08c3a",
    "src/coherent_canary_preflight.py":
        "94735df650135f976dba59748d16a0bceed3543d566a64549d090b156cb3a6f8",
    "src/coherent_state_tokens.py":
        "b6a052bcbe451084fb48ee203c8460f29c7b4a2ad3575287bd3c8771a5f65d15",
}
_LEGACY_BINDING = {
    "path": LEGACY_MODEL_BITS_PATH.as_posix(),
    "file_sha256":
        "ee05d45a55bb75e810897c1590a23bb0d3edfbf8ec8add05215c54edaa9a8677",
    "schema": "coherent_state_decision_canary_v12_model_snapshot_contract_v1",
    "design_id": "coherent-state-decision-canary-v12",
    "exact_subject_entry_sha256":
        "834d8ac0d533c0177d7b8e6d1150808098339d3b51f271d1cab3eb83b16161d2",
    "protocol_tokenizer_entry_sha256":
        "fd3cec807d09dea54aaaa99295e6291edc9f36ef407a4ae08796115cef79a570",
}
_CONVERSION_RECIPE_SHA256 = (
    "3ce851414b649d9eff93aaa302ab4801ebefcd931fcb0e8e5997491278500efa"
)
_SENTINEL_POSITIONS = [[0, 0], [24, 64], [47, 127]]
_SENTINEL_POSITIONS_SHA256 = (
    "e2dc589a861c2ffb3bd03ade3db6b0c31253b55ebe5e55db9f0f8a7e6bfedd3f"
)
_TOKENIZER = {
    "tokenizer_class":
        "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer",
    "backend_class": "tokenizers.Tokenizer",
    "backend_serialization_sha256":
        "41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04",
    "vocab_sha256":
        "f488fa45d324a8bc64c84f0e27b47223872d550f2a6900564f77dfa67ca5ff4d",
    "chat_template_sha256":
        "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326",
    "length": 151669,
    "vocab_size": 151643,
    "eos_token": "<|im_end|>",
    "eos_token_id": 151645,
    "pad_token_id": 151643,
    "all_special_ids": [
        151645, 151643, 151644, 151646, 151647, 151648, 151649,
        151650, 151651, 151652, 151653, 151654, 151655, 151656,
    ],
    "special_token_ids": {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
    },
    "files": {
        "merges.txt": {
            "sha256":
                "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
            "size_bytes": 1_671_839,
        },
        "tokenizer.json": {
            "sha256":
                "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            "size_bytes": 11_422_654,
        },
        "tokenizer_config.json": {
            "sha256":
                "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
            "size_bytes": 9_377,
        },
        "vocab.json": {
            "sha256":
                "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
            "size_bytes": 2_776_833,
        },
    },
}
_REQUIRED_STAGE_T_PATHS = frozenset({
    "COHERENT-STATE-POWERED-SUCCESSOR-V13-PREREGISTRATION.md",
    STAGE_T_INVENTORY_CONTRACT_PATH.as_posix(),
    SUBJECT_CONTRACT_PATH.as_posix(),
    LEGACY_MODEL_BITS_PATH.as_posix(),
    "pyproject.toml",
    "uv.lock",
    "src/powered_v13_release.py",
    "src/powered_v13_subject.py",
    *_FACTUAL_CODE,
})
_FORBIDDEN_IMPORT_STEMS = (
    "powered_v13_pool", "powered_v13_recipe", "powered_v13_permutation",
)
_GPU_UUID_RE = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z"
)


class PoweredV13SubjectError(RuntimeError):
    """The Stage-T release, environment, snapshot, or loaded subject differed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PoweredV13SubjectError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PoweredV13SubjectError(
            f"attestation is not canonical-JSON safe: {exc}"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise PoweredV13SubjectError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoweredV13SubjectError(f"{label} is not UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root is not an object")
    return value


def _regular_file(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is absent, non-regular, or a symlink")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PoweredV13SubjectError(f"cannot read {label}: {exc}") from exc


def _safe_relative_path(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} is not a path")
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute() and pure.as_posix() == value and pure.parts
        and all(part not in {"", ".", ".."} for part in pure.parts)
        and pure.parts[0] != ".git",
        f"{label} is not a safe normalized repository path",
    )
    return value


def _run_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        raise PoweredV13SubjectError(
            f"command could not start ({command[0]}): {exc}"
        ) from exc
    if result.returncode != 0:
        raise PoweredV13SubjectError(
            f"command failed ({command[0]}, {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _git(
    repo: Path, *args: str, check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run Git against ``repo`` without ambient repository redirection.

    Git honors a broad family of ``GIT_*`` variables that can redirect the
    repository, work tree, index, object database, namespace, replacements,
    and pathspec behavior even when ``cwd`` names the intended checkout.  This
    loader's pre/post-load boundary must describe the passed checkout itself.
    """
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_LITERAL_PATHSPECS"] = "1"
    try:
        result = subprocess.run(
            ["git", *args], cwd=Path(repo), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        raise PoweredV13SubjectError(f"git invocation failed: {exc}") from exc
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise PoweredV13SubjectError(
            f"git {' '.join(args)} failed with {result.returncode}: {detail}"
        )
    return result


def _git_text(repo: Path, *args: str) -> str:
    try:
        return _git(repo, *args).stdout.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise PoweredV13SubjectError("git output is not UTF-8") from exc


def _fixed_stage_t_release(repo: Path, receipt_directory: Path) -> dict[str, Any]:
    """Adapt the fixed receipt to the fixed release verifier, with no override."""
    import powered_v13_release as release

    _require(Path(release.__file__).resolve() ==
             Path(repo).resolve() / "src/powered_v13_release.py",
             "executing Stage-T release module is not the inventoried repo file")
    _require(release.DESIGN_ID == DESIGN_ID,
             "Stage-T release module design identity differs")
    receipt_directory = Path(receipt_directory).resolve()
    receipt_path = receipt_directory / release.STAGE_T_RECEIPT_BASENAME
    raw = _regular_file(receipt_path, "fixed Stage-T launch receipt")
    receipt = _json_object(raw, "fixed Stage-T launch receipt")
    _require(raw == release.canonical_json_bytes(receipt) + b"\n",
             "fixed Stage-T launch receipt is not canonical JSON plus LF")
    _require(receipt.get("design_id") == DESIGN_ID and
             receipt.get("stage") == release.STAGE_T,
             "fixed Stage-T receipt identity differs")
    authorization_commit = receipt.get("authorization_commit")
    manifest_path = receipt.get("manifest_path")
    _require(isinstance(authorization_commit, str) and
             re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", authorization_commit),
             "fixed Stage-T receipt authorization commit is malformed")
    manifest_path = _safe_relative_path(manifest_path, "Stage-T manifest path")

    # No clock, tolerance, spec, or expected-hash capability reaches this call.
    evidence = release.verify_stage_t_checkout(
        Path(repo), authorization_commit=authorization_commit,
        manifest_path=manifest_path, receipt_directory=receipt_directory,
    )
    _require(isinstance(evidence, Mapping),
             "Stage-T checkout verifier returned no evidence")
    result = dict(evidence)
    authorization = result.get("authorization")
    _require(isinstance(authorization, Mapping),
             "Stage-T authorization evidence is absent")
    static_root = authorization.get("static_root_commit")
    _require(isinstance(static_root, str) and
             re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", static_root),
             "Stage-T static root is malformed")
    contract_path = _safe_relative_path(
        release.STAGE_T_INVENTORY_CONTRACT_PATH,
        "Stage-T inventory-contract path",
    )
    inventory_raw = _git(
        Path(repo), "cat-file", "blob", f"{static_root}:{contract_path}"
    )
    inventory_doc = _json_object(
        inventory_raw.stdout, "Stage-T inventory contract"
    )
    _require(inventory_raw.stdout ==
             release.canonical_json_bytes(inventory_doc) + b"\n",
             "Stage-T inventory contract is not canonical JSON plus LF")
    paths = inventory_doc.get("inventory_paths")
    _require(isinstance(paths, list) and paths == sorted(set(paths)) and
             all(isinstance(path, str) for path in paths),
             "Stage-T inventory paths are not sorted unique strings")
    result["subject_inventory_paths"] = paths
    return result


def _probe_dependencies(repo: Path) -> dict[str, Any]:
    distributions: dict[str, str] = {}
    for name in _DEPENDENCIES["distributions"]:
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise PoweredV13SubjectError(
                f"locked runtime distribution is absent: {name}"
            ) from exc
    uv_output = _run_text(["uv", "--version"]).strip().split()
    _require(len(uv_output) >= 2 and uv_output[0] == "uv",
             "uv version output is malformed")
    locks = {
        relative: _sha256_file(Path(repo) / relative)
        for relative in _DEPENDENCIES["lock_files"]
    }
    return {
        "python": platform.python_version(),
        "uv": uv_output[1],
        "distributions": distributions,
        "lock_files": locks,
    }


@dataclass(frozen=True, slots=True)
class _Runtime:
    torch: Any
    auto_model: Any
    auto_tokenizer: Any
    factual: Any
    runtime_fingerprint: Callable[..., dict[str, Any]]


def _load_runtime() -> _Runtime:
    # This is deliberately lazy: importing the historical factual module also
    # imports torch, and therefore may only occur after Stage-T and lock checks.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import coherent_canary_loader as factual
    from coherent_canary_preflight import runtime_fingerprint

    return _Runtime(
        torch=torch, auto_model=AutoModelForCausalLM,
        auto_tokenizer=AutoTokenizer, factual=factual,
        runtime_fingerprint=runtime_fingerprint,
    )


def _attest_runtime_origins(runtime: _Runtime, repo: Path) -> dict[str, str]:
    expected = {
        "subject": Path(repo) / "src/powered_v13_subject.py",
        "factual": Path(repo) / "src/coherent_canary_loader.py",
        "preflight": Path(repo) / "src/coherent_canary_preflight.py",
    }
    preflight = sys.modules.get(runtime.runtime_fingerprint.__module__)
    observed = {
        "subject": Path(__file__).resolve(),
        "factual": Path(str(getattr(runtime.factual, "__file__", ""))).resolve(),
        "preflight": Path(str(getattr(preflight, "__file__", ""))).resolve(),
    }
    _require(observed == {key: value.resolve() for key, value in expected.items()},
             f"executing subject/factual module origins differ: {observed}")
    return {key: str(value) for key, value in observed.items()}


def _nvidia_smi() -> str:
    return _run_text([
        "nvidia-smi",
        "--query-gpu=uuid,driver_version,name,memory.total",
        "--format=csv,noheader,nounits",
    ])


@dataclass(frozen=True, slots=True)
class _ContractBundle:
    subject_contract: Mapping[str, Any]
    legacy_contract: Mapping[str, Any]
    subject_contract_sha256: str
    legacy_contract_sha256: str


def _load_contracts(repo: Path) -> _ContractBundle:
    subject_raw = _regular_file(
        Path(repo) / SUBJECT_CONTRACT_PATH, "v13 subject contract"
    )
    subject = _json_object(subject_raw, "v13 subject contract")
    _require(subject_raw == _canonical(subject) + b"\n",
             "v13 subject contract is not canonical JSON plus LF")
    expected_fields = {
        "schema", "design_id", "subject", "tokenizer", "cuda",
        "dependencies", "load_policy", "legacy_model_bits", "factual_code",
        "topology",
    }
    _require(set(subject) == expected_fields and
             subject.get("schema") == "powered-v13-exact-subject-contract-v1" and
             subject.get("design_id") == DESIGN_ID,
             "v13 subject contract identity or fields differ")
    _require(subject.get("subject") == _SUBJECT,
             "v13 exact-subject selector/geometry differs")
    _require(subject.get("tokenizer") == _TOKENIZER,
             "v13 tokenizer contract differs")
    _require(subject.get("cuda") == _CUDA_CONTRACT,
             "v13 CUDA contract differs")
    _require(subject.get("dependencies") == _DEPENDENCIES,
             "v13 dependency contract differs")
    _require(subject.get("load_policy") == _LOAD_POLICY,
             "v13 load policy differs")
    _require(subject.get("legacy_model_bits") == _LEGACY_BINDING,
             "v13 legacy model-bit binding differs")
    _require(subject.get("factual_code") == _FACTUAL_CODE,
             "v13 factual-code binding differs")
    topology = subject.get("topology")
    _require(topology == {
        "representation": "transformers-5-qwen2-style-moe-packed",
        "conversion_recipe_sha256": _CONVERSION_RECIPE_SHA256,
        "content_sentinel_positions": _SENTINEL_POSITIONS,
        "content_sentinel_positions_sha256": _SENTINEL_POSITIONS_SHA256,
    }, "v13 Transformers-5 topology contract differs")
    for relative, digest in _FACTUAL_CODE.items():
        path = Path(repo) / relative
        _regular_file(path, f"factual code {relative}")
        _require(_sha256_file(path) == digest,
                 f"factual code hash differs: {relative}")

    legacy_raw = _regular_file(
        Path(repo) / LEGACY_MODEL_BITS_PATH, "historical model-bit contract"
    )
    _require(_sha256_bytes(legacy_raw) == _LEGACY_BINDING["file_sha256"],
             "historical model-bit contract hash differs")
    legacy = _json_object(legacy_raw, "historical model-bit contract")
    _require(legacy.get("schema") == _LEGACY_BINDING["schema"] and
             legacy.get("design_id") == _LEGACY_BINDING["design_id"],
             "historical model-bit contract identity differs")
    exact_entry = legacy.get("subjects", {}).get("exact-subject")
    tokenizer_entry = legacy.get("protocol_tokenizer")
    _require(isinstance(exact_entry, Mapping) and
             _sha256_bytes(_canonical(exact_entry)) ==
             _LEGACY_BINDING["exact_subject_entry_sha256"],
             "historical exact-subject entry differs")
    _require(isinstance(tokenizer_entry, Mapping) and
             _sha256_bytes(_canonical(tokenizer_entry)) ==
             _LEGACY_BINDING["protocol_tokenizer_entry_sha256"],
             "historical tokenizer entry differs")
    return _ContractBundle(
        subject_contract=subject, legacy_contract=legacy,
        subject_contract_sha256=_sha256_bytes(subject_raw),
        legacy_contract_sha256=_sha256_bytes(legacy_raw),
    )


def _recheck_stage_t_binding(
    repo: Path, release: Mapping[str, Any], bundle: _ContractBundle,
) -> dict[str, Any]:
    """Recheck immutable checkout/contract state without refreshing receipt age."""
    authorization = release.get("authorization")
    _require(isinstance(authorization, Mapping),
             "post-load Stage-T authorization evidence is absent")
    expected_head = authorization.get("authorization_commit")
    head = _git_text(repo, "rev-parse", "HEAD").strip()
    _require(head == expected_head,
             "Stage-T checkout HEAD changed during subject load")
    symbolic = _git(repo, "symbolic-ref", "-q", "HEAD", check=False)
    _require(symbolic.returncode == 1,
             "Stage-T checkout stopped being detached during subject load")
    status = _git(
        repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    _require(not status.stdout,
             "Stage-T checkout changed or became dirty during subject load")
    after = _load_contracts(repo)
    _require(after == bundle,
             "subject/model-bit contracts changed during subject load")
    return {
        "status": "PASS",
        "detached_head": head,
        "clean_tree": True,
        "subject_contract_sha256": after.subject_contract_sha256,
        "legacy_model_bits_sha256": after.legacy_contract_sha256,
    }


@dataclass(frozen=True, slots=True)
class _SubjectSeams:
    verify_release: Callable[[Path, Path], Mapping[str, Any]]
    probe_dependencies: Callable[[Path], Mapping[str, Any]]
    load_runtime: Callable[[], _Runtime]
    nvidia_smi: Callable[[], str]
    load_contracts: Callable[[Path], _ContractBundle]
    recheck_release: Callable[
        [Path, Mapping[str, Any], _ContractBundle], Mapping[str, Any]
    ]


_PRODUCTION_SEAMS = _SubjectSeams(
    verify_release=_fixed_stage_t_release,
    probe_dependencies=_probe_dependencies,
    load_runtime=_load_runtime,
    nvidia_smi=_nvidia_smi,
    load_contracts=_load_contracts,
    recheck_release=_recheck_stage_t_binding,
)


def _validate_release(evidence: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(evidence, Mapping), "Stage-T evidence is absent")
    authorization = evidence.get("authorization")
    receipt = evidence.get("receipt")
    paths = evidence.get("subject_inventory_paths")
    _require(evidence.get("status") == "PASS" and
             evidence.get("design_id") == DESIGN_ID and
             evidence.get("stage") == "TECHNICAL_CANARY" and
             evidence.get("clean_tree") is True,
             "fixed Stage-T checkout did not pass")
    _require(isinstance(authorization, Mapping) and
             evidence.get("detached_head") ==
             authorization.get("authorization_commit"),
             "Stage-T detached HEAD/authorization binding differs")
    _require(isinstance(receipt, Mapping) and
             isinstance(receipt.get("receipt_sha256"), str) and
             re.fullmatch(r"[0-9a-f]{64}", receipt["receipt_sha256"]),
             "Stage-T receipt evidence differs")
    _require(isinstance(paths, list) and
             _REQUIRED_STAGE_T_PATHS.issubset(set(paths)),
             "Stage-T inventory omits exact-subject loader dependencies")
    return dict(evidence)


def _validate_dependencies(evidence: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "python": evidence.get("python"),
        "uv": evidence.get("uv"),
        "distributions": evidence.get("distributions"),
        "lock_files": evidence.get("lock_files"),
    }
    _require(observed == _DEPENDENCIES,
             f"locked dependency environment differs: {observed}")
    return observed


def _parse_driver(value: str) -> list[int]:
    parts = value.split(".")
    _require(len(parts) >= 2 and all(part.isdigit() for part in parts),
             "NVIDIA driver version is malformed")
    return [int(part) for part in parts]


def _attest_cuda(runtime: _Runtime, nvidia_output: str) -> dict[str, Any]:
    rows = [row.strip() for row in nvidia_output.splitlines() if row.strip()]
    _require(len(rows) == 1, "host does not expose exactly one NVIDIA GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    _require(len(fields) == 4, "nvidia-smi GPU row is malformed")
    uuid, driver, gpu_name, memory_text = fields
    _require(_GPU_UUID_RE.fullmatch(uuid) is not None,
             "NVIDIA GPU UUID is malformed")
    _require(gpu_name == _CUDA_CONTRACT["gpu_name"],
             f"NVIDIA GPU class differs: {gpu_name}")
    _require(memory_text.isdigit() and
             int(memory_text) == _CUDA_CONTRACT["memory_mib"],
             f"NVIDIA GPU memory differs: {memory_text}")
    driver_parts = _parse_driver(driver)
    _require(tuple(driver_parts[:2]) >=
             tuple(_CUDA_CONTRACT["driver_minimum"]),
             f"NVIDIA driver is below the CUDA-13 minimum: {driver}")

    torch = runtime.torch
    _require(str(getattr(torch, "__version__", "")) ==
             _CUDA_CONTRACT["torch_version"],
             f"Torch CUDA build differs: {getattr(torch, '__version__', None)}")
    _require(str(getattr(getattr(torch, "version", None), "cuda", None)) ==
             _CUDA_CONTRACT["torch_cuda_version"],
             "Torch CUDA runtime version differs")
    cuda = getattr(torch, "cuda", None)
    _require(cuda is not None and cuda.is_available(),
             "Torch CUDA is unavailable")
    _require(cuda.device_count() == 1 and cuda.current_device() == 0,
             "Torch does not expose exactly CUDA device 0")
    properties = cuda.get_device_properties(0)
    torch_name = str(cuda.get_device_name(0))
    torch_memory = int(getattr(properties, "total_memory", -1))
    capability = [int(getattr(properties, "major", -1)),
                  int(getattr(properties, "minor", -1))]
    _require(torch_name == gpu_name == _CUDA_CONTRACT["gpu_name"],
             "Torch and nvidia-smi GPU names differ")
    _require(torch_memory == _CUDA_CONTRACT["torch_total_memory_bytes"],
             f"Torch CUDA memory differs: {torch_memory}")
    _require(capability == _CUDA_CONTRACT["compute_capability"],
             f"CUDA compute capability differs: {capability}")
    cudnn = int(torch.backends.cudnn.version())
    _require(cudnn == _CUDA_CONTRACT["cudnn_version"],
             f"cuDNN version differs: {cudnn}")
    return {
        "gpu_uuid": uuid,
        "driver_version": driver,
        "driver_components": driver_parts,
        "gpu_name": gpu_name,
        "memory_mib": int(memory_text),
        "torch_total_memory_bytes": torch_memory,
        "compute_capability": capability,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": cudnn,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _validate_contract_dependency_binding(bundle: _ContractBundle) -> None:
    _require(bundle.subject_contract.get("dependencies") == _DEPENDENCIES,
             "subject contract does not bind observed dependency lock")
    _require(bundle.subject_contract.get("cuda") == _CUDA_CONTRACT,
             "subject contract does not bind CUDA admission")


def _resolve_snapshot(runtime: _Runtime) -> Path:
    try:
        snapshot = Path(runtime.factual.resolve_pinned_snapshot(
            MODEL_ID, MODEL_REVISION, local_files_only=False,
        ))
    except Exception as exc:
        raise PoweredV13SubjectError(
            f"exact subject revision resolution failed: {exc}"
        ) from exc
    _require(snapshot.name == MODEL_REVISION,
             "resolved snapshot revision differs")
    try:
        runtime.factual.validate_snapshot_location(
            snapshot, MODEL_REVISION, repo_id=MODEL_ID,
        )
    except Exception as exc:
        raise PoweredV13SubjectError(
            f"resolved snapshot location differs: {exc}"
        ) from exc
    return snapshot


def _inventory(runtime: _Runtime, snapshot: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        model = runtime.factual.inventory_snapshot(
            snapshot, revision=MODEL_REVISION, kind="model",
            spec=runtime.factual.EXACT_SPEC,
        )
        tokenizer = runtime.factual.inventory_snapshot(
            snapshot, revision=MODEL_REVISION, kind="protocol_tokenizer",
            repo_id=MODEL_ID,
        )
    except Exception as exc:
        raise PoweredV13SubjectError(f"snapshot inventory failed: {exc}") from exc
    _require(isinstance(model, dict) and isinstance(tokenizer, dict),
             "snapshot inventory evidence is absent")
    return model, tokenizer


def _verify_inventories(
    model: Mapping[str, Any], tokenizer: Mapping[str, Any], *,
    snapshot: Path, bundle: _ContractBundle,
) -> dict[str, Any]:
    exact_entry = bundle.legacy_contract.get("subjects", {}).get("exact-subject")
    tokenizer_entry = bundle.legacy_contract.get("protocol_tokenizer")
    _require(isinstance(exact_entry, Mapping) and
             isinstance(tokenizer_entry, Mapping),
             "model-bit contract entries are absent")
    snapshot_path = str(snapshot.resolve())
    _require(
        model.get("kind") == "model" and model.get("repo_id") == MODEL_ID
        and model.get("revision") == MODEL_REVISION
        and str(Path(str(model.get("snapshot_path", ""))).resolve()) == snapshot_path,
        "model inventory selector/path differs",
    )
    _require(
        tokenizer.get("kind") == "protocol_tokenizer"
        and tokenizer.get("repo_id") == MODEL_ID
        and tokenizer.get("revision") == MODEL_REVISION
        and str(Path(str(tokenizer.get("snapshot_path", ""))).resolve()) ==
        snapshot_path,
        "tokenizer inventory selector/path differs",
    )
    _require(model.get("files") == exact_entry.get("files") and
             model.get("inventory_sha256") ==
             exact_entry.get("inventory_sha256"),
             "bf16 model file inventory differs from immutable contract")
    _require(tokenizer.get("files") == tokenizer_entry.get("files") and
             tokenizer.get("inventory_sha256") ==
             tokenizer_entry.get("inventory_sha256"),
             "tokenizer file inventory differs from immutable contract")
    tensors = model.get("weight_tensors")
    _require(isinstance(tensors, list) and
             len(tensors) == exact_entry.get("weight_tensor_count") ==
             bundle.subject_contract.get("subject", {}).get("weight_tensor_count"),
             "bf16 checkpoint tensor count differs")
    seen: set[str] = set()
    name_to_shard: list[dict[str, str]] = []
    parameter_count = 0
    for row in tensors:
        _require(isinstance(row, Mapping) and
                 isinstance(row.get("name"), str) and row.get("name") not in seen and
                 isinstance(row.get("shard"), str) and
                 row.get("dtype") == "BF16" and
                 isinstance(row.get("shape"), list) and row.get("shape") and
                 all(type(value) is int and value >= 1 for value in row["shape"]),
                 "bf16 checkpoint topology row differs")
        seen.add(row["name"])
        parameter_count += math.prod(row["shape"])
        name_to_shard.append({"name": row["name"], "shard": row["shard"]})
    name_to_shard.sort(key=lambda row: row["name"])
    map_sha = _sha256_bytes(_canonical(name_to_shard))
    _require(parameter_count == exact_entry.get("parameter_count") ==
             bundle.subject_contract.get("subject", {}).get("parameter_count"),
             "bf16 checkpoint parameter count differs")
    _require(map_sha == exact_entry.get("weight_name_to_shard_sha256") ==
             bundle.subject_contract.get("subject", {}).get(
                 "weight_name_to_shard_sha256"),
             "bf16 checkpoint name/shard map differs")
    return {
        "subject_contract_entry_sha256": _sha256_bytes(_canonical(exact_entry)),
        "tokenizer_contract_entry_sha256":
            _sha256_bytes(_canonical(tokenizer_entry)),
        "parameter_count": parameter_count,
        "weight_tensor_count": len(tensors),
        "weight_name_to_shard_sha256": map_sha,
    }


def _load_local(runtime: _Runtime, snapshot: Path) -> tuple[Any, Any, Mapping[str, Any]]:
    try:
        tokenizer = runtime.auto_tokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False,
        )
        loaded = runtime.auto_model.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False,
            dtype=runtime.torch.bfloat16, attn_implementation="eager",
            low_cpu_mem_usage=True, device_map="cuda:0",
            output_loading_info=True,
        )
    except Exception as exc:
        raise PoweredV13SubjectError(f"local exact-subject load failed: {exc}") from exc
    _require(isinstance(loaded, tuple) and len(loaded) == 2,
             "Transformers did not return model loading information")
    model, loading_info = loaded
    _require(isinstance(loading_info, Mapping),
             "Transformers model loading information is absent")
    try:
        model.eval()
        model.requires_grad_(False)
    except Exception as exc:
        raise PoweredV13SubjectError(
            f"cannot place exact subject in frozen evaluation mode: {exc}"
        ) from exc
    return model, tokenizer, loading_info


def _qualified_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _attest_loading_info(loading_info: Mapping[str, Any]) -> dict[str, list[Any]]:
    keys = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    _require(set(loading_info) == keys,
             f"model loading-info fields differ: {sorted(loading_info)}")
    result: dict[str, list[Any]] = {}
    for key in sorted(keys):
        value = loading_info[key]
        _require(isinstance(value, (list, tuple, set)) and not value,
                 f"model loading reported {key}: {value}")
        result[key] = []
    return result


def _config_value(config: Any, name: str) -> Any:
    return getattr(config, name, None)


def _attest_config_and_backend(model: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(type(model).__name__ == _SUBJECT["architecture"],
             f"loaded model class differs: {type(model).__name__}")
    model_config = getattr(model, "config", None)
    _require(model_config is not None, "loaded model config is absent")
    config = getattr(model_config, "text_config", model_config)
    rope_parameters = _config_value(config, "rope_parameters")
    rope_theta = (
        rope_parameters.get("rope_theta")
        if isinstance(rope_parameters, Mapping)
        else _config_value(config, "rope_theta")
    )
    observed = {
        "architecture": type(model).__name__,
        "architectures": _config_value(model_config, "architectures"),
        "model_type": _config_value(model_config, "model_type"),
        "layers": _config_value(config, "num_hidden_layers"),
        "attention_heads": _config_value(config, "num_attention_heads"),
        "kv_heads": _config_value(config, "num_key_value_heads"),
        "head_dim": _config_value(config, "head_dim"),
        "rope_theta": float(rope_theta) if rope_theta is not None else None,
        "hidden_size": _config_value(config, "hidden_size"),
        "moe_intermediate_size": _config_value(config, "moe_intermediate_size"),
        "num_experts": _config_value(config, "num_experts"),
        "decoder_sparse_step": _config_value(config, "decoder_sparse_step"),
        "tie_word_embeddings": _config_value(config, "tie_word_embeddings"),
    }
    expected = {
        "architecture": _SUBJECT["architecture"],
        "architectures": [_SUBJECT["architecture"]],
        "model_type": _SUBJECT["model_type"],
        **_SUBJECT["geometry"],
        **_SUBJECT["moe"],
    }
    _require(observed == expected,
             f"loaded exact-subject class/config/geometry differs: {observed}")
    commit_hash = _config_value(model_config, "_commit_hash")
    _require(commit_hash in (None, MODEL_REVISION),
             "loaded model config revision differs")
    for scope, value in (("model", model_config), ("text", config)):
        backends = {
            str(_config_value(value, "_attn_implementation")),
            str(_config_value(value, "_attn_implementation_internal")),
        }
        _require(backends == {"eager"},
                 f"{scope} attention backend differs: {backends}")
    attention_layers: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if (hasattr(module, "q_proj") and hasattr(module, "k_proj") and
                "Attention" in type(module).__name__):
            index = getattr(module, "layer_idx", None)
            local = getattr(module, "config", None)
            _require(type(index) is int and local is not None,
                     f"attention layer metadata is absent: {name}")
            _require(str(getattr(local, "_attn_implementation", None)) == "eager" and
                     str(getattr(local, "_attn_implementation_internal", None)) ==
                     "eager", f"attention layer backend differs: {name}")
            attention_layers.append({
                "layer": index, "name": name,
                "class": type(module).__name__, "backend": "eager",
            })
    _require([row["layer"] for row in attention_layers] == list(range(48)),
             "attention-layer eager coverage/order differs")
    return observed, attention_layers


def _attest_unquantized(model: Any) -> dict[str, Any]:
    config = model.config
    fields = {
        "config_quantization": getattr(config, "quantization_config", None),
        "is_quantized": bool(getattr(model, "is_quantized", False)),
        "hf_quantizer": getattr(model, "hf_quantizer", None),
        "quantization_method": getattr(model, "quantization_method", None),
        "load_in_4bit": bool(getattr(model, "load_in_4bit", False)),
        "load_in_8bit": bool(getattr(model, "load_in_8bit", False)),
    }
    _require(fields == {
        "config_quantization": None, "is_quantized": False,
        "hf_quantizer": None, "quantization_method": None,
        "load_in_4bit": False, "load_in_8bit": False,
    }, f"loaded exact subject is quantized: {fields}")
    forbidden = ("bitsandbytes", "gptq", "awq")
    _require(not any(
        any(token in _qualified_name(module).lower() for token in forbidden)
        for module in model.modules()
    ), "loaded exact subject contains a quantization module")
    return fields


def _attest_parameters(
    model: Any, *, runtime: _Runtime,
    model_inventory: Mapping[str, Any], bundle: _ContractBundle,
    snapshot: Path,
) -> dict[str, Any]:
    named = list(model.named_parameters(remove_duplicate=False))
    _require(bool(named) and len({name for name, _ in named}) == len(named),
             "loaded parameter names are empty or duplicated")
    parameters = [parameter for _, parameter in named]
    buffers = list(model.buffers())
    _require(not model.training and all(not module.training for module in model.modules()),
             "loaded exact subject is not entirely in evaluation mode")
    _require(all(parameter.is_floating_point() and
                 parameter.dtype == runtime.torch.bfloat16
                 for parameter in parameters),
             "loaded exact-subject parameter dtype differs from bf16")
    _require(all(not parameter.requires_grad for parameter in parameters),
             "loaded exact-subject parameter is trainable")
    _require(all(getattr(parameter.device, "type", None) != "meta"
                 for parameter in parameters) and
             all(getattr(buffer.device, "type", None) != "meta"
                 for buffer in buffers),
             "loaded exact subject contains a meta tensor")
    parameter_devices = {str(parameter.device) for parameter in parameters}
    buffer_devices = {str(buffer.device) for buffer in buffers}
    _require(parameter_devices == {"cuda:0"} and
             buffer_devices.issubset({"cuda:0"}),
             f"loaded exact-subject devices differ: "
             f"{parameter_devices}/{buffer_devices}")
    _require(getattr(model, "hf_device_map", None) == {"": "cuda:0"},
             f"loaded exact-subject device map differs: "
             f"{getattr(model, 'hf_device_map', None)}")
    adverse_hooks = []
    for name, module in model.named_modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        execution_raw = getattr(hook, "execution_device", None)
        execution = None if execution_raw is None else str(execution_raw)
        if (bool(getattr(hook, "offload", False)) or
                bool(getattr(hook, "offload_buffers", False)) or
                execution not in (None, "cuda", "cuda:0")):
            adverse_hooks.append({"module": name, "hook": type(hook).__name__})
    _require(not adverse_hooks, f"loaded exact subject has offload hooks: {adverse_hooks}")
    _require(not getattr(model, "offload_index", None) and
             not getattr(model, "offload_folder", None),
             "loaded exact subject has offload state")

    loaded_rows = sorted([{
        "name": name,
        "shape": [int(value) for value in parameter.shape],
        "dtype": "BF16" if parameter.dtype == runtime.torch.bfloat16
        else str(parameter.dtype),
    } for name, parameter in named], key=lambda row: row["name"])
    try:
        expected_rows, topology = runtime.factual.expected_loaded_parameter_topology(
            model_inventory["weight_tensors"], spec=runtime.factual.EXACT_SPEC,
        )
    except Exception as exc:
        raise PoweredV13SubjectError(
            f"Transformers-5 expected topology derivation failed: {exc}"
        ) from exc
    _require(loaded_rows == expected_rows,
             "loaded parameter topology differs from checkpoint conversion")
    loaded_count = sum(math.prod(row["shape"]) for row in loaded_rows)
    _require(loaded_count == bundle.subject_contract["subject"]["parameter_count"],
             "loaded parameter count differs from exact contract")
    try:
        conversion = runtime.factual.attest_weight_conversion_recipe(
            model, spec=runtime.factual.EXACT_SPEC,
        )
        sentinels = runtime.factual.attest_moe_content_sentinels(
            model, spec=runtime.factual.EXACT_SPEC, model_snapshot=snapshot,
            weight_tensors=model_inventory["weight_tensors"],
        )
    except Exception as exc:
        raise PoweredV13SubjectError(
            f"Transformers-5 conversion/content attestation failed: {exc}"
        ) from exc
    _require(isinstance(conversion, Mapping) and
             conversion.get("representation") ==
             "transformers-5-qwen2-style-moe-packed" and
             _sha256_bytes(_canonical(conversion.get("converters"))) ==
             _CONVERSION_RECIPE_SHA256,
             "Transformers-5 MoE conversion recipe differs")
    sentinel_rows = sentinels.get("sentinels") if isinstance(sentinels, Mapping) else None
    _require(isinstance(sentinel_rows, list) and len(sentinel_rows) == 9,
             "MoE content-sentinel coverage differs")
    positions = sorted({
        (row.get("layer"), row.get("expert"))
        for row in sentinel_rows if isinstance(row, Mapping)
    })
    _require(positions == [tuple(value) for value in _SENTINEL_POSITIONS] and
             all(row.get("projection") in {"gate_proj", "up_proj", "down_proj"}
                 and row.get("dtype") == "BF16" and
                 re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))
                 for row in sentinel_rows),
             "MoE content-sentinel identity/evidence differs")
    return {
        "parameter_count": loaded_count,
        "parameter_devices": sorted(parameter_devices),
        "buffer_devices": sorted(buffer_devices),
        "loaded_parameter_topology_sha256": _sha256_bytes(_canonical(loaded_rows)),
        "topology": topology,
        "conversion": dict(conversion),
        "content_sentinels": dict(sentinels),
    }


def _attest_tokenizer(
    tokenizer: Any, *, snapshot: Path,
    tokenizer_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    _require(_qualified_name(tokenizer) == _TOKENIZER["tokenizer_class"],
             f"tokenizer wrapper class differs: {_qualified_name(tokenizer)}")
    _require(Path(str(getattr(tokenizer, "name_or_path", ""))).resolve() ==
             snapshot.resolve(), "tokenizer runtime source differs")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    _require(isinstance(init_kwargs, Mapping),
             "tokenizer initialization provenance is absent")
    roots: set[Path] = set()
    for key, filename in (("vocab_file", "vocab.json"),
                          ("merges_file", "merges.txt")):
        value = init_kwargs.get(key)
        _require(isinstance(value, str) and Path(value).name == filename,
                 f"tokenizer {key} provenance differs")
        # HF snapshot files are normally symlinks into ``blobs``.  Preserve
        # the lexical snapshot parent here; snapshot inventory separately
        # verifies that every resolved target remains in the model cache.
        roots.add(Path(value).expanduser().absolute().parent)
    _require(roots == {snapshot.expanduser().absolute()},
             "tokenizer files do not originate in the resolved snapshot")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    _require(backend is not None and callable(getattr(backend, "to_str", None)) and
             _qualified_name(backend) == _TOKENIZER["backend_class"],
             "tokenizer backend class/serialization differs")
    backend_text = backend.to_str()
    _require(isinstance(backend_text, str) and backend_text and
             _sha256_bytes(backend_text.encode("utf-8")) ==
             _TOKENIZER["backend_serialization_sha256"],
             "tokenizer backend serialization hash differs")
    vocab_hash = _sha256_bytes(_canonical(tokenizer.get_vocab()))
    template_hash = _sha256_bytes(
        str(getattr(tokenizer, "chat_template", "") or "").encode("utf-8")
    )
    observed = {
        "tokenizer_class": _qualified_name(tokenizer),
        "backend_class": _qualified_name(backend),
        "backend_serialization_utf8_bytes": len(backend_text.encode("utf-8")),
        "backend_serialization_sha256":
            _sha256_bytes(backend_text.encode("utf-8")),
        "vocab_sha256": vocab_hash,
        "chat_template_sha256": template_hash,
        "length": len(tokenizer),
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token": str(tokenizer.eos_token),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "all_special_ids": [int(value) for value in tokenizer.all_special_ids],
        "special_token_ids": {
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in _TOKENIZER["special_token_ids"]
        },
        "files": {
            row["path"]: {
                "sha256": row["sha256"], "size_bytes": row["size_bytes"],
            }
            for row in tokenizer_inventory["files"]
        },
    }
    expected = dict(_TOKENIZER)
    expected.pop("backend_serialization_sha256")
    expected["backend_serialization_sha256"] = (
        _TOKENIZER["backend_serialization_sha256"]
    )
    expected["backend_serialization_utf8_bytes"] = (
        observed["backend_serialization_utf8_bytes"]
    )
    _require(observed == expected,
             f"tokenizer files/backend/template/vocab/EOS differ: {observed}")
    return observed


def _attest_runtime_fingerprint(
    runtime: _Runtime, model: Any, tokenizer: Any, *, snapshot: Path,
) -> dict[str, Any]:
    try:
        observed = runtime.runtime_fingerprint(
            model, tokenizer, requested_model=MODEL_ID,
            requested_revision=MODEL_REVISION,
            resolved_snapshot=snapshot.name,
            expected_geometry=_SUBJECT["geometry"],
        )
    except Exception as exc:
        raise PoweredV13SubjectError(f"runtime fingerprint failed: {exc}") from exc
    _require(isinstance(observed, dict) and
             observed.get("requested_model") == MODEL_ID and
             observed.get("requested_revision") == MODEL_REVISION and
             observed.get("resolved_snapshot") == MODEL_REVISION and
             observed.get("dtype") == "torch.bfloat16" and
             observed.get("attention_backend") == "eager" and
             observed.get("geometry") == _SUBJECT["geometry"] and
             observed.get("torch_version") == _CUDA_CONTRACT["torch_version"] and
             observed.get("transformers_version") ==
             _DEPENDENCIES["distributions"]["transformers"],
             "runtime fingerprint identity/backend/geometry differs")
    return observed


def _attest_loaded(
    model: Any, tokenizer: Any, loading_info: Mapping[str, Any], *,
    runtime: _Runtime, snapshot: Path, model_inventory: Mapping[str, Any],
    tokenizer_inventory: Mapping[str, Any], inventory_binding: Mapping[str, Any],
    bundle: _ContractBundle, release: Mapping[str, Any],
    dependencies: Mapping[str, Any], cuda: Mapping[str, Any], gate_trace: list[str],
) -> dict[str, Any]:
    loading = _attest_loading_info(loading_info)
    config, attention_layers = _attest_config_and_backend(model)
    unquantized = _attest_unquantized(model)
    for candidate in (
        getattr(model, "name_or_path", None),
        getattr(model.config, "_name_or_path", None),
    ):
        if candidate:
            _require(Path(str(candidate)).resolve() == snapshot.resolve(),
                     "loaded model path differs from resolved snapshot")
    parameters = _attest_parameters(
        model, runtime=runtime, model_inventory=model_inventory,
        bundle=bundle, snapshot=snapshot,
    )
    tokenizer_binding = _attest_tokenizer(
        tokenizer, snapshot=snapshot,
        tokenizer_inventory=tokenizer_inventory,
    )
    runtime_binding = _attest_runtime_fingerprint(
        runtime, model, tokenizer, snapshot=snapshot,
    )
    embeddings = model.get_input_embeddings().weight
    _require(int(embeddings.shape[0]) >= len(tokenizer),
             "tokenizer vocabulary exceeds model embeddings")
    eos_ids = runtime_binding.get("eos_ids")
    _require(isinstance(eos_ids, list) and eos_ids and
             all(type(value) is int and value in tokenizer.all_special_ids and
                 0 <= value < len(tokenizer) and value < int(embeddings.shape[0])
                 for value in eos_ids),
             "runtime EOS set is outside tokenizer/model special-token domain")
    release_binding = {
        "authorization_commit": release["authorization"]["authorization_commit"],
        "static_root_commit": release["authorization"]["static_root_commit"],
        "manifest_path": release["authorization"]["manifest_path"],
        "manifest_sha256": release["authorization"]["manifest_sha256"],
        "inventory_sha256": release["authorization"]["inventory_sha256"],
        "receipt_sha256": release["receipt"]["receipt_sha256"],
        "release_evidence_sha256": _sha256_bytes(_canonical(release)),
        "subject_contract_sha256": bundle.subject_contract_sha256,
        "legacy_model_bits_sha256": bundle.legacy_contract_sha256,
        **dict(inventory_binding),
    }
    result = dict(runtime_binding)
    result.update({
        "schema": "powered-v13-exact-subject-runtime-attestation-v1",
        "design_id": DESIGN_ID,
        "gate_trace": [*gate_trace, "full_post_load_attestation"],
        "subject": dict(_SUBJECT),
        "dependencies": dict(dependencies),
        "cuda": dict(cuda),
        "snapshot_path": str(snapshot.resolve()),
        "model_inventory_sha256": model_inventory["inventory_sha256"],
        "tokenizer_inventory_sha256": tokenizer_inventory["inventory_sha256"],
        "loading_info": loading,
        "config": config,
        "attention_layers": attention_layers,
        "unquantized": unquantized,
        "parameters": parameters,
        "tokenizer": tokenizer_binding,
        "release_binding": release_binding,
    })
    result["fingerprint_sha256"] = _sha256_bytes(_canonical(result))
    return result


_ACTIVE_LOCK = threading.Lock()
_ACTIVE_TOKEN: object | None = None


def _reserve_subject() -> object:
    global _ACTIVE_TOKEN
    with _ACTIVE_LOCK:
        _require(_ACTIVE_TOKEN is None,
                 "one exact-subject handle is already live in this process")
        token = object()
        _ACTIVE_TOKEN = token
        return token


def _release_subject(token: object) -> None:
    global _ACTIVE_TOKEN
    with _ACTIVE_LOCK:
        if _ACTIVE_TOKEN is token:
            _ACTIVE_TOKEN = None


def _discard_loaded(model: Any, tokenizer: Any, runtime: _Runtime | None) -> None:
    del model, tokenizer
    gc.collect()
    if runtime is not None:
        try:
            if runtime.torch.cuda.is_available():
                runtime.torch.cuda.empty_cache()
        except Exception:
            pass


class ExactSubjectHandle:
    """The sole live exact subject; explicit close releases its process slot."""

    __slots__ = (
        "_token", "_model", "_tokenizer", "_runtime", "_snapshot",
        "_fingerprint", "_model_inventory", "_tokenizer_inventory", "_closed",
        "_model_finalizer",
    )

    def __init__(
        self, token: object, model: Any, tokenizer: Any, runtime: _Runtime,
        snapshot: Path, fingerprint: Mapping[str, Any],
        model_inventory: Mapping[str, Any],
        tokenizer_inventory: Mapping[str, Any],
    ) -> None:
        self._token = token
        self._model = model
        self._tokenizer = tokenizer
        self._runtime = runtime
        self._snapshot = snapshot
        self._fingerprint = dict(fingerprint)
        self._model_inventory = dict(model_inventory)
        self._tokenizer_inventory = dict(tokenizer_inventory)
        self._closed = False
        try:
            # If a caller retains ``handle.model`` after close, its CUDA state
            # remains live and therefore keeps the global slot.  The slot is
            # evicted only when the actual model object becomes unreachable.
            self._model_finalizer = weakref.finalize(
                model, _release_subject, token,
            )
        except TypeError as exc:
            raise PoweredV13SubjectError(
                "loaded exact-subject model does not support lifecycle tracking"
            ) from exc

    def _require_open(self) -> None:
        _require(not self._closed, "exact-subject handle is closed")

    @property
    def model(self) -> Any:
        self._require_open()
        return self._model

    @property
    def tokenizer(self) -> Any:
        self._require_open()
        return self._tokenizer

    @property
    def snapshot(self) -> Path:
        self._require_open()
        return self._snapshot

    @property
    def runtime_fingerprint(self) -> dict[str, Any]:
        self._require_open()
        return json.loads(_canonical(self._fingerprint))

    @property
    def model_inventory(self) -> dict[str, Any]:
        self._require_open()
        return json.loads(_canonical(self._model_inventory))

    @property
    def tokenizer_inventory(self) -> dict[str, Any]:
        self._require_open()
        return json.loads(_canonical(self._tokenizer_inventory))

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        model, tokenizer, runtime = self._model, self._tokenizer, self._runtime
        self._model = None
        self._tokenizer = None
        self._runtime = None
        _discard_loaded(model, tokenizer, runtime)

    def __enter__(self) -> "ExactSubjectHandle":
        self._require_open()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def _open_exact_subject(
    repo: Path, receipt_directory: Path, *, seams: _SubjectSeams,
) -> ExactSubjectHandle:
    repo = Path(repo).resolve()
    receipt_directory = Path(receipt_directory).resolve()
    gate_trace: list[str] = []
    release = _validate_release(seams.verify_release(repo, receipt_directory))
    gate_trace.append("stage_t_detached_clean_receipt")
    dependencies = _validate_dependencies(seams.probe_dependencies(repo))
    gate_trace.append("exact_locked_dependencies")
    runtime = seams.load_runtime()
    runtime_origins = _attest_runtime_origins(runtime, repo)
    cuda = _attest_cuda(runtime, seams.nvidia_smi())
    gate_trace.append("one_real_cuda_gpu")
    token = _reserve_subject()
    gate_trace.append("one_live_handle_reserved")
    model = tokenizer = None
    try:
        snapshot = _resolve_snapshot(runtime)
        gate_trace.append("exact_revision_resolved")
        bundle = seams.load_contracts(repo)
        _validate_contract_dependency_binding(bundle)
        model_inventory, tokenizer_inventory = _inventory(runtime, snapshot)
        inventory_binding = _verify_inventories(
            model_inventory, tokenizer_inventory, snapshot=snapshot, bundle=bundle,
        )
        gate_trace.append("pre_load_bf16_snapshot_and_tokenizer_inventory")
        model, tokenizer, loading_info = _load_local(runtime, snapshot)
        gate_trace.append("resolved_local_snapshot_loaded")
        model_after, tokenizer_after = _inventory(runtime, snapshot)
        _require(model_after == model_inventory,
                 "model snapshot inventory changed across load")
        _require(tokenizer_after == tokenizer_inventory,
                 "tokenizer snapshot inventory changed across load")
        _verify_inventories(
            model_after, tokenizer_after, snapshot=snapshot, bundle=bundle,
        )
        gate_trace.append("post_load_snapshot_reinventory")
        post_load_release = seams.recheck_release(repo, release, bundle)
        _require(isinstance(post_load_release, Mapping) and
                 post_load_release.get("status") == "PASS" and
                 post_load_release.get("detached_head") ==
                 release["authorization"]["authorization_commit"] and
                 post_load_release.get("clean_tree") is True,
                 "post-load Stage-T release binding did not pass")
        release["post_load_binding"] = dict(post_load_release)
        gate_trace.append("post_load_stage_t_binding")
        fingerprint = _attest_loaded(
            model, tokenizer, loading_info, runtime=runtime, snapshot=snapshot,
            model_inventory=model_inventory,
            tokenizer_inventory=tokenizer_inventory,
            inventory_binding=inventory_binding, bundle=bundle, release=release,
            dependencies=dependencies, cuda=cuda, gate_trace=gate_trace,
        )
        fingerprint["runtime_module_origins"] = runtime_origins
        fingerprint["fingerprint_sha256"] = _sha256_bytes(_canonical({
            key: value for key, value in fingerprint.items()
            if key != "fingerprint_sha256"
        }))
        return ExactSubjectHandle(
            token, model, tokenizer, runtime, snapshot, fingerprint,
            model_inventory, tokenizer_inventory,
        )
    except Exception:
        _discard_loaded(model, tokenizer, runtime)
        _release_subject(token)
        raise


def open_exact_subject(repo: Path, receipt_directory: Path) -> ExactSubjectHandle:
    """Verify fixed Stage-T and open the one exact powered-v13 subject."""
    return _open_exact_subject(repo, receipt_directory, seams=_PRODUCTION_SEAMS)


__all__ = [
    "ExactSubjectHandle", "MODEL_ID", "MODEL_REVISION",
    "PoweredV13SubjectError", "open_exact_subject",
]
