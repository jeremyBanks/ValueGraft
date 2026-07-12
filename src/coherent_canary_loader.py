"""Pinned subject loading and fail-closed provenance for canary v12.

The experimental subject and the protocol tokenizer are separate immutable
specifications.  Local apparatus runs use Qwen3-0.6B weights but still render
the frozen protocol with the exact 30B Instruct-2507 tokenizer snapshot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
from typing import Any, Mapping, Sequence

import torch

from coherent_canary_preflight import runtime_fingerprint


MODEL_30B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION_30B = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
MODEL_LOCAL = "Qwen/Qwen3-0.6B"
REVISION_LOCAL = "c1899de289a04d12100db370d81485cdf75e47ca"
EXPECTED_DEPENDENCIES = {
    "torch": "2.12.1",
    "transformers": "5.0.0",
    "accelerate": "1.14.0",
    "huggingface-hub": "1.22.0",
    "safetensors": "0.8.0",
}
TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json",
}


class CanaryLoaderError(RuntimeError):
    """The subject, tokenizer, environment, or repository differs from seal."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryLoaderError(message)


@dataclass(frozen=True)
class SubjectSpec:
    key: str
    model_id: str
    revision: str
    architecture: str
    model_type: str
    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    rope_theta: float
    device_type: str

    @property
    def geometry(self) -> dict[str, int | float]:
        return {
            "layers": self.layers,
            "attention_heads": self.attention_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
        }


LOCAL_SPEC = SubjectSpec(
    key="local-apparatus", model_id=MODEL_LOCAL, revision=REVISION_LOCAL,
    architecture="Qwen3ForCausalLM", model_type="qwen3",
    layers=28, attention_heads=16, kv_heads=8, head_dim=128,
    rope_theta=1_000_000.0, device_type="cpu")
EXACT_SPEC = SubjectSpec(
    key="exact-subject", model_id=MODEL_30B, revision=REVISION_30B,
    architecture="Qwen3MoeForCausalLM", model_type="qwen3_moe",
    layers=48, attention_heads=32, kv_heads=4, head_dim=128,
    rope_theta=10_000_000.0, device_type="cuda")
PROTOCOL_TOKENIZER = {
    "model_id": MODEL_30B,
    "revision": REVISION_30B,
}
PROTOCOL_TOKENIZER_ATTESTATION = {
    "class": "Qwen2Tokenizer",
    "length": 151669,
    "vocab_size": 151643,
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
    "chat_template_sha256":
        "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326",
    "vocab_sha256":
        "f488fa45d324a8bc64c84f0e27b47223872d550f2a6900564f77dfa67ca5ff4d",
    "identity_prefix_token_count": 27,
    "identity_prefix_ids_sha256":
        "44bc1b106347e9008c6d7f1d30ab51166942c419ad66819eae733d2e990c3881",
}
MODEL_SNAPSHOT_CONTRACT_PATH = Path(
    "data/coherent_canary_v12/model_snapshot_contract.json")
FROZEN_AUTHORIZATION_PATH = Path(
    "COHERENT-STATE-DECISION-CANARY-V12-FROZEN-AUTHORIZATION-3.json")
MANDATORY_FROZEN_PATHS = {
    "AGENTS.md", "pyproject.toml", "uv.lock",
    "COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md",
    "COHERENT-STATE-DECISION-CANARY-V12-STIMULUS-CONTRACT.md",
    MODEL_SNAPSHOT_CONTRACT_PATH.as_posix(),
    "data/coherent_canary_v12/fixed_text_token_evidence_v2.json",
    "data/coherent_canary_v12/generated_forced_identity_fixture.json",
    "data/coherent_canary_v12/technical_control_fixture.json",
    "data/coherent_canary_v12/revision2/session_d/e01.json",
    "data/coherent_canary_v12/revision2/session_d/e02.json",
    "data/coherent_canary_v12/revision2/session_e/e03.json",
    "data/coherent_canary_v12/revision4/session_h/e04.json",
    "data/coherent_canary_v12/revision2/session_f/e05.json",
    "data/coherent_canary_v12/revision2/session_f/e06.json",
    "notes/2026071152-sol-ultra-regroup-decision.md",
    "notes/2026071153-fable-ultra-regroup-review.md",
    "notes/2026071155-sol-fable-review-disposition-and-canary-closure.md",
    "notes/2026071156-canary-preregistration-independent-audit.md",
    "notes/2026071167-sol-v12-execution-audit-and-runtime-closure.md",
    "notes/2026071168-sol-v12-harvest-and-release-closure.md",
    "results/coherent_canary_validation/"
    "coherent_canary_revision4_full_manifest_Qwen3-30B-A3B-Instruct-2507_"
    "20260711T205951Z.json",
    "results/coherent_canary_validation/"
    "coherent_canary_control_fixtures_literal_Qwen3-30B-A3B-Instruct-2507_"
    "20260711T205951Z.json",
    "results/coherent_canary_validation/"
    "coherent_canary_v12_codex_runtime_provenance_correction_gpt-5.6-sol_"
    "20260711T214629Z.json",
    "results/coherent_canary_reviews/reviews/"
    "revision4_blind_singleton_independent_codex_20260711T210013Z.json",
    "results/coherent_canary_reviews/reviews/"
    "revision4_paired_diversity_independent_codex_20260711T210013Z.json",
    "results/coherent_canary_reviews/packets/"
    "coherent_canary_v12_blind_singleton_20260711T210013Z_7f36752b3295.json",
    "results/coherent_canary_reviews/packets/"
    "coherent_canary_v12_target_aware_paired_20260711T210013Z_55779fc94383.json",
    "results/coherent_canary_reviews/packets/"
    "coherent_canary_v12_cross_case_diversity_20260711T210013Z_8d7e38e6e2e4.json",
    "scripts/build_coherent_canary_review_packets.py",
    "scripts/aggregate_coherent_canary_v12_harvest.py",
    "scripts/validate_coherent_canary_control_fixtures.py",
    "scripts/validate_coherent_canary_stimuli.py",
    "scripts/harvest_coherent_canary_v12.py",
    "scripts/run_coherent_canary_v12_technical.py",
    "scripts/validate_coherent_canary_v12_technical.py",
    "scripts/run_coherent_canary_v12_phase_a.py",
    "scripts/validate_coherent_canary_v12_phase_a.py",
    "scripts/run_coherent_canary_v12_treatment.py",
    "src/coherent_canary_artifacts.py", "src/coherent_canary_case.py",
    "src/coherent_canary_controls.py",
    "src/coherent_canary_loader.py", "src/coherent_canary_path_control.py",
    "src/coherent_canary_preflight.py", "src/coherent_canary_runtime.py",
    "src/coherent_canary_schema.py", "src/coherent_canary_stimuli.py",
    "src/coherent_canary_store.py", "src/coherent_canary_technical.py",
    "src/coherent_canary_tokens.py", "src/coherent_state_tokens.py",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _frozen_spec(spec: SubjectSpec) -> SubjectSpec:
    _require(spec in (LOCAL_SPEC, EXACT_SPEC),
             "subject specification is not one of the two frozen literals")
    return spec


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise CanaryLoaderError(f"invalid JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def resolve_pinned_snapshot(repo_id: str, revision: str, *,
                            local_files_only: bool) -> Path:
    _require(len(revision) == 40 and all(ch in "0123456789abcdef" for ch in revision),
             "snapshot revision is not a lowercase 40-hex commit")
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(
        repo_id, revision=revision, local_files_only=local_files_only))
    validate_snapshot_location(path, revision, repo_id=repo_id)
    return path


def validate_snapshot_location(snapshot: Path, revision: str, *,
                               repo_id: str | None = None) -> Path:
    snapshot = snapshot.resolve(strict=True)
    _require(snapshot.is_dir() and snapshot.name == revision,
             f"resolved snapshot {snapshot.name} != pinned revision {revision}")
    _require(snapshot.parent.name == "snapshots",
             "snapshot is not under an HF snapshots directory")
    model_cache_root = snapshot.parent.parent.resolve(strict=True)
    if repo_id is not None:
        expected_root = "models--" + repo_id.replace("/", "--")
        _require(model_cache_root.name == expected_root,
                 f"snapshot repository root differs: {model_cache_root.name} "
                 f"!= {expected_root}")
    for entry in snapshot.rglob("*"):
        if entry.is_symlink():
            resolved = entry.resolve(strict=True)
            try:
                resolved.relative_to(model_cache_root)
            except ValueError as exc:
                raise CanaryLoaderError(
                    f"snapshot link escapes model cache: {entry} -> {resolved}") from exc
            _require(resolved.is_file(),
                     f"snapshot link target is not a regular file: {entry}")
        elif entry.is_dir():
            continue
        else:
            _require(entry.is_file(),
                     f"snapshot entry is not a regular file: {entry}")
    return model_cache_root


def _safe_relative_file(name: str) -> str:
    pure = PurePosixPath(name)
    _require(not pure.is_absolute() and pure.parts and ".." not in pure.parts and
             len(pure.parts) == 1,
             f"unsafe snapshot file reference: {name}")
    return pure.as_posix()


def validate_config(snapshot: Path, spec: SubjectSpec) -> dict[str, Any]:
    _frozen_spec(spec)
    config = _read_json(snapshot / "config.json")
    observed = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "layers": config.get("num_hidden_layers"),
        "attention_heads": config.get("num_attention_heads"),
        "kv_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "rope_theta": float(config.get("rope_theta", -1)),
        "torch_dtype": config.get("torch_dtype"),
        "quantization_config": config.get("quantization_config"),
    }
    expected = {
        "architectures": [spec.architecture],
        "model_type": spec.model_type,
        "layers": spec.layers,
        "attention_heads": spec.attention_heads,
        "kv_heads": spec.kv_heads,
        "head_dim": spec.head_dim,
        "rope_theta": spec.rope_theta,
        "torch_dtype": "bfloat16",
        "quantization_config": None,
    }
    _require(observed == expected,
             f"snapshot config differs for {spec.key}: {observed} != {expected}")
    return observed


def _weight_files(snapshot: Path) -> tuple[list[str], dict[str, Any] | None]:
    _require(not list(snapshot.glob("*.bin")), "PyTorch .bin weights are forbidden")
    index_path = snapshot / "model.safetensors.index.json"
    single = snapshot / "model.safetensors"
    all_shards = sorted(path.name for path in snapshot.glob("*.safetensors"))
    if index_path.is_file():
        _require(not single.exists(), "single and sharded safetensors coexist")
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        _require(isinstance(weight_map, Mapping) and weight_map,
                 "safetensors index weight_map is empty")
        referenced = sorted({_safe_relative_file(str(value))
                             for value in weight_map.values()})
        _require(referenced == all_shards,
                 "safetensors index shard set differs from snapshot")
        return ["model.safetensors.index.json", *referenced], index
    _require(single.is_file() and all_shards == ["model.safetensors"],
             "snapshot lacks exactly one safetensors model or valid index")
    return ["model.safetensors"], None


def _weight_tensor_index(snapshot: Path, weight_files: Sequence[str],
                         index: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    from safetensors import safe_open
    shard_names = [name for name in weight_files if name.endswith(".safetensors")]
    rows = []
    observed_map = {}
    seen = set()
    for shard in shard_names:
        try:
            with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                _require(bool(keys), f"safetensors shard is empty: {shard}")
                for key in keys:
                    _require(key not in seen, f"duplicate weight tensor key: {key}")
                    seen.add(key)
                    tensor_slice = handle.get_slice(key)
                    shape = [int(value) for value in tensor_slice.get_shape()]
                    dtype = str(tensor_slice.get_dtype())
                    _require(shape and all(value >= 1 for value in shape),
                             f"weight tensor shape is invalid: {key}")
                    _require(dtype == "BF16",
                             f"weight tensor is not bf16: {key}={dtype}")
                    observed_map[key] = shard
                    rows.append({"name": key, "shard": shard,
                                 "dtype": dtype, "shape": shape})
        except CanaryLoaderError:
            raise
        except Exception as exc:
            raise CanaryLoaderError(
                f"invalid safetensors shard {shard}: {exc}") from exc
    if index is not None:
        declared = index.get("weight_map")
        _require(isinstance(declared, Mapping) and
                 {str(key): _safe_relative_file(str(value))
                  for key, value in declared.items()} == observed_map,
                 "safetensors tensor-key map differs from index")
    return sorted(rows, key=lambda row: row["name"])


def inventory_snapshot(snapshot: Path, *, revision: str, kind: str,
                       spec: SubjectSpec | None = None,
                       repo_id: str | None = None) -> dict[str, Any]:
    expected_repo = spec.model_id if spec is not None else repo_id
    _require(expected_repo is not None,
             "snapshot inventory lacks an expected repository ID")
    validate_snapshot_location(snapshot, revision, repo_id=expected_repo)
    _require(kind in ("model", "protocol_tokenizer"),
             f"unknown snapshot inventory kind: {kind}")
    names: set[str] = set()
    if kind == "model":
        _require(spec is not None and _frozen_spec(spec).revision == revision,
                 "model inventory lacks matching subject specification")
        validate_config(snapshot, spec)
        names.add("config.json")
        weights, weight_index = _weight_files(snapshot)
        names.update(weights)
        if (snapshot / "generation_config.json").is_file():
            names.add("generation_config.json")
    else:
        _require(spec is None, "protocol tokenizer inventory received model spec")
        _require(expected_repo == MODEL_30B and revision == REVISION_30B,
                 "protocol tokenizer is not the frozen 30B snapshot")
        _require((snapshot / "tokenizer.json").is_file() and
                 (snapshot / "tokenizer_config.json").is_file(),
                 "protocol tokenizer core files are absent")
        names.update(name for name in TOKENIZER_FILES if (snapshot / name).is_file())
    rows = []
    for name in sorted(names):
        path = snapshot / _safe_relative_file(name)
        _require(path.is_file(), f"required snapshot file absent: {name}")
        digest = file_sha256(path)
        resolved_name = path.resolve().name
        if name.endswith(".safetensors") and len(resolved_name) == 64 and all(
                ch in "0123456789abcdef" for ch in resolved_name):
            _require(digest == resolved_name,
                     f"weight shard differs from HF LFS SHA-256: {name}")
        rows.append({"path": name, "size_bytes": path.stat().st_size,
                     "sha256": digest})
    _require(bool(rows), "snapshot inventory is empty")
    inventory_hash = hashlib.sha256(_canonical(rows)).hexdigest()
    result = {
        "kind": kind, "repo_id": expected_repo, "revision": revision,
        "snapshot_path": str(snapshot.resolve()), "files": rows,
        "inventory_sha256": inventory_hash,
    }
    if kind == "model":
        tensor_rows = _weight_tensor_index(snapshot, weights, weight_index)
        result["weight_tensors"] = tensor_rows
        result["weight_tensors_sha256"] = hashlib.sha256(
            _canonical(tensor_rows)).hexdigest()
    return result


def load_model_snapshot_contract(repo: Path) -> dict[str, Any]:
    path = repo / MODEL_SNAPSHOT_CONTRACT_PATH
    contract = _read_json(path)
    _require(contract.get("schema") ==
             "coherent_state_decision_canary_v12_model_snapshot_contract_v1" and
             contract.get("design_id") == "coherent-state-decision-canary-v12" and
             set(contract.get("subjects", {})) == {LOCAL_SPEC.key, EXACT_SPEC.key},
             "model snapshot contract schema/subject set differs")
    return contract


def verify_inventory_contract(inventory: Mapping[str, Any], *,
                              spec: SubjectSpec | None,
                              contract: Mapping[str, Any]) -> dict[str, Any]:
    if spec is None:
        expected = contract.get("protocol_tokenizer")
        _require(inventory.get("kind") == "protocol_tokenizer" and
                 inventory.get("repo_id") == MODEL_30B and
                 inventory.get("revision") == REVISION_30B,
                 "protocol inventory selector differs from contract")
        _require(isinstance(expected, Mapping) and
                 inventory.get("files") == expected.get("files") and
                 inventory.get("inventory_sha256") ==
                 expected.get("inventory_sha256"),
                 "protocol tokenizer file inventory differs from frozen contract")
        return dict(expected)
    _frozen_spec(spec)
    expected = contract.get("subjects", {}).get(spec.key)
    _require(isinstance(expected, Mapping) and
             inventory.get("kind") == "model" and
             inventory.get("repo_id") == expected.get("model_id") == spec.model_id and
             inventory.get("revision") == expected.get("revision") == spec.revision and
             inventory.get("files") == expected.get("files") and
             inventory.get("inventory_sha256") == expected.get("inventory_sha256"),
             f"{spec.key} model file inventory differs from frozen contract")
    tensors = inventory.get("weight_tensors")
    _require(isinstance(tensors, list) and
             len(tensors) == expected.get("weight_tensor_count"),
             f"{spec.key} weight tensor count differs from contract")
    parameter_count = sum(math.prod(row["shape"]) for row in tensors)
    _require(parameter_count == expected.get("parameter_count"),
             f"{spec.key} parameter count differs from contract")
    if spec == LOCAL_SPEC:
        _require(inventory.get("weight_tensors_sha256") ==
                 expected.get("weight_tensors_sha256"),
                 "local weight tensor inventory differs from frozen contract")
    else:
        name_to_shard = [{"name": row["name"], "shard": row["shard"]}
                         for row in tensors]
        digest = hashlib.sha256(_canonical(name_to_shard)).hexdigest()
        _require(digest == expected.get("weight_name_to_shard_sha256"),
                 "exact weight name/shard map differs from frozen contract")
    return dict(expected)


def dependency_fingerprint() -> dict[str, str]:
    observed = {name: importlib.metadata.version(name)
                for name in EXPECTED_DEPENDENCIES}
    _require(observed == EXPECTED_DEPENDENCIES,
             f"locked dependency versions differ: {observed}")
    return observed


def verify_frozen_repository(repo: Path) -> dict[str, Any]:
    """Verify the two-stage, non-self-referential FROZEN authorization.

    Apparatus commit A contains every sealed input/code byte.  The immediately
    following authorization commit B adds only the fixed authorization JSON,
    which names A and its file inventory.  Later descendants may add committed
    result records only; every sealed apparatus byte must remain equal to A.
    """
    branch = subprocess.check_output(
        ["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    _require(branch == "trunk", "repository branch is not trunk")
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1"], text=True)
    _require(not status.strip(), "repository has tracked or untracked changes")
    auth_relative = FROZEN_AUTHORIZATION_PATH.as_posix()
    additions = subprocess.check_output([
        "git", "-C", str(repo), "log", "--diff-filter=A", "--format=%H",
        "--", auth_relative], text=True).splitlines()
    _require(len(additions) == 1,
             "FROZEN authorization file lacks one unique add commit")
    auth_commit = additions[0]
    _require(subprocess.run([
        "git", "-C", str(repo), "merge-base", "--is-ancestor",
        auth_commit, head], check=False).returncode == 0,
        "FROZEN authorization commit is not an ancestor of HEAD")
    parent = subprocess.check_output([
        "git", "-C", str(repo), "rev-parse", f"{auth_commit}^"],
        text=True).strip()
    auth_parents = subprocess.check_output([
        "git", "-C", str(repo), "rev-list", "--parents", "-n", "1",
        auth_commit], text=True).split()
    _require(auth_parents == [auth_commit, parent],
             "authorization commit is not a one-parent child of apparatus")
    auth_commit_changes = subprocess.check_output([
        "git", "-C", str(repo), "diff-tree", "--no-commit-id", "--name-status",
        "--no-renames", "-r", auth_commit], text=True).splitlines()
    _require(auth_commit_changes == [f"A\t{auth_relative}"],
             "authorization commit changed more than the authorization file")
    # Audit the complete history, not only the endpoint diff.  Endpoint-only
    # validation falsely accepts apparatus code introduced for a result and
    # reverted later.  Every descendant must form one linear first-parent chain
    # and may only ADD a never-before-used result JSON/Markdown path.
    descendants = subprocess.check_output([
        "git", "-C", str(repo), "rev-list", "--reverse",
        f"{auth_commit}..{head}"], text=True).splitlines()
    previous = auth_commit
    result_additions = []
    result_blobs: list[tuple[str, bytes]] = []
    seen_result_paths: set[str] = set()
    for commit in descendants:
        commit_and_parents = subprocess.check_output([
            "git", "-C", str(repo), "rev-list", "--parents", "-n", "1",
            commit], text=True).split()
        _require(commit_and_parents == [commit, previous],
                 "post-authorization history is not a linear one-parent chain")
        changes = subprocess.check_output([
            "git", "-C", str(repo), "diff-tree", "--no-commit-id",
            "--name-status", "--no-renames", "-r", commit],
            text=True).splitlines()
        _require(bool(changes),
                 "post-authorization history contains an empty commit")
        for change in changes:
            fields = change.split("\t")
            _require(len(fields) == 2 and fields[0] == "A",
                     f"post-authorization change is not an addition: {change}")
            relative = _safe_relative_repo_path(fields[1])
            _require(relative.startswith("results/coherent_canary_") and
                     Path(relative).suffix in {".json", ".md"},
                     f"post-authorization addition is not a result: {relative}")
            _require(relative not in seen_result_paths,
                     f"post-authorization result path is reused: {relative}")
            tree_entry = subprocess.check_output([
                "git", "-C", str(repo), "ls-tree", commit, "--", relative],
                text=True).strip().split(maxsplit=3)
            _require(len(tree_entry) == 4 and tree_entry[0] == "100644" and
                     tree_entry[1] == "blob" and
                     tree_entry[3].split("\t", 1)[-1] == relative,
                     f"post-authorization result is not a regular data blob: {relative}")
            result_bytes = subprocess.check_output([
                "git", "-C", str(repo), "show", f"{commit}:{relative}"])
            seen_result_paths.add(relative)
            result_blobs.append((relative, result_bytes))
            result_additions.append({
                "commit": commit, "path": relative,
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
                "size_bytes": len(result_bytes),
            })
        previous = commit
    for relative, result_bytes in result_blobs:
        head_result_bytes = subprocess.check_output([
            "git", "-C", str(repo), "show", f"HEAD:{relative}"])
        result_path = repo / relative
        _require(result_path.is_file() and not result_path.is_symlink() and
                 result_path.read_bytes() == result_bytes == head_result_bytes,
                 f"post-authorization result worktree bytes differ: {relative}")
    auth_head = subprocess.check_output([
        "git", "-C", str(repo), "show", f"{auth_commit}:{auth_relative}"])
    auth_now = subprocess.check_output([
        "git", "-C", str(repo), "show", f"HEAD:{auth_relative}"])
    auth_path = repo / auth_relative
    auth_tree_entry = subprocess.check_output([
        "git", "-C", str(repo), "ls-tree", auth_commit, "--", auth_relative],
        text=True).strip().split(maxsplit=3)
    _require(len(auth_tree_entry) == 4 and auth_tree_entry[0] == "100644" and
             auth_tree_entry[1] == "blob",
             "FROZEN authorization is not a regular data blob")
    _require(auth_head == auth_now and auth_path.is_file() and
             not auth_path.is_symlink() and
             auth_path.read_bytes() == auth_head,
             "FROZEN authorization bytes changed after authorization")
    try:
        seal = json.loads(auth_head)
    except Exception as exc:
        raise CanaryLoaderError(f"FROZEN authorization JSON is invalid: {exc}") from exc
    expected_fields = {
        "schema", "design_id", "status", "branch", "apparatus_commit",
        "files", "inventory_sha256",
    }
    _require(isinstance(seal, Mapping) and set(seal) == expected_fields and
             seal["schema"] ==
             "coherent_state_decision_canary_v12_frozen_authorization_v1" and
             seal["design_id"] == "coherent-state-decision-canary-v12" and
             seal["status"] == "FROZEN" and seal["branch"] == "trunk" and
             seal["apparatus_commit"] == parent,
             "FROZEN authorization fields/apparatus parent differ")
    rows = seal["files"]
    _require(isinstance(rows, list) and rows,
             "repository sealed inventory is empty")
    normalized = []
    seen = set()
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256"},
                 "repository sealed file fields differ")
        relative = _safe_relative_repo_path(str(row["path"]))
        _require(relative not in seen, "repository sealed file is duplicated")
        seen.add(relative)
        apparatus_bytes = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{parent}:{relative}"])
        head_bytes = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"HEAD:{relative}"])
        tree_entry = subprocess.check_output([
            "git", "-C", str(repo), "ls-tree", parent, "--", relative],
            text=True).strip().split(maxsplit=3)
        worktree = repo / relative
        _require(len(tree_entry) == 4 and tree_entry[0] in {"100644", "100755"} and
                 tree_entry[1] == "blob",
                 f"sealed apparatus path is not a regular file: {relative}")
        _require(head_bytes == apparatus_bytes and worktree.is_file() and
                 not worktree.is_symlink() and
                 worktree.read_bytes() == apparatus_bytes,
                 f"sealed apparatus file changed after parent: {relative}")
        digest = hashlib.sha256(apparatus_bytes).hexdigest()
        _require(digest == row["sha256"],
                 f"sealed file hash differs: {relative}")
        normalized.append({"path": relative, "sha256": digest})
    normalized.sort(key=lambda row: row["path"])
    _require(rows == normalized,
             "repository sealed file inventory is not canonical")
    observed_hash = hashlib.sha256(_canonical(normalized)).hexdigest()
    _require(observed_hash == seal["inventory_sha256"],
             "repository sealed inventory hash differs")
    _require(MANDATORY_FROZEN_PATHS.issubset(seen),
             "FROZEN authorization omits mandatory apparatus paths")
    return {
        "branch": branch, "head": head, "apparatus_commit": parent,
        "authorization_commit": auth_commit, "files": normalized,
        "inventory_sha256": observed_hash,
        "authorization_sha256": hashlib.sha256(auth_head).hexdigest(),
        "post_authorization_result_additions": result_additions,
    }


def _safe_relative_repo_path(value: str) -> str:
    pure = PurePosixPath(value)
    _require(not pure.is_absolute() and pure.parts and ".." not in pure.parts and
             pure.parts[0] != ".git",
             f"unsafe repository path: {value}")
    return pure.as_posix()


def _quantization_absent(model) -> dict[str, Any]:
    config = model.config
    fields = {
        "config_quantization": getattr(config, "quantization_config", None),
        "is_quantized": bool(getattr(model, "is_quantized", False)),
        "hf_quantizer": getattr(model, "hf_quantizer", None),
        "quantization_method": getattr(model, "quantization_method", None),
        "load_in_4bit": bool(getattr(model, "load_in_4bit", False)),
        "load_in_8bit": bool(getattr(model, "load_in_8bit", False)),
    }
    _require(fields["config_quantization"] is None and
             fields["is_quantized"] is False and
             fields["hf_quantizer"] is None and
             fields["quantization_method"] is None and
             fields["load_in_4bit"] is False and
             fields["load_in_8bit"] is False,
             f"loaded subject is quantized: {fields}")
    forbidden = ("bitsandbytes", "gptq", "awq")
    _require(not any(any(word in type(module).__module__.lower() or
                         word in type(module).__name__.lower()
                         for word in forbidden)
                     for module in model.modules()),
             "loaded subject contains a quantization module")
    return fields


def attest_protocol_tokenizer(tokenizer, *, snapshot: Path,
                              inventory: Mapping[str, Any]) -> dict[str, Any]:
    _require(inventory.get("kind") == "protocol_tokenizer" and
             inventory.get("repo_id") == MODEL_30B and
             inventory.get("revision") == REVISION_30B and
             Path(inventory.get("snapshot_path", "")).resolve() == snapshot.resolve(),
             "protocol tokenizer inventory differs")
    name_or_path = getattr(tokenizer, "name_or_path", None)
    _require(name_or_path and Path(name_or_path).resolve() == snapshot.resolve(),
             "protocol tokenizer loaded path differs")
    from coherent_state_tokens import generation_prefix_ids
    identity_messages = [
        {"role": "system", "content": "Answer plainly."},
        {"role": "user", "content":
         "Write one short neutral sentence acknowledging that a record exists."},
    ]
    identity_ids = [int(value) for value in generation_prefix_ids(
        tokenizer, identity_messages)]
    observed = {
        "class": type(tokenizer).__name__,
        "length": len(tokenizer),
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "all_special_ids": [int(value) for value in tokenizer.all_special_ids],
        "special_token_ids": {
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in PROTOCOL_TOKENIZER_ATTESTATION["special_token_ids"]
        },
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")).hexdigest(),
        "vocab_sha256": hashlib.sha256(_canonical(tokenizer.get_vocab())).hexdigest(),
        "identity_prefix_token_count": len(identity_ids),
        "identity_prefix_ids_sha256": hashlib.sha256(b"".join(
            value.to_bytes(8, "little", signed=True)
            for value in identity_ids)).hexdigest(),
    }
    _require(observed == PROTOCOL_TOKENIZER_ATTESTATION,
             f"protocol tokenizer attestation differs: {observed}")
    return observed


def attest_loaded_subject(model, tokenizer, *, spec: SubjectSpec,
                          repo: Path,
                          repository_authorization: Mapping[str, Any],
                          snapshot_contract: Mapping[str, Any],
                          model_snapshot: Path,
                          model_inventory: Mapping[str, Any],
                          protocol_tokenizer_snapshot: Path,
                          protocol_tokenizer_inventory: Mapping[str, Any],
                          loading_info: Mapping[str, Any]
                          ) -> dict[str, Any]:
    _frozen_spec(spec)
    # Re-run both release checks inside attestation.  This makes a fingerprint
    # from a direct private-loader call distinguishable from one that passed the
    # frozen repository and snapshot-contract gates.
    observed_repository = verify_frozen_repository(repo)
    _require(observed_repository == repository_authorization,
             "repository authorization changed before runtime attestation")
    observed_contract = load_model_snapshot_contract(repo)
    _require(observed_contract == snapshot_contract,
             "model snapshot contract changed before runtime attestation")
    _require(type(model).__name__ == spec.architecture,
             f"loaded model class differs: {type(model).__name__}")
    config = getattr(model.config, "text_config", model.config)
    _require(getattr(model.config, "model_type", None) == spec.model_type,
             "loaded model_type differs")
    _require(getattr(model.config, "_commit_hash", None) in (None, spec.revision),
             "loaded config commit differs")
    for candidate in (getattr(model, "name_or_path", None),
                      getattr(model.config, "_name_or_path", None)):
        if candidate:
            _require(Path(candidate).resolve() == model_snapshot.resolve(),
                     "loaded model path differs from pinned snapshot")
    observed_model_inventory = inventory_snapshot(
        model_snapshot, revision=spec.revision, kind="model", spec=spec)
    observed_tokenizer_inventory = inventory_snapshot(
        protocol_tokenizer_snapshot, revision=REVISION_30B,
        kind="protocol_tokenizer", repo_id=MODEL_30B)
    _require(observed_model_inventory == model_inventory,
             "model snapshot inventory changed across load")
    _require(observed_tokenizer_inventory == protocol_tokenizer_inventory,
             "protocol tokenizer inventory changed across load")
    subject_contract_entry = verify_inventory_contract(
        model_inventory, spec=spec, contract=snapshot_contract)
    tokenizer_contract_entry = verify_inventory_contract(
        protocol_tokenizer_inventory, spec=None, contract=snapshot_contract)
    _require(model_inventory.get("kind") == "model" and
             model_inventory.get("repo_id") == spec.model_id and
             model_inventory.get("revision") == spec.revision and
             Path(model_inventory.get("snapshot_path", "")).resolve() ==
             model_snapshot.resolve(),
             "model inventory does not bind loaded snapshot")
    _require(protocol_tokenizer_inventory.get("kind") == "protocol_tokenizer" and
             protocol_tokenizer_inventory.get("repo_id") == MODEL_30B and
             protocol_tokenizer_inventory.get("revision") == REVISION_30B and
             Path(protocol_tokenizer_inventory.get("snapshot_path", "")).resolve() ==
             protocol_tokenizer_snapshot.resolve(),
             "protocol tokenizer inventory does not bind loaded snapshot")
    tokenizer_attestation = attest_protocol_tokenizer(
        tokenizer, snapshot=protocol_tokenizer_snapshot,
        inventory=protocol_tokenizer_inventory)
    _require(isinstance(loading_info, Mapping), "model loading info is absent")
    loading_info_keys = (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    _require(set(loading_info) == set(loading_info_keys),
             f"model loading info fields differ: {sorted(loading_info)}")
    for key in loading_info_keys:
        value = loading_info[key]
        _require(isinstance(value, (list, tuple, set)) and not value,
                 f"model loading reported {key}: {value}")
    _quantization_absent(model)
    model.eval()
    model.requires_grad_(False)
    _require(not model.training and all(not module.training for module in model.modules()),
             "loaded subject is not entirely in evaluation mode")
    named_parameters = list(model.named_parameters(remove_duplicate=False))
    parameters = [parameter for _, parameter in named_parameters]
    _require(parameters and all(parameter.is_floating_point() and
                                parameter.dtype == torch.bfloat16
                                for parameter in parameters),
             "loaded subject parameter dtype/type differs")
    _require(all(not parameter.requires_grad for parameter in parameters),
             "loaded subject parameters are not frozen")
    devices = {str(parameter.device) for parameter in parameters}
    buffer_devices = {str(buffer.device) for buffer in model.buffers()}
    _require(all(parameter.device.type != "meta" for parameter in parameters) and
             all(buffer.device.type != "meta" for buffer in model.buffers()),
             "loaded subject contains meta tensors")
    loaded_parameter_rows = sorted(({
        "name": name, "shape": [int(value) for value in parameter.shape],
        "dtype": "BF16" if parameter.dtype == torch.bfloat16
        else str(parameter.dtype),
    } for name, parameter in named_parameters), key=lambda row: row["name"])
    weight_parameter_rows = sorted(({
        "name": str(row["name"]),
        "shape": [int(value) for value in row["shape"]],
        "dtype": str(row["dtype"]),
    } for row in model_inventory["weight_tensors"]), key=lambda row: row["name"])
    _require(loaded_parameter_rows == weight_parameter_rows,
             "loaded parameter names/shapes/dtypes differ from weight inventory")
    loaded_parameter_count = sum(
        math.prod(row["shape"]) for row in loaded_parameter_rows)
    _require(loaded_parameter_count == subject_contract_entry["parameter_count"],
             "loaded parameter count differs from snapshot contract")
    if spec.device_type == "cpu":
        _require(devices == {"cpu"} and buffer_devices.issubset({"cpu"}),
                 f"local subject device coverage differs: {devices}/{buffer_devices}")
    else:
        _require(devices == {"cuda:0"} and buffer_devices.issubset({"cuda:0"}),
                 f"exact subject device coverage differs: {devices}/{buffer_devices}")
        device_map = getattr(model, "hf_device_map", None)
        _require(device_map in (None, {"": 0}, {"": "cuda:0"}),
                 f"exact subject offload/device map differs: {device_map}")
    adverse_hooks = []
    for name, module in model.named_modules():
        hook = getattr(module, "_hf_hook", None)
        execution_device = (None if hook is None else
                            getattr(hook, "execution_device", None))
        execution_device = None if execution_device is None else str(execution_device)
        allowed_device = "cuda:0" if spec.device_type == "cuda" else "cpu"
        if hook is not None and (
                bool(getattr(hook, "offload", False)) or
                bool(getattr(hook, "offload_buffers", False)) or
                execution_device not in (None, spec.device_type, allowed_device)):
            adverse_hooks.append({"module": name, "hook": type(hook).__name__})
    _require(not adverse_hooks, f"loaded subject has offload hooks: {adverse_hooks}")
    embeddings = model.get_input_embeddings().weight
    _require(int(embeddings.shape[0]) >= len(tokenizer),
             "protocol tokenizer vocabulary exceeds model embeddings")
    fingerprint = runtime_fingerprint(
        model, tokenizer, requested_model=spec.model_id,
        requested_revision=spec.revision,
        resolved_snapshot=model_snapshot.name,
        expected_geometry=spec.geometry)
    fingerprint.update({
        "subject_spec": asdict(spec),
        "model_inventory_sha256": model_inventory["inventory_sha256"],
        "protocol_tokenizer_spec": dict(PROTOCOL_TOKENIZER),
        "protocol_tokenizer_snapshot": str(protocol_tokenizer_snapshot.resolve()),
        "protocol_tokenizer_inventory_sha256":
            protocol_tokenizer_inventory["inventory_sha256"],
        "protocol_tokenizer_attestation": tokenizer_attestation,
        "weight_tensors_sha256": model_inventory["weight_tensors_sha256"],
        "loading_info": {key: list(loading_info[key]) for key in
                         loading_info_keys},
        "parameter_count": loaded_parameter_count,
        "loaded_parameter_topology_sha256": hashlib.sha256(
            _canonical(loaded_parameter_rows)).hexdigest(),
        "parameter_devices": sorted(devices),
        "buffer_devices": sorted(buffer_devices),
        "training": model.training,
        "all_parameters_frozen": True,
        "release_binding": {
            "apparatus_commit": repository_authorization["apparatus_commit"],
            "authorization_commit":
                repository_authorization["authorization_commit"],
            "authorization_sha256":
                repository_authorization["authorization_sha256"],
            "sealed_inventory_sha256":
                repository_authorization["inventory_sha256"],
            "model_snapshot_contract_sha256": file_sha256(
                repo / MODEL_SNAPSHOT_CONTRACT_PATH),
            "subject_contract_entry_sha256": hashlib.sha256(
                _canonical(subject_contract_entry)).hexdigest(),
            "protocol_tokenizer_contract_entry_sha256": hashlib.sha256(
                _canonical(tokenizer_contract_entry)).hexdigest(),
        },
    })
    eos_ids = fingerprint["eos_ids"]
    special_ids = set(tokenizer_attestation["all_special_ids"])
    _require(all(isinstance(value, int) and 0 <= value < len(tokenizer) and
                 value < int(embeddings.shape[0]) and value in special_ids
                 for value in eos_ids),
             "authoritative EOS set is outside tokenizer/model special-token domain")
    fingerprint["fingerprint_sha256"] = hashlib.sha256(
        _canonical(fingerprint)).hexdigest()
    return fingerprint


def _load_pinned_subject(*, spec: SubjectSpec, model_snapshot: Path,
                         protocol_tokenizer_snapshot: Path):
    """Load only from already-resolved and inventoried immutable snapshots."""
    validate_snapshot_location(
        model_snapshot, spec.revision, repo_id=spec.model_id)
    validate_snapshot_location(
        protocol_tokenizer_snapshot, REVISION_30B, repo_id=MODEL_30B)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(protocol_tokenizer_snapshot), local_files_only=True,
        trust_remote_code=False)
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }
    if spec.device_type == "cuda":
        kwargs.update({"low_cpu_mem_usage": True, "device_map": {"": 0}})
    loaded = AutoModelForCausalLM.from_pretrained(
        str(model_snapshot), output_loading_info=True, **kwargs)
    _require(isinstance(loaded, tuple) and len(loaded) == 2,
             "Transformers did not return model loading information")
    model, loading_info = loaded
    if spec.device_type == "cpu":
        model.to("cpu")
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer, loading_info


def prepare_pinned_subject(*, spec: SubjectSpec, repo: Path,
                           local_files_only: bool) -> dict[str, Any]:
    """The sole production order: resolve, seal, inventory, load, reverify."""
    _frozen_spec(spec)
    repository = verify_frozen_repository(repo)
    snapshot_contract = load_model_snapshot_contract(repo)
    dependencies = dependency_fingerprint()
    device = device_fingerprint(spec)
    model_snapshot = resolve_pinned_snapshot(
        spec.model_id, spec.revision, local_files_only=local_files_only)
    tokenizer_snapshot = resolve_pinned_snapshot(
        MODEL_30B, REVISION_30B, local_files_only=local_files_only)
    model_inventory = inventory_snapshot(
        model_snapshot, revision=spec.revision, kind="model", spec=spec)
    tokenizer_inventory = inventory_snapshot(
        tokenizer_snapshot, revision=REVISION_30B,
        kind="protocol_tokenizer", repo_id=MODEL_30B)
    verify_inventory_contract(
        model_inventory, spec=spec, contract=snapshot_contract)
    verify_inventory_contract(
        tokenizer_inventory, spec=None, contract=snapshot_contract)
    model, tokenizer, loading_info = _load_pinned_subject(
        spec=spec, model_snapshot=model_snapshot,
        protocol_tokenizer_snapshot=tokenizer_snapshot)
    fingerprint = attest_loaded_subject(
        model, tokenizer, spec=spec, repo=repo,
        repository_authorization=repository,
        snapshot_contract=snapshot_contract,
        model_snapshot=model_snapshot,
        model_inventory=model_inventory,
        protocol_tokenizer_snapshot=tokenizer_snapshot,
        protocol_tokenizer_inventory=tokenizer_inventory,
        loading_info=loading_info)
    return {
        "model": model, "tokenizer": tokenizer,
        "model_snapshot": model_snapshot,
        "protocol_tokenizer_snapshot": tokenizer_snapshot,
        "model_inventory": model_inventory,
        "protocol_tokenizer_inventory": tokenizer_inventory,
        "runtime_fingerprint": fingerprint,
        "repository": repository, "dependencies": dependencies,
        "device": device, "model_snapshot_contract": snapshot_contract,
    }


def device_fingerprint(spec: SubjectSpec) -> dict[str, Any]:
    base = {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if spec.device_type == "cpu":
        _require(not torch.cuda.is_available(),
                 "local apparatus unexpectedly has CUDA available")
        return {**base, "device_type": "cpu"}
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "exact subject requires exactly one visible CUDA device")
    props = torch.cuda.get_device_properties(0)
    query = subprocess.run([
        "nvidia-smi", "--query-gpu=uuid,driver_version,name,memory.total",
        "--format=csv,noheader,nounits"], capture_output=True, text=True,
        check=False)
    _require(query.returncode == 0 and len(query.stdout.strip().splitlines()) == 1,
             "nvidia-smi exact-device query failed")
    uuid, driver, name, memory = [part.strip()
                                  for part in query.stdout.strip().split(",", 3)]
    return {**base, "device_type": "cuda", "device_index": 0,
            "gpu_name": name, "gpu_uuid": uuid, "driver_version": driver,
            "memory_total_mib": int(memory),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "compute_capability": [props.major, props.minor],
            "torch_device_name": props.name,
            "torch_total_memory": props.total_memory}
