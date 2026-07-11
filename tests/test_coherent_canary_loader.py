from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from safetensors.torch import save as save_safetensors
import torch
from transformers import AutoTokenizer
import coherent_canary_loader as loader_module

from coherent_canary_loader import (
    EXACT_SPEC, LOCAL_SPEC, MODEL_30B, PROTOCOL_TOKENIZER_ATTESTATION,
    REVISION_30B, SubjectSpec, CanaryLoaderError, attest_protocol_tokenizer,
    attest_loaded_subject, dependency_fingerprint, inventory_snapshot,
    load_model_snapshot_contract, validate_config, validate_snapshot_location,
    verify_frozen_repository, verify_inventory_contract,
)


class Qwen3Attention(torch.nn.Module):
    def __init__(self, index, config):
        super().__init__()
        self.layer_idx = index
        self.config = config
        self.q_proj = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)


class Qwen3ForCausalLM(torch.nn.Module):
    def __init__(self, snapshot: Path):
        super().__init__()
        config = SimpleNamespace(
            model_type="qwen3", num_hidden_layers=28,
            num_attention_heads=16, num_key_value_heads=8, head_dim=128,
            rope_theta=1_000_000.0, eos_token_id=151645,
            _attn_implementation="eager",
            _attn_implementation_internal="eager",
            _commit_hash=LOCAL_SPEC.revision,
            _name_or_path=str(snapshot), quantization_config=None,
        )
        config.text_config = config
        self.config = config
        self.name_or_path = str(snapshot)
        self.generation_config = SimpleNamespace(eos_token_id=[151645, 151643])
        self.embed_tokens = torch.nn.Embedding(
            151669, 1, dtype=torch.bfloat16)
        self.layers = torch.nn.ModuleList([
            Qwen3Attention(index, config) for index in range(28)])

    def get_input_embeddings(self):
        return self.embed_tokens


def write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def fake_snapshot(tmp_path: Path, *, spec=LOCAL_SPEC, sharded=False) -> Path:
    model_root = tmp_path / ("models--" + spec.model_id.replace("/", "--"))
    blobs = model_root / "blobs"
    snapshot = model_root / "snapshots" / spec.revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    config = {
        "architectures": [spec.architecture], "model_type": spec.model_type,
        "num_hidden_layers": spec.layers,
        "num_attention_heads": spec.attention_heads,
        "num_key_value_heads": spec.kv_heads, "head_dim": spec.head_dim,
        "rope_theta": spec.rope_theta, "torch_dtype": "bfloat16",
        "quantization_config": None,
    }
    files = {
        "config.json": json.dumps(config).encode(),
        "generation_config.json": json.dumps({"eos_token_id": [9, 7]}).encode(),
        "tokenizer.json": b"{}", "tokenizer_config.json": b"{}",
        "vocab.json": b"{}", "merges.txt": b"",
    }
    if sharded:
        files["model-00001-of-00002.safetensors"] = save_safetensors({
            "a": torch.zeros((1,), dtype=torch.bfloat16)})
        files["model-00002-of-00002.safetensors"] = save_safetensors({
            "b": torch.ones((1,), dtype=torch.bfloat16)})
        files["model.safetensors.index.json"] = json.dumps({
            "weight_map": {"a": "model-00001-of-00002.safetensors",
                           "b": "model-00002-of-00002.safetensors"}
        }).encode()
    else:
        files["model.safetensors"] = save_safetensors({
            "weight": torch.zeros((1,), dtype=torch.bfloat16)})
    for index, (name, content) in enumerate(files.items()):
        blob = blobs / f"blob-{index}"
        write(blob, content)
        (snapshot / name).symlink_to(Path("../../blobs") / blob.name)
    return snapshot


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def authorize_test_repo(repo: Path) -> tuple[str, dict]:
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    git(repo, "branch", "-m", "trunk")
    tracked = repo / "a.py"
    tracked.write_text("x = 1\n")
    git(repo, "add", "a.py")
    git(repo, "commit", "-m", "initial")
    apparatus_commit = git(repo, "rev-parse", "HEAD")
    rows = [{"path": "a.py", "sha256": hashlib.sha256(
        tracked.read_bytes()).hexdigest()}]
    seal = {
        "schema": "coherent_state_decision_canary_v12_frozen_authorization_v1",
        "design_id": "coherent-state-decision-canary-v12",
        "status": "FROZEN", "branch": "trunk",
        "apparatus_commit": apparatus_commit, "files": rows,
        "inventory_sha256": hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    authorization = repo / loader_module.FROZEN_AUTHORIZATION_PATH
    authorization.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    git(repo, "add", authorization.name)
    git(repo, "commit", "-m", "freeze")
    return apparatus_commit, seal


def test_cached_configs_match_frozen_local_and_exact_specs():
    cache = Path.home() / ".cache/huggingface/hub"
    local = cache / "models--Qwen--Qwen3-0.6B/snapshots" / LOCAL_SPEC.revision
    exact = (cache / "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots" /
             EXACT_SPEC.revision)
    assert validate_config(local, LOCAL_SPEC)["architectures"] == \
        ["Qwen3ForCausalLM"]
    assert validate_config(exact, EXACT_SPEC)["architectures"] == \
        ["Qwen3MoeForCausalLM"]


def test_model_and_protocol_tokenizer_inventories_are_content_addressed(tmp_path):
    snapshot = fake_snapshot(tmp_path, sharded=True)
    protocol_snapshot = fake_snapshot(tmp_path / "protocol", spec=EXACT_SPEC)
    model = inventory_snapshot(
        snapshot, revision=LOCAL_SPEC.revision, kind="model", spec=LOCAL_SPEC)
    tokenizer = inventory_snapshot(
        protocol_snapshot, revision=REVISION_30B, kind="protocol_tokenizer",
        repo_id=MODEL_30B)
    assert model["inventory_sha256"] == hashlib.sha256(json.dumps(
        model["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert {row["path"] for row in tokenizer["files"]} == {
        "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}


def test_snapshot_rejects_escape_and_shard_index_tampering(tmp_path):
    snapshot = fake_snapshot(tmp_path, sharded=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    (snapshot / "escape.txt").symlink_to(outside)
    with pytest.raises(CanaryLoaderError, match="escapes"):
        validate_snapshot_location(
            snapshot, LOCAL_SPEC.revision, repo_id=LOCAL_SPEC.model_id)
    (snapshot / "escape.txt").unlink()
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (snapshot / "escape-dir").symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(CanaryLoaderError, match="escapes"):
        validate_snapshot_location(
            snapshot, LOCAL_SPEC.revision, repo_id=LOCAL_SPEC.model_id)
    (snapshot / "escape-dir").unlink()
    index_blob = (snapshot / "model.safetensors.index.json").resolve()
    index_blob.write_text(json.dumps({"weight_map": {"a": "../escape.safetensors"}}))
    with pytest.raises(CanaryLoaderError, match="unsafe"):
        inventory_snapshot(
            snapshot, revision=LOCAL_SPEC.revision, kind="model", spec=LOCAL_SPEC)


def test_config_rejects_wrong_architecture_and_quantization(tmp_path):
    snapshot = fake_snapshot(tmp_path)
    config_blob = (snapshot / "config.json").resolve()
    config = json.loads(config_blob.read_text())
    config["quantization_config"] = {"bits": 4}
    config_blob.write_text(json.dumps(config))
    with pytest.raises(CanaryLoaderError, match="config differs"):
        validate_config(snapshot, LOCAL_SPEC)


def test_custom_same_shape_subject_spec_cannot_self_authorize(tmp_path):
    snapshot = fake_snapshot(tmp_path)
    custom = SubjectSpec(**{
        **LOCAL_SPEC.__dict__, "model_id": "attacker/same-shape-model"})
    with pytest.raises(CanaryLoaderError, match="two frozen literals"):
        validate_config(snapshot, custom)


def test_dependency_versions_match_lock():
    observed = dependency_fingerprint()
    assert observed["transformers"] == "5.0.0"
    assert observed["torch"] == "2.12.1"


def test_two_stage_repository_authorization_and_dirty_rejection(
        tmp_path, monkeypatch):
    apparatus_commit, _ = authorize_test_repo(tmp_path)
    monkeypatch.setattr(loader_module, "MANDATORY_FROZEN_PATHS", {"a.py"})
    observed = verify_frozen_repository(tmp_path)
    assert observed["apparatus_commit"] == apparatus_commit
    (tmp_path / "untracked.txt").write_text("no")
    with pytest.raises(CanaryLoaderError, match="changes"):
        verify_frozen_repository(tmp_path)


def test_post_authorization_history_is_linear_and_results_are_append_only(
        tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "MANDATORY_FROZEN_PATHS", {"a.py"})
    valid = tmp_path / "valid"
    valid.mkdir()
    authorize_test_repo(valid)
    result = valid / "results/coherent_canary_unit/raw.json"
    write(result, b"{}\n")
    git(valid, "add", result.relative_to(valid).as_posix())
    git(valid, "commit", "-m", "add result")
    observed = verify_frozen_repository(valid)
    assert observed["post_authorization_result_additions"] == [{
        "commit": git(valid, "rev-parse", "HEAD"),
        "path": "results/coherent_canary_unit/raw.json",
        "sha256": hashlib.sha256(b"{}\n").hexdigest(),
        "size_bytes": 3,
    }]
    result.write_text('{"changed": true}\n')
    git(valid, "add", result.relative_to(valid).as_posix())
    git(valid, "commit", "-m", "mutate result")
    with pytest.raises(CanaryLoaderError, match="not an addition"):
        verify_frozen_repository(valid)

    reverted = tmp_path / "reverted"
    reverted.mkdir()
    authorize_test_repo(reverted)
    (reverted / "a.py").write_text("forbidden = True\n")
    hidden_result = reverted / "results/coherent_canary_unit/tainted.json"
    write(hidden_result, b"{}\n")
    git(reverted, "add", "a.py", hidden_result.relative_to(reverted).as_posix())
    git(reverted, "commit", "-m", "run under changed apparatus")
    (reverted / "a.py").write_text("x = 1\n")
    git(reverted, "add", "a.py")
    git(reverted, "commit", "-m", "hide apparatus change")
    with pytest.raises(CanaryLoaderError, match="not an addition"):
        verify_frozen_repository(reverted)


def test_post_authorization_rejects_delete_rename_merge_and_symlink(
        tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "MANDATORY_FROZEN_PATHS", {"a.py"})

    deleted = tmp_path / "deleted"
    deleted.mkdir()
    authorize_test_repo(deleted)
    deleted_result = deleted / "results/coherent_canary_unit/deleted.json"
    write(deleted_result, b"{}\n")
    relative = deleted_result.relative_to(deleted).as_posix()
    git(deleted, "add", relative)
    git(deleted, "commit", "-m", "add result")
    deleted_result.unlink()
    git(deleted, "add", "-u", relative)
    git(deleted, "commit", "-m", "delete result")
    with pytest.raises(CanaryLoaderError):
        verify_frozen_repository(deleted)

    renamed = tmp_path / "renamed"
    renamed.mkdir()
    authorize_test_repo(renamed)
    old_result = renamed / "results/coherent_canary_unit/old.json"
    new_result = renamed / "results/coherent_canary_unit/new.json"
    write(old_result, b"{}\n")
    old_relative = old_result.relative_to(renamed).as_posix()
    new_relative = new_result.relative_to(renamed).as_posix()
    git(renamed, "add", old_relative)
    git(renamed, "commit", "-m", "add result")
    old_result.rename(new_result)
    git(renamed, "add", old_relative, new_relative)
    git(renamed, "commit", "-m", "rename result")
    with pytest.raises(CanaryLoaderError):
        verify_frozen_repository(renamed)

    merged = tmp_path / "merged"
    merged.mkdir()
    authorize_test_repo(merged)
    git(merged, "checkout", "-b", "side")
    side_result = merged / "results/coherent_canary_unit/side.json"
    write(side_result, b"{}\n")
    git(merged, "add", side_result.relative_to(merged).as_posix())
    git(merged, "commit", "-m", "side result")
    git(merged, "checkout", "trunk")
    main_result = merged / "results/coherent_canary_unit/main.json"
    write(main_result, b"{}\n")
    git(merged, "add", main_result.relative_to(merged).as_posix())
    git(merged, "commit", "-m", "main result")
    git(merged, "merge", "--no-ff", "side", "-m", "merge result branches")
    with pytest.raises(CanaryLoaderError, match="linear"):
        verify_frozen_repository(merged)

    linked = tmp_path / "linked"
    linked.mkdir()
    authorize_test_repo(linked)
    external = tmp_path / "external.json"
    external.write_text("{}\n")
    link = linked / "results/coherent_canary_unit/link.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(external)
    git(linked, "add", link.relative_to(linked).as_posix())
    git(linked, "commit", "-m", "add result symlink")
    with pytest.raises(CanaryLoaderError, match="regular data blob"):
        verify_frozen_repository(linked)


def test_protocol_tokenizer_exact_attestation_from_pinned_snapshot():
    snapshot = (Path.home() / ".cache/huggingface/hub/"
                "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots" /
                REVISION_30B)
    inventory = inventory_snapshot(
        snapshot, revision=REVISION_30B, kind="protocol_tokenizer",
        repo_id=MODEL_30B)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    assert attest_protocol_tokenizer(
        tokenizer, snapshot=snapshot,
        inventory=inventory) == PROTOCOL_TOKENIZER_ATTESTATION


def test_production_repository_fails_closed_without_authorization_file(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "branch", "-m", "trunk")
    (tmp_path / "a").write_text("x")
    git(tmp_path, "add", "a")
    git(tmp_path, "commit", "-m", "initial")
    with pytest.raises(CanaryLoaderError, match="authorization"):
        verify_frozen_repository(tmp_path)


def test_frozen_snapshot_contract_matches_actual_local_and_protocol_files():
    repo = Path(__file__).resolve().parents[1]
    contract = load_model_snapshot_contract(repo)
    cache = Path.home() / ".cache/huggingface/hub"
    local = cache / "models--Qwen--Qwen3-0.6B/snapshots" / LOCAL_SPEC.revision
    protocol = (cache /
                "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots" /
                REVISION_30B)
    local_inventory = inventory_snapshot(
        local, revision=LOCAL_SPEC.revision, kind="model", spec=LOCAL_SPEC)
    protocol_inventory = inventory_snapshot(
        protocol, revision=REVISION_30B, kind="protocol_tokenizer",
        repo_id=MODEL_30B)
    assert verify_inventory_contract(
        local_inventory, spec=LOCAL_SPEC,
        contract=contract)["parameter_count"] == 751632384
    assert verify_inventory_contract(
        protocol_inventory, spec=None,
        contract=contract)["inventory_sha256"] == \
        "aaf92eec1d388cb80d43138f3a578f206b7e6988b704c8a41f050ea17c3ab412"


def test_exact_contract_opens_against_pinned_index_and_canonical_file_hash():
    repo = Path(__file__).resolve().parents[1]
    contract = load_model_snapshot_contract(repo)
    exact = contract["subjects"][EXACT_SPEC.key]
    canonical_files = json.dumps(
        exact["files"], sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical_files).hexdigest() == \
        exact["inventory_sha256"]
    snapshot = (Path.home() / ".cache/huggingface/hub/"
                "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots" /
                REVISION_30B)
    index_path = snapshot / "model.safetensors.index.json"
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == next(
        row["sha256"] for row in exact["files"]
        if row["path"] == "model.safetensors.index.json")
    weight_map = json.loads(index_path.read_bytes())["weight_map"]
    name_to_shard = sorted(
        ({"name": name, "shard": shard} for name, shard in weight_map.items()),
        key=lambda row: row["name"])
    canonical_map = json.dumps(
        name_to_shard, sort_keys=True, separators=(",", ":")).encode()
    assert len(name_to_shard) == exact["weight_tensor_count"]
    assert hashlib.sha256(canonical_map).hexdigest() == \
        exact["weight_name_to_shard_sha256"]


def test_loaded_subject_attestation_rehashes_weights_and_rejects_quantization(
        tmp_path, monkeypatch):
    snapshot = fake_snapshot(tmp_path)
    model = Qwen3ForCausalLM(snapshot)
    weight_blob = (snapshot / "model.safetensors").resolve()
    weight_blob.write_bytes(save_safetensors({
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }))
    model_inventory = inventory_snapshot(
        snapshot, revision=LOCAL_SPEC.revision, kind="model", spec=LOCAL_SPEC)
    protocol_snapshot = (Path.home() / ".cache/huggingface/hub/"
                         "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots" /
                         REVISION_30B)
    protocol_inventory = inventory_snapshot(
        protocol_snapshot, revision=REVISION_30B,
        kind="protocol_tokenizer", repo_id=MODEL_30B)
    tokenizer = AutoTokenizer.from_pretrained(
        str(protocol_snapshot), local_files_only=True, trust_remote_code=False)
    loading_info = {"missing_keys": [], "unexpected_keys": [],
                    "mismatched_keys": [], "error_msgs": []}
    model_rows = model_inventory["weight_tensors"]
    local_contract_entry = {
        "model_id": LOCAL_SPEC.model_id, "revision": LOCAL_SPEC.revision,
        "files": model_inventory["files"],
        "inventory_sha256": model_inventory["inventory_sha256"],
        "parameter_count": sum(
            torch.tensor(row["shape"]).prod().item() for row in model_rows),
        "weight_tensor_count": len(model_rows),
        "weight_tensors_sha256": model_inventory["weight_tensors_sha256"],
    }
    protocol_contract_entry = {
        "model_id": MODEL_30B, "revision": REVISION_30B,
        "files": protocol_inventory["files"],
        "inventory_sha256": protocol_inventory["inventory_sha256"],
    }
    snapshot_contract = {
        "schema": "coherent_state_decision_canary_v12_model_snapshot_contract_v1",
        "design_id": "coherent-state-decision-canary-v12",
        "subjects": {
            LOCAL_SPEC.key: local_contract_entry,
            EXACT_SPEC.key: {"unused_in_fake_attestation": True},
        },
        "protocol_tokenizer": protocol_contract_entry,
    }
    repo = tmp_path / "release-repo"
    contract_path = repo / loader_module.MODEL_SNAPSHOT_CONTRACT_PATH
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps(snapshot_contract, sort_keys=True) + "\n")
    repository_authorization = {
        "branch": "trunk", "head": "d" * 40,
        "apparatus_commit": "a" * 40,
        "authorization_commit": "b" * 40,
        "files": [{"path": loader_module.MODEL_SNAPSHOT_CONTRACT_PATH.as_posix(),
                   "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest()}],
        "inventory_sha256": "c" * 64,
        "authorization_sha256": "e" * 64,
        "post_authorization_result_additions": [],
    }
    monkeypatch.setattr(loader_module, "verify_frozen_repository",
                        lambda observed_repo: repository_authorization)
    monkeypatch.setattr(loader_module, "load_model_snapshot_contract",
                        lambda observed_repo: snapshot_contract)
    release = {
        "repo": repo,
        "repository_authorization": repository_authorization,
        "snapshot_contract": snapshot_contract,
    }
    result = attest_loaded_subject(
        model, tokenizer, spec=LOCAL_SPEC, model_snapshot=snapshot, **release,
        model_inventory=model_inventory,
        protocol_tokenizer_snapshot=protocol_snapshot,
        protocol_tokenizer_inventory=protocol_inventory,
        loading_info=loading_info)
    assert result["all_parameters_frozen"] is True
    assert result["protocol_tokenizer_attestation"] == \
        PROTOCOL_TOKENIZER_ATTESTATION
    assert result["release_binding"]["apparatus_commit"] == "a" * 40
    assert result["release_binding"]["subject_contract_entry_sha256"] == \
        hashlib.sha256(json.dumps(
            local_contract_entry, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(CanaryLoaderError, match="authorization changed"):
        attest_loaded_subject(
            model, tokenizer, spec=LOCAL_SPEC, model_snapshot=snapshot,
            **{**release, "repository_authorization": {
                **repository_authorization, "head": "f" * 40}},
            model_inventory=model_inventory,
            protocol_tokenizer_snapshot=protocol_snapshot,
            protocol_tokenizer_inventory=protocol_inventory,
            loading_info=loading_info)
    wrong_shape_model = Qwen3ForCausalLM(snapshot)
    wrong_shape_model.embed_tokens = torch.nn.Embedding(
        151669, 2, dtype=torch.bfloat16)
    with pytest.raises(CanaryLoaderError, match="names/shapes/dtypes"):
        attest_loaded_subject(
            wrong_shape_model, tokenizer, spec=LOCAL_SPEC,
            model_snapshot=snapshot, **release,
            model_inventory=model_inventory,
            protocol_tokenizer_snapshot=protocol_snapshot,
            protocol_tokenizer_inventory=protocol_inventory,
            loading_info=loading_info)
    with pytest.raises(CanaryLoaderError, match="missing_keys"):
        attest_loaded_subject(
            model, tokenizer, spec=LOCAL_SPEC, model_snapshot=snapshot, **release,
            model_inventory=model_inventory,
            protocol_tokenizer_snapshot=protocol_snapshot,
            protocol_tokenizer_inventory=protocol_inventory,
            loading_info={**loading_info, "missing_keys": ["missing.weight"]})
    model.is_quantized = True
    with pytest.raises(CanaryLoaderError, match="quantized"):
        attest_loaded_subject(
            model, tokenizer, spec=LOCAL_SPEC, model_snapshot=snapshot, **release,
            model_inventory=model_inventory,
            protocol_tokenizer_snapshot=protocol_snapshot,
            protocol_tokenizer_inventory=protocol_inventory,
            loading_info=loading_info)
    model.is_quantized = False
    changed = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    first = next(iter(changed))
    changed[first].view(-1)[0] = torch.nextafter(
        changed[first].view(-1)[0],
        torch.tensor(float("inf"), dtype=torch.bfloat16))
    weight_blob.write_bytes(save_safetensors(changed))
    with pytest.raises(CanaryLoaderError, match="changed across load"):
        attest_loaded_subject(
            model, tokenizer, spec=LOCAL_SPEC, model_snapshot=snapshot, **release,
            model_inventory=model_inventory,
            protocol_tokenizer_snapshot=protocol_snapshot,
            protocol_tokenizer_inventory=protocol_inventory,
            loading_info=loading_info)
