"""Pre-text tests for the powered-v13 seed/permutation boundary."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess

import pytest

import powered_v13_permutation as permutation
import powered_v13_recipe as recipe


TEST_PRESEED_COMMIT = "a" * 40
TEST_SEED_COMMIT = "b" * 40
GOLDEN_SEED = "000102030405060708090a0b0c0d0e0f"
GOLDEN_FIRST_TEN = [
    3873, 1811, 1140, 3884, 1876, 3037, 1328, 3852, 1620, 3559,
]
GOLDEN_FIRST_THREE_IDS = [
    "86203d61286654f48cb5313c770047f22feb0a1a0bd3d7fbbcc81d162544467d",
    "7735fc89b851d2debb5c85c326df4eae95161909d661ead7030d3cca620fe0ae",
    "606fec2ba4cb691a3ec7b6e303fd89e8d5b393802ea76634d4d4b47bd3590d48",
]
GOLDEN_INDEX_SHA = (
    "2946e202e58eee1d0e2b3be6904bdbfe55b2d2beb9355f5de5e997d3bd4714f0")
GOLDEN_RANKED_ID_SHA = (
    "7c5c26ee50fa88b221368b4e65f02e24acdb9bfdbfc4a36c078706855d609e8c")


def _inventory() -> list[dict[str, str]]:
    return [
        {"path": path, "sha256": f"{index + 1:064x}"}
        for index, path in enumerate(permutation.PRESEED_INPUT_PATHS)
    ]


def _seeds() -> dict[str, str]:
    return {
        stratum: f"{index + 1:032x}"
        for index, stratum in enumerate(recipe.STRATA)
    }


def _pretty_json(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False).encode("utf-8") + b"\n"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.strip()


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pushed_preseed_repo(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "origin.git"
    remote.mkdir()
    _git(remote, "init", "--bare")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "trunk")
    _git(repo, "config", "user.email", "permutation-test@example.com")
    _git(repo, "config", "user.name", "Permutation Test")
    for index, relative in enumerate(permutation.PRESEED_INPUT_PATHS):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic preseed input {index}\n".encode())
    _git(repo, "add", *permutation.PRESEED_INPUT_PATHS)
    _git(repo, "commit", "-m", "synthetic preseed root")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "trunk")
    return repo, remote


def _write_pushed_seed_c1(repo: Path) -> tuple[bytes, str, object]:
    script = _load_script("write_powered_v13_seed_manifest")
    script.REPO = repo
    assert script.main([]) == 0
    seed_path = repo / permutation.SEED_MANIFEST_RELATIVE_PATH
    seed_raw = seed_path.read_bytes()
    _git(repo, "add", permutation.SEED_MANIFEST_RELATIVE_PATH)
    _git(repo, "commit", "-m", "freeze synthetic seeds")
    _git(repo, "push", "origin", "trunk")
    return seed_raw, _git(repo, "rev-parse", "HEAD"), script


def _write_literal_worktree(repo: Path) -> tuple[Path, object]:
    script = _load_script("write_powered_v13_literal_permutations")
    script.REPO = repo
    assert script.main([]) == 0
    return repo / permutation.LITERAL_PERMUTATION_RELATIVE_PATH, script


def _commit_literal_c2(repo: Path, *, push: bool,
                       extra_path: str | None = None) -> str:
    _git(repo, "add", permutation.LITERAL_PERMUTATION_RELATIVE_PATH)
    if extra_path is not None:
        path = repo / extra_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("extra C2 content\n")
        _git(repo, "add", extra_path)
    _git(repo, "commit", "-m", "freeze synthetic literal permutations")
    if push:
        _git(repo, "push", "origin", "trunk")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def seed_manifest() -> dict:
    return permutation.build_seed_manifest(permutation.SeedManifestInput(
        observed_utc="2026-07-12T18:00:00Z",
        seeds=_seeds(),
        preseed_git_commit=TEST_PRESEED_COMMIT,
        input_inventory=_inventory(),
        git_clean_before_write=True,
    ))


@pytest.fixture(scope="module")
def seed_raw(seed_manifest: dict) -> bytes:
    return _pretty_json(seed_manifest)


@pytest.fixture(scope="module")
def literal(seed_raw: bytes) -> dict:
    return permutation.build_literal_permutations(
        seed_raw,
        seed_git_commit=TEST_SEED_COMMIT,
        observed_utc="2026-07-12T18:01:00Z",
    )


def test_construction_is_pre_text_and_git_materializer_has_no_binding_inputs():
    source = inspect.getsource(permutation)
    assert "materialize_development_sentinel" not in source
    assert "_messages_for_variant" not in source
    assert "import numpy as np" not in source
    assert "def _numpy_module" in source
    signature = inspect.signature(
        permutation.materialize_committed_ranked_candidate)
    assert list(signature.parameters) == ["stratum", "rank"]
    assert not hasattr(permutation, "ranked_materialization_authorization")


def test_seed_manifest_binds_exact_frame_entropy_and_no_later_state(
        seed_manifest: dict):
    assert permutation.validate_seed_manifest(seed_manifest) == _seeds()
    assert seed_manifest["entropy"] == {
        "api": "secrets.token_bytes",
        "bytes_per_request": 16,
        "requests": 8,
        "stratum_order": list(recipe.STRATA),
    }
    assert seed_manifest["repository_boundary"] == {
        "branch": "trunk",
        "git_clean_before_write": True,
        "preseed_git_commit": TEST_PRESEED_COMMIT,
    }
    assert [row["path"] for row in seed_manifest["input_inventory"]] == list(
        permutation.PRESEED_INPUT_PATHS)
    assert seed_manifest["randomization_boundary"] == permutation.SEED_BOUNDARY
    encoded = _pretty_json(seed_manifest).lower()
    assert b"permutation-seeds-v1" not in encoded
    assert b"ranked_candidate" not in encoded
    assert b"messages" not in encoded


def test_seed_manifest_rejects_dirty_claim_duplicate_seed_and_inventory_drift():
    common = dict(
        observed_utc="2026-07-12T18:00:00Z",
        seeds=_seeds(),
        preseed_git_commit=TEST_PRESEED_COMMIT,
        input_inventory=_inventory(),
    )
    with pytest.raises(permutation.V13PermutationError, match="clean"):
        permutation.build_seed_manifest(permutation.SeedManifestInput(
            **common, git_clean_before_write=False))
    duplicate = _seeds()
    duplicate[recipe.STRATA[1]] = duplicate[recipe.STRATA[0]]
    with pytest.raises(permutation.V13PermutationError, match="not unique"):
        permutation.build_seed_manifest(permutation.SeedManifestInput(
            **(common | {"seeds": duplicate}), git_clean_before_write=True))
    inventory = _inventory()
    inventory[0] = {"path": "../escape", "sha256": "0" * 64}
    with pytest.raises(permutation.V13PermutationError, match="path 0 differs"):
        permutation.build_seed_manifest(permutation.SeedManifestInput(
            **(common | {"input_inventory": inventory}),
            git_clean_before_write=True))


def test_strict_seed_schema_rejects_extra_and_nested_extra(seed_manifest: dict):
    changed = copy.deepcopy(seed_manifest)
    changed["covert_history"] = "text"
    with pytest.raises(permutation.V13PermutationError, match="keys differ"):
        permutation.validate_seed_manifest(changed)
    changed = copy.deepcopy(seed_manifest)
    changed["repository_boundary"]["caller_claim"] = True
    with pytest.raises(permutation.V13PermutationError, match="keys differ"):
        permutation.validate_seed_manifest(changed)


def test_strict_json_loader_rejects_duplicate_keys():
    with pytest.raises(permutation.V13PermutationError, match="duplicate JSON key"):
        permutation.parse_json_mapping_bytes(b'{"a":1,"a":2}', "duplicate")


def test_numpy_251_golden_vector_is_exact_canonical_json():
    result = permutation.build_stratum_permutation(
        "threshold_eligibility", GOLDEN_SEED)
    assert result["compact_indices"][:10] == GOLDEN_FIRST_TEN
    assert result["ranked_candidate_ids"][:3] == GOLDEN_FIRST_THREE_IDS
    assert result["compact_indices_sha256"] == GOLDEN_INDEX_SHA
    assert result["ranked_candidate_ids_sha256"] == GOLDEN_RANKED_ID_SHA
    assert result["compact_indices_sha256"] == permutation.sha256_bytes(
        permutation.canonical_json_bytes(result["compact_indices"]))


def test_literal_is_compact_complete_and_contains_only_first_ten_tuples(
        literal: dict):
    encoded = _pretty_json(literal)
    assert len(encoded) < 4_000_000
    assert literal["global_candidate_count"] == 32768
    all_ids: list[str] = []
    for stratum in recipe.STRATA:
        record = literal["strata"][stratum]
        assert len(record["compact_indices"]) == 4096
        assert sorted(record["compact_indices"]) == list(range(4096))
        assert len(record["ranked_candidate_ids"]) == 4096
        assert len(record["first_ten"]) == 10
        assert [row["rank"] for row in record["first_ten"]] == list(range(1, 11))
        all_ids.extend(record["ranked_candidate_ids"])
    assert len(all_ids) == len(set(all_ids)) == 32768


def test_literal_structural_and_construction_replay_both_pass(
        literal: dict, seed_raw: bytes):
    permutation.validate_literal_permutations(
        literal, seed_raw, expected_seed_git_commit=TEST_SEED_COMMIT)
    permutation.validate_construction_rng_replay(
        literal, seed_raw, expected_seed_git_commit=TEST_SEED_COMMIT)


def test_durable_structural_verifier_never_calls_numpy(
        literal: dict, seed_raw: bytes, monkeypatch: pytest.MonkeyPatch):
    def forbidden_numpy():
        raise AssertionError("durable verification called NumPy")

    monkeypatch.setattr(permutation, "_numpy_module", forbidden_numpy)
    permutation.validate_literal_permutations(
        literal, seed_raw, expected_seed_git_commit=TEST_SEED_COMMIT)
    with pytest.raises(AssertionError, match="called NumPy"):
        permutation.validate_construction_rng_replay(
            literal, seed_raw, expected_seed_git_commit=TEST_SEED_COMMIT)


def test_structural_verifier_rejects_bool_index_and_forged_id(
        literal: dict, seed_raw: bytes):
    changed = copy.deepcopy(literal)
    stratum = recipe.STRATA[0]
    changed["strata"][stratum]["compact_indices"][0] = True
    with pytest.raises(permutation.V13PermutationError, match="indices are invalid"):
        permutation.validate_literal_permutations(
            changed, seed_raw, expected_seed_git_commit=TEST_SEED_COMMIT)
    changed = copy.deepcopy(literal)
    changed["strata"][stratum]["ranked_candidate_ids"][0] = "f" * 64
    with pytest.raises(permutation.V13PermutationError,
                       match="ranked candidate IDs differ"):
        permutation.validate_literal_permutations(
            changed, seed_raw, expected_seed_git_commit=TEST_SEED_COMMIT)


def test_literal_rejects_extra_field_wrong_seed_bytes_and_wrong_commit(
        literal: dict, seed_manifest: dict, seed_raw: bytes):
    changed = dict(literal)
    changed["unreviewed_text"] = "leak"
    with pytest.raises(permutation.V13PermutationError, match="keys differ"):
        permutation.validate_literal_permutations(
            changed, seed_raw, expected_seed_git_commit=TEST_SEED_COMMIT)
    other_seed = copy.deepcopy(seed_manifest)
    other_seed["seeds"][recipe.STRATA[0]] = "f" * 32
    with pytest.raises(permutation.V13PermutationError, match="SHA binding"):
        permutation.validate_literal_permutations(
            literal, _pretty_json(other_seed),
            expected_seed_git_commit=TEST_SEED_COMMIT)
    with pytest.raises(permutation.V13PermutationError, match="commit binding"):
        permutation.validate_literal_permutations(
            literal, seed_raw, expected_seed_git_commit="c" * 40)


def test_git_materializer_hard_caps_rank_before_git_or_text(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Git or history materializer was reached")

    monkeypatch.setattr(permutation, "_git_text", forbidden)
    monkeypatch.setattr(permutation, "_materialize_ranked_candidate", forbidden)
    with pytest.raises(permutation.V13PermutationError, match="rank is invalid"):
        permutation.materialize_committed_ranked_candidate(
            stratum=recipe.STRATA[0], rank=11)


def test_exclusive_writer_never_overwrites(tmp_path: Path, seed_manifest: dict):
    path = tmp_path / "seed.json"
    permutation.write_json_exclusive(path, seed_manifest)
    original = path.read_bytes()
    assert permutation.sha256_file(path) == permutation.sha256_bytes(original)
    with pytest.raises(FileExistsError):
        permutation.write_json_exclusive(path, {"different": True})
    assert path.read_bytes() == original


def test_seed_generator_requests_exactly_eight_independent_16_byte_values(
        monkeypatch: pytest.MonkeyPatch):
    script = _load_script("write_powered_v13_seed_manifest")
    calls: list[int] = []

    def synthetic_entropy(width: int) -> bytes:
        calls.append(width)
        return len(calls).to_bytes(width, "big")

    monkeypatch.setattr(script.secrets, "token_bytes", synthetic_entropy)
    seeds = script._os_random_seeds()
    assert calls == [16] * len(recipe.STRATA)
    assert list(seeds) == list(recipe.STRATA)
    assert len(set(seeds.values())) == len(recipe.STRATA)


def test_seed_generator_aborts_duplicate_after_exactly_eight_requests(
        monkeypatch: pytest.MonkeyPatch):
    script = _load_script("write_powered_v13_seed_manifest")
    calls: list[int] = []

    def duplicate_entropy(width: int) -> bytes:
        calls.append(width)
        value = 1 if len(calls) in (1, 8) else len(calls)
        return value.to_bytes(width, "big")

    monkeypatch.setattr(script.secrets, "token_bytes", duplicate_entropy)
    with pytest.raises(script.SeedWriterError, match="without resampling"):
        script._os_random_seeds()
    assert calls == [16] * len(recipe.STRATA)


def test_real_git_object_c0_c1_c2_reaches_only_nontext_internal_compiler(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    repo, _remote = _pushed_preseed_repo(tmp_path)
    seed_raw, seed_commit, _seed_script = _write_pushed_seed_c1(repo)
    seed_path = repo / permutation.SEED_MANIFEST_RELATIVE_PATH
    assert seed_path.exists()
    assert not (repo / permutation.LITERAL_PERMUTATION_RELATIVE_PATH).exists()
    assert _git(repo, "status", "--short") == ""

    literal_path, _literal_script = _write_literal_worktree(repo)
    manifest = permutation.parse_json_mapping_bytes(
        seed_raw, "synthetic seed manifest")
    assert manifest["repository_boundary"]["preseed_git_commit"] == _git(
        repo, "rev-parse", f"{seed_commit}^")
    assert literal_path.exists() and literal_path.stat().st_size < 4_000_000
    literal = permutation.parse_json_mapping_bytes(
        literal_path.read_bytes(), "synthetic literal")
    permutation.validate_literal_permutations(
        literal, seed_raw, expected_seed_git_commit=seed_commit)
    permutation.validate_construction_rng_replay(
        literal, seed_raw, expected_seed_git_commit=seed_commit)
    literal_commit = _commit_literal_c2(repo, push=True)
    monkeypatch.setattr(permutation, "REPO_ROOT", repo)
    reached: dict[str, object] = {}

    def nontext_compiler(candidate, authorization, *, capability):
        reached.update({
            "candidate": candidate,
            "authorization": authorization,
            "capability": capability,
        })
        return {"status": "NON_TEXT_INTERNAL_COMPILER_REACHED"}

    monkeypatch.setattr(
        permutation, "_materialize_ranked_candidate", nontext_compiler)
    result = permutation.materialize_committed_ranked_candidate(
        stratum=recipe.STRATA[0], rank=10)
    assert result == {"status": "NON_TEXT_INTERNAL_COMPILER_REACHED"}
    candidate = reached["candidate"]
    authorization = reached["authorization"]
    assert authorization.candidate_id == recipe.stable_candidate_id(candidate)
    assert authorization.permutation_rank == 10
    assert authorization.seed_manifest_sha256 == permutation.sha256_bytes(seed_raw)
    assert authorization.seed_git_commit == seed_commit
    assert authorization.literal_permutation_sha256 == permutation.sha256_file(
        literal_path)
    assert authorization.literal_git_commit == literal_commit
    assert reached["capability"] is recipe._GIT_ANCHORED_MATERIALIZATION_CAPABILITY
    output = capsys.readouterr().out
    assert "ranked_conversation_text_materialized" in output
    assert '"ranked_conversation_text_materialized": false' in output


def test_uncommitted_arbitrary_seed_literal_pair_cannot_resolve(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, _remote = _pushed_preseed_repo(tmp_path)
    _write_pushed_seed_c1(repo)
    literal_path, _script = _write_literal_worktree(repo)
    literal_path.write_bytes(literal_path.read_bytes() + b" ")
    monkeypatch.setattr(permutation, "REPO_ROOT", repo)
    monkeypatch.setattr(
        permutation, "_materialize_ranked_candidate",
        lambda *_args, **_kwargs: pytest.fail("text compiler was reached"))
    with pytest.raises(permutation.V13PermutationError, match="clean tree"):
        permutation.materialize_committed_ranked_candidate(
            stratum=recipe.STRATA[0], rank=1)


def test_unpushed_literal_c2_cannot_resolve(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, _remote = _pushed_preseed_repo(tmp_path)
    _write_pushed_seed_c1(repo)
    _write_literal_worktree(repo)
    _commit_literal_c2(repo, push=False)
    monkeypatch.setattr(permutation, "REPO_ROOT", repo)
    monkeypatch.setattr(
        permutation, "_materialize_ranked_candidate",
        lambda *_args, **_kwargs: pytest.fail("text compiler was reached"))
    with pytest.raises(permutation.V13PermutationError, match="not pushed"):
        permutation.materialize_committed_ranked_candidate(
            stratum=recipe.STRATA[0], rank=1)


def test_non_dedicated_literal_c2_cannot_resolve(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, _remote = _pushed_preseed_repo(tmp_path)
    _write_pushed_seed_c1(repo)
    _write_literal_worktree(repo)
    _commit_literal_c2(repo, push=True, extra_path="extra-c2.txt")
    monkeypatch.setattr(permutation, "REPO_ROOT", repo)
    monkeypatch.setattr(
        permutation, "_materialize_ranked_candidate",
        lambda *_args, **_kwargs: pytest.fail("text compiler was reached"))
    with pytest.raises(permutation.V13PermutationError,
                       match="must add only"):
        permutation.materialize_committed_ranked_candidate(
            stratum=recipe.STRATA[0], rank=1)


def test_non_dedicated_seed_c1_cannot_resolve(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, _remote = _pushed_preseed_repo(tmp_path)
    seed_script = _load_script("write_powered_v13_seed_manifest")
    seed_script.REPO = repo
    assert seed_script.main([]) == 0
    seed_path = repo / permutation.SEED_MANIFEST_RELATIVE_PATH
    seed_raw = seed_path.read_bytes()
    (repo / "extra-c1.txt").write_text("extra C1 content\n")
    _git(repo, "add", permutation.SEED_MANIFEST_RELATIVE_PATH, "extra-c1.txt")
    _git(repo, "commit", "-m", "non-dedicated synthetic seed commit")
    _git(repo, "push", "origin", "trunk")
    seed_commit = _git(repo, "rev-parse", "HEAD")
    literal = permutation.build_literal_permutations(
        seed_raw,
        seed_git_commit=seed_commit,
        observed_utc="2026-07-12T18:01:00Z",
    )
    permutation.write_json_exclusive(
        repo / permutation.LITERAL_PERMUTATION_RELATIVE_PATH, literal)
    _commit_literal_c2(repo, push=True)
    monkeypatch.setattr(permutation, "REPO_ROOT", repo)
    monkeypatch.setattr(
        permutation, "_materialize_ranked_candidate",
        lambda *_args, **_kwargs: pytest.fail("text compiler was reached"))
    with pytest.raises(permutation.V13PermutationError,
                       match="seed manifest commit must add only"):
        permutation.materialize_committed_ranked_candidate(
            stratum=recipe.STRATA[0], rank=1)


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("parent", "manifest-bound C0"),
        ("inventory", "C0 inventory Git-object hash differs"),
    ),
)
def test_git_materializer_rejects_forged_c0_boundary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, match):
    repo, _remote = _pushed_preseed_repo(tmp_path)
    seed_script = _load_script("write_powered_v13_seed_manifest")
    seed_script.REPO = repo
    assert seed_script.main([]) == 0
    seed_path = repo / permutation.SEED_MANIFEST_RELATIVE_PATH
    manifest = permutation.parse_json_mapping_bytes(
        seed_path.read_bytes(), "synthetic seed manifest")
    if mutation == "parent":
        manifest["repository_boundary"]["preseed_git_commit"] = "a" * 40
    else:
        manifest["input_inventory"][0]["sha256"] = "0" * 64
    seed_path.write_bytes(_pretty_json(manifest))
    seed_raw = seed_path.read_bytes()
    _git(repo, "add", permutation.SEED_MANIFEST_RELATIVE_PATH)
    _git(repo, "commit", "-m", "forged synthetic seed boundary")
    _git(repo, "push", "origin", "trunk")
    seed_commit = _git(repo, "rev-parse", "HEAD")
    literal = permutation.build_literal_permutations(
        seed_raw,
        seed_git_commit=seed_commit,
        observed_utc="2026-07-12T18:01:00Z",
    )
    permutation.write_json_exclusive(
        repo / permutation.LITERAL_PERMUTATION_RELATIVE_PATH, literal)
    _commit_literal_c2(repo, push=True)
    monkeypatch.setattr(permutation, "REPO_ROOT", repo)
    monkeypatch.setattr(
        permutation, "_materialize_ranked_candidate",
        lambda *_args, **_kwargs: pytest.fail("text compiler was reached"))
    with pytest.raises(permutation.V13PermutationError, match=match):
        permutation.materialize_committed_ranked_candidate(
            stratum=recipe.STRATA[0], rank=1)


def test_dirty_pushed_literal_c2_cannot_resolve(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, _remote = _pushed_preseed_repo(tmp_path)
    _write_pushed_seed_c1(repo)
    _write_literal_worktree(repo)
    _commit_literal_c2(repo, push=True)
    (repo / "untracked-after-c2.txt").write_text("dirty\n")
    monkeypatch.setattr(permutation, "REPO_ROOT", repo)
    monkeypatch.setattr(
        permutation, "_materialize_ranked_candidate",
        lambda *_args, **_kwargs: pytest.fail("text compiler was reached"))
    with pytest.raises(permutation.V13PermutationError, match="clean tree"):
        permutation.materialize_committed_ranked_candidate(
            stratum=recipe.STRATA[0], rank=1)


def test_seed_writer_rejects_dirty_untracked_preseed_repo(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, _remote = _pushed_preseed_repo(tmp_path)
    script = _load_script("write_powered_v13_seed_manifest")
    monkeypatch.setattr(script, "REPO", repo)
    (repo / "untracked.txt").write_text("dirty\n")
    with pytest.raises(script.SeedWriterError, match="clean"):
        script._require_clean_integration_head()
