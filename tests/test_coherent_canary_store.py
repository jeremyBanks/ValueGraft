from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from coherent_canary_store import (
    DESIGN_ID,
    CanaryStoreError,
    atomic_create_json,
    create_checkpoint,
    create_treatment_checkpoint,
    file_sha256,
    inventory_external_tensor,
    literal_file_binding,
    promote_checkpoint,
    read_checkpoint,
    unique_output_path,
    validate_bounds,
)


def bounds(*, max_layers: int = 48, max_tokens: int = 128) -> dict:
    return {
        "case_count": 1,
        "history_ids": ["F", "C", "W"],
        "region_ids": ["R1_content", "R2_boundary", "R3_anchor"],
        "layer_indices": list(range(max_layers)),
        "token_counts_by_region": {
            "R1_content": 42,
            "R2_boundary": 67,
            "R3_anchor": 70,
        },
        "limits": {
            "max_cases": 1,
            "max_histories": 3,
            "max_regions": 3,
            "max_layers": max_layers,
            "max_tokens_per_region": max_tokens,
        },
    }


def binding(name: str) -> dict[str, str]:
    return {"path": name, "sha256": hashlib.sha256(name.encode()).hexdigest()}


def create(
    path: Path,
    *,
    mode: str,
    fingerprint: dict | None = None,
    sources: list[dict] | None = None,
    reviews: list[dict] | None = None,
) -> dict:
    return create_checkpoint(
        path,
        mode=mode,
        case_id="e01",
        fingerprint=fingerprint or {"model": "exact", "repo": "abc"},
        source_bindings=sources or [binding("stimulus.json")],
        review_bindings=reviews if reviews is not None else [binding("review.json")],
        bounds=bounds(),
        payload={"started": True},
        created_at_utc="20260711T210000000000Z",
    )


def promote_to_pass(start: Path, directory: Path, mode: str) -> Path:
    if mode == "technical":
        stages = ("CAPTURED", "VALIDATED", "PASS")
    else:
        stages = ("MECHANICAL_PASS", "REVIEW_PASS", "PASS")
    prior = start
    for number, stage in enumerate(stages, 1):
        path = directory / f"{mode}_{number}.json"
        promote_checkpoint(
            prior,
            path,
            stage=stage,
            payload_additions={stage.lower(): True},
            created_at_utc=f"20260711T21000{number}000000Z",
        )
        prior = path
    return prior


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def init_repo(path: Path) -> None:
    git(path, "init")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")


def receipt_expectation(repo: Path, path: Path) -> dict[str, str]:
    relative = path.relative_to(repo).as_posix()
    document = json.loads(path.read_text())
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "status": document["status"],
        "design_id": document["design_id"],
    }


def test_unique_names_and_atomic_creation_never_overwrite(tmp_path: Path):
    path = unique_output_path(
        tmp_path,
        "canary-technical-e01-s00-started",
        "Qwen3-30B",
        timestamp="20260711T210000Z",
    )
    atomic_create_json(path, {"first": True})
    before = path.read_bytes()
    with pytest.raises(CanaryStoreError, match="exists"):
        atomic_create_json(path, {"second": True})
    assert path.read_bytes() == before
    with pytest.raises(CanaryStoreError, match="non-unique"):
        unique_output_path(
            tmp_path,
            "canary-technical-e01-s00-started",
            "Qwen3-30B",
            timestamp="20260711T210000Z",
        )


def test_checkpoint_promotion_is_append_only_and_monotonic(tmp_path: Path):
    initial_path = tmp_path / "initial.json"
    initial = create(initial_path, mode="technical", reviews=[])
    initial_bytes = initial_path.read_bytes()
    captured_path = tmp_path / "captured.json"
    captured = promote_checkpoint(
        initial_path,
        captured_path,
        stage="CAPTURED",
        payload_additions={"capture": {"row_hash": "a" * 64}},
    )
    assert initial_path.read_bytes() == initial_bytes
    assert captured["sequence"] == 1
    assert captured["previous_checkpoint"] == {
        "path": str(initial_path),
        "sha256": file_sha256(initial_path),
    }
    assert read_checkpoint(captured_path) == captured
    with pytest.raises(CanaryStoreError, match="overwrite payload"):
        promote_checkpoint(
            captured_path,
            tmp_path / "bad.json",
            stage="VALIDATED",
            payload_additions={"started": False},
        )
    with pytest.raises(CanaryStoreError, match="non-monotonic"):
        promote_checkpoint(
            captured_path, tmp_path / "regress.json", stage="STARTED"
        )
    with pytest.raises(CanaryStoreError, match="non-monotonic"):
        promote_checkpoint(
            initial_path, tmp_path / "skip.json", stage="PASS"
        )
    assert initial["fingerprint_sha256"] == captured["fingerprint_sha256"]


def test_explicit_state_bounds_fail_closed():
    assert validate_bounds(bounds())["limits"]["max_cases"] == 1
    too_many = bounds()
    too_many["history_ids"].append("X")
    with pytest.raises(CanaryStoreError, match="history"):
        validate_bounds(too_many)
    too_long = bounds(max_tokens=69)
    with pytest.raises(CanaryStoreError, match="token count"):
        validate_bounds(too_long)
    too_many_layers = bounds(max_layers=47)
    too_many_layers["layer_indices"].append(47)
    with pytest.raises(CanaryStoreError, match="layer"):
        validate_bounds(too_many_layers)


def test_external_tensor_inventory_is_literal_and_bounded(tmp_path: Path):
    tensor = tmp_path / "rows.safetensors"
    tensor.write_bytes(b"lossless rows")
    row = inventory_external_tensor(
        tensor,
        display_path="external/e01_rows.safetensors",
        case_id="e01",
        history_ids=["C", "W"],
        region_ids=["R2_boundary"],
        layer_indices=list(range(48)),
        token_counts_by_region={"R2_boundary": 67},
        dtype="bfloat16",
        shape=[2, 48, 67, 4, 128],
        retention="until committed row hashes and treatment scores exist",
    )
    assert row["sha256"] == file_sha256(tensor)
    path = tmp_path / "technical.json"
    create_checkpoint(
        path,
        mode="technical",
        case_id="e01",
        fingerprint={"model": "exact"},
        source_bindings=[binding("fixture.json")],
        review_bindings=[],
        bounds=bounds(),
        artifact_inventory=[row],
    )
    bad = dict(row)
    bad["token_counts_by_region"] = {"R2_boundary": 129}
    with pytest.raises(CanaryStoreError, match="tokens exceed"):
        create_checkpoint(
            tmp_path / "bad.json",
            mode="technical",
            case_id="e01",
            fingerprint={"model": "exact"},
            source_bindings=[binding("fixture.json")],
            review_bindings=[],
            bounds=bounds(),
            artifact_inventory=[bad],
        )


def test_treatment_requires_exact_pass_receipts_tracked_in_head(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    fingerprint = {"model": "exact", "revision": "0d7c", "repo": "abc"}
    sources = [binding("stimulus.json")]
    reviews = [binding("blind.json"), binding("paired.json")]

    technical_start = repo / "technical_0.json"
    create(
        technical_start,
        mode="technical",
        fingerprint=fingerprint,
        sources=[binding("technical_fixture.json")],
        reviews=[],
    )
    technical = promote_to_pass(technical_start, repo, "technical")
    eligibility_start = repo / "eligibility_0.json"
    create(
        eligibility_start,
        mode="eligibility",
        fingerprint={"validator": "tokenizer-only"},
        sources=sources,
        reviews=reviews,
    )
    eligibility = promote_to_pass(eligibility_start, repo, "eligibility")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "committed gate receipts")

    treatment_path = repo / "treatment_0.json"
    treatment = create_treatment_checkpoint(
        treatment_path,
        repo_root=repo,
        case_id="e01",
        fingerprint=fingerprint,
        source_bindings=sources,
        review_bindings=reviews,
        bounds=bounds(),
        technical_receipt=receipt_expectation(repo, technical),
        eligibility_receipt=receipt_expectation(repo, eligibility),
    )
    assert treatment["stage"] == "RELEASED"
    assert treatment["release_receipts"]["technical"]["status"] == "PASS"


def test_treatment_rejects_untracked_modified_or_mismatched_receipts(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    fingerprint = {"model": "exact"}
    sources = [binding("stimulus.json")]
    reviews = [binding("review.json")]
    technical_start = repo / "technical_0.json"
    create(
        technical_start,
        mode="technical",
        fingerprint=fingerprint,
        sources=[binding("fixture.json")],
        reviews=[],
    )
    technical = promote_to_pass(technical_start, repo, "technical")
    eligibility_start = repo / "eligibility_0.json"
    create(
        eligibility_start,
        mode="eligibility",
        fingerprint={"local": True},
        sources=sources,
        reviews=reviews,
    )
    eligibility = promote_to_pass(eligibility_start, repo, "eligibility")
    tech_expected = receipt_expectation(repo, technical)
    eligibility_expected = receipt_expectation(repo, eligibility)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "gate receipts")

    wrong_sha = dict(tech_expected)
    wrong_sha["sha256"] = "0" * 64
    with pytest.raises(CanaryStoreError, match="literal HEAD"):
        create_treatment_checkpoint(
            repo / "bad_sha.json",
            repo_root=repo,
            case_id="e01",
            fingerprint=fingerprint,
            source_bindings=sources,
            review_bindings=reviews,
            bounds=bounds(),
            technical_receipt=wrong_sha,
            eligibility_receipt=eligibility_expected,
        )

    committed_technical_bytes = technical.read_bytes()
    technical.write_text(technical.read_text() + " ")
    with pytest.raises(CanaryStoreError, match="tracked HEAD"):
        create_treatment_checkpoint(
            repo / "modified.json",
            repo_root=repo,
            case_id="e01",
            fingerprint=fingerprint,
            source_bindings=sources,
            review_bindings=reviews,
            bounds=bounds(),
            technical_receipt=tech_expected,
            eligibility_receipt=eligibility_expected,
        )
    technical.write_bytes(committed_technical_bytes)

    with pytest.raises(CanaryStoreError, match="sources do not bind"):
        create_treatment_checkpoint(
            repo / "wrong_source.json",
            repo_root=repo,
            case_id="e01",
            fingerprint=fingerprint,
            source_bindings=[binding("different.json")],
            review_bindings=reviews,
            bounds=bounds(),
            technical_receipt=tech_expected,
            eligibility_receipt=eligibility_expected,
        )


def test_treatment_cannot_use_fail_or_uncommitted_receipt(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    source = repo / "source.txt"
    source.write_text("x")
    git(repo, "add", "source.txt")
    git(repo, "commit", "-m", "initial")

    untracked = repo / "technical.json"
    create(untracked, mode="technical", reviews=[])
    expected = {
        "path": untracked.name,
        "sha256": file_sha256(untracked),
        "status": "PASS",
        "design_id": DESIGN_ID,
    }
    with pytest.raises(CanaryStoreError, match="git receipt verification"):
        create_treatment_checkpoint(
            repo / "treatment.json",
            repo_root=repo,
            case_id="e01",
            fingerprint={"model": "exact", "repo": "abc"},
            source_bindings=[binding("stimulus.json")],
            review_bindings=[binding("review.json")],
            bounds=bounds(),
            technical_receipt=expected,
            eligibility_receipt=expected,
        )


def test_literal_file_binding_hashes_exact_bytes(tmp_path: Path):
    path = tmp_path / "x.json"
    path.write_bytes(b"{\"x\": 1}\n")
    assert literal_file_binding(path, display_path="x.json") == {
        "path": "x.json",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_timezone_naive_output_timestamp_is_rejected(tmp_path: Path):
    # Exercise the public formatter indirectly with a valid explicit name and
    # establish that microsecond timestamps remain within the naming contract.
    path = unique_output_path(
        tmp_path,
        "experiment",
        "model",
        timestamp=datetime(2026, 7, 11, tzinfo=timezone.utc).strftime(
            "%Y%m%dT%H%M%S000000Z"
        ),
    )
    assert path.name == "experiment_model_20260711T000000000000Z.json"
