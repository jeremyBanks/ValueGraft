from __future__ import annotations

import hashlib

import pytest
import torch

from coherent_canary_artifacts import (
    ARM_SOURCES, DESIGN_ID, SCHEMA_ID, CanaryArtifactError,
    inspect_tensor_bundle, load_tensor_bundle, reconstruct_arm_boundary,
    save_tensor_bundle,
)
from coherent_canary_runtime import tensor_sha256
from coherent_canary_schema import R1, R2, R3


def snapshot(*, layers: int, tokens: int, offset: int) -> list[tuple[torch.Tensor,
                                                                       torch.Tensor]]:
    rows = []
    for layer in range(layers):
        base = torch.arange(tokens * 2 * 3, dtype=torch.float32).reshape(
            1, 2, tokens, 3) + offset + layer * 100
        rows.append((base.to(torch.bfloat16), (base + 25).to(torch.bfloat16)))
    return rows


def descriptor() -> dict:
    return {
        "schema": SCHEMA_ID,
        "design_id": DESIGN_ID,
        "case_id": "e01",
        "fingerprint_sha256": "a" * 64,
        "fresh_plan_sha256": "b" * 64,
        "source_plan_sha256s": {
            "C_N": "c" * 64, "W_N": "d" * 64,
            "C_P": "e" * 64, "W_P": "f" * 64,
        },
        "source_capture_bindings": {
            source_id: {
                "plan_sha256": plan_sha,
                "history": source_id.split("_")[0],
                "schedule": source_id.split("_")[1],
                "logical_start": 100,
                "logical_end": 106,
                "selected_token_ids_sha256": "1" * 64,
                "source_execution_snapshot_sha256": "2" * 64,
                "extraction_call_trace_sha256": "3" * 64,
            }
            for source_id, plan_sha in {
                "C_N": "c" * 64, "W_N": "d" * 64,
                "C_P": "e" * 64, "W_P": "f" * 64,
            }.items()
        },
        "cache_geometry": {
            "layers": 2, "batch": 1, "kv_heads": 2,
            "head_dim": 3, "dtype": "torch.bfloat16",
        },
        "fresh_physical_regions": {
            R1: [4, 6], R2: [4, 8], R3: [4, 10],
        },
        "source_logical_regions": {
            "N": {R1: [100, 102], R2: [100, 104], R3: [100, 106]},
            "P": {R1: [100, 102], R2: [100, 104], R3: [100, 106]},
        },
    }


def source_snapshots() -> dict:
    return {
        "C_N": snapshot(layers=2, tokens=6, offset=1000),
        "W_N": snapshot(layers=2, tokens=6, offset=2000),
        "C_P": snapshot(layers=2, tokens=6, offset=3000),
        "W_P": snapshot(layers=2, tokens=6, offset=4000),
    }


def test_lossless_bundle_round_trip_and_no_overwrite(tmp_path):
    path = tmp_path / "e01_rows.safetensors"
    fresh = snapshot(layers=2, tokens=10, offset=0)
    sources = source_snapshots()
    inventory = save_tensor_bundle(
        path, fresh_r3=fresh, sources=sources, descriptor=descriptor())
    loaded = load_tensor_bundle(
        path, expected_file_sha256=inventory["sha256"],
        expected_descriptor=descriptor())
    assert loaded["manifest"] == inventory["manifest"]
    for observed, expected in zip(loaded["fresh"], fresh):
        assert torch.equal(observed[0], expected[0])
        assert torch.equal(observed[1], expected[1])
    with pytest.raises(CanaryArtifactError, match="overwrite"):
        save_tensor_bundle(
            path, fresh_r3=fresh, sources=sources, descriptor=descriptor())


def test_arm_reconstruction_proves_channel_and_outside_lineage(tmp_path):
    path = tmp_path / "e01_rows.safetensors"
    fresh = snapshot(layers=2, tokens=10, offset=0)
    sources = source_snapshots()
    inventory = save_tensor_bundle(
        path, fresh_r3=fresh, sources=sources, descriptor=descriptor())
    loaded = load_tensor_bundle(
        path, expected_file_sha256=inventory["sha256"],
        expected_descriptor=descriptor())
    arm, evidence = reconstruct_arm_boundary(
        loaded, region=R2, schedule="N", cell="FC")
    assert evidence["key_source"] == "F"
    assert evidence["value_source"] == "C"
    assert evidence["destination_start"] == 4
    assert evidence["destination_end"] == 8
    for layer, (keys, values) in enumerate(arm):
        assert torch.equal(keys, fresh[layer][0][..., :8, :])
        assert torch.equal(values[..., :4, :], fresh[layer][1][..., :4, :])
        assert torch.equal(values[..., 4:8, :], sources["C_N"][layer][1][..., :4, :])
        row = evidence["per_layer"][layer]
        assert row["outside_prefix_k_before_sha256"] == \
            row["outside_prefix_k_after_sha256"]
        assert row["outside_prefix_v_before_sha256"] == \
            row["outside_prefix_v_after_sha256"]
        assert row["inserted_selected_k_sha256"] == \
            row["fresh_selected_k_sha256"]
        assert row["inserted_selected_v_sha256"] == tensor_sha256(
            sources["C_N"][layer][1][..., :4, :])


def test_bundle_hash_descriptor_and_geometry_tampering_fail_closed(tmp_path):
    path = tmp_path / "e01_rows.safetensors"
    inventory = save_tensor_bundle(
        path, fresh_r3=snapshot(layers=2, tokens=10, offset=0),
        sources=source_snapshots(), descriptor=descriptor())
    with pytest.raises(CanaryArtifactError, match="file hash"):
        load_tensor_bundle(
            path, expected_file_sha256="0" * 64,
            expected_descriptor=descriptor())
    wrong = descriptor()
    wrong["case_id"] = "e02"
    with pytest.raises(CanaryArtifactError, match="descriptor differs"):
        load_tensor_bundle(
            path, expected_file_sha256=inventory["sha256"],
            expected_descriptor=wrong)
    bad_sources = source_snapshots()
    bad_sources["C_N"] = snapshot(layers=2, tokens=5, offset=1000)
    with pytest.raises(CanaryArtifactError, match="content-start through R3"):
        save_tensor_bundle(
            tmp_path / "bad.safetensors",
            fresh_r3=snapshot(layers=2, tokens=10, offset=0),
            sources=bad_sources, descriptor=descriptor())
    wrong_geometry = descriptor()
    wrong_geometry["cache_geometry"]["kv_heads"] = 3
    with pytest.raises(CanaryArtifactError, match="differs from descriptor"):
        save_tensor_bundle(
            tmp_path / "bad_geometry.safetensors",
            fresh_r3=snapshot(layers=2, tokens=10, offset=0),
            sources=source_snapshots(), descriptor=wrong_geometry)


def test_reconstruction_rejects_mutated_loaded_tensor_mapping(tmp_path):
    path = tmp_path / "e01_rows.safetensors"
    inventory = save_tensor_bundle(
        path, fresh_r3=snapshot(layers=2, tokens=10, offset=0),
        sources=source_snapshots(), descriptor=descriptor())
    loaded = load_tensor_bundle(
        path, expected_file_sha256=inventory["sha256"],
        expected_descriptor=descriptor())
    loaded["tensors"]["C_N.layer_000.v"] = \
        loaded["tensors"]["C_N.layer_000.v"].clone()
    loaded["tensors"]["C_N.layer_000.v"][..., 0, 0] = torch.nextafter(
        loaded["tensors"]["C_N.layer_000.v"][..., 0, 0],
        torch.tensor(float("inf"), dtype=torch.bfloat16),
    )
    with pytest.raises(CanaryArtifactError, match="bound index"):
        reconstruct_arm_boundary(loaded, region=R2, schedule="N", cell="FC")


def test_tensor_index_detects_payload_change_even_with_new_file_hash(tmp_path):
    path = tmp_path / "e01_rows.safetensors"
    inventory = save_tensor_bundle(
        path, fresh_r3=snapshot(layers=2, tokens=10, offset=0),
        sources=source_snapshots(), descriptor=descriptor())
    loaded = load_tensor_bundle(
        path, expected_file_sha256=inventory["sha256"],
        expected_descriptor=descriptor())
    original = loaded["tensors"]["C_N.layer_000.v"]
    assert tensor_sha256(original) == next(
        row["sha256"] for row in loaded["manifest"]["tensor_index"]
        if row["name"] == "C_N.layer_000.v")
    # The index is content-addressed independently of the outer file hash.
    changed = original.clone()
    changed[..., 0, 0] = torch.nextafter(
        changed[..., 0, 0],
        torch.tensor(float("inf"), dtype=torch.bfloat16),
    )
    assert tensor_sha256(changed) != tensor_sha256(original)


def test_descriptor_hash_is_not_inferred_from_case_or_filename(tmp_path):
    path = tmp_path / "misleading-name.safetensors"
    desc = descriptor()
    desc["fingerprint_sha256"] = hashlib.sha256(b"exact runtime").hexdigest()
    inventory = save_tensor_bundle(
        path, fresh_r3=snapshot(layers=2, tokens=10, offset=0),
        sources=source_snapshots(), descriptor=desc)
    observed = load_tensor_bundle(
        path, expected_file_sha256=inventory["sha256"],
        expected_descriptor=desc)
    assert observed["manifest"]["descriptor"]["case_id"] == "e01"
    assert observed["manifest"]["descriptor"]["fingerprint_sha256"] == \
        desc["fingerprint_sha256"]


def test_unbound_inspection_cannot_authorize_reconstruction(tmp_path):
    path = tmp_path / "e01_rows.safetensors"
    save_tensor_bundle(
        path, fresh_r3=snapshot(layers=2, tokens=10, offset=0),
        sources=source_snapshots(), descriptor=descriptor())
    inspected = inspect_tensor_bundle(path)
    with pytest.raises(CanaryArtifactError, match="structure differs"):
        reconstruct_arm_boundary(
            inspected, region=R2, schedule="N", cell="FC")


def test_all_frozen_arm_boundary_selectors_reconstruct(tmp_path):
    path = tmp_path / "e01_rows.safetensors"
    inventory = save_tensor_bundle(
        path, fresh_r3=snapshot(layers=2, tokens=10, offset=0),
        sources=source_snapshots(), descriptor=descriptor())
    loaded = load_tensor_bundle(
        path, expected_file_sha256=inventory["sha256"],
        expected_descriptor=descriptor())
    for region in (R1, R2, R3):
        for schedule in ("N", "P"):
            for cell in ARM_SOURCES:
                arm, evidence = reconstruct_arm_boundary(
                    loaded, region=region, schedule=schedule, cell=cell)
                assert len(arm) == 2
                assert evidence["region"] == region
                assert evidence["schedule"] == schedule
                assert evidence["cell"] == cell


def test_descriptor_rejects_n_p_region_shift(tmp_path):
    shifted = descriptor()
    shifted["source_logical_regions"]["P"] = {
        R1: [200, 202], R2: [200, 204], R3: [200, 206],
    }
    for row in shifted["source_capture_bindings"].values():
        if row["schedule"] == "P":
            row["logical_start"], row["logical_end"] = 200, 206
    with pytest.raises(CanaryArtifactError, match="N/P logical"):
        save_tensor_bundle(
            tmp_path / "shifted.safetensors",
            fresh_r3=snapshot(layers=2, tokens=10, offset=0),
            sources=source_snapshots(), descriptor=shifted)


def test_safetensors_round_trip_preserves_exact_bf16_payload_bits(tmp_path):
    path = tmp_path / "bf16_bits.safetensors"
    fresh = snapshot(layers=2, tokens=10, offset=0)
    patterns = torch.tensor(
        [0x0000, 0x0001, 0x3F80, 0x7F7F, 0x8000, 0xBF80],
        dtype=torch.uint16,
    )
    fresh[0][0].view(torch.uint16).flatten()[:len(patterns)] = patterns
    inventory = save_tensor_bundle(
        path, fresh_r3=fresh, sources=source_snapshots(),
        descriptor=descriptor())
    loaded = load_tensor_bundle(
        path, expected_file_sha256=inventory["sha256"],
        expected_descriptor=descriptor())
    observed = loaded["fresh"][0][0].view(torch.uint16).flatten()[:len(patterns)]
    assert torch.equal(observed, patterns)
