from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import powered_v13_bundle as bundle
from powered_v13_placebo import build_value_row_placebo
from powered_v13_schema import DESIGN_ID, N_SCHEDULE, PRIMARY_ARMS


LAYERS = 48
HEADS = 1
HEAD_DIM = 1
FRESH_TOKENS = 5
R1_START = 1
R1_END = 3
R2_END = 5
SELECTED = R2_END - R1_START
LOGICAL_START = 100
CASE_ID = "0" * 64
TOKEN_IDS = (41, 42, 43, 44)
POSITION_IDS = tuple(range(LOGICAL_START, LOGICAL_START + SELECTED))


def _key_tensor(layer: int, tokens: int, source: int) -> torch.Tensor:
    return torch.full(
        (1, HEADS, tokens, HEAD_DIM),
        source * 8 + layer / 64,
        dtype=torch.bfloat16,
    )


def _fresh_snapshot() -> list[tuple[torch.Tensor, torch.Tensor]]:
    result = []
    for layer in range(LAYERS):
        keys = _key_tensor(layer, FRESH_TOKENS, 0)
        values = torch.zeros_like(keys)
        values[..., :R1_START, :] = torch.tensor(
            7 + layer / 64, dtype=torch.bfloat16)
        result.append((keys, values))
    return result


def _selected_snapshot(history: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    assert history in {"C", "W"}
    result = []
    source = 1 if history == "C" else 2
    value_rows = [1.0, 2.0, 3.0, 4.0] if history == "C" else [0.0, 4.0, 3.0, 4.0]
    for layer in range(LAYERS):
        keys = _key_tensor(layer, SELECTED, source)
        values = torch.tensor(value_rows, dtype=torch.bfloat16).reshape(
            1, HEADS, SELECTED, HEAD_DIM)
        result.append((keys, values))
    return result


@pytest.fixture
def fresh():
    return _fresh_snapshot()


@pytest.fixture
def sources():
    return {"C": _selected_snapshot("C"), "W": _selected_snapshot("W")}


def _make_descriptor(
    sources,
    *,
    case_id: str = CASE_ID,
    stable_candidate_id: str | None = None,
    token_ids: dict[str, tuple[int, ...]] | None = None,
    position_ids: dict[str, tuple[int, ...]] | None = None,
    source_intervals: dict[str, tuple[int, int]] | None = None,
    primary_batch_id: str = "primary-001",
):
    tokens = token_ids or {history: TOKEN_IDS for history in ("F", "C", "W")}
    positions = position_ids or {
        history: POSITION_IDS for history in ("F", "C", "W")}
    intervals = source_intervals or {
        history: (LOGICAL_START, LOGICAL_START + SELECTED)
        for history in ("C", "W")
    }
    return bundle.build_descriptor(
        release_sha256="1" * 64,
        runtime_fingerprint_sha256="2" * 64,
        primary_batch_id=primary_batch_id,
        gpu_uuid="GPU-deadbeef",
        case_id=case_id,
        stable_candidate_id=stable_candidate_id,
        render_id="r1",
        schedule=N_SCHEDULE,
        plan_sha256s={"F": "3" * 64, "C": "4" * 64, "W": "5" * 64},
        foundation_bindings={
            "fixture_sha256": "6" * 64,
            "render_attempt_sha256": "7" * 64,
            "review_sha256": "8" * 64,
            "phase_a_sha256": "9" * 64,
        },
        selected_map_sha256="b" * 64,
        selected_token_ids=tokens,
        logical_position_ids=positions,
        fresh_boundary_token_count=FRESH_TOKENS,
        r1_start=R1_START,
        r1_end=R1_END,
        r2_end=R2_END,
        source_intervals=intervals,
        source_rows_sha256={
            "C": bundle.snapshot_sha256(sources["C"]),
            "W": bundle.snapshot_sha256(sources["W"]),
        },
        kv_heads=HEADS,
        head_dim=HEAD_DIM,
    )


@pytest.fixture
def descriptor(sources):
    return _make_descriptor(sources)


@pytest.fixture
def saved(tmp_path: Path, fresh, sources, descriptor):
    path = tmp_path / "foundation.safetensors"
    binding = bundle.save_bundle(
        path, fresh_boundary=fresh, selected_sources=sources,
        descriptor=descriptor)
    return path, binding


@pytest.fixture
def loaded(saved, descriptor):
    path, binding = saved
    return bundle.load_verified_bundle(
        path, expected_file_sha256=binding["sha256"],
        expected_descriptor=descriptor)


@pytest.fixture
def materialized(saved, descriptor):
    path, binding = saved
    return bundle._materialize_verified_bundle_for_test(
        path, expected_file_sha256=binding["sha256"],
        expected_descriptor=descriptor)


@pytest.fixture
def vp_artifact(saved, descriptor):
    path, binding = saved
    return bundle.build_placebo_receipt(
        path, expected_file_sha256=binding["sha256"],
        expected_descriptor=descriptor)


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(left.view(torch.uint16), right.view(torch.uint16))


def _receipt_sha(receipt) -> str:
    return bundle.sha256_bytes(bundle.canonical_json_bytes(receipt))


def test_descriptor_has_exact_state_keys_capture_lineage_and_bounds(descriptor):
    assert bundle.validate_descriptor(descriptor) == descriptor
    assert descriptor["history_ids"] == ["F", "C", "W"]
    for history in descriptor["history_ids"]:
        assert descriptor["state_keys"][history] == [
            DESIGN_ID, "1" * 64, "2" * 64, CASE_ID, "r1",
            N_SCHEDULE, history,
        ]
        capture = descriptor["capture_bindings"][history]
        assert capture["logical_start"] == LOGICAL_START
        assert capture["logical_r1_end"] == LOGICAL_START + 2
        assert capture["logical_end"] == LOGICAL_START + SELECTED
        assert capture["selected_token_ids_sha256"] == (
            descriptor["selected_token_ids_sha256"])
        assert capture["logical_position_ids_sha256"] == (
            descriptor["logical_position_ids_sha256"])
    assert descriptor["cache_geometry"] == {
        "layers": 48, "batch": 1, "kv_heads": HEADS,
        "head_dim": HEAD_DIM, "dtype": "torch.bfloat16",
    }


def test_descriptor_rejects_extra_fields_bad_render_and_row_overflow(descriptor):
    changed = deepcopy(descriptor)
    changed["extra"] = True
    with pytest.raises(bundle.V13BundleError, match="field set"):
        bundle.validate_descriptor(changed)
    changed = deepcopy(descriptor)
    changed["render_id"] = "r3"
    for history in ("F", "C", "W"):
        changed["state_keys"][history][4] = "r3"
    with pytest.raises(bundle.V13BundleError, match="r1 or r2"):
        bundle.validate_descriptor(changed)
    changed = deepcopy(descriptor)
    changed["fresh_boundary_token_count"] = 300
    changed["destination"] = {"r1_start": 0, "r1_end": 1, "r2_end": 300}
    with pytest.raises(bundle.V13BundleError, match="selected R2"):
        bundle.validate_descriptor(changed)


def test_descriptor_rejects_c_w_f_logical_or_token_mismatch(sources):
    positions = {history: POSITION_IDS for history in ("F", "C", "W")}
    positions["W"] = tuple(value + 10 for value in POSITION_IDS)
    with pytest.raises(bundle.V13BundleError, match="logical position IDs differ"):
        _make_descriptor(sources, position_ids=positions)
    tokens = {history: TOKEN_IDS for history in ("F", "C", "W")}
    tokens["C"] = (41, 42, 43, 99)
    with pytest.raises(bundle.V13BundleError, match="selected token IDs differ"):
        _make_descriptor(sources, token_ids=tokens)
    with pytest.raises(bundle.V13BundleError, match="logical R2 intervals differ"):
        _make_descriptor(sources, source_intervals={
            "C": (LOGICAL_START, LOGICAL_START + SELECTED),
            "W": (LOGICAL_START + 20, LOGICAL_START + 20 + SELECTED),
        })
    descriptor = _make_descriptor(sources)
    descriptor["logical_position_ids"][2] += 1
    with pytest.raises(bundle.V13BundleError, match="not contiguous|IDs/hash"):
        bundle.validate_descriptor(descriptor)


def test_descriptor_requires_stable_semantic_id_or_explicit_technical_id(sources):
    with pytest.raises(bundle.V13BundleError, match="stable 64-hex"):
        _make_descriptor(sources, case_id="semantic-alias")
    technical = _make_descriptor(
        sources, case_id="technical_e01", stable_candidate_id="e" * 64)
    assert technical["case_id"] == "technical_e01"
    assert technical["stable_candidate_id"] == "e" * 64
    with pytest.raises(bundle.V13BundleError, match="128 bytes"):
        _make_descriptor(sources, primary_batch_id="x" * 129)


def test_save_load_is_lossless_cpu_and_immutably_anchored(
        saved, descriptor, fresh, sources):
    path, binding = saved
    assert binding["sha256"] == bundle.file_sha256(path)
    assert binding["size_bytes"] == path.stat().st_size < bundle.MAX_BUNDLE_BYTES
    loaded = bundle.load_verified_bundle(
        path, expected_file_sha256=binding["sha256"],
        expected_descriptor=descriptor)
    assert loaded.verified is True and loaded.materialized is False
    assert loaded.descriptor == descriptor
    for layer in range(LAYERS):
        for observed, expected in zip(loaded.fresh[layer], fresh[layer]):
            assert observed.device.type == "cpu" and _bits_equal(observed, expected)
        for history in ("C", "W"):
            for observed, expected in zip(
                    loaded.sources[history][layer], sources[history][layer]):
                assert _bits_equal(observed, expected)


def test_save_is_exclusive_and_rejects_wrong_layer_count(
        saved, fresh, sources, descriptor):
    path, _binding = saved
    with pytest.raises(bundle.V13BundleError, match="overwrite"):
        bundle.save_bundle(
            path, fresh_boundary=fresh, selected_sources=sources,
            descriptor=descriptor)
    with pytest.raises(bundle.V13BundleError, match="exactly 48"):
        bundle.save_bundle(
            path.with_name("short.safetensors"), fresh_boundary=fresh[:-1],
            selected_sources=sources, descriptor=descriptor)


def test_serialized_size_is_checked_before_publish(tmp_path: Path):
    path = tmp_path / "too-large.safetensors"
    with pytest.raises(bundle.V13BundleError, match="before publish"):
        bundle._atomic_safetensors(
            path,
            {"x": torch.zeros(1024, dtype=torch.bfloat16)},
            {"schema": "test"},
            max_bytes=16,
        )
    assert not path.exists()


def test_load_requires_external_file_and_descriptor_commitments(saved, descriptor):
    path, binding = saved
    with pytest.raises(bundle.V13BundleError, match="outer file hash"):
        bundle.load_verified_bundle(
            path, expected_file_sha256="f" * 64,
            expected_descriptor=descriptor)
    changed = deepcopy(descriptor)
    changed["selected_map_sha256"] = "e" * 64
    with pytest.raises(bundle.V13BundleError, match="descriptor differs"):
        bundle.load_verified_bundle(
            path, expected_file_sha256=binding["sha256"],
            expected_descriptor=changed)


def test_serialized_corruption_and_change_during_read_fail(
        saved, descriptor, monkeypatch):
    path, binding = saved
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(bundle.V13BundleError, match="outer file hash"):
        bundle.load_verified_bundle(
            path, expected_file_sha256=binding["sha256"],
            expected_descriptor=descriptor)

    # Restore a valid file, then inject a mutation after safetensors loading.
    path.unlink()
    fresh = _fresh_snapshot()
    sources = {"C": _selected_snapshot("C"), "W": _selected_snapshot("W")}
    binding = bundle.save_bundle(
        path, fresh_boundary=fresh, selected_sources=sources,
        descriptor=descriptor)
    original = bundle._load_manifest

    def mutate_after_read(target):
        result = original(target)
        changed = bytearray(target.read_bytes())
        changed[-1] ^= 1
        target.write_bytes(changed)
        return result

    monkeypatch.setattr(bundle, "_load_manifest", mutate_after_read)
    with pytest.raises(bundle.V13BundleError, match="changed while"):
        bundle.load_verified_bundle(
            path, expected_file_sha256=binding["sha256"],
            expected_descriptor=descriptor)


def test_loaded_manifest_index_and_lineage_copies_cannot_reanchor(loaded):
    manifest = loaded.manifest
    manifest["tensor_index_sha256"] = "f" * 64
    manifest["descriptor"]["foundation_bindings"]["fixture_sha256"] = "e" * 64
    assert loaded.manifest["tensor_index_sha256"] != "f" * 64
    assert loaded.descriptor["foundation_bindings"]["fixture_sha256"] == "6" * 64
    bundle._revalidate_loaded(loaded)
    with pytest.raises(FrozenInstanceError):
        loaded.sha256 = "f" * 64


def test_loaded_tensor_mutation_is_detected_even_if_index_copy_is_changed(loaded):
    loaded.fresh[0][0].view(torch.uint16).reshape(-1)[0] ^= 1
    index_copy = loaded.tensor_index
    index_copy[0]["sha256"] = bundle.tensor_sha256(loaded.fresh[0][0])
    with pytest.raises(bundle.V13BundleError, match="changed after verification"):
        bundle._revalidate_loaded(loaded)


def test_public_materialization_has_no_cpu_or_loaded_handle_bypass(
        saved, loaded, descriptor, materialized):
    path, binding = saved
    with pytest.raises(bundle.V13BundleError, match="requires a CUDA device"):
        bundle.materialize_verified_bundle(
            path, expected_file_sha256=binding["sha256"],
            expected_descriptor=descriptor, device="cpu")
    with pytest.raises(bundle.V13BundleError, match="path-like"):
        bundle.materialize_verified_bundle(
            loaded, expected_file_sha256=binding["sha256"],
            expected_descriptor=descriptor, device="cuda")
    with pytest.raises(bundle.V13BundleError, match="CUDA-authorized"):
        bundle.reconstruct_arm_boundary(materialized, arm="FF")


def test_private_cpu_materialization_is_bit_bound_but_not_publicly_authorized(
        materialized):
    assert materialized.materialized and materialized.device == "cpu"
    assert dict(materialized.transfer_hashes).keys() == {"F", "C", "W"}
    output, evidence = bundle._reconstruct_arm_boundary_for_test(
        materialized, arm="FF")
    assert len(output) == LAYERS and evidence["arm"] == "FF"


def test_postmaterialization_tensor_mutation_is_detected(materialized):
    materialized.fresh[0][0].view(torch.uint16).reshape(-1)[0] ^= 1
    with pytest.raises(bundle.V13BundleError, match="changed after device transfer"):
        bundle._reconstruct_arm_boundary_for_test(materialized, arm="FF")


def test_postmaterialization_descriptor_copy_cannot_relabel(materialized):
    descriptor = materialized.descriptor
    descriptor["foundation_bindings"]["fixture_sha256"] = "e" * 64
    output, _evidence = bundle._reconstruct_arm_boundary_for_test(
        materialized, arm="FF")
    assert len(output) == LAYERS
    assert materialized.descriptor["foundation_bindings"]["fixture_sha256"] == "6" * 64


def test_placebo_adapter_exactly_splits_content_and_structural_rows(materialized):
    values = bundle._placebo_class_values_for_test(materialized)
    assert set(values) == {"fresh", "correct", "wrong"}
    for source in values.values():
        assert source["content"].shape == (LAYERS, 2, HEADS, HEAD_DIM)
        assert source["structural"].shape == (LAYERS, 2, HEADS, HEAD_DIM)
        assert source["content"].dtype == torch.bfloat16
    expected = materialized.fresh[0][1][..., R1_START:R1_END, :]
    observed = values["fresh"]["content"][0].permute(1, 0, 2).unsqueeze(0)
    assert _bits_equal(observed, expected)


def test_phase_a_vp_receipt_is_available_and_binds_every_origin(
        vp_artifact, saved, descriptor):
    _path, binding = saved
    receipt = vp_artifact["receipt"]
    assert receipt["status"] == "AVAILABLE"
    assert vp_artifact["receipt_sha256"] == _receipt_sha(receipt)
    assert receipt["bundle_file_sha256"] == binding["sha256"]
    assert receipt["descriptor_sha256"] == bundle.sha256_bytes(
        bundle.canonical_json_bytes(descriptor))
    assert receipt["case_id"] == receipt["stable_candidate_id"] == CASE_ID
    assert receipt["row_map"] is not None
    assert set(receipt["result_layer_sha256"]) == {"content", "structural"}
    assert all(len(rows) == LAYERS
               for rows in receipt["result_layer_sha256"].values())
    assert vp_artifact["diagnostics_sha256"] == receipt["diagnostics_sha256"]


def test_valid_vp_receipt_is_recomputed_and_inserted_exactly(
        materialized, vp_artifact):
    receipt = vp_artifact["receipt"]
    output, evidence = bundle._reconstruct_arm_boundary_for_test(
        materialized,
        arm="VP",
        placebo_receipt=receipt,
        expected_placebo_receipt_sha256=vp_artifact["receipt_sha256"],
    )
    assert evidence["vp_receipt_sha256"] == vp_artifact["receipt_sha256"]
    classes = bundle._placebo_class_values_for_test(materialized)
    result = build_value_row_placebo(
        classes["fresh"], classes["correct"], classes["wrong"],
        design_id=DESIGN_ID, stable_candidate_id=CASE_ID, render_id="r1")
    assert result.status == "AVAILABLE" and result.values is not None
    expected = torch.cat(
        (result.values["content"], result.values["structural"]), dim=1)
    for layer, (_keys, values) in enumerate(output):
        observed = values[..., R1_START:R2_END, :]
        expected_layer = expected[layer].permute(1, 0, 2).unsqueeze(0)
        assert _bits_equal(observed, expected_layer)


@pytest.mark.parametrize("field", ["diagnostics_sha256", "row_map", "result_sha256"])
def test_forged_vp_receipt_fails_even_with_forged_matching_outer_sha(
        materialized, vp_artifact, field):
    forged = deepcopy(vp_artifact["receipt"])
    if field == "row_map":
        forged[field]["content"][0]["donor"] ^= 1
    elif field == "result_sha256":
        forged[field]["content"] = "e" * 64
    else:
        forged[field] = "e" * 64
    with pytest.raises(bundle.V13BundleError, match="deterministic recomputation"):
        bundle._reconstruct_arm_boundary_for_test(
            materialized,
            arm="VP",
            placebo_receipt=forged,
            expected_placebo_receipt_sha256=_receipt_sha(forged),
        )


def test_zeroed_vp_receipt_hashes_fail_deterministic_recomputation(
        materialized, vp_artifact):
    forged = deepcopy(vp_artifact["receipt"])
    forged["result_sha256"] = {
        class_name: "0" * 64 for class_name in ("content", "structural")}
    forged["result_layer_sha256"] = {
        class_name: ["0" * 64] * LAYERS
        for class_name in ("content", "structural")
    }
    with pytest.raises(bundle.V13BundleError, match="deterministic recomputation"):
        bundle._reconstruct_arm_boundary_for_test(
            materialized,
            arm="VP",
            placebo_receipt=forged,
            expected_placebo_receipt_sha256=_receipt_sha(forged),
        )


def test_arbitrary_vp_tensor_payload_has_no_api_path(materialized):
    classes = bundle._placebo_class_values_for_test(materialized)
    bogus = {name: torch.zeros_like(value)
             for name, value in classes["fresh"].items()}
    with pytest.raises(TypeError, match="placebo_values"):
        bundle._reconstruct_arm_boundary_for_test(
            materialized, arm="VP", placebo_values=bogus)


def test_unavailable_vp_receipt_is_bound_but_cannot_construct_arm(
        tmp_path: Path, fresh, descriptor):
    unavailable_sources = {
        "C": [(keys[..., R1_START:R2_END, :].clone(),
               fresh_values[..., R1_START:R2_END, :].clone())
              for keys, fresh_values in fresh],
        "W": [(keys[..., R1_START:R2_END, :].clone(),
               fresh_values[..., R1_START:R2_END, :].clone())
              for keys, fresh_values in fresh],
    }
    unavailable_descriptor = deepcopy(descriptor)
    for history in ("C", "W"):
        unavailable_descriptor["source"][history]["selected_rows_sha256"] = (
            bundle.snapshot_sha256(unavailable_sources[history]))
    unavailable_descriptor = bundle.validate_descriptor(unavailable_descriptor)
    path = tmp_path / "unavailable.safetensors"
    binding = bundle.save_bundle(
        path, fresh_boundary=fresh, selected_sources=unavailable_sources,
        descriptor=unavailable_descriptor)
    artifact = bundle.build_placebo_receipt(
        path, expected_file_sha256=binding["sha256"],
        expected_descriptor=unavailable_descriptor)
    assert artifact["receipt"]["status"] == "PLACEBO_UNAVAILABLE"
    assert artifact["receipt"]["row_map"] is None
    materialized = bundle._materialize_verified_bundle_for_test(
        path, expected_file_sha256=binding["sha256"],
        expected_descriptor=unavailable_descriptor)
    with pytest.raises(bundle.V13BundleError, match="not AVAILABLE"):
        bundle._reconstruct_arm_boundary_for_test(
            materialized,
            arm="VP",
            placebo_receipt=artifact["receipt"],
            expected_placebo_receipt_sha256=artifact["receipt_sha256"],
        )


@pytest.mark.parametrize("arm", PRIMARY_ARMS)
def test_every_arm_reconstructs_from_fresh_with_exact_sources_and_hashes(
        materialized, vp_artifact, arm):
    kwargs = {}
    if arm == "VP":
        kwargs = {
            "placebo_receipt": vp_artifact["receipt"],
            "expected_placebo_receipt_sha256": vp_artifact["receipt_sha256"],
        }
    output, evidence = bundle._reconstruct_arm_boundary_for_test(
        materialized, arm=arm, **kwargs)
    assert evidence["arm"] == arm and len(evidence["per_layer"]) == LAYERS
    for layer, (keys, values) in enumerate(output):
        fresh_k, fresh_v = materialized.fresh[layer]
        assert keys.data_ptr() != fresh_k.data_ptr()
        assert values.data_ptr() != fresh_v.data_ptr()
        assert _bits_equal(keys[..., :R1_START, :], fresh_k[..., :R1_START, :])
        assert _bits_equal(values[..., :R1_START, :], fresh_v[..., :R1_START, :])
        row = evidence["per_layer"][layer]
        assert row["inserted_selected_k_sha256"] == row["source_selected_k_sha256"]
        assert row["inserted_selected_v_sha256"] == row["source_selected_v_sha256"]
        if arm in {"CC", "WW"}:
            source = arm[0]
            assert _bits_equal(keys[..., R1_START:R2_END, :],
                               materialized.sources[source][layer][0])
        else:
            assert _bits_equal(keys, fresh_k)
        if arm in {"CC", "FC"}:
            assert _bits_equal(values[..., R1_START:R2_END, :],
                               materialized.sources["C"][layer][1])
        elif arm in {"WW", "FW"}:
            assert _bits_equal(values[..., R1_START:R2_END, :],
                               materialized.sources["W"][layer][1])
        elif arm == "FF":
            assert _bits_equal(values, fresh_v)


def test_arm_output_mutation_does_not_change_immutable_foundation(materialized):
    before = bundle.snapshot_sha256(materialized.fresh)
    output, _evidence = bundle._reconstruct_arm_boundary_for_test(
        materialized, arm="CC")
    output[0][0].view(torch.uint16).reshape(-1)[0] ^= 1
    assert bundle.snapshot_sha256(materialized.fresh) == before


def test_non_vp_arm_rejects_receipt(materialized, vp_artifact):
    with pytest.raises(bundle.V13BundleError, match="non-VP"):
        bundle._reconstruct_arm_boundary_for_test(
            materialized,
            arm="FF",
            placebo_receipt=vp_artifact["receipt"],
            expected_placebo_receipt_sha256=vp_artifact["receipt_sha256"],
        )
