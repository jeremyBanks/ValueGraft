from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

import powered_v13_bundle as bundle
from powered_v13_placebo import build_value_row_placebo
from powered_v13_schema import DESIGN_ID, N_SCHEDULE, PRIMARY_ARMS


LAYERS = 48
HEADS = 2
HEAD_DIM = 4
FRESH_TOKENS = 8
R1_START = 2
R1_END = 5
R2_END = 8
SELECTED = R2_END - R1_START


def _tensor(layer: int, tokens: int, source: int, channel: int) -> torch.Tensor:
    positions = torch.arange(tokens * HEADS * HEAD_DIM, dtype=torch.float32)
    value = positions.reshape(1, HEADS, tokens, HEAD_DIM)
    return (value / 64 + source * 16 + channel * 4 + layer / 64).to(
        torch.bfloat16)


def _snapshot(tokens: int, source: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(_tensor(layer, tokens, source, 0),
             _tensor(layer, tokens, source, 1)) for layer in range(LAYERS)]


@pytest.fixture
def fresh():
    return _snapshot(FRESH_TOKENS, 0)


@pytest.fixture
def sources():
    return {"C": _snapshot(SELECTED, 1), "W": _snapshot(SELECTED, 2)}


@pytest.fixture
def descriptor(sources):
    return bundle.build_descriptor(
        release_sha256="1" * 64,
        runtime_fingerprint_sha256="2" * 64,
        primary_batch_id="primary-001",
        gpu_uuid="GPU-deadbeef",
        case_id="a" * 64,
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
        selected_token_ids_sha256="c" * 64,
        fresh_boundary_token_count=FRESH_TOKENS,
        r1_start=R1_START,
        r1_end=R1_END,
        r2_end=R2_END,
        source_intervals={"C": (100, 100 + SELECTED),
                          "W": (100, 100 + SELECTED)},
        source_rows_sha256={
            "C": bundle.snapshot_sha256(sources["C"]),
            "W": bundle.snapshot_sha256(sources["W"]),
        },
        kv_heads=HEADS,
        head_dim=HEAD_DIM,
    )


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
def materialized(loaded):
    return bundle.materialize_verified_bundle(
        loaded, device="cpu", require_accelerator=False)


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(left.view(torch.uint16), right.view(torch.uint16))


def test_descriptor_has_exact_state_keys_and_frozen_bounds(descriptor):
    assert bundle.validate_descriptor(descriptor) == descriptor
    assert descriptor["history_ids"] == ["F", "C", "W"]
    for history in descriptor["history_ids"]:
        assert descriptor["state_keys"][history] == [
            DESIGN_ID, "1" * 64, "2" * 64, "a" * 64, "r1",
            N_SCHEDULE, history,
        ]
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
    changed["state_keys"]["F"][4] = "r3"
    changed["state_keys"]["C"][4] = "r3"
    changed["state_keys"]["W"][4] = "r3"
    with pytest.raises(bundle.V13BundleError, match="r1 or r2"):
        bundle.validate_descriptor(changed)
    changed = deepcopy(descriptor)
    changed["fresh_boundary_token_count"] = 300
    changed["destination"] = {"r1_start": 0, "r1_end": 1, "r2_end": 300}
    with pytest.raises(bundle.V13BundleError, match="selected R2"):
        bundle.validate_descriptor(changed)


def test_save_load_is_lossless_cpu_and_outer_bound(saved, descriptor, fresh, sources):
    path, binding = saved
    assert binding["sha256"] == bundle.file_sha256(path)
    assert binding["size_bytes"] == path.stat().st_size < bundle.MAX_BUNDLE_BYTES
    loaded = bundle.load_verified_bundle(
        path, expected_file_sha256=binding["sha256"],
        expected_descriptor=descriptor)
    assert loaded["verified"] is True and loaded["materialized"] is False
    for layer in range(LAYERS):
        for observed, expected in zip(loaded["fresh"][layer], fresh[layer]):
            assert observed.device.type == "cpu" and _bits_equal(observed, expected)
        for history in ("C", "W"):
            for observed, expected in zip(
                    loaded["sources"][history][layer], sources[history][layer]):
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


def test_loaded_cpu_bundle_cannot_execute_without_explicit_materialization(loaded):
    with pytest.raises(bundle.V13BundleError, match="CUDA"):
        bundle.materialize_verified_bundle(loaded, device="cpu")
    moved = bundle.materialize_verified_bundle(
        loaded, device="cpu", require_accelerator=False)
    assert moved["materialized"] is True and moved["device"] == "cpu"
    assert set(moved["transfer_hashes"]) == {"F", "C", "W"}


def test_postload_tensor_mutation_is_detected(loaded):
    tensor = next(iter(loaded["tensors"].values()))
    tensor.view(torch.uint16).reshape(-1)[0] ^= 1
    with pytest.raises(bundle.V13BundleError, match="changed after verification"):
        bundle.materialize_verified_bundle(
            loaded, device="cpu", require_accelerator=False)


def test_postmaterialization_fresh_mutation_is_detected(materialized):
    materialized["fresh"][0][0].view(torch.uint16).reshape(-1)[0] ^= 1
    with pytest.raises(bundle.V13BundleError, match="changed after device transfer"):
        bundle.reconstruct_arm_boundary(materialized, arm="FF")


def test_placebo_adapter_exactly_splits_content_and_structural_rows(
        materialized):
    values = bundle.placebo_class_values(materialized)
    assert set(values) == {"fresh", "correct", "wrong"}
    for source in values.values():
        assert source["content"].shape == (LAYERS, 3, HEADS, HEAD_DIM)
        assert source["structural"].shape == (LAYERS, 3, HEADS, HEAD_DIM)
        assert source["content"].dtype == torch.bfloat16
    expected = materialized["fresh"][0][1][..., R1_START:R1_END, :]
    observed = values["fresh"]["content"][0].permute(1, 0, 2).unsqueeze(0)
    assert _bits_equal(observed, expected)
    # Compatibility with the independently audited pure selector is itself a
    # gate; tiny synthetic geometry may honestly return unavailable.
    result = build_value_row_placebo(
        values["fresh"], values["correct"], values["wrong"],
        design_id=DESIGN_ID, stable_candidate_id="a" * 64, render_id="r1")
    assert result.status in {"AVAILABLE", "PLACEBO_UNAVAILABLE"}


@pytest.mark.parametrize("arm", PRIMARY_ARMS)
def test_every_arm_reconstructs_from_a_fresh_clone(materialized, arm):
    placebo = None
    if arm == "VP":
        classes = bundle.placebo_class_values(materialized)
        # A geometry-only adapter test: use correct rows as a stand-in for an
        # already independently selected VP result.
        placebo = classes["correct"]
    output, evidence = bundle.reconstruct_arm_boundary(
        materialized, arm=arm, placebo_values=placebo)
    assert evidence["arm"] == arm and len(evidence["per_layer"]) == LAYERS
    start, end = R1_START, R2_END
    for layer, (keys, values) in enumerate(output):
        fresh_k, fresh_v = materialized["fresh"][layer]
        assert _bits_equal(keys[..., :start, :], fresh_k[..., :start, :])
        assert _bits_equal(values[..., :start, :], fresh_v[..., :start, :])
        if arm in {"CC", "WW"}:
            source = arm[0]
            assert _bits_equal(keys[..., start:end, :],
                               materialized["sources"][source][layer][0])
        else:
            assert _bits_equal(keys, fresh_k)
        if arm in {"CC", "FC", "VP"}:
            assert _bits_equal(values[..., start:end, :],
                               materialized["sources"]["C"][layer][1])
        elif arm in {"WW", "FW"}:
            assert _bits_equal(values[..., start:end, :],
                               materialized["sources"]["W"][layer][1])
        else:
            assert _bits_equal(values, fresh_v)


def test_non_vp_arm_rejects_placebo_payload(materialized):
    classes = bundle.placebo_class_values(materialized)
    with pytest.raises(bundle.V13BundleError, match="non-VP"):
        bundle.reconstruct_arm_boundary(
            materialized, arm="FF", placebo_values=classes["correct"])
