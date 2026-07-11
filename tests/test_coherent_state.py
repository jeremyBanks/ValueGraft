from __future__ import annotations

import pytest
import torch

from coherent_state_hf import (
    CoherentStateError,
    compare_rows,
    delta_deranged_snapshot,
    fixed_point_free_permutation,
    move_key_rows,
    replace_summary_rows,
    require_exact_span,
    sha256_ids,
)


def _snapshot(layers=2, heads=2, length=7, dim=4):
    out = []
    for li in range(layers):
        base = torch.arange(heads * length * dim, dtype=torch.float32).reshape(
            1, heads, length, dim) + li * 1000
        out.append((base.clone(), (base + 0.25).clone()))
    return out


@pytest.mark.parametrize("n", [2, 3, 4, 17])
def test_derangement_has_no_fixed_points_and_is_reproducible(n):
    a = fixed_point_free_permutation(n, 42)
    b = fixed_point_free_permutation(n, 42)
    assert a == b
    assert sorted(a) == list(range(n))
    assert all(i != p for i, p in enumerate(a))


def test_derangement_rejects_single_row():
    with pytest.raises(CoherentStateError, match="at least two"):
        fixed_point_free_permutation(1, 0)


def test_replace_summary_rows_changes_only_selected_components_and_span():
    fresh = _snapshot()
    source = [(k[..., :3, :] + 10_000, v[..., :3, :] + 20_000)
              for k, v in fresh]
    v_only = replace_summary_rows(fresh, source, 2, use_keys=False, use_values=True)
    k_only = replace_summary_rows(fresh, source, 2, use_keys=True, use_values=False)
    for li, ((kf, vf), (kv, vv), (kk, vk)) in enumerate(zip(fresh, v_only, k_only)):
        assert torch.equal(kv, kf)
        assert torch.equal(vk, vf)
        assert torch.equal(vv[..., :2, :], vf[..., :2, :])
        assert torch.equal(vv[..., 5:, :], vf[..., 5:, :])
        assert torch.equal(vv[..., 2:5, :], source[li][1])
        assert torch.equal(kk[..., 2:5, :], source[li][0])


def test_delta_placebo_is_exact_per_head_delta_permutation():
    fresh = _snapshot(layers=2, heads=2, length=8, dim=4)
    start, n = 2, 4
    correct = []
    for li, (k, v) in enumerate(fresh):
        rows = v[..., start:start + n, :].clone()
        # Unique row deltas per layer/head/position.
        delta = torch.arange(rows.numel(), dtype=rows.dtype).reshape_as(rows) + 1
        correct.append((k[..., start:start + n, :].clone(), rows + delta))
    placebo, diagnostics = delta_deranged_snapshot(fresh, correct, start, seed=17)
    assert len(diagnostics) == 4
    for d in diagnostics:
        assert d.fixed_points == 0
        assert d.max_multiset_diff == 0.0
        assert d.mean_diff < 1e-5
        assert d.covariance_diff < 1e-4
    for li, ((kf, vf), (kp, vp), (_, vc)) in enumerate(zip(fresh, placebo, correct)):
        assert torch.equal(kp, kf)
        assert torch.equal(vp[..., :start, :], vf[..., :start, :])
        assert torch.equal(vp[..., start + n:, :], vf[..., start + n:, :])
        for head in range(vf.shape[1]):
            original = vc[:, head].float() - vf[:, head, start:start + n].float()
            observed = vp[:, head, start:start + n].float() - \
                vf[:, head, start:start + n].float()
            # Row sums are unique here, so sorted sums prove exact row-multiset reuse.
            assert torch.equal(
                torch.sort(original.sum(-1)).values,
                torch.sort(observed.sum(-1)).values)


def test_key_move_roundtrip_is_small():
    rows = [(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))]
    moved = move_key_rows(rows, 37, 1_000_000.0)
    back = move_key_rows(moved, -37, 1_000_000.0)
    diffs = compare_rows(rows, back)
    assert diffs[0]["k_max_abs"] < 1e-5
    assert diffs[0]["v_max_abs"] == 0.0


def test_exact_span_requires_unique_declared_occurrence():
    assert require_exact_span([1, 2, 3, 4, 5], [3, 4], 2) == (2, 4)
    with pytest.raises(CoherentStateError, match="absent"):
        require_exact_span([1, 2, 8, 4, 5], [3, 4], 2)
    with pytest.raises(CoherentStateError, match="not unique"):
        require_exact_span([1, 3, 4, 3, 4], [3, 4], 1)


def test_id_hash_is_order_and_value_sensitive():
    assert sha256_ids([1, 2, 3]) == sha256_ids([1, 2, 3])
    assert sha256_ids([1, 2, 3]) != sha256_ids([3, 2, 1])
    assert sha256_ids([1, 2, 3]) != sha256_ids([1, 2, 4])
