import copy

import pytest
import torch

from coherent_state_hf import CoherentStateError
from coherent_state_runtime import arm_snapshot, validate_generated_replay


def _snap(layers=2, rows=5):
    out = []
    for li in range(layers):
        k = torch.arange(1 * 2 * rows * 4, dtype=torch.float32).reshape(1, 2, rows, 4)
        v = k + 100 * (li + 1)
        out.append((k.clone(), v.clone()))
    return out


def _capture(rows, lp=(0.1, 0.2)):
    class C:
        pass
    c = C()
    c.summary_ids = [4, 5]
    c.prefix_ids = [1, 2, 3]
    c.trace = {"token_logprobs": list(lp)}
    c.rows = rows
    return c


def test_generated_replay_gate_is_numeric_not_hash_only():
    rows = [(k[..., 1:3, :].clone(), v[..., 1:3, :].clone()) for k, v in _snap()]
    result = validate_generated_replay(_capture(rows), _capture(copy.deepcopy(rows)))
    assert result["k_max_abs"] == 0
    broken = copy.deepcopy(rows)
    broken[0][1][..., 0, 0] += 0.01
    with pytest.raises(CoherentStateError, match="identity failed"):
        validate_generated_replay(_capture(rows), _capture(broken))


def test_arm_constructor_changes_only_requested_summary_channels():
    fresh = _snap()
    correct = [(k[..., :2, :].clone(), v[..., :2, :].clone() + 7)
               for k, v in fresh]
    wrong = [(k[..., :2, :].clone(), v[..., :2, :].clone() - 9)
             for k, v in fresh]
    v_arm, _ = arm_snapshot("V_only", fresh, correct, wrong, 2, 2, 2,
                            10_000.0, 17)
    for (kf, vf), (ka, va), (_, vc) in zip(fresh, v_arm, correct):
        assert torch.equal(kf, ka)
        assert torch.equal(vf[..., :2, :], va[..., :2, :])
        assert torch.equal(vf[..., 4:, :], va[..., 4:, :])
        assert torch.equal(va[..., 2:4, :], vc)


def test_unknown_or_full_arm_cannot_enter_compacted_constructor():
    fresh = _snap()
    rows = [(k[..., :2, :], v[..., :2, :]) for k, v in fresh]
    with pytest.raises(CoherentStateError, match="unsupported"):
        arm_snapshot("A_full", fresh, rows, rows, 2, 0, 0, 10_000.0, 1)
