import copy
from types import SimpleNamespace

import pytest
import torch

from coherent_state_hf import CoherentStateError, IncrementalTrace
import coherent_state_runtime as runtime
from coherent_state_runtime import (
    append_gapped_post_summary,
    arm_snapshot,
    gapped_arm_boundary,
    snapshot_physical_length,
    validate_position_schedule,
    validate_generated_replay,
)


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


def test_gapped_arms_copy_without_key_rotation():
    fresh = _snap(rows=4)
    correct = [(k[..., :2, :].clone() + 11, v[..., :2, :].clone() + 7)
               for k, v in fresh]
    wrong = [(k[..., :2, :].clone() - 13, v[..., :2, :].clone() - 9)
             for k, v in fresh]
    coherent, _ = gapped_arm_boundary(
        "G_correct", fresh, correct, wrong, 2, 17)
    wrong_arm, _ = gapped_arm_boundary(
        "G_wrong", fresh, correct, wrong, 2, 17)
    for (kf, vf), (kc, vc), (kw, vw), (ks, vs), (kx, vx) in zip(
            fresh, coherent, wrong_arm, correct, wrong):
        assert torch.equal(kc[..., 2:4, :], ks)
        assert torch.equal(vc[..., 2:4, :], vs)
        assert torch.equal(kw[..., 2:4, :], kx)
        assert torch.equal(vw[..., 2:4, :], vx)
        assert torch.equal(kc[..., :2, :], kf[..., :2, :])
        assert torch.equal(vc[..., :2, :], vf[..., :2, :])
        assert torch.equal(kc[..., 4:, :], kf[..., 4:, :])
        assert torch.equal(vc[..., 4:, :], vf[..., 4:, :])


def test_full_arm_cannot_enter_gapped_constructor():
    fresh = _snap()
    rows = [(k[..., :2, :], v[..., :2, :]) for k, v in fresh]
    with pytest.raises(CoherentStateError, match="unsupported gapped"):
        gapped_arm_boundary("A_full", fresh, rows, rows, 2, 1)


def test_logical_positions_cannot_be_used_as_cache_positions():
    ok = validate_position_schedule([80, 81, 82], [4, 5, 6], physical_start=4)
    assert ok["gap_from_physical"] == 76
    with pytest.raises(CoherentStateError, match="cache_position"):
        validate_position_schedule([80, 81, 82], [80, 81, 82], physical_start=4)


def test_actual_gapped_destination_schedule_compares_every_saved_summary_row(
        monkeypatch):
    layout = SimpleNamespace(
        prefix_ids=[10, 11, 12, 13],
        prefix_position_ids=[0, 1, 80, 81],
        summary_position_ids=[82, 83],
        system_end=2,
        request_logical_start=80,
        source_summary_start=82,
        physical_summary_start=4,
        physical_summary_end=6,
    )
    calls = []

    def branch(_model, _layout, summary, partitions):
        calls.append(list(partitions))
        trace = IncrementalTrace(
            token_ids=list(summary), token_logprobs=[-1.0, -2.0],
            start_position=82, end_position=84, ended_on_eos=False)
        rows = [(torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 3))]
        return trace, rows

    monkeypatch.setattr(runtime, "_run_gapped_schedule_branch", branch)
    observed = runtime.measure_gapped_destination_schedule(
        object(), layout, [21, 22])
    assert calls == [[2, 2], [4]]
    assert observed["status"] == "PASS"
    assert observed["summary_step_widths"] == [1, 1]
    assert observed["prefix_position_ids"] == [0, 1, 80, 81]
    assert observed["per_layer"][0]["k_per_summary_token_max_abs"] == [0.0, 0.0]
    assert observed["token_logprob_abs_differences"] == [0.0, 0.0]


def test_actual_gapped_destination_schedule_persists_error(monkeypatch):
    layout = SimpleNamespace(
        prefix_ids=[10, 11], prefix_position_ids=[0, 80],
        summary_position_ids=[81], system_end=1,
        request_logical_start=80, source_summary_start=81,
        physical_summary_start=2, physical_summary_end=3,
    )
    monkeypatch.setattr(
        runtime, "_run_gapped_schedule_branch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            torch.OutOfMemoryError("injected")))
    progress = []
    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        runtime.measure_gapped_destination_schedule(
            object(), layout, [21], progress=progress.append)
    assert progress[-1]["status"] == "ERROR"
    assert progress[-1]["failure_evidence"]["error_type"] == "OutOfMemoryError"


def test_snapshot_length_and_gapped_boundary_reject_pretailed_cache():
    fresh = _snap(rows=5)
    assert snapshot_physical_length(fresh) == 5
    rows = [(k[..., :2, :], v[..., :2, :]) for k, v in fresh]
    with pytest.raises(CoherentStateError, match="immediate summary boundary"):
        gapped_arm_boundary("G_correct", fresh, rows, rows, 2, 1)

    layout = SimpleNamespace(physical_summary_end=4)
    with pytest.raises(CoherentStateError, match="exactly at the summary boundary"):
        append_gapped_post_summary(None, fresh, layout)


def test_snapshot_length_rejects_inconsistent_layers():
    malformed = _snap()
    malformed[1] = (malformed[1][0][..., :-1, :], malformed[1][1][..., :-1, :])
    with pytest.raises(CoherentStateError, match="inconsistent physical lengths"):
        snapshot_physical_length(malformed)
