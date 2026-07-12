from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import precision_probe_p02_case as module


class Regions:
    def interval(self, region):
        assert region == module.R2
        return (3, 7)


def _plans():
    ordinary = SimpleNamespace(token_ids=[1, 2])
    fresh = SimpleNamespace(
        token_ids=[5, 6, 7],
        logical_positions=[10, 11, 12],
        physical_regions=Regions(),
    )
    return {
        "C_N": ordinary,
        "W_N": ordinary,
        "C_P": ordinary,
        "W_P": ordinary,
        "F": fresh,
    }


def _install_fakes(monkeypatch, events):
    monkeypatch.setattr(module, "case_histories", lambda case: (
        [{"role": "user", "content": "correct"}],
        [{"role": "user", "content": "wrong"}],
        1,
    ))
    monkeypatch.setattr(module, "build_case_plans", lambda tokenizer, case: _plans())
    monkeypatch.setattr(
        module, "execute_replay_plan",
        lambda model, plan: SimpleNamespace(kind="source"),
    )

    def fresh(model, plan, stop_at=None):
        events.append(("fresh", stop_at))
        return SimpleNamespace(kind="fresh", snapshot=f"snapshot-{stop_at}")

    monkeypatch.setattr(module, "execute_fresh_plan", fresh)
    monkeypatch.setattr(module, "execution_record", lambda row: {"kind": row.kind})
    monkeypatch.setattr(module, "plan_record", lambda name, plan: {"name": name})
    monkeypatch.setattr(module, "compact_messages", lambda history, middle: history)
    monkeypatch.setattr(
        module, "probe_record",
        lambda *args, **kwargs: {
            "margin_float32_bits": "00000000",
            "generation": {"content_ids": [1], "decoded_content": "x"},
        },
    )

    def arm(model, tokenizer, case, plans, executions, boundaries, *,
            schedule, region, cell, eos_ids):
        selector = (schedule, region, cell)
        events.append(("execute_arm", selector))
        return {
            "arm_kind": "primary",
            "schedule": schedule,
            "region": region,
            "cell": cell,
            "scores": {"focal": {}, "nonfocal": {}},
        }

    monkeypatch.setattr(module, "_arm_record", arm)

    def placebo(*args, region, **kwargs):
        events.append(("execute_placebo", region))
        return {
            "arm_kind": "placebo_control",
            "schedule": "N",
            "region": region,
            "cell": "V_PLACEBO",
            "control_status": "AVAILABLE",
            "scores": {"focal": {}, "nonfocal": {}},
        }

    monkeypatch.setattr(module, "_placebo_arm_record", placebo)


def _case():
    return {
        "case_id": "e01",
        "focal": {
            "probe": "focal?",
            "correct_target": "yes",
            "counterfactual_target": "no",
        },
        "nonfocal_control": {
            "probe": "control?", "target": "up", "countertarget": "down",
        },
    }


def test_repeat1_runs_exact_six_grafts_then_one_placebo_with_sync_callbacks(
    monkeypatch,
):
    events = []
    _install_fakes(monkeypatch, events)

    def foundation_callback(value):
        assert value["source_plan_order"] == ["C_N", "W_N", "C_P", "W_P"]
        events.append(("persist_foundation", None))

    def arm_callback(index, value):
        events.append(("persist_arm", index,
                       (value["schedule"], value["region"], value["cell"])))

    result = module.run_targeted_treatment_case(
        object(), object(), _case(), eos_ids=[1, 2], repeat_index=1,
        on_foundation=foundation_callback, on_arm=arm_callback,
    )
    assert result["schema"] == module.TREATMENT_SCHEMA
    assert result["arms"][0]["execution_kind"] == "zero_increment_reuse"
    assert result["arms"][0]["shared_fresh_baseline"] is True
    assert [
        (row["schedule"], row["region"], row["cell"])
        for row in result["arms"][1:7]
    ] == list(module.TARGETED_GRAFTS)
    assert result["arms"][7]["cell"] == "V_PLACEBO"
    assert result["executed_graft_count"] == 6
    assert result["zero_increment_anchor_count"] == 1
    assert result["placebo_attempt_count"] == 1

    assert events.index(("persist_foundation", None)) < events.index(
        ("execute_arm", module.TARGETED_GRAFTS[0])
    )
    for index, selector in enumerate(module.TARGETED_GRAFTS, start=1):
        execute = events.index(("execute_arm", selector))
        persist = events.index(("persist_arm", index, selector))
        assert execute < persist
        if index < 6:
            assert persist < events.index(
                ("execute_arm", module.TARGETED_GRAFTS[index])
            )
    placebo_execute = events.index(("execute_placebo", module.R2))
    placebo_persist = events.index(
        ("persist_arm", 7, ("N", module.R2, "V_PLACEBO"))
    )
    assert events.index(("persist_arm", 6, module.TARGETED_GRAFTS[5])) < placebo_execute
    assert placebo_execute < placebo_persist


def test_repeat2_omits_placebo_but_keeps_exact_primary_set(monkeypatch):
    events = []
    _install_fakes(monkeypatch, events)
    result = module.run_targeted_treatment_case(
        object(), object(), _case(), eos_ids=[1], repeat_index=2,
    )
    assert len(result["arms"]) == 7
    assert result["placebo_attempt_count"] == 0
    assert not any(event[0] == "execute_placebo" for event in events)
    assert [
        (row["schedule"], row["region"], row["cell"])
        for row in result["arms"][1:]
    ] == list(module.TARGETED_GRAFTS)


def test_contract_verifier_rejects_repeat3_and_nonreused_ff(monkeypatch):
    events = []
    _install_fakes(monkeypatch, events)
    result = module.run_targeted_treatment_case(
        object(), object(), _case(), eos_ids=[1], repeat_index=2,
    )
    malformed_repeat = dict(result)
    malformed_repeat["repeat_index"] = 3
    malformed_repeat["foundation"] = dict(result["foundation"])
    malformed_repeat["foundation"]["repeat_index"] = 3
    with pytest.raises(module.PrecisionProbeP02CaseError,
                       match="repeat index"):
        module.assert_targeted_treatment(
            malformed_repeat, case_id="e01", repeat_index=3
        )

    nonreused = dict(result)
    nonreused["arms"] = list(result["arms"])
    nonreused["arms"][0] = dict(result["arms"][0])
    nonreused["arms"][0]["scores"] = dict(result["fresh_scores"])
    with pytest.raises(module.PrecisionProbeP02CaseError,
                       match="FF anchor"):
        module.assert_targeted_treatment(
            nonreused, case_id="e01", repeat_index=2
        )


def test_callback_failure_stops_before_next_expensive_arm(monkeypatch):
    events = []
    _install_fakes(monkeypatch, events)

    def fail_second(index, value):
        events.append(("persist_attempt", index))
        if index == 2:
            raise RuntimeError("durable write failed")

    with pytest.raises(RuntimeError, match="durable write failed"):
        module.run_targeted_treatment_case(
            object(), object(), _case(), eos_ids=[1], repeat_index=1,
            on_arm=fail_second,
        )
    executed = [event[1] for event in events if event[0] == "execute_arm"]
    assert executed == list(module.TARGETED_GRAFTS[:2])


@pytest.mark.parametrize("repeat_index", [0, 3, -1])
def test_repeat_index_is_frozen(repeat_index):
    with pytest.raises(module.PrecisionProbeP02CaseError,
                       match="repeat index"):
        module.run_targeted_treatment_case(
            object(), object(), _case(), eos_ids=[1],
            repeat_index=repeat_index,
        )
