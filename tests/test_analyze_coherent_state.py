from __future__ import annotations

import pytest

from analyze_coherent_state import AnalysisError, analyze, bootstrap_interval, t_interval


def _docs(n=6, *, cf=0.4, cw=0.3, vf=0.0, calibration=True,
          technical=True, headroom=0.6):
    docs = []
    for i in range(n):
        f = 0.2 + i * 0.01
        c = f + cf
        w = c - cw
        docs.append({
            "conversation_id": f"c{i+1:02d}",
            "order_position": i + 1,
            "status": "scored",
            "gates": {"technical_pass": technical,
                      "failures": [] if technical else ["injected"]},
            "conversation_outcomes": {
                "A_full": f + headroom,
                "F_fresh": f,
                "C_coherent": c,
                "W_wrong": w,
                "V_only": f + vf,
                "K_only": f,
                "D_delta": f,
            },
            "calibration_outcomes": {
                "C_coherent": 0.3 if calibration else 0.0,
                "F_fresh": 0.0,
                "W_wrong": 0.0 if calibration else 0.1,
            },
        })
    return docs


def test_intervals_are_conversation_level_and_deterministic():
    vals = [1, 2, 3, 4, 5, 6]
    t = t_interval(vals)
    assert t["n"] == 6 and t["lo"] < t["mean"] < t["hi"]
    assert bootstrap_interval(vals) == bootstrap_interval(vals)


def test_n6_extends_when_channel_direction_or_calibration_fires():
    out = analyze(_docs())
    assert out["serial_decision"] == "EXTEND_TO_12"
    assert out["regime_gate"]["passes"]
    assert out["calibration"]["fires"]


def test_n6_futility_requires_both_nonpositive_and_calibration_failure():
    out = analyze(_docs(cf=-0.1, cw=-0.2, calibration=False))
    assert out["serial_decision"] == "STOP_FUTILITY"


def test_technical_and_regime_failures_take_precedence():
    assert analyze(_docs(technical=False))["serial_decision"] == "STOP_TECHNICAL"
    assert analyze(_docs(headroom=0.1))["serial_decision"] == "STOP_REGIME"


def test_final_intersection_channel_claim_needs_both_intervals():
    # Identical per-row effects have zero variance and positive lower bounds.
    out = analyze(_docs(n=12, cf=0.4, cw=0.3, vf=0.0))
    assert out["serial_decision"] == "FINAL_N12"
    assert out["contrasts"]["CF"]["t"]["lo"] > 0
    assert out["contrasts"]["CW"]["t"]["lo"] > 0
    assert out["interpretation"] == "HISTORY_CHANNEL_KV_SPLIT_LOSES_IT"


def test_n12_regime_gate_remains_frozen_to_first_six():
    docs = _docs(n=12, headroom=0.6)
    for doc in docs[6:]:
        doc["conversation_outcomes"]["A_full"] = -10.0
    out = analyze(docs)
    assert out["regime_gate"]["passes"]
    assert out["regime_gate"]["frozen_at_n"] == 6


def test_missing_arm_and_noncontiguous_order_fail_closed():
    docs = _docs()
    del docs[0]["conversation_outcomes"]["K_only"]
    with pytest.raises(AnalysisError, match="missing arms"):
        analyze(docs)
    docs = _docs()
    docs[2]["order_position"] = 7
    with pytest.raises(AnalysisError, match="non-contiguous"):
        analyze(docs)
