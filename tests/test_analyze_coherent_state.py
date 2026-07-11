from __future__ import annotations

import pytest

from analyze_coherent_state import AnalysisError, analyze, bootstrap_interval, t_interval


def _docs(n=6, *, cf=0.4, cw=0.3, vf=0.0, calibration=True,
          technical=True, headroom=0.6):
    docs = []
    calibration_labels = ["A", "A", "A", "A", "B", "A",
                          "B", "A", "B", "A", "B", "A"]
    for i in range(n):
        f = 0.2 + i * 0.01
        c = f + cf
        w = c - cw
        docs.append({
            "schema": 2,
            "design_id": "coherent-state-gapped-v2",
            "amendment_id": "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2",
            "conversation_id": f"c{i+1:02d}",
            "order_position": i + 1,
            "status": "scored",
            "gates": {"technical_pass": technical,
                      "failures": [] if technical else ["injected"]},
            "conversation_outcomes": {
                "A_full": f + headroom,
                "G_fresh": f,
                "G_correct": c,
                "G_wrong": w,
                "G_Vcorrect": f + vf,
                "G_Kcorrect": f,
                "G_delta": f,
            },
            "calibration_outcomes": {
                "G_correct": 0.3 if calibration else 0.0,
                "G_fresh": 0.0,
                "G_wrong": 0.0 if calibration else 0.1,
            },
            "calibration": {"correct_label": calibration_labels[i]},
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
    assert out["calibration"]["variants"]["A"]["n_repeated_executions"] == 5
    assert out["calibration"]["variants"]["B"]["n_repeated_executions"] == 1


def test_calibration_does_not_pseudoreplicate_majority_label():
    docs = _docs()
    b = next(doc for doc in docs if doc["calibration"]["correct_label"] == "B")
    b["calibration_outcomes"] = {
        "G_correct": 0.0, "G_fresh": 0.1, "G_wrong": 0.1}
    out = analyze(docs)
    assert not out["calibration"]["fires"]
    assert out["calibration"]["both_directional_variants"] == 1


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
    assert out["contrasts"]["GF"]["t"]["lo"] > 0
    assert out["contrasts"]["GW"]["t"]["lo"] > 0
    assert out["interpretation"] == "HISTORY_SPECIFIC_CHANNEL"


def test_n12_regime_gate_remains_frozen_to_first_six():
    docs = _docs(n=12, headroom=0.6)
    for doc in docs[6:]:
        doc["conversation_outcomes"]["A_full"] = -10.0
    out = analyze(docs)
    assert out["regime_gate"]["passes"]
    assert out["regime_gate"]["frozen_at_n"] == 6


def test_n12_calibration_gate_remains_frozen_to_first_six():
    docs = _docs(n=12, calibration=False)
    for doc in docs[6:]:
        doc["calibration_outcomes"] = {
            "G_correct": 1.0, "G_fresh": 0.0, "G_wrong": 0.0}
    out = analyze(docs)
    assert not out["calibration"]["fires"]
    assert out["calibration"]["n"] == 6
    assert out["calibration"]["frozen_at_n"] == 6


def test_missing_arm_and_noncontiguous_order_fail_closed():
    docs = _docs()
    del docs[0]["conversation_outcomes"]["G_Kcorrect"]
    with pytest.raises(AnalysisError, match="missing arms"):
        analyze(docs)
    docs = _docs()
    docs[2]["order_position"] = 7
    with pytest.raises(AnalysisError, match="non-contiguous"):
        analyze(docs)


def test_old_packed_or_unversioned_documents_fail_closed():
    docs = _docs()
    docs[0]["schema"] = 1
    with pytest.raises(AnalysisError, match="schema 2"):
        analyze(docs)
    docs = _docs()
    docs[0]["design_id"] = "coherent-state-packed-v0"
    with pytest.raises(AnalysisError, match="Amendments-1-2"):
        analyze(docs)
