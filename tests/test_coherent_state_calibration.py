from coherent_state_calibration import (
    CALIBRATION_SUMMARY,
    calibration_fresh_messages,
    calibration_labels,
    calibration_source_messages,
)


def test_label_assignment_is_deterministic_and_binary():
    assert calibration_labels("c01") == calibration_labels("c01")
    assert set(calibration_labels("c01")) == {"A", "B"}


def test_correct_and_wrong_sources_differ_only_in_label_fact():
    a = calibration_source_messages("A")
    b = calibration_source_messages("B")
    assert a[0] == b[0]
    assert a[2:] == b[2:]
    assert "Label A is approved" in a[1]["content"]
    assert "Label B is approved" in b[1]["content"]


def test_fresh_source_contains_no_label_assignment():
    fresh = calibration_fresh_messages()
    text = " ".join(m["content"] for m in fresh)
    assert "Label A" not in text and "Label B" not in text
    assert CALIBRATION_SUMMARY in text
