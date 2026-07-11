import pytest

from coherent_canary_schema import (
    ARM_GRID,
    PRIMARY_REGION,
    R1,
    R2,
    R3,
    CanarySchemaError,
    CarrierRegions,
    ReplayEvent,
    validate_arm_name,
)


def test_frozen_arm_grid_is_complete_and_ordered():
    assert ARM_GRID == (
        "FF", "FC", "FW",
        "CF", "CC", "CW",
        "WF", "WC", "WW",
    )
    assert PRIMARY_REGION == R2


def test_replay_event_requires_explicit_q1_and_bounded_prefill():
    assert ReplayEvent("q1", "content", "assistant", 2, 10, 11).validate().width == 1
    assert ReplayEvent("prefill", "user", "user", 1, 0, 4096).validate().width == 4096
    with pytest.raises(CanarySchemaError):
        ReplayEvent("q1", "content", "assistant", 2, 10, 12).validate()
    with pytest.raises(CanarySchemaError):
        ReplayEvent("prefill", "user", "user", 1, 0, 4097).validate()


def test_carrier_regions_are_nested_and_named():
    regions = CarrierRegions(100, 140, 142, 173).validate()
    assert regions.interval(R1) == (100, 140)
    assert regions.interval(R2) == (100, 142)
    assert regions.interval(R3) == (100, 173)
    with pytest.raises(CanarySchemaError):
        CarrierRegions(100, 140, 140, 173).validate()


def test_arm_sources_are_explicit():
    assert validate_arm_name("FC") == ("F", "C")
    with pytest.raises(CanarySchemaError):
        validate_arm_name("G_correct")
