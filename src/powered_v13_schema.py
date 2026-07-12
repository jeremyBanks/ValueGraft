"""Immutable names and tokenizer-plan records for powered successor v13.

This namespace is deliberately independent of the historical v12 authorization
and contains no model execution or retained serving state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


DESIGN_ID = "coherent-state-powered-successor-v13"
CASE_SCHEMA = "coherent-state-powered-successor-v13-fixture-draft-v1"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"

N_SCHEDULE = "role_native_q1_replay"
P_SCHEDULE = "turn_aligned_replay"
SCHEDULES = (N_SCHEDULE, P_SCHEDULE)

R1 = "R1_carrier_content"
R2 = "R2_carrier_boundary"
R3 = "R3_acknowledged_content"
REGIONS = (R1, R2, R3)
PRIMARY_REGION = R2

HISTORY_SOURCES = ("F", "C", "W")
PRIMARY_ARMS = ("FF", "CC", "WW", "FC", "FW", "VP")

CARRIER_REQUEST = (
    "Write a 40–60 word handoff that mentions only shared background and the "
    "existence—not the content—of prior decision criteria. Do not state or "
    "imply any option, answer, name, number, value, rule branch, outcome, or "
    "nonfocal fact. Output only the handoff."
)
ANCHOR_USER = (
    "Acknowledge receipt of this handoff without adding or repeating any "
    "factual detail."
)
ANCHOR_ASSISTANT = "Acknowledged."

EventKind = Literal["prefill", "q1"]


class V13SchemaError(RuntimeError):
    """A v13 immutable record or tokenizer geometry is invalid."""


@dataclass(frozen=True)
class ReplayEvent:
    kind: EventKind
    label: str
    role: str
    message_index: int
    token_start: int
    token_end: int

    @property
    def width(self) -> int:
        return self.token_end - self.token_start

    def validate(self) -> "ReplayEvent":
        if self.kind not in ("prefill", "q1"):
            raise V13SchemaError(f"invalid replay event kind: {self.kind}")
        if not self.label or not self.role or self.message_index < 0:
            raise V13SchemaError("invalid replay event metadata")
        if self.token_start < 0 or self.width < 1:
            raise V13SchemaError(
                f"invalid replay event interval: {asdict(self)}")
        if self.kind == "q1" and self.width != 1:
            raise V13SchemaError("q1 replay event does not have width one")
        if self.kind == "prefill" and self.width > 4096:
            raise V13SchemaError("prefill replay event exceeds 4096 tokens")
        return self


@dataclass(frozen=True)
class CarrierRegions:
    content_start: int
    content_end: int
    anchor_prefix_end: int
    anchor_content_end: int

    def validate(self) -> "CarrierRegions":
        points = (self.content_start, self.content_end,
                  self.anchor_prefix_end, self.anchor_content_end)
        if self.content_start < 0 or any(
                right <= left for left, right in zip(points, points[1:])):
            raise V13SchemaError(f"carrier regions are not nested: {points}")
        return self

    def interval(self, region: str) -> tuple[int, int]:
        self.validate()
        if region == R1:
            return self.content_start, self.content_end
        if region == R2:
            return self.content_start, self.anchor_prefix_end
        if region == R3:
            return self.content_start, self.anchor_content_end
        raise V13SchemaError(f"unknown carrier region: {region}")


@dataclass(frozen=True)
class ReplayPlan:
    token_ids: list[int]
    message_start_positions: list[int]
    events: list[ReplayEvent]
    regions: CarrierRegions

    def validate(self) -> "ReplayPlan":
        if not self.token_ids or not self.events:
            raise V13SchemaError("replay plan is empty")
        if (not self.message_start_positions or
                self.message_start_positions[0] != 0):
            raise V13SchemaError("message starts do not begin at zero")
        if any(right <= left for left, right in zip(
                self.message_start_positions,
                self.message_start_positions[1:])):
            raise V13SchemaError("message starts are not strictly increasing")
        cursor = 0
        for event in self.events:
            event.validate()
            if event.token_start != cursor:
                raise V13SchemaError("replay event coverage has a gap/overlap")
            cursor = event.token_end
        if cursor != len(self.token_ids):
            raise V13SchemaError("replay events do not cover the token stream")
        self.regions.validate()
        if self.regions.anchor_content_end > len(self.token_ids):
            raise V13SchemaError("carrier regions exceed the replay stream")
        starts = {event.token_start for event in self.events}
        ends = {event.token_end for event in self.events}
        if self.regions.content_start not in starts:
            raise V13SchemaError("R1 does not begin at an event boundary")
        for label, point in (
            (R1, self.regions.content_end),
            (R2, self.regions.anchor_prefix_end),
            (R3, self.regions.anchor_content_end),
        ):
            if point not in ends:
                raise V13SchemaError(f"{label} does not end at an event boundary")
        return self

    def geometry(self) -> dict:
        self.validate()
        return {
            "token_count": len(self.token_ids),
            "message_start_positions": list(self.message_start_positions),
            "events": [asdict(event) | {"width": event.width}
                       for event in self.events],
            "regions": asdict(self.regions),
        }


@dataclass(frozen=True)
class DestinationEvent:
    kind: EventKind
    label: str
    role: str
    message_index: int
    physical_start: int
    physical_end: int
    logical_start: int
    logical_end: int

    @property
    def width(self) -> int:
        return self.physical_end - self.physical_start

    def validate(self) -> "DestinationEvent":
        if self.kind not in ("prefill", "q1"):
            raise V13SchemaError("invalid destination event kind")
        if not self.label or not self.role or self.message_index < 0:
            raise V13SchemaError("invalid destination event metadata")
        if (self.physical_start < 0 or self.logical_start < 0 or
                self.width < 1 or
                self.logical_end - self.logical_start != self.width):
            raise V13SchemaError("invalid destination event interval")
        if self.kind == "q1" and self.width != 1:
            raise V13SchemaError("destination q1 event width differs")
        if self.kind == "prefill" and self.width > 4096:
            raise V13SchemaError("destination prefill exceeds 4096 tokens")
        return self


@dataclass(frozen=True)
class FreshDestinationPlan:
    token_ids: list[int]
    logical_positions: list[int]
    physical_positions: list[int]
    source_token_indices: list[int]
    events: list[DestinationEvent]
    source_regions: CarrierRegions
    physical_regions: CarrierRegions
    system_width: int
    suffix_source_start: int

    def validate(self) -> "FreshDestinationPlan":
        width = len(self.token_ids)
        if width < 1 or not (
            len(self.logical_positions) == len(self.physical_positions) ==
            len(self.source_token_indices) == width
        ):
            raise V13SchemaError("fresh destination arrays differ in length")
        if self.physical_positions != list(range(width)):
            raise V13SchemaError("fresh physical positions are not packed")
        if self.source_token_indices != self.logical_positions:
            raise V13SchemaError("fresh source/logical mapping differs")
        if not (0 < self.system_width < width and
                self.suffix_source_start > self.system_width):
            raise V13SchemaError("fresh system/gap bounds are invalid")
        if self.logical_positions[:self.system_width] != list(
                range(self.system_width)):
            raise V13SchemaError("fresh system logical positions differ")
        suffix_width = width - self.system_width
        if self.logical_positions[self.system_width:] != list(range(
                self.suffix_source_start,
                self.suffix_source_start + suffix_width)):
            raise V13SchemaError("fresh suffix logical positions differ")
        cursor = 0
        for event in self.events:
            event.validate()
            if event.physical_start != cursor:
                raise V13SchemaError("fresh event coverage differs")
            expected = self.logical_positions[
                event.physical_start:event.physical_end]
            if expected != list(range(event.logical_start,
                                      event.logical_end)):
                raise V13SchemaError("fresh event logical span differs")
            cursor = event.physical_end
        if cursor != width:
            raise V13SchemaError("fresh events do not cover tokens")
        self.source_regions.validate()
        self.physical_regions.validate()
        for region in REGIONS:
            source_start, source_end = self.source_regions.interval(region)
            physical_start, physical_end = self.physical_regions.interval(region)
            if source_start != self.logical_positions[physical_start]:
                raise V13SchemaError(f"fresh {region} start mapping differs")
            if source_end - source_start != physical_end - physical_start:
                raise V13SchemaError(f"fresh {region} width mapping differs")
        return self

    def geometry(self) -> dict:
        self.validate()
        return {
            "token_count": len(self.token_ids),
            "logical_positions": list(self.logical_positions),
            "physical_positions": list(self.physical_positions),
            "source_token_indices": list(self.source_token_indices),
            "events": [asdict(event) | {"width": event.width}
                       for event in self.events],
            "source_regions": asdict(self.source_regions),
            "physical_regions": asdict(self.physical_regions),
            "system_width": self.system_width,
            "suffix_source_start": self.suffix_source_start,
        }


def require_matching_geometry(left: ReplayPlan, right: ReplayPlan) -> None:
    if left.geometry() != right.geometry():
        raise V13SchemaError("v13 replay geometry differs")
