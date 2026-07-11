"""Frozen names and small immutable records for decision-canary v12.

This module is deliberately independent of every v10/v11 arm and store schema.
It contains no model execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence


DESIGN_ID = "coherent-state-decision-canary-v12"
CASE_SCHEMA = "coherent_state_decision_canary_v12_case_draft_v1"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"

PRIMARY_CASE_IDS = ("e01", "e02", "e03", "e04")
RESERVE_CASE_IDS = ("e05", "e06")
ENGINEERED_CASE_IDS = PRIMARY_CASE_IDS + RESERVE_CASE_IDS
CONVERSATION_CASE_IDS = ("d01", "d02", "d03")

N_SCHEDULE = "role_native_q1_replay"
P_SCHEDULE = "turn_aligned_replay"
SCHEDULES = (N_SCHEDULE, P_SCHEDULE)

R1 = "R1_content"
R2 = "R2_boundary"
R3 = "R3_anchor"
REGIONS = (R1, R2, R3)
PRIMARY_REGION = R2

ROW_SOURCES = ("F", "C", "W")
ARM_GRID = tuple(f"{key}{value}" for key in ROW_SOURCES for value in ROW_SOURCES)
P_ARM_SUBSET = ("CC", "WW", "FC", "FW")

ENGINEERED_CARRIER_REQUEST = (
    "Write the fixed neutral handoff note for the next assistant. "
    "Output only that note."
)
ENGINEERED_CARRIER_CONTENT = (
    "The prior discussion established the operating context and relevant "
    "decision criteria. Continue from this handoff, preserve the existing "
    "constraints, and answer later questions from the state available here. "
    "No unresolved action is introduced by this note."
)
ANCHOR_USER = (
    "Acknowledge receipt of this handoff without adding or repeating any "
    "factual detail."
)
ANCHOR_ASSISTANT = "Acknowledged."

EventKind = Literal["prefill", "q1"]


class CanarySchemaError(RuntimeError):
    """The additive canary schema or an immutable interval is invalid."""


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
            raise CanarySchemaError(f"invalid replay event kind: {self.kind}")
        if not self.label or not self.role:
            raise CanarySchemaError("replay event label/role is empty")
        if self.message_index < 0 or self.token_start < 0 or self.width < 1:
            raise CanarySchemaError(f"invalid replay event interval: {asdict(self)}")
        if self.kind == "q1" and self.width != 1:
            raise CanarySchemaError("q1 replay event does not have width one")
        if self.kind == "prefill" and self.width > 4096:
            raise CanarySchemaError("prefill replay event exceeds explicit 4096 bound")
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
            raise CanarySchemaError(f"carrier regions are not nested: {points}")
        return self

    def interval(self, region: str) -> tuple[int, int]:
        self.validate()
        if region == R1:
            return self.content_start, self.content_end
        if region == R2:
            return self.content_start, self.anchor_prefix_end
        if region == R3:
            return self.content_start, self.anchor_content_end
        raise CanarySchemaError(f"unknown carrier region: {region}")


@dataclass(frozen=True)
class ReplayPlan:
    token_ids: list[int]
    message_start_positions: list[int]
    events: list[ReplayEvent]
    regions: CarrierRegions

    def validate(self) -> "ReplayPlan":
        if not self.token_ids or not self.events:
            raise CanarySchemaError("replay plan is empty")
        if not self.message_start_positions or self.message_start_positions[0] != 0:
            raise CanarySchemaError("message starts do not begin at zero")
        if any(right <= left for left, right in zip(
                self.message_start_positions, self.message_start_positions[1:])):
            raise CanarySchemaError("message starts are not strictly increasing")
        expected = 0
        for event in self.events:
            event.validate()
            if event.token_start != expected:
                raise CanarySchemaError(
                    f"event coverage gap/overlap: {event.token_start} != {expected}")
            expected = event.token_end
        if expected != len(self.token_ids):
            raise CanarySchemaError(
                f"event coverage ends at {expected}, tokens end at {len(self.token_ids)}")
        if self.regions.anchor_content_end > len(self.token_ids):
            raise CanarySchemaError("carrier regions exceed replay token stream")
        self.regions.validate()
        event_starts = {event.token_start for event in self.events}
        event_ends = {event.token_end for event in self.events}
        if self.regions.content_start not in event_starts:
            raise CanarySchemaError("R1 does not begin at an event boundary")
        for label, point in (
            ("R1", self.regions.content_end),
            ("R2", self.regions.anchor_prefix_end),
            ("R3", self.regions.anchor_content_end),
        ):
            if point not in event_ends:
                raise CanarySchemaError(f"{label} does not end at an event boundary")
        return self

    def geometry(self) -> dict:
        self.validate()
        return {
            "token_count": len(self.token_ids),
            "message_start_positions": list(self.message_start_positions),
            "events": [
                {
                    "kind": event.kind,
                    "label": event.label,
                    "role": event.role,
                    "message_index": event.message_index,
                    "token_start": event.token_start,
                    "token_end": event.token_end,
                    "width": event.width,
                }
                for event in self.events
            ],
            "regions": asdict(self.regions),
        }


def validate_arm_name(arm: str) -> tuple[str, str]:
    if arm not in ARM_GRID:
        raise CanarySchemaError(f"unknown canary arm: {arm}")
    return arm[0], arm[1]


def require_matching_geometry(left: ReplayPlan, right: ReplayPlan) -> None:
    if left.geometry() != right.geometry():
        raise CanarySchemaError("role-native replay geometry differs")
