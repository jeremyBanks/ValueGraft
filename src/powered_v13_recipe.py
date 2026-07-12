"""Deterministic, treatment-blind recipe foundation for powered successor v13.

This module defines the compact 32,768-candidate sampling frame (4,096
parameter tuples in each of eight strata), its canonical candidate-ID
algorithm, pure rule evaluators, and the conversation-template compiler.

The randomization boundary is deliberate: enumerating and auditing the compact
parameter frame does *not* materialize any in-pool conversation.  In-pool text
can be compiled only with an explicit post-permutation authorization record.
Tests exercise the templates with component values that are outside every
production pool.

No tokenizer, subject model, result, RNG, or permutation seed is used here.
Tokenizer geometry and literal ranked-candidate materialization are later
layers and remain pending.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from powered_v13_schema import CASE_SCHEMA as FIXTURE_SCHEMA
from powered_v13_schema import DESIGN_ID

RECIPE_SCHEMA = "coherent-state-powered-successor-v13-recipe-v1"
TEMPLATE_VERSION = "powered-v13-eight-family-template-v1"
GENERATOR_VERSION = "powered-v13-recipe-compiler-1"
POOL_SIZE_PER_STRATUM = 4096


class RecipeError(ValueError):
    """Fail-closed error for an invalid recipe tuple or draft fixture."""


@dataclass(frozen=True, slots=True)
class CandidateTuple:
    """Compact canonical tuple; it contains no conversation text."""

    stratum_id: str
    resolution_mode_id: str
    domain_skin_id: str
    label_pair_id: str
    logic_pack_id: str
    nonfocal_tail_pack_id: str

    def canonical_object(self) -> dict[str, str]:
        return {
            "domain_skin_id": self.domain_skin_id,
            "label_pair_id": self.label_pair_id,
            "logic_pack_id": self.logic_pack_id,
            "nonfocal_tail_pack_id": self.nonfocal_tail_pack_id,
            "resolution_mode_id": self.resolution_mode_id,
        }


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    stratum_id: str
    variant: str
    state_id: str
    outcome_index: int
    changed_value: Any
    evidence: Mapping[str, Any]
    proof: str


@dataclass(frozen=True, slots=True)
class RankedMaterializationAuthorization:
    """Receipt required before any in-pool tuple can become conversation text.

    This record does not create or inspect a permutation.  A later release
    verifier is responsible for binding its two hashes to the committed seed
    and literal permutation and for establishing that candidate_id is truly at
    permutation_rank.  The compiler merely refuses unreceipted expansion.
    """

    status: str
    candidate_id: str
    permutation_rank: int
    seed_manifest_sha256: str
    seed_git_commit: str
    literal_permutation_sha256: str


@dataclass(frozen=True, slots=True)
class _DomainSkin:
    id: str
    organization: str
    ledger: str
    coordinator: str
    subject: str
    site: str
    distractor_a: str
    distractor_b: str


@dataclass(frozen=True, slots=True)
class _LabelPair:
    id: str
    zero: str
    one: str


@dataclass(frozen=True, slots=True)
class _NonfocalTailPack:
    id: str
    location: str
    fact_name: str
    target: str
    countertarget: str
    probe: str


STRATA: tuple[str, ...] = (
    "threshold_eligibility",
    "sequential_tiebreak",
    "conjunction_all_conditions",
    "interval_schedule_overlap",
    "arithmetic_capacity_budget",
    "categorical_set_membership",
    "ordered_priority_exception",
    "referent_alias_resolution",
)

STRATUM_TITLES: Mapping[str, str] = MappingProxyType({
    "threshold_eligibility": "inclusive threshold eligibility",
    "sequential_tiebreak": "sequential decision with deterministic tie-break",
    "conjunction_all_conditions": "conjunction of all three conditions",
    "interval_schedule_overlap": "minimum-duration and larger-overlap schedule",
    "arithmetic_capacity_budget": "capacity after ceiling and reserve",
    "categorical_set_membership": "required and forbidden set membership",
    "ordered_priority_exception": "ordered priority with a named exception",
    "referent_alias_resolution": "two- or three-hop alias resolution",
})

RESOLUTION_MODES: tuple[str, ...] = (
    "explicit_resolution",
    "unstated_resolution",
)

DOMAIN_SKINS: tuple[_DomainSkin, ...] = (
    _DomainSkin("canal_cooperative", "Canal Cooperative", "lockside worksheet",
                "rotation steward", "incoming parcel", "east lock",
                "canvas-cover inventory", "volunteer radio roster"),
    _DomainSkin("museum_workroom", "Museum Workroom", "handling register",
                "collections scheduler", "loan crate", "north annex",
                "humidity-card replacement", "docent badge renewal"),
    _DomainSkin("library_network", "Library Network", "transfer docket",
                "circulation planner", "traveling collection", "river branch",
                "book-cart wheel service", "evening key census"),
    _DomainSkin("hill_observatory", "Hill Observatory", "observation notebook",
                "survey convener", "equipment bundle", "upland shelter",
                "rain-gauge calibration", "trail-marker repainting"),
    _DomainSkin("civic_archive", "Civic Archive", "accession worksheet",
                "records coordinator", "document carton", "map room",
                "scanner lamp replacement", "reading-room locker audit"),
    _DomainSkin("festival_logistics", "Festival Logistics Desk", "routing folio",
                "site dispatcher", "vendor kit", "garden concourse",
                "awning-tag inventory", "usher lanyard exchange"),
    _DomainSkin("estuary_observers", "Estuary Observers", "inspection journal",
                "route facilitator", "monitoring case", "marsh outpost",
                "sample-bottle count", "footbridge notice renewal"),
    _DomainSkin("community_kitchen", "Community Kitchen", "service worksheet",
                "shift coordinator", "pantry tote", "courtyard pantry",
                "apron-mending queue", "delivery-bicycle service"),
)

LABEL_PAIRS: tuple[_LabelPair, ...] = (
    _LabelPair("garnet_mallow", "Garnet", "Mallow"),
    _LabelPair("heron_otter", "Heron", "Otter"),
    _LabelPair("willow_copper", "Willow", "Copper"),
    _LabelPair("falcon_maple", "Falcon", "Maple"),
    _LabelPair("riverstone_sunbeam", "Riverstone", "Sunbeam"),
    _LabelPair("juniper_alder", "Juniper", "Alder"),
    _LabelPair("quartz_flint", "Quartz", "Flint"),
    _LabelPair("comet_violet", "Comet", "Violet"),
)

NONFOCAL_TAIL_PACKS: tuple[_NonfocalTailPack, ...] = (
    _NonfocalTailPack(
        "system_correspondence_word", "system", "correspondence word",
        "Morrow", "Trellis", "What is the standing correspondence word?"),
    _NonfocalTailPack(
        "system_receipt_mark", "system", "receipt mark",
        "Pewter", "Dovetail", "What is the standing receipt mark?"),
    _NonfocalTailPack(
        "tail_shelf_word", "tail_user", "shelf word",
        "Nimbus", "Solace", "What shelf word is stated in the retained exchange?"),
    _NonfocalTailPack(
        "tail_desk_mark", "tail_assistant", "desk mark",
        "Russet", "Ivory", "What desk mark is stated in the retained exchange?"),
)

DOMAIN_SKIN_IDS = tuple(item.id for item in DOMAIN_SKINS)
LABEL_PAIR_IDS = tuple(item.id for item in LABEL_PAIRS)
NONFOCAL_TAIL_PACK_IDS = tuple(item.id for item in NONFOCAL_TAIL_PACKS)

_DOMAIN_BY_ID = MappingProxyType({item.id: item for item in DOMAIN_SKINS})
_LABEL_BY_ID = MappingProxyType({item.id: item for item in LABEL_PAIRS})
_NONFOCAL_BY_ID = MappingProxyType({item.id: item for item in NONFOCAL_TAIL_PACKS})

_ORIENTATIONS: tuple[str, ...] = ("c0_w1", "c1_w0")
LOGIC_PACK_IDS: tuple[str, ...] = tuple(
    f"magnitude_{magnitude}_{orientation}"
    for magnitude in range(1, 5)
    for orientation in _ORIENTATIONS
)


# The four magnitude records in every family define a state that evaluates to
# outcome 0 and a state that evaluates to outcome 1.  Orientation only assigns
# those fixed states to C and W; it never changes the rule.
_MAGNITUDES: Mapping[str, tuple[Mapping[str, Any], ...]] = MappingProxyType({
    "threshold_eligibility": (
        {"threshold": 60, "state0": 60, "state1": 59},
        {"threshold": 72, "state0": 74, "state1": 69},
        {"threshold": 84, "state0": 88, "state1": 79},
        {"threshold": 91, "state0": 98, "state1": 83},
    ),
    "sequential_tiebreak": (
        {"cutoff": 70, "primary": 82, "first_time": "10:30",
         "state0": "10:40", "state1": "10:20"},
        {"cutoff": 74, "primary": 86, "first_time": "11:25",
         "state0": "11:45", "state1": "11:05"},
        {"cutoff": 78, "primary": 90, "first_time": "13:15",
         "state0": "13:50", "state1": "12:40"},
        {"cutoff": 82, "primary": 94, "first_time": "15:10",
         "state0": "15:55", "state1": "14:25"},
    ),
    "conjunction_all_conditions": (
        {"queue": 8, "queue_max": 10, "minimum": 60,
         "state0": 64, "state1": 56},
        {"queue": 11, "queue_max": 13, "minimum": 70,
         "state0": 76, "state1": 64},
        {"queue": 14, "queue_max": 17, "minimum": 80,
         "state0": 88, "state1": 72},
        {"queue": 17, "queue_max": 21, "minimum": 90,
         "state0": 99, "state1": 81},
    ),
    "interval_schedule_overlap": (
        {"reference": (540, 1020), "first": (600, 840), "minimum": 120,
         "state0": (600, 780), "state1": (600, 900)},
        {"reference": (480, 960), "first": (540, 810), "minimum": 150,
         "state0": (570, 780), "state1": (510, 870)},
        {"reference": (600, 1080), "first": (660, 960), "minimum": 180,
         "state0": (690, 930), "state1": (630, 990)},
        {"reference": (420, 900), "first": (480, 810), "minimum": 180,
         "state0": (510, 780), "state1": (450, 840)},
    ),
    "arithmetic_capacity_budget": (
        {"capacity": 12, "reserve": 2, "group": 5,
         "state0": 50, "state1": 51},
        {"capacity": 14, "reserve": 2, "group": 4,
         "state0": 48, "state1": 49},
        {"capacity": 16, "reserve": 3, "group": 5,
         "state0": 65, "state1": 66},
        {"capacity": 18, "reserve": 4, "group": 6,
         "state0": 84, "state1": 85},
    ),
    "categorical_set_membership": (
        {"required": ("calm", "dry", "verified"), "fixed": ("calm", "dry"),
         "state0": "verified", "state1": "restricted",
         "forbidden": ("restricted", "quarantined")},
        {"required": ("logged", "sealed", "local"), "fixed": ("logged", "sealed"),
         "state0": "local", "state1": "foreign",
         "forbidden": ("foreign", "damaged")},
        {"required": ("stable", "marked", "cleared"), "fixed": ("stable", "marked"),
         "state0": "cleared", "state1": "held",
         "forbidden": ("held", "expired")},
        {"required": ("listed", "covered", "quiet"), "fixed": ("listed", "covered"),
         "state0": "quiet", "state1": "noisy",
         "forbidden": ("noisy", "unlisted")},
    ),
    "ordered_priority_exception": (
        {"first_trigger": "time-sensitive", "second_trigger": "scheduled",
         "exception": "Lantern Pause"},
        {"first_trigger": "safety review", "second_trigger": "routine review",
         "exception": "Meadow Hold"},
        {"first_trigger": "same-day", "second_trigger": "next-cycle",
         "exception": "Bramble Rest"},
        {"first_trigger": "priority notice", "second_trigger": "standard notice",
         "exception": "Orchard Quiet"},
    ),
    "referent_alias_resolution": (
        {"query": "folio", "relay": "rowan index", "branch0": "west alcove",
         "branch1": "east alcove", "extra0": None, "extra1": None},
        {"query": "parcel", "relay": "glass register", "branch0": "upper bay",
         "branch1": "lower bay", "extra0": "amber card", "extra1": "blue card"},
        {"query": "kit", "relay": "reed directory", "branch0": "stone cabinet",
         "branch1": "clay cabinet", "extra0": None, "extra1": None},
        {"query": "packet", "relay": "copper list", "branch0": "morning drawer",
         "branch1": "evening drawer", "extra0": "round slip", "extra1": "square slip"},
    ),
})


_DEVELOPMENT_DOMAIN = _DomainSkin(
    "development_orchard_exchange", "Orchard Exchange", "sorting memorandum",
    "plot convenor", "seed envelope", "pear gallery",
    "ladder-rung inspection", "watering-can repair",
)
_DEVELOPMENT_LABEL = _LabelPair(
    "development_tokenizer_qualified_quartz_flint", "Quartz", "Flint")
_DEVELOPMENT_NONFOCAL = _NonfocalTailPack(
    "development_tail_token", "tail_user", "filing token",
    "Bramble", "Mica", "What filing token is stated in the retained exchange?",
)
_DEVELOPMENT_LOGIC_PACK_ID = "development_sentinel_c0_w1"

_DEVELOPMENT_MAGNITUDES: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "threshold_eligibility": {"threshold": 67, "state0": 68, "state1": 66},
    "sequential_tiebreak": {"cutoff": 73, "primary": 87,
                            "first_time": "14:20", "state0": "14:35",
                            "state1": "14:05"},
    "conjunction_all_conditions": {"queue": 9, "queue_max": 12,
                                    "minimum": 65, "state0": 69, "state1": 61},
    "interval_schedule_overlap": {"reference": (510, 990),
                                   "first": (570, 840), "minimum": 135,
                                   "state0": (600, 810), "state1": (540, 900)},
    "arithmetic_capacity_budget": {"capacity": 17, "reserve": 2, "group": 7,
                                    "state0": 105, "state1": 106},
    "categorical_set_membership": {
        "required": ("counted", "wrapped", "current"),
        "fixed": ("counted", "wrapped"), "state0": "current",
        "state1": "deferred", "forbidden": ("deferred", "missing")},
    "ordered_priority_exception": {"first_trigger": "immediate sorting",
                                   "second_trigger": "ordinary sorting",
                                   "exception": "Pear Rest"},
    "referent_alias_resolution": {"query": "envelope", "relay": "orchard card",
                                  "branch0": "north basket", "branch1": "south basket",
                                  "extra0": "woven tag", "extra1": "plain tag"},
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecipeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the protocol's canonical UTF-8 JSON encoding."""

    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RecipeError(f"value is not canonical-JSON encodable: {exc}") from exc
    return rendered.encode("utf-8")


def stable_candidate_id(candidate: CandidateTuple) -> str:
    _require(candidate.stratum_id in STRATA, "unknown stratum")
    payload = [
        DESIGN_ID,
        candidate.stratum_id,
        TEMPLATE_VERSION,
        candidate.canonical_object(),
    ]
    return sha256(canonical_json_bytes(payload)).hexdigest()


def is_pool_member(candidate: CandidateTuple) -> bool:
    return (
        candidate.stratum_id in STRATA
        and candidate.resolution_mode_id in RESOLUTION_MODES
        and candidate.domain_skin_id in DOMAIN_SKIN_IDS
        and candidate.label_pair_id in LABEL_PAIR_IDS
        and candidate.logic_pack_id in LOGIC_PACK_IDS
        and candidate.nonfocal_tail_pack_id in NONFOCAL_TAIL_PACK_IDS
    )


def enumerate_candidate_tuples(stratum_id: str) -> Iterator[CandidateTuple]:
    """Enumerate compact parameters only; never expand conversation text."""

    _require(stratum_id in STRATA, f"unknown stratum: {stratum_id}")
    for mode, domain, labels, logic, nonfocal in product(
        RESOLUTION_MODES,
        DOMAIN_SKIN_IDS,
        LABEL_PAIR_IDS,
        LOGIC_PACK_IDS,
        NONFOCAL_TAIL_PACK_IDS,
    ):
        yield CandidateTuple(
            stratum_id=stratum_id,
            resolution_mode_id=mode,
            domain_skin_id=domain,
            label_pair_id=labels,
            logic_pack_id=logic,
            nonfocal_tail_pack_id=nonfocal,
        )


def _logic_definition(candidate: CandidateTuple) -> tuple[Mapping[str, Any], str]:
    if candidate.logic_pack_id == _DEVELOPMENT_LOGIC_PACK_ID:
        return _DEVELOPMENT_MAGNITUDES[candidate.stratum_id], "c0_w1"
    match = re.fullmatch(r"magnitude_([1-4])_(c0_w1|c1_w0)",
                         candidate.logic_pack_id)
    _require(match is not None, "invalid logic pack ID")
    magnitude_index = int(match.group(1)) - 1
    return _MAGNITUDES[candidate.stratum_id][magnitude_index], match.group(2)


def _state_for_variant(orientation: str, variant: str) -> str:
    _require(variant in ("C", "W"), "variant must be C or W")
    if orientation == "c0_w1":
        return "state0" if variant == "C" else "state1"
    _require(orientation == "c1_w0", "invalid orientation")
    return "state1" if variant == "C" else "state0"


def _minutes(text: str) -> int:
    hour, minute = (int(part) for part in text.split(":"))
    return hour * 60 + minute


def _clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _overlap(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def evaluate_rule(candidate: CandidateTuple, variant: str) -> RuleEvaluation:
    """Purely evaluate a candidate's focal rule for C or W."""

    _require(candidate.stratum_id in STRATA, "unknown stratum")
    definition, orientation = _logic_definition(candidate)
    state_id = _state_for_variant(orientation, variant)
    changed = definition[state_id] if state_id in definition else (state_id == "state1")
    stratum = candidate.stratum_id

    if stratum == "threshold_eligibility":
        score = int(changed)
        threshold = int(definition["threshold"])
        eligible = score >= threshold
        outcome = 0 if eligible else 1
        evidence = {"score": score, "inclusive_threshold": threshold,
                    "eligible": eligible}
        proof = f"{score} {'>=' if eligible else '<'} {threshold}"

    elif stratum == "sequential_tiebreak":
        first_time = str(definition["first_time"])
        second_time = str(changed)
        cutoff = int(definition["cutoff"])
        primary = int(definition["primary"])
        _require(primary >= cutoff, "both candidates must pass the cutoff")
        first_wins = _minutes(first_time) < _minutes(second_time)
        _require(_minutes(first_time) != _minutes(second_time), "tie-break timestamps tie")
        outcome = 0 if first_wins else 1
        evidence = {
            "cutoff": cutoff,
            "candidate_primary_scores": [primary, primary],
            "candidate_timestamps": [first_time, second_time],
            "tie_break": "earlier_timestamp",
        }
        proof = ("both pass and primary scores tie; "
                 f"{'first' if first_wins else 'second'} timestamp is earlier")

    elif stratum == "conjunction_all_conditions":
        active = True
        queue_ok = int(definition["queue"]) <= int(definition["queue_max"])
        inspection_ok = int(changed) >= int(definition["minimum"])
        passes = active and queue_ok and inspection_ok
        outcome = 0 if passes else 1
        evidence = {
            "certification_active": active,
            "queue": int(definition["queue"]),
            "queue_max": int(definition["queue_max"]),
            "inspection": int(changed),
            "inspection_minimum": int(definition["minimum"]),
            "all_conditions": passes,
        }
        proof = f"active={active}, queue_ok={queue_ok}, inspection_ok={inspection_ok}"

    elif stratum == "interval_schedule_overlap":
        reference = tuple(int(value) for value in definition["reference"])
        first = tuple(int(value) for value in definition["first"])
        second = tuple(int(value) for value in changed)
        minimum = int(definition["minimum"])
        overlaps = (_overlap(reference, first), _overlap(reference, second))
        _require(all(value >= minimum for value in overlaps),
                 "both windows must meet minimum overlap")
        _require(overlaps[0] != overlaps[1], "overlap comparison ties")
        outcome = 0 if overlaps[0] > overlaps[1] else 1
        evidence = {
            "reference_minutes": reference,
            "candidate_minutes": [first, second],
            "minimum_overlap_minutes": minimum,
            "overlap_minutes": overlaps,
        }
        proof = f"both overlaps meet {minimum}; larger overlap is candidate {outcome + 1}"

    elif stratum == "arithmetic_capacity_budget":
        capacity = int(definition["capacity"])
        reserve = int(definition["reserve"])
        group = int(definition["group"])
        requests = int(changed)
        usable = capacity - reserve
        needed = math.ceil(requests / group)
        fits = needed <= usable
        outcome = 0 if fits else 1
        evidence = {
            "capacity": capacity,
            "reserve": reserve,
            "requests": requests,
            "requests_per_group": group,
            "usable_slots": usable,
            "needed_slots": needed,
            "fits": fits,
        }
        proof = f"ceil({requests}/{group})={needed}; {needed} {'<=' if fits else '>'} {usable}"

    elif stratum == "categorical_set_membership":
        tags = set(str(value) for value in definition["fixed"])
        tags.add(str(changed))
        required = set(str(value) for value in definition["required"])
        forbidden = set(str(value) for value in definition["forbidden"])
        all_required = required.issubset(tags)
        no_forbidden = tags.isdisjoint(forbidden)
        passes = all_required and no_forbidden
        outcome = 0 if passes else 1
        evidence = {
            "observed_tags": tuple(sorted(tags)),
            "required_tags": tuple(sorted(required)),
            "forbidden_tags": tuple(sorted(forbidden)),
            "all_required": all_required,
            "no_forbidden": no_forbidden,
        }
        proof = f"all_required={all_required}, no_forbidden={no_forbidden}"

    elif stratum == "ordered_priority_exception":
        exception_active = state_id == "state1"
        outcome = 1 if exception_active else 0
        evidence = {
            "first_trigger_present": True,
            "second_trigger_present": True,
            "named_exception": str(definition["exception"]),
            "exception_active": exception_active,
            "rule": "first applicable action unless named exception suppresses it",
        }
        proof = (f"both triggers present; {definition['exception']} is "
                 f"{'active, suppressing first' if exception_active else 'inactive'}")
        changed = "active" if exception_active else "inactive"

    else:
        _require(stratum == "referent_alias_resolution", "unhandled stratum")
        branch = str(definition["branch0"] if state_id == "state0" else
                     definition["branch1"])
        extra = definition["extra0"] if state_id == "state0" else definition["extra1"]
        graph: dict[str, str] = {
            str(definition["query"]): str(definition["relay"]),
            str(definition["relay"]): branch,
        }
        if extra is not None:
            graph[branch] = str(extra)
        cursor = str(definition["query"])
        path = [cursor]
        while cursor in graph:
            cursor = graph[cursor]
            path.append(cursor)
        outcome = 0 if state_id == "state0" else 1
        evidence = {"alias_graph": graph, "resolution_path": tuple(path),
                    "hop_count": len(path) - 1,
                    "terminal_outcome": outcome}
        _require(len(path) - 1 in (2, 3), "alias fixture is not a 2--3-hop alias chain")
        proof = " -> ".join(path) + f"; terminal maps to outcome {outcome}"
        changed = branch

    expected = 0 if state_id == "state0" else 1
    _require(outcome == expected, "magnitude state does not prove its declared outcome")
    return RuleEvaluation(
        stratum_id=stratum,
        variant=variant,
        state_id=state_id,
        outcome_index=outcome,
        changed_value=changed,
        evidence=MappingProxyType(dict(evidence)),
        proof=proof,
    )


def evaluate_pair(candidate: CandidateTuple) -> tuple[RuleEvaluation, RuleEvaluation]:
    correct = evaluate_rule(candidate, "C")
    wrong = evaluate_rule(candidate, "W")
    _require({correct.outcome_index, wrong.outcome_index} == {0, 1},
             "C and W do not prove opposite exact outcomes")
    return correct, wrong


def _evaluation_record(evaluation: RuleEvaluation) -> dict[str, Any]:
    return {
        "stratum_id": evaluation.stratum_id,
        "variant": evaluation.variant,
        "state_id": evaluation.state_id,
        "outcome_index": evaluation.outcome_index,
        "changed_value": evaluation.changed_value,
        "evidence": dict(evaluation.evidence),
        "proof": evaluation.proof,
    }


def compact_pool_audit(stratum_id: str) -> dict[str, Any]:
    """Audit IDs, tuples, and logic without expanding a single history."""

    tuple_bytes_seen: set[bytes] = set()
    ids_seen: set[str] = set()
    ordered_ids: list[str] = []
    correct_outcomes = {0: 0, 1: 0}
    wrong_outcomes = {0: 0, 1: 0}
    for candidate in enumerate_candidate_tuples(stratum_id):
        tuple_bytes = canonical_json_bytes(candidate.canonical_object())
        _require(tuple_bytes not in tuple_bytes_seen, "canonical tuple collision")
        tuple_bytes_seen.add(tuple_bytes)
        identifier = stable_candidate_id(candidate)
        _require(identifier not in ids_seen, "stable candidate-ID collision")
        ids_seen.add(identifier)
        ordered_ids.append(identifier)
        correct, wrong = evaluate_pair(candidate)
        correct_outcomes[correct.outcome_index] += 1
        wrong_outcomes[wrong.outcome_index] += 1

    _require(len(ordered_ids) == POOL_SIZE_PER_STRATUM,
             "stratum does not contain exactly 4096 tuples")
    _require(correct_outcomes == {0: 2048, 1: 2048},
             "correct-history orientation is not exactly balanced")
    _require(wrong_outcomes == {0: 2048, 1: 2048},
             "wrong-history orientation is not exactly balanced")
    return {
        "schema": RECIPE_SCHEMA,
        "design_id": DESIGN_ID,
        "stratum_id": stratum_id,
        "compact_tuple_count": len(tuple_bytes_seen),
        "stable_candidate_id_count": len(ids_seen),
        "ordered_candidate_ids_sha256": sha256(
            canonical_json_bytes(ordered_ids)).hexdigest(),
        "correct_outcome_counts": correct_outcomes,
        "wrong_outcome_counts": wrong_outcomes,
        "conversation_text_materialized": False,
        "literal_history_collision_audit":
            "DEFERRED_UNTIL_RANKED_MATERIALIZATION_AFTER_PERMUTATION_COMMIT",
    }


def development_sentinel(stratum_id: str, *, explicit: bool) -> CandidateTuple:
    _require(stratum_id in STRATA, "unknown stratum")
    return CandidateTuple(
        stratum_id=stratum_id,
        resolution_mode_id=("development_explicit" if explicit else
                            "development_unstated"),
        domain_skin_id=_DEVELOPMENT_DOMAIN.id,
        label_pair_id=_DEVELOPMENT_LABEL.id,
        logic_pack_id=_DEVELOPMENT_LOGIC_PACK_ID,
        nonfocal_tail_pack_id=_DEVELOPMENT_NONFOCAL.id,
    )


def _resolve_components(candidate: CandidateTuple) -> tuple[
        str, _DomainSkin, _LabelPair, _NonfocalTailPack]:
    if is_pool_member(candidate):
        return (
            candidate.resolution_mode_id,
            _DOMAIN_BY_ID[candidate.domain_skin_id],
            _LABEL_BY_ID[candidate.label_pair_id],
            _NONFOCAL_BY_ID[candidate.nonfocal_tail_pack_id],
        )
    expected_development = development_sentinel(
        candidate.stratum_id,
        explicit=candidate.resolution_mode_id == "development_explicit")
    _require(candidate == expected_development,
             "non-pool tuple is not the exact development sentinel")
    return (candidate.resolution_mode_id, _DEVELOPMENT_DOMAIN,
            _DEVELOPMENT_LABEL, _DEVELOPMENT_NONFOCAL)


def _format_interval(interval: Sequence[int]) -> str:
    return f"{_clock(int(interval[0]))}–{_clock(int(interval[1]))}"


def _family_narrative(candidate: CandidateTuple, evaluation: RuleEvaluation,
                      labels: _LabelPair, skin: _DomainSkin) -> dict[str, str]:
    definition, _ = _logic_definition(candidate)
    zero, one = labels.zero, labels.one
    stratum = candidate.stratum_id

    if stratum == "threshold_eligibility":
        rule = (f"Use an inclusive threshold of {definition['threshold']}. A score at or above "
                f"that line receives {zero}; a score below it receives {one}.")
        changed = (f"The verified score for the {skin.subject} is "
                   f"{evaluation.evidence['score']}.")
        record = (f"score {evaluation.evidence['score']}; inclusive threshold "
                  f"{definition['threshold']}")
        order = "rule, background record, changed score, audit"
    elif stratum == "sequential_tiebreak":
        scores = evaluation.evidence["candidate_primary_scores"]
        times = evaluation.evidence["candidate_timestamps"]
        rule = (f"Compare two candidate records in sequence. Both must meet {definition['cutoff']}; "
                f"higher primary score wins, and equal primary scores are broken by the earlier "
                f"timestamp. The first record maps to {zero}; the second maps to {one}.")
        changed = (f"The second candidate's verified timestamp is {times[1]}; its primary score "
                   f"remains {scores[1]}.")
        record = (f"first score/time {scores[0]}/{times[0]}; second score/time "
                  f"{scores[1]}/{times[1]}")
        order = "candidate records, qualification, equal score, timestamp tie-break"
    elif stratum == "conjunction_all_conditions":
        rule = (f"Award {zero} only when certification is active, the queue is no more than "
                f"{definition['queue_max']}, and inspection is at least {definition['minimum']}. "
                f"If any one condition fails, award {one}.")
        changed = (f"The newly verified inspection reading for the {skin.subject} is "
                   f"{evaluation.evidence['inspection']}.")
        record = (f"certification active; queue {definition['queue']}; inspection "
                  f"{evaluation.evidence['inspection']}")
        order = "three predicate records, changed third predicate, all-conditions rule"
    elif stratum == "interval_schedule_overlap":
        second = evaluation.evidence["candidate_minutes"][1]
        rule = (f"Each candidate window must overlap the reference window for at least "
                f"{definition['minimum']} minutes. If both qualify, choose the larger overlap. "
                f"The first candidate maps to {zero}; the second maps to {one}.")
        changed = (f"The verified second candidate window is {_format_interval(second)}.")
        record = (f"reference {_format_interval(definition['reference'])}; first "
                  f"{_format_interval(definition['first'])}; second {_format_interval(second)}")
        order = "reference window, candidate windows, minimum duration, larger overlap"
    elif stratum == "arithmetic_capacity_budget":
        rule = (f"First subtract the reserve of {definition['reserve']} from capacity "
                f"{definition['capacity']}. Then compute ceiling(requests/{definition['group']}). "
                f"If needed slots do not exceed usable slots, record {zero}; otherwise record {one}.")
        changed = (f"The reconciled request count for this cycle is "
                   f"{evaluation.evidence['requests']}.")
        record = (f"capacity {definition['capacity']}; reserve {definition['reserve']}; "
                  f"group size {definition['group']}; requests {evaluation.evidence['requests']}")
        order = "request evidence, capacity ledger, ceiling calculation, reserve inequality"
    elif stratum == "categorical_set_membership":
        rule = (f"Record {zero} only if every required tag is present and no forbidden tag is "
                f"present; otherwise record {one}. Required tags are "
                f"{', '.join(definition['required'])}; forbidden tags are "
                f"{', '.join(definition['forbidden'])}.")
        changed = (f"The final observed tag on the {skin.subject} is "
                   f"{evaluation.changed_value}.")
        record = "observed tags " + ", ".join(evaluation.evidence["observed_tags"])
        order = "observed categories, required set, forbidden set, membership rule"
    elif stratum == "ordered_priority_exception":
        rule = (f"Apply priorities in order. The {definition['first_trigger']} action comes first "
                f"and maps to {zero}; the {definition['second_trigger']} action comes next and maps "
                f"to {one}. The named {definition['exception']} exception suppresses only the first "
                f"action, after which evaluation continues to the second.")
        changed = (f"The {definition['exception']} exception is verified "
                   f"{evaluation.changed_value} for this record.")
        record = (f"both priority triggers present; {definition['exception']} "
                  f"{evaluation.changed_value}")
        order = "named exception, ordered triggers, suppression, next applicable action"
    else:
        query = str(definition["query"])
        relay = str(definition["relay"])
        if definition["extra0"] is None:
            terminal_rules = (
                f"{definition['branch0']} resolves to outcome:0; "
                f"{definition['branch1']} resolves to outcome:1")
        else:
            terminal_rules = (
                f"{definition['branch0']} refers to {definition['extra0']}, which resolves to "
                f"outcome:0; {definition['branch1']} refers to {definition['extra1']}, which "
                "resolves to outcome:1")
        rule = (f"Resolve the requested alias by following each recorded reference until an output "
                f"is reached. outcome:0 means {zero}; outcome:1 means {one}. The fixed register "
                f"says {query} refers to {relay}; {terminal_rules}.")
        changed = (f"The revised intermediate link says {relay} refers to "
                   f"{evaluation.changed_value}.")
        record = (f"query alias {query}; intermediate register {relay}; changed link "
                  f"to {evaluation.changed_value}")
        order = "query referent, outer alias, changed intermediate link, terminal resolution"

    return {"rule": rule, "changed": changed, "record": record,
            "discourse_order": order}


def _nonfocal_sentence(pack: _NonfocalTailPack) -> str:
    return (f"The standing {pack.fact_name} is {pack.target}; this direct fact is unrelated "
            "to the earlier classification record.")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise RecipeError(f"unsupported semantic inventory value: {type(value).__name__}")


def _semantic_atoms(value: Any) -> tuple[set[str], set[int | float]]:
    strings: set[str] = set()
    numbers: set[int | float] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                visit(nested)
        elif isinstance(item, bool):
            strings.add("active" if item else "inactive")
        elif isinstance(item, (int, float)):
            numbers.add(item)
        elif isinstance(item, str):
            strings.add(item)
        elif item is not None:
            raise RecipeError(
                f"unsupported carrier-forbidden atom: {type(item).__name__}")

    visit(value)
    return strings, numbers


def _changed_value_surface(candidate: CandidateTuple,
                           evaluation: RuleEvaluation) -> str:
    if candidate.stratum_id == "interval_schedule_overlap":
        return _format_interval(evaluation.changed_value)
    return str(evaluation.changed_value)


def _carrier_forbidden_inventory(
    candidate: CandidateTuple,
    skin: _DomainSkin,
    labels: _LabelPair,
    nonfocal: _NonfocalTailPack,
    correct: RuleEvaluation,
    wrong: RuleEvaluation,
) -> dict[str, Any]:
    """Enumerate semantic carrier exclusions before any carrier is generated."""

    definition, _ = _logic_definition(candidate)
    definition_plain = _plain_json(definition)
    evidence_plain = {
        "C": _plain_json(correct.evidence),
        "W": _plain_json(wrong.evidence),
    }
    semantic_strings, semantic_numbers = _semantic_atoms(
        {"definition": definition_plain, "evidence": evidence_plain})

    time_forms = {
        value for value in semantic_strings
        if re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", value)
    }
    rendered_interval_forms: set[str] = set()
    if candidate.stratum_id == "interval_schedule_overlap":
        for key in ("reference", "first", "state0", "state1"):
            interval = definition[key]
            rendered_interval_forms.add(_format_interval(interval))
            time_forms.update(_clock(int(endpoint)) for endpoint in interval)

    contextual_names = {
        "organization": skin.organization,
        "ledger": skin.ledger,
        "coordinator": skin.coordinator,
        "subject": skin.subject,
        "site": skin.site,
        "distractor_a": skin.distractor_a,
        "distractor_b": skin.distractor_b,
    }
    changed_values = {
        "C": {
            "raw": _plain_json(correct.changed_value),
            "rendered": _changed_value_surface(candidate, correct),
        },
        "W": {
            "raw": _plain_json(wrong.changed_value),
            "rendered": _changed_value_surface(candidate, wrong),
        },
    }
    exact_forms = {
        labels.zero,
        labels.one,
        nonfocal.target,
        nonfocal.countertarget,
        *contextual_names.values(),
        *semantic_strings,
        *time_forms,
        *rendered_interval_forms,
        *(str(value) for value in semantic_numbers),
        changed_values["C"]["rendered"],
        changed_values["W"]["rendered"],
    }
    exact_forms.discard("")

    core = {
        "schema": "powered-v13-carrier-forbidden-inventory-v1",
        "focal_targets": {"label_0": labels.zero, "label_1": labels.one,
                           "C": labels.zero if correct.outcome_index == 0 else labels.one,
                           "W": labels.zero if wrong.outcome_index == 0 else labels.one},
        "changed_values": changed_values,
        "nonfocal": {
            "fact_name": nonfocal.fact_name,
            "target": nonfocal.target,
            "countertarget": nonfocal.countertarget,
        },
        "contextual_names": contextual_names,
        "rule_semantics": {
            "definition": definition_plain,
            "evaluated_evidence": evidence_plain,
            "numeric_values": sorted(semantic_numbers, key=lambda value: (float(value), str(value))),
            "time_forms": sorted(time_forms),
            "rendered_interval_forms": sorted(rendered_interval_forms),
            "category_and_alias_values": sorted(semantic_strings - time_forms),
        },
        "exact_forbidden_forms": sorted(exact_forms, key=lambda value: (value.casefold(), value)),
        "unicode_casefolded_exact_forms": sorted(
            {value.casefold() for value in exact_forms}),
        "surface_expansion": {
            "exact_semantic_values_and_names_enumerated": True,
            "punctuation_variants_status": "PENDING_PRE_CARRIER_GATE_EXPANSION",
            "spelled_number_variants_status": "PENDING_PRE_CARRIER_GATE_EXPANSION",
            "unicode_normalization_variants_status": "PENDING_PRE_CARRIER_GATE_EXPANSION",
            "production_tokenizer_subsequence_status": "PENDING_PRE_CARRIER_GATE_EXPANSION",
            "target_aware_semantic_review_required": True,
        },
    }
    return {
        **core,
        "semantic_inventory_sha256": sha256(canonical_json_bytes(core)).hexdigest(),
    }


def _messages_for_variant(candidate: CandidateTuple, variant: str,
                          mode: str, skin: _DomainSkin, labels: _LabelPair,
                          nonfocal: _NonfocalTailPack) -> tuple[list[dict[str, str]], dict[str, Any]]:
    evaluation = evaluate_rule(candidate, variant)
    narrative = _family_narrative(candidate, evaluation, labels, skin)
    target = labels.zero if evaluation.outcome_index == 0 else labels.one
    nonfocal_system = (_nonfocal_sentence(nonfocal)
                       if nonfocal.location == "system" else "")
    nonfocal_user = (_nonfocal_sentence(nonfocal)
                     if nonfocal.location == "tail_user" else "")
    nonfocal_assistant = (_nonfocal_sentence(nonfocal)
                          if nonfocal.location == "tail_assistant" else "")

    system = (
        f"You maintain a careful internal record for the {skin.organization}. Preserve stated "
        "rules, distinguish verified evidence from commentary, and do not invent missing facts. "
        "A later handoff may summarize shared background, so the record must remain coherent even "
        "when an earlier decision is no longer quoted. Treat classification labels as exact ledger "
        "entries, not as suggestions or prose. Keep unrelated operational threads separate from "
        f"the focal decision. {nonfocal_system}"
    ).strip()
    opening = (
        f"We are reconciling the {skin.ledger} at the {skin.site}. The focal entry concerns one "
        f"{skin.subject}, and the {skin.coordinator} has asked us to apply a deterministic rule. "
        f"{narrative['rule']} Read the mapping literally and retain the exact capitalization of the "
        "two possible ledger words. The first unrelated thread concerns "
        f"{skin.distractor_a}: its count was copied from yesterday and will be checked separately. "
        "That inventory neither changes a predicate nor selects a classification label."
    )
    acknowledge = (
        f"I have separated the classification rule from the {skin.distractor_a} thread. The latter "
        "still needs a physical recount before anyone closes its worksheet, but it supplies no input "
        "to the focal rule. A second unrelated thread concerns "
        f"{skin.distractor_b}. Its appointment was moved to the following week, and the move affects "
        "staff availability only. I will preserve both threads as context without using either one "
        "as evidence for the focal answer. The operations margin also records who will perform each "
        "follow-up and which worksheet will receive the confirmation. Those notes make the conversation "
        "realistically interleaved, yet they do not add a criterion, change a comparison, or create an "
        "exception."
    )
    changed_input = (
        f"Here is the one revised focal input supplied by the {skin.coordinator}. "
        f"{narrative['changed']} All other focal records and both unrelated threads remain as already "
        "stated. Use the revised input for the ledger decision, but do not treat the act of revision "
        "itself as an extra priority, tag, score, interval, capacity item, or alias edge."
    )
    hold_record = (
        f"The revised input is now attached to the focal worksheet. I am leaving the "
        f"{skin.distractor_a} recount open and keeping the {skin.distractor_b} appointment in its own "
        "operations section. Neither thread changes the decision rule, the fixed records, or the "
        "label mapping. The audit trail distinguishes the submitted evidence from the calculation "
        "that follows, so a reviewer can reconstruct the result without relying on tone or order. "
        "The worksheet has separate boxes for source evidence, the deterministic transformation, and "
        "the final ledger entry. Only the source-evidence box receives the revision; the rule box stays "
        "fixed. This separation prevents an incidental scheduling remark from masquerading as focal data."
    )
    audit_request = (
        "Before this portion of the discussion is compacted, make a final decision record. Confirm "
        "that the focal rule was applied only to its declared evidence. Also note that the first "
        f"unrelated chain, {skin.distractor_a}, remains pending a recount, while the second, "
        f"{skin.distractor_b}, has only a later appointment. Close both distractor chains explicitly "
        "as irrelevant. Do not turn either operational detail into a hidden exception."
    )
    if mode in ("explicit_resolution", "development_explicit"):
        final_evicted = (
            f"Final evicted decision record: {narrative['record']}. The pure rule therefore yields "
            f"the exact focal ledger word {target}. The {skin.distractor_a} recount and the "
            f"{skin.distractor_b} appointment are closed here as unrelated operational chains; "
            "neither was consulted in deriving the label. This entry names the outcome explicitly "
            "and preserves the evidence-to-rule path for review before the boundary."
        )
        subtype = "explicit_final_evicted_outcome"
    else:
        _require(mode in ("unstated_resolution", "development_unstated"),
                 "invalid resolution mode")
        final_evicted = (
            f"Final evicted predicate record: {narrative['record']}. I preserved the evidence and "
            "the rule application but, under this subtype, deliberately withheld the resulting focal "
            f"ledger word. The {skin.distractor_a} recount and the {skin.distractor_b} appointment are "
            "closed here as unrelated operational chains; neither was consulted in the derivation. "
            "A later answer must resolve the rule from this record rather than quote a written outcome."
        )
        subtype = "unstated_final_evicted_outcome"

    tail_user = (
        "We are now past the decision-record boundary. Continue with a neutral handoff check that "
        "does not restate, imply, revise, or challenge the earlier focal rule, its inputs, or its "
        "result. The remaining exchange concerns only direct administrative facts that stay visible "
        f"after compaction. {nonfocal_user} Please acknowledge the direct fact and the separation of "
        "topics without summarizing any earlier option, number, condition, branch, referent, or label. "
        "This request intentionally leaves the prior decision opaque. It asks for no calculation and "
        "offers no substitute evidence; it only establishes what remains safe to discuss in the visible "
        "administrative tail."
    ).strip()
    tail_assistant = (
        "Acknowledged. This retained exchange is administratively separate from the earlier decision "
        "and supplies no evidence about that decision. I will preserve the directly stated visible "
        f"fact exactly and will not reconstruct or hint at the focal classification. {nonfocal_assistant} "
        "The retained tail therefore remains coherent on its own while leaving the earlier rule and "
        "outcome untouched. Nothing in this acknowledgement ranks candidates, evaluates eligibility, "
        "combines predicates, compares windows, performs arithmetic, checks categories, invokes an "
        "exception, or resolves a reference. It is a direct administrative acknowledgement only."
    ).strip()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": opening},
        {"role": "assistant", "content": acknowledge},
        {"role": "user", "content": changed_input},
        {"role": "assistant", "content": hold_record},
        {"role": "user", "content": audit_request},
        {"role": "assistant", "content": final_evicted},
        {"role": "user", "content": tail_user},
        {"role": "assistant", "content": tail_assistant},
    ]
    return messages, {
        "evaluation": evaluation,
        "target": target,
        "subtype": subtype,
        "discourse_order": narrative["discourse_order"],
    }


def _authorization_record(candidate: CandidateTuple,
                          authorization: RankedMaterializationAuthorization | None,
                          *, development: bool) -> dict[str, Any]:
    identifier = stable_candidate_id(candidate)
    if development:
        _require(not is_pool_member(candidate),
                 "development expansion cannot use an in-pool tuple")
        _require(authorization is None,
                 "development sentinel must not carry a production authorization")
        return {
            "kind": "OUT_OF_POOL_DEVELOPMENT_SENTINEL",
            "permutation_rank": None,
            "seed_manifest_sha256": None,
            "seed_git_commit": None,
            "literal_permutation_sha256": None,
        }
    _require(is_pool_member(candidate), "ranked expansion requires an in-pool tuple")
    _require(authorization is not None, "in-pool text expansion requires authorization")
    _require(authorization.status == "PERMUTATION_COMMITTED_RANKED_CANDIDATE",
             "materialization authorization status differs")
    _require(authorization.candidate_id == identifier,
             "authorization candidate ID differs")
    _require(1 <= authorization.permutation_rank <= 10,
             "only ranks one through ten may be materialized")
    for value in (authorization.seed_manifest_sha256,
                  authorization.literal_permutation_sha256):
        _require(re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                 "authorization hash is not lowercase SHA-256")
    _require(re.fullmatch(r"[0-9a-f]{40}", authorization.seed_git_commit)
             is not None, "authorization seed git commit is invalid")
    return {"kind": authorization.status, **asdict(authorization)}


def _materialize(candidate: CandidateTuple,
                 authorization: RankedMaterializationAuthorization | None,
                 *, development: bool) -> dict[str, Any]:
    mode, skin, labels, nonfocal = _resolve_components(candidate)
    authorization_record = _authorization_record(
        candidate, authorization, development=development)
    correct_messages, correct = _messages_for_variant(
        candidate, "C", mode, skin, labels, nonfocal)
    wrong_messages, wrong = _messages_for_variant(
        candidate, "W", mode, skin, labels, nonfocal)
    _require(correct["evaluation"].outcome_index != wrong["evaluation"].outcome_index,
             "materialized variants do not have opposite outcomes")
    candidate_id = stable_candidate_id(candidate)
    nonfocal_index = {"system": 0, "tail_user": 7, "tail_assistant": 8}[
        nonfocal.location]
    fixture = {
        "schema": FIXTURE_SCHEMA,
        "design_id": DESIGN_ID,
        "template_version": TEMPLATE_VERSION,
        "generator_version": GENERATOR_VERSION,
        "stable_candidate_id": candidate_id,
        "pool_member": is_pool_member(candidate),
        "canonical_tuple": candidate.canonical_object(),
        "stratum_id": candidate.stratum_id,
        "stratum_title": STRATUM_TITLES[candidate.stratum_id],
        "status": "DRAFT_UNREVIEWED_NOT_EXECUTABLE",
        "execution_ready": False,
        "materialization_authorization": authorization_record,
        "resolution_mode": mode,
        "resolution_subtype": correct["subtype"],
        "domain_skin": asdict(skin),
        "middle_end_msg": 7,
        "variants": {
            "C": {"messages": correct_messages},
            "W": {"messages": wrong_messages},
        },
        "changed_input": {
            "count": 1,
            "message_index": 3,
            "role": "user",
            "C_value": _plain_json(correct["evaluation"].changed_value),
            "W_value": _plain_json(wrong["evaluation"].changed_value),
            "before_compaction_boundary": True,
        },
        "changed_message_allowlist": [3, 6],
        "focal": {
            "category": "derived_exact_decision",
            "probe": "What exact focal ledger word follows from the recorded rule?",
            "C_target": correct["target"],
            "W_target": wrong["target"],
            "target": correct["target"],
            "countertarget": wrong["target"],
            "target_strings_nonprefix": True,
            "production_tokenizer_status": "PENDING_1_TO_4_TOKEN_CHECK",
            "evidence_message_indices": [1, 3, 6],
            "rule_proof_C": _evaluation_record(correct["evaluation"]),
            "rule_proof_W": _evaluation_record(wrong["evaluation"]),
            "discourse_order": correct["discourse_order"],
        },
        "nonfocal_control": {
            "category": "visible_direct_fact",
            "probe": nonfocal.probe,
            "target": nonfocal.target,
            "countertarget": nonfocal.countertarget,
            "target_strings_nonprefix": True,
            "production_tokenizer_status": "PENDING_1_TO_4_TOKEN_CHECK",
            "location": nonfocal.location,
            "establishing_message_indices": [nonfocal_index],
            "visible_after_compaction": True,
            "byte_identical_between_variants": True,
            "why_independent": "The direct administrative fact is not an input to any focal rule.",
        },
        "carrier_forbidden_inventory": _carrier_forbidden_inventory(
            candidate,
            skin,
            labels,
            nonfocal,
            correct["evaluation"],
            wrong["evaluation"],
        ),
        "distractor_chains": [
            {
                "chain_id": "distractor_a",
                "topic": skin.distractor_a,
                "message_indices": [1, 2, 5, 6],
                "why_irrelevant": "Inventory reconciliation is absent from the focal evaluator.",
            },
            {
                "chain_id": "distractor_b",
                "topic": skin.distractor_b,
                "message_indices": [2, 4, 5, 6],
                "why_irrelevant": "Appointment timing is absent from the focal evaluator.",
            },
        ],
        "retained_tail": {
            "start_message_index": 7,
            "starts_with_user": True,
            "ends_with_assistant": True,
            "byte_identical_between_variants": True,
            "focal_neutrality_status": "PENDING_INDEPENDENT_CONTENT_REVIEW",
        },
        "length_design": {
            "scope": "noncarrier messages through final retained-tail assistant",
            "production_tokenizer_target": [940, 1180],
            "production_tokenizer_status": "PENDING_NOT_MEASURED_OR_CLAIMED",
            "whitespace_word_count_C": sum(
                len(item["content"].split()) for item in correct_messages),
            "whitespace_word_count_W": sum(
                len(item["content"].split()) for item in wrong_messages),
        },
        "authoring_provenance": {
            "compiler": "src/powered_v13_recipe.py",
            "generator_version": GENERATOR_VERSION,
            "method": "deterministic parameterized human-readable templates",
            "subject_model_used": False,
            "randomness_used": False,
            "old_case_text_used": False,
            "content_review_status": "PENDING",
            "tokenizer_geometry_status": "PENDING",
        },
        "warning": (
            "Draft compiler output only. It has not passed the production tokenizer, "
            "blind content review, Phase A, or treatment release."
        ),
    }
    fixture["literal_history_pair_sha256"] = sha256(canonical_json_bytes({
        "C": correct_messages,
        "W": wrong_messages,
    })).hexdigest()
    validate_fixture_schema(fixture)
    return fixture


def materialize_development_sentinel(stratum_id: str, *, explicit: bool) -> dict[str, Any]:
    """Compile an out-of-pool fixture for unpaid template/schema tests."""

    return _materialize(
        development_sentinel(stratum_id, explicit=explicit),
        None,
        development=True,
    )


def materialize_ranked_candidate(
    candidate: CandidateTuple,
    authorization: RankedMaterializationAuthorization,
) -> dict[str, Any]:
    """Compile one already-ranked tuple after an external permutation commit.

    Calling this function is intentionally outside the foundation test suite;
    doing so before the real seed and literal permutation are committed would
    violate the sampling protocol.
    """

    return _materialize(candidate, authorization, development=False)


def _messages(fixture: Mapping[str, Any], variant: str) -> list[Mapping[str, Any]]:
    raw = fixture.get("variants", {}).get(variant, {}).get("messages")
    _require(isinstance(raw, list), f"{variant} messages missing")
    return raw


def _nonprefix(left: str, right: str) -> bool:
    a, b = left.casefold(), right.casefold()
    return a != b and not a.startswith(b) and not b.startswith(a)


def validate_fixture_schema(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Tokenizer-free, fail-closed validation of one materialized draft."""

    _require(fixture.get("schema") == FIXTURE_SCHEMA, "fixture schema differs")
    _require(fixture.get("design_id") == DESIGN_ID, "fixture design ID differs")
    _require(fixture.get("template_version") == TEMPLATE_VERSION,
             "template version differs")
    _require(fixture.get("generator_version") == GENERATOR_VERSION,
             "generator version differs")
    _require(fixture.get("status") == "DRAFT_UNREVIEWED_NOT_EXECUTABLE",
             "fixture status incorrectly authorizes execution")
    _require(fixture.get("execution_ready") is False,
             "fixture incorrectly authorizes execution")
    _require(fixture.get("stratum_id") in STRATA, "unknown fixture stratum")
    _require(fixture.get("authoring_provenance", {}).get("subject_model_used") is False,
             "subject model provenance differs")
    _require(fixture.get("authoring_provenance", {}).get("randomness_used") is False,
             "recipe compiler unexpectedly used randomness")

    canonical_tuple = fixture.get("canonical_tuple")
    _require(isinstance(canonical_tuple, Mapping), "canonical tuple missing")
    _require(set(canonical_tuple) == {
        "resolution_mode_id", "domain_skin_id", "label_pair_id",
        "logic_pack_id", "nonfocal_tail_pack_id",
    }, "canonical tuple fields differ")
    candidate = CandidateTuple(
        stratum_id=str(fixture["stratum_id"]),
        resolution_mode_id=str(canonical_tuple["resolution_mode_id"]),
        domain_skin_id=str(canonical_tuple["domain_skin_id"]),
        label_pair_id=str(canonical_tuple["label_pair_id"]),
        logic_pack_id=str(canonical_tuple["logic_pack_id"]),
        nonfocal_tail_pack_id=str(canonical_tuple["nonfocal_tail_pack_id"]),
    )
    _require(fixture.get("stable_candidate_id") == stable_candidate_id(candidate),
             "stable candidate ID differs")
    resolved_mode, resolved_skin, resolved_labels, resolved_nonfocal = \
        _resolve_components(candidate)
    _require(fixture.get("resolution_mode") == resolved_mode,
             "resolution mode differs from canonical tuple")
    _require(fixture.get("domain_skin") == asdict(resolved_skin),
             "domain skin differs from canonical tuple")
    expected_c, expected_w = evaluate_pair(candidate)
    expected_forbidden = _carrier_forbidden_inventory(
        candidate, resolved_skin, resolved_labels, resolved_nonfocal,
        expected_c, expected_w)
    _require(fixture.get("carrier_forbidden_inventory") == expected_forbidden,
             "carrier-forbidden inventory differs")

    correct = _messages(fixture, "C")
    wrong = _messages(fixture, "W")
    _require(len(correct) == len(wrong) == 9, "message counts differ")
    expected_roles = ["system"] + ["user" if index % 2 else "assistant"
                                   for index in range(1, 9)]
    _require([item.get("role") for item in correct] == expected_roles,
             "C role sequence differs")
    _require([item.get("role") for item in wrong] == expected_roles,
             "W role sequence differs")
    for variant, messages in (("C", correct), ("W", wrong)):
        _require(all(isinstance(item.get("content"), str) and
                     item["content"].strip() for item in messages),
                 f"{variant} has an empty message")
        _require(all("<|im_" not in item["content"] for item in messages),
                 f"{variant} contains a special-token literal")

    middle = fixture.get("middle_end_msg")
    _require(middle == 7, "retained-tail boundary differs")
    _require(correct[0] == wrong[0], "system messages differ")
    _require(correct[middle:] == wrong[middle:], "retained tails differ")
    _require(correct[middle]["role"] == "user" and
             correct[-1]["role"] == "assistant",
             "retained tail roles differ")
    changed_indices = [index for index, (left, right) in
                       enumerate(zip(correct, wrong)) if left != right]
    _require(changed_indices == [3, 6],
             f"unexpected changed messages: {changed_indices}")
    changed_user_indices = [index for index in changed_indices
                            if correct[index]["role"] == "user"]
    _require(changed_user_indices == [3],
             "fixture does not have exactly one changed user input")
    changed = fixture.get("changed_input", {})
    _require(changed.get("count") == 1 and changed.get("message_index") == 3 and
             changed.get("role") == "user" and
             changed.get("before_compaction_boundary") is True,
             "changed-input metadata differs")
    _require(fixture.get("changed_message_allowlist") == [3, 6],
             "changed-message allowlist differs")

    focal = fixture.get("focal", {})
    target = str(focal.get("target", ""))
    countertarget = str(focal.get("countertarget", ""))
    _require(_nonprefix(target, countertarget), "focal targets are equal/prefixing")
    _require(target == focal.get("C_target") and
             countertarget == focal.get("W_target"),
             "focal target orientation differs")
    probe_folded = str(focal.get("probe", "")).casefold()
    _require(target.casefold() not in probe_folded and
             countertarget.casefold() not in probe_folded,
             "focal probe contains a choice")
    proof_c = focal.get("rule_proof_C", {})
    proof_w = focal.get("rule_proof_W", {})
    _require({proof_c.get("outcome_index"), proof_w.get("outcome_index")} == {0, 1},
             "saved rule proofs are not opposite")
    mode = str(fixture.get("resolution_mode", ""))
    if "explicit" in mode:
        _require(target in correct[6]["content"] and
                 countertarget in wrong[6]["content"],
                 "explicit final evicted outcome is absent")
    else:
        _require(target not in correct[6]["content"] and
                 countertarget not in wrong[6]["content"],
                 "unstated final evicted outcome leaks a label")

    nonfocal = fixture.get("nonfocal_control", {})
    nf_target = str(nonfocal.get("target", ""))
    nf_counter = str(nonfocal.get("countertarget", ""))
    _require(_nonprefix(nf_target, nf_counter),
             "nonfocal targets are equal/prefixing")
    nf_probe = str(nonfocal.get("probe", "")).casefold()
    _require(nf_target.casefold() not in nf_probe and
             nf_counter.casefold() not in nf_probe,
             "nonfocal probe contains a choice")
    locations = nonfocal.get("establishing_message_indices")
    _require(locations in ([0], [7], [8]), "nonfocal location is not retained")
    index = int(locations[0])
    _require(nf_target in correct[index]["content"] and
             correct[index] == wrong[index],
             "direct nonfocal fact is absent or not byte-identical")
    _require(nonfocal.get("visible_after_compaction") is True,
             "nonfocal fact is not declared visible")

    fresh_visible_text = "\n".join(
        [correct[0]["content"]] +
        [item["content"] for item in correct[middle:]]
    ).casefold()
    _require(target.casefold() not in fresh_visible_text and
             countertarget.casefold() not in fresh_visible_text,
             "system or retained tail leaks focal labels")
    chains = fixture.get("distractor_chains")
    _require(isinstance(chains, list) and len(chains) == 2,
             "fixture does not have exactly two distractor chains")
    _require({item.get("chain_id") for item in chains} ==
             {"distractor_a", "distractor_b"},
             "distractor chain IDs differ")
    for chain in chains:
        topic = str(chain.get("topic", ""))
        indices = chain.get("message_indices")
        _require(topic and isinstance(indices, list) and len(indices) >= 3,
                 "distractor chain is incomplete")
        _require(all(topic in correct[int(index)]["content"] for index in indices),
                 "distractor chain is not literally threaded")

    _require(fixture.get("length_design", {}).get("production_tokenizer_status") ==
             "PENDING_NOT_MEASURED_OR_CLAIMED",
             "fixture incorrectly claims a tokenizer length pass")
    expected_history_hash = sha256(canonical_json_bytes({
        "C": correct,
        "W": wrong,
    })).hexdigest()
    _require(fixture.get("literal_history_pair_sha256") == expected_history_hash,
             "literal history pair hash differs")
    return {
        "status": "TOKENIZER_FREE_DRAFT_SCHEMA_PASS",
        "stable_candidate_id": fixture.get("stable_candidate_id"),
        "changed_message_indices": changed_indices,
        "retained_tail_identical": True,
        "rule_outcomes_opposite": True,
        "distractor_chain_count": 2,
        "production_tokenizer_status": "PENDING",
        "content_review_status": "PENDING",
    }


def recipe_definition() -> dict[str, Any]:
    """Return the compact, JSON-serializable frame definition."""

    return {
        "schema": RECIPE_SCHEMA,
        "design_id": DESIGN_ID,
        "template_version": TEMPLATE_VERSION,
        "generator_version": GENERATOR_VERSION,
        "strata": list(STRATA),
        "resolution_modes": list(RESOLUTION_MODES),
        "domain_skins": [asdict(item) for item in DOMAIN_SKINS],
        "label_pairs": [asdict(item) for item in LABEL_PAIRS],
        "logic_pack_ids": list(LOGIC_PACK_IDS),
        "logic_magnitude_definitions": {
            stratum: _plain_json(definitions)
            for stratum, definitions in _MAGNITUDES.items()
        },
        "nonfocal_tail_packs": [asdict(item) for item in NONFOCAL_TAIL_PACKS],
        "factorization": {
            "resolution_modes": 2,
            "domain_skins": 8,
            "label_pairs": 8,
            "logic_evidence_packs": 8,
            "nonfocal_tail_packs": 4,
            "canonical_tuples_per_stratum": 4096,
            "strata": 8,
            "canonical_tuples_total": 32768,
        },
        "randomization_boundary": {
            "compact_pool_enumeration_allowed": True,
            "in_pool_conversation_materialization_before_seed_commit": False,
            "development_template_tests": "OUT_OF_POOL_SENTINELS_ONLY",
        },
        "production_tokenizer_geometry": "PENDING",
        "content_review": "PENDING",
        "execution_authorized": False,
    }


__all__ = [
    "CandidateTuple",
    "DESIGN_ID",
    "FIXTURE_SCHEMA",
    "GENERATOR_VERSION",
    "LOGIC_PACK_IDS",
    "POOL_SIZE_PER_STRATUM",
    "RECIPE_SCHEMA",
    "RESOLUTION_MODES",
    "RankedMaterializationAuthorization",
    "RecipeError",
    "STRATA",
    "TEMPLATE_VERSION",
    "canonical_json_bytes",
    "compact_pool_audit",
    "development_sentinel",
    "enumerate_candidate_tuples",
    "evaluate_pair",
    "evaluate_rule",
    "is_pool_member",
    "materialize_development_sentinel",
    "materialize_ranked_candidate",
    "recipe_definition",
    "stable_candidate_id",
    "validate_fixture_schema",
]
