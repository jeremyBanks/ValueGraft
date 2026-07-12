"""Unpaid, tokenizer-free tests for the v13 compact recipe foundation.

The production pool is never expanded here.  Conversation/schema tests use
the explicitly out-of-pool development sentinel only.
"""

from __future__ import annotations

import copy
from hashlib import sha256
import inspect
import json
from pathlib import Path

import pytest

import powered_v13_recipe as recipe
import powered_v13_schema as schema


EXPECTED_POOL_DIGESTS = {
    "threshold_eligibility":
        "be87659352ce3cf7ed8dacb41db6b63d66ed0405211ca952854a5694ba7e2187",
    "sequential_tiebreak":
        "ad82ad6308f3694f94df8ef0f474411cb3fa3ef6383433d0a436849b2c5b931c",
    "conjunction_all_conditions":
        "7a1a1e70a8f2c4a92f4730dc19b079917e920512ebe8d1520a127cedd4ba6686",
    "interval_schedule_overlap":
        "886c0e1d80626a5dabd5d5afe93dcd5a5a5164902b1d093ab2d15778422f77bf",
    "arithmetic_capacity_budget":
        "490b531211aa27889570579eee076976c5bf33211bf45d6a74128f08ae3436f1",
    "categorical_set_membership":
        "f166ea70b0d30358fe14898307f51f68758df0f9d7658b9da82fc61cf4badd98",
    "ordered_priority_exception":
        "1664c70f4674294556f7ff12019a6e44fbc2d13cad6f8165de43eb4b3a61bdec",
    "referent_alias_resolution":
        "bfc44d3609f2e8d54d0dd7324b1c134338f9bf0ba32df56a0b90575e60a9113c",
}


def _pool_candidate(stratum: str, logic_pack: str):
    return recipe.CandidateTuple(
        stratum_id=stratum,
        resolution_mode_id="explicit_resolution",
        domain_skin_id="canal_cooperative",
        label_pair_id="garnet_mallow",
        logic_pack_id=logic_pack,
        nonfocal_tail_pack_id="system_correspondence_word",
    )


def _dev_fixture(stratum: str = "threshold_eligibility", *, explicit=True):
    return recipe.materialize_development_sentinel(stratum, explicit=explicit)


def test_factorization_is_literal_two_by_eight_by_eight_by_eight_by_four():
    definition = recipe.recipe_definition()
    assert definition["factorization"] == {
        "resolution_modes": 2,
        "domain_skins": 8,
        "label_pairs": 8,
        "logic_evidence_packs": 8,
        "nonfocal_tail_packs": 4,
        "canonical_tuples_per_stratum": 4096,
        "strata": 8,
        "canonical_tuples_total": 32768,
    }
    assert len(recipe.STRATA) == 8
    assert len(recipe.RESOLUTION_MODES) == 2
    assert len(recipe.LOGIC_PACK_IDS) == 8
    assert definition["randomization_boundary"] == {
        "compact_pool_enumeration_allowed": True,
        "in_pool_conversation_materialization_before_seed_commit": False,
        "development_template_tests": "OUT_OF_POOL_SENTINELS_ONLY",
    }
    assert definition["execution_authorized"] is False


def test_recipe_uses_central_v13_design_and_fixture_schema_constants():
    assert recipe.DESIGN_ID == schema.DESIGN_ID
    assert recipe.FIXTURE_SCHEMA == schema.CASE_SCHEMA
    source = inspect.getsource(recipe)
    assert "from powered_v13_schema import CASE_SCHEMA as FIXTURE_SCHEMA" in source
    assert "from powered_v13_schema import DESIGN_ID" in source


def test_data_foundation_matches_compiler_and_contains_no_seed_or_permutation():
    path = (Path(__file__).resolve().parents[1] / "data" /
            "coherent_state_powered_v13" / "recipe-foundation-v1.json")
    persisted = json.loads(path.read_text())
    definition = recipe.recipe_definition()
    assert persisted["design_id"] == definition["design_id"]
    assert persisted["template_version"] == definition["template_version"]
    assert persisted["generator_version"] == definition["generator_version"]
    assert persisted["factorization"] == definition["factorization"]
    assert persisted["resolution_modes"] == definition["resolution_modes"]
    assert persisted["domain_skin_ids"] == [
        item["id"] for item in definition["domain_skins"]]
    assert persisted["domain_skins"] == definition["domain_skins"]
    assert persisted["label_pair_ids"] == [
        item["id"] for item in definition["label_pairs"]]
    assert persisted["label_pairs"] == definition["label_pairs"]
    assert persisted["logic_evidence_pack_ids"] == definition["logic_pack_ids"]
    assert persisted["nonfocal_tail_pack_ids"] == [
        item["id"] for item in definition["nonfocal_tail_packs"]]
    assert persisted["nonfocal_tail_packs"] == definition["nonfocal_tail_packs"]
    assert persisted["compact_pool_digests"] == EXPECTED_POOL_DIGESTS
    assert persisted["randomization_boundary"]["permutation_seed_present"] is False
    assert persisted["randomization_boundary"]["literal_permutation_present"] is False
    assert persisted["execution_authorized"] is False


def test_compact_named_components_do_not_reuse_v12_domains_or_exact_answers():
    repo = Path(__file__).resolve().parents[1]
    legacy_values: set[str] = set()
    for path in (repo / "data" / "coherent_canary_v12").rglob("*.json"):
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        for value in (raw.get("domain"),):
            if isinstance(value, str):
                legacy_values.add(value.casefold())
        for record_name in ("focal", "nonfocal_control"):
            record = raw.get(record_name)
            if not isinstance(record, dict):
                continue
            for key in ("correct_target", "counterfactual_target",
                        "target", "countertarget"):
                value = record.get(key)
                if isinstance(value, str):
                    legacy_values.add(value.casefold())

    definition = recipe.recipe_definition()
    new_named_values: set[str] = set()
    for domain in definition["domain_skins"]:
        new_named_values.update(str(value).casefold() for value in domain.values())
    for pair in definition["label_pairs"]:
        new_named_values.update(str(value).casefold() for value in pair.values())
    for pack in definition["nonfocal_tail_packs"]:
        for key in ("target", "countertarget"):
            new_named_values.add(str(pack[key]).casefold())
    assert new_named_values.isdisjoint(legacy_values)


@pytest.mark.parametrize("stratum", recipe.STRATA)
def test_compact_pool_is_collision_free_balanced_and_digest_bound(stratum):
    audit = recipe.compact_pool_audit(stratum)
    assert audit["compact_tuple_count"] == 4096
    assert audit["stable_candidate_id_count"] == 4096
    assert audit["correct_outcome_counts"] == {0: 2048, 1: 2048}
    assert audit["wrong_outcome_counts"] == {0: 2048, 1: 2048}
    assert audit["ordered_candidate_ids_sha256"] == EXPECTED_POOL_DIGESTS[stratum]
    assert audit["conversation_text_materialized"] is False
    assert audit["literal_history_collision_audit"].startswith("DEFERRED_")


def test_canonical_id_uses_exact_sorted_utf8_json_array_and_has_no_rank():
    candidate = _pool_candidate("threshold_eligibility", "magnitude_1_c0_w1")
    expected_payload = [
        recipe.DESIGN_ID,
        "threshold_eligibility",
        recipe.TEMPLATE_VERSION,
        {
            "domain_skin_id": "canal_cooperative",
            "label_pair_id": "garnet_mallow",
            "logic_pack_id": "magnitude_1_c0_w1",
            "nonfocal_tail_pack_id": "system_correspondence_word",
            "resolution_mode_id": "explicit_resolution",
        },
    ]
    exact = json.dumps(
        expected_payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    assert recipe.canonical_json_bytes(expected_payload) == exact
    assert recipe.stable_candidate_id(candidate) == sha256(exact).hexdigest()
    assert "rank" not in candidate.canonical_object()
    assert "seed" not in candidate.canonical_object()


def test_canonical_json_rejects_nonfinite_numbers():
    with pytest.raises(recipe.RecipeError, match="canonical-JSON"):
        recipe.canonical_json_bytes({"bad": float("nan")})


@pytest.mark.parametrize("stratum", recipe.STRATA)
@pytest.mark.parametrize("orientation", ("c0_w1", "c1_w0"))
def test_every_family_proves_opposite_outcomes_in_both_orientations(
        stratum, orientation):
    for magnitude in range(1, 5):
        candidate = _pool_candidate(
            stratum, f"magnitude_{magnitude}_{orientation}")
        correct, wrong = recipe.evaluate_pair(candidate)
        assert {correct.outcome_index, wrong.outcome_index} == {0, 1}
        assert correct.variant == "C"
        assert wrong.variant == "W"
        assert correct.changed_value != wrong.changed_value
        if orientation == "c0_w1":
            assert (correct.outcome_index, wrong.outcome_index) == (0, 1)
        else:
            assert (correct.outcome_index, wrong.outcome_index) == (1, 0)


def test_threshold_rule_is_inclusive_at_equality():
    candidate = _pool_candidate(
        "threshold_eligibility", "magnitude_1_c0_w1")
    correct, wrong = recipe.evaluate_pair(candidate)
    assert correct.evidence["score"] == correct.evidence["inclusive_threshold"] == 60
    assert correct.evidence["eligible"] is True
    assert wrong.evidence["eligible"] is False


def test_sequential_rule_has_two_passes_equal_primary_and_equal_width_times():
    candidate = _pool_candidate(
        "sequential_tiebreak", "magnitude_3_c0_w1")
    correct, wrong = recipe.evaluate_pair(candidate)
    for evaluation in (correct, wrong):
        scores = evaluation.evidence["candidate_primary_scores"]
        times = evaluation.evidence["candidate_timestamps"]
        assert scores[0] == scores[1]
        assert scores[0] >= evaluation.evidence["cutoff"]
        assert len(times[0]) == len(times[1]) == 5
        assert evaluation.evidence["tie_break"] == "earlier_timestamp"
    assert correct.evidence["candidate_timestamps"][1] == "13:50"
    assert wrong.evidence["candidate_timestamps"][1] == "12:40"


def test_conjunction_rule_requires_all_three_predicates():
    candidate = _pool_candidate(
        "conjunction_all_conditions", "magnitude_2_c0_w1")
    correct, wrong = recipe.evaluate_pair(candidate)
    assert correct.evidence["certification_active"] is True
    assert correct.evidence["queue"] <= correct.evidence["queue_max"]
    assert correct.evidence["inspection"] >= correct.evidence["inspection_minimum"]
    assert correct.evidence["all_conditions"] is True
    assert wrong.evidence["inspection"] < wrong.evidence["inspection_minimum"]
    assert wrong.evidence["all_conditions"] is False


def test_interval_rule_applies_minimum_then_larger_overlap():
    candidate = _pool_candidate(
        "interval_schedule_overlap", "magnitude_2_c0_w1")
    correct, wrong = recipe.evaluate_pair(candidate)
    for evaluation in (correct, wrong):
        minimum = evaluation.evidence["minimum_overlap_minutes"]
        overlaps = evaluation.evidence["overlap_minutes"]
        assert all(value >= minimum for value in overlaps)
        assert overlaps[0] != overlaps[1]
    assert correct.evidence["overlap_minutes"][0] > \
        correct.evidence["overlap_minutes"][1]
    assert wrong.evidence["overlap_minutes"][1] > \
        wrong.evidence["overlap_minutes"][0]


def test_capacity_rule_uses_ceiling_and_reserve_inequality():
    candidate = _pool_candidate(
        "arithmetic_capacity_budget", "magnitude_1_c0_w1")
    correct, wrong = recipe.evaluate_pair(candidate)
    assert correct.evidence["usable_slots"] == 10
    assert correct.evidence["needed_slots"] == 10
    assert correct.evidence["fits"] is True
    assert wrong.evidence["needed_slots"] == 11
    assert wrong.evidence["fits"] is False


def test_set_rule_requires_every_required_and_no_forbidden_tag():
    candidate = _pool_candidate(
        "categorical_set_membership", "magnitude_4_c0_w1")
    correct, wrong = recipe.evaluate_pair(candidate)
    assert correct.evidence["all_required"] is True
    assert correct.evidence["no_forbidden"] is True
    assert wrong.evidence["all_required"] is False
    assert wrong.evidence["no_forbidden"] is False


def test_priority_rule_suppresses_only_first_action_under_named_exception():
    candidate = _pool_candidate(
        "ordered_priority_exception", "magnitude_3_c0_w1")
    correct, wrong = recipe.evaluate_pair(candidate)
    assert correct.evidence["first_trigger_present"] is True
    assert correct.evidence["second_trigger_present"] is True
    assert correct.evidence["exception_active"] is False
    assert wrong.evidence["exception_active"] is True
    assert correct.outcome_index == 0
    assert wrong.outcome_index == 1


@pytest.mark.parametrize(
    ("magnitude", "expected_hops"), ((1, 2), (2, 3), (3, 2), (4, 3)))
def test_alias_rule_has_changed_intermediate_link_and_two_or_three_hops(
        magnitude, expected_hops):
    candidate = _pool_candidate(
        "referent_alias_resolution", f"magnitude_{magnitude}_c0_w1")
    correct, wrong = recipe.evaluate_pair(candidate)
    assert correct.evidence["hop_count"] == expected_hops
    assert wrong.evidence["hop_count"] == expected_hops
    assert correct.changed_value != wrong.changed_value
    assert correct.evidence["resolution_path"][0] == \
        wrong.evidence["resolution_path"][0]
    assert correct.evidence["resolution_path"][1] == \
        wrong.evidence["resolution_path"][1]
    assert correct.evidence["resolution_path"][2] != \
        wrong.evidence["resolution_path"][2]


def test_in_pool_text_expansion_fails_before_message_compilation(monkeypatch):
    candidate = next(recipe.enumerate_candidate_tuples("threshold_eligibility"))
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("production conversation text was inspected")

    monkeypatch.setattr(recipe, "_messages_for_variant", forbidden)
    with pytest.raises(recipe.RecipeError, match="requires authorization"):
        recipe.materialize_ranked_candidate(candidate, None)  # type: ignore[arg-type]
    assert called is False


@pytest.mark.parametrize("stratum", recipe.STRATA)
@pytest.mark.parametrize("explicit", (True, False))
def test_out_of_pool_development_fixture_passes_static_schema(stratum, explicit):
    candidate = recipe.development_sentinel(stratum, explicit=explicit)
    assert recipe.is_pool_member(candidate) is False
    fixture = recipe.materialize_development_sentinel(stratum, explicit=explicit)
    report = recipe.validate_fixture_schema(fixture)
    assert report["status"] == "TOKENIZER_FREE_DRAFT_SCHEMA_PASS"
    assert report["changed_message_indices"] == [3, 6]
    assert report["retained_tail_identical"] is True
    assert report["rule_outcomes_opposite"] is True
    assert report["distractor_chain_count"] == 2
    assert fixture["pool_member"] is False
    assert fixture["materialization_authorization"]["kind"] == \
        "OUT_OF_POOL_DEVELOPMENT_SENTINEL"
    assert fixture["execution_ready"] is False
    assert fixture["length_design"]["production_tokenizer_status"] == \
        "PENDING_NOT_MEASURED_OR_CLAIMED"
    # Roughly 800 English words is an intentional pre-tokenizer design target;
    # this is not presented as a production-token count.
    assert 730 <= fixture["length_design"]["whitespace_word_count_C"] <= 840


def test_development_history_pairs_are_collision_free_without_touching_pool_text():
    hashes = []
    for stratum in recipe.STRATA:
        for explicit in (True, False):
            fixture = recipe.materialize_development_sentinel(
                stratum, explicit=explicit)
            hashes.append(fixture["literal_history_pair_sha256"])
    assert len(hashes) == 16
    assert len(set(hashes)) == 16


def test_fixture_has_one_changed_input_two_chains_and_byte_identical_tail():
    fixture = _dev_fixture("interval_schedule_overlap", explicit=True)
    correct = fixture["variants"]["C"]["messages"]
    wrong = fixture["variants"]["W"]["messages"]
    assert fixture["changed_input"]["count"] == 1
    assert fixture["changed_input"]["role"] == "user"
    assert [index for index, (left, right) in enumerate(zip(correct, wrong))
            if left != right] == [3, 6]
    assert correct[7:] == wrong[7:]
    assert len(fixture["distractor_chains"]) == 2
    assert fixture["retained_tail"]["starts_with_user"] is True
    assert fixture["retained_tail"]["ends_with_assistant"] is True


def test_explicit_names_outcome_and_unstated_withholds_it():
    explicit = _dev_fixture("arithmetic_capacity_budget", explicit=True)
    unstated = _dev_fixture("arithmetic_capacity_budget", explicit=False)
    for variant in ("C", "W"):
        e_target = explicit["focal"][f"{variant}_target"]
        u_target = unstated["focal"][f"{variant}_target"]
        assert e_target in explicit["variants"][variant]["messages"][6]["content"]
        assert u_target not in unstated["variants"][variant]["messages"][6]["content"]
        assert "withheld" in unstated["variants"][variant]["messages"][6]["content"]


def test_direct_nonfocal_fact_is_visible_and_probe_omits_choices():
    fixture = _dev_fixture("categorical_set_membership", explicit=False)
    control = fixture["nonfocal_control"]
    index = control["establishing_message_indices"][0]
    correct = fixture["variants"]["C"]["messages"]
    wrong = fixture["variants"]["W"]["messages"]
    assert index >= fixture["middle_end_msg"] or index == 0
    assert control["target"] in correct[index]["content"]
    assert correct[index] == wrong[index]
    assert control["target"].casefold() not in control["probe"].casefold()
    assert control["countertarget"].casefold() not in control["probe"].casefold()


@pytest.mark.parametrize("stratum", recipe.STRATA)
def test_carrier_forbidden_inventory_enumerates_case_semantics(stratum):
    fixture = _dev_fixture(stratum, explicit=True)
    inventory = fixture["carrier_forbidden_inventory"]
    focal = fixture["focal"]
    control = fixture["nonfocal_control"]
    skin = fixture["domain_skin"]
    assert inventory["focal_targets"]["C"] == focal["C_target"]
    assert inventory["focal_targets"]["W"] == focal["W_target"]
    assert inventory["changed_values"]["C"]["raw"] == \
        fixture["changed_input"]["C_value"]
    assert inventory["changed_values"]["W"]["raw"] == \
        fixture["changed_input"]["W_value"]
    assert inventory["nonfocal"]["target"] == control["target"]
    assert inventory["nonfocal"]["countertarget"] == control["countertarget"]
    assert inventory["contextual_names"] == {
        key: skin[key] for key in (
            "organization", "ledger", "coordinator", "subject", "site",
            "distractor_a", "distractor_b")
    }
    exact = set(inventory["exact_forbidden_forms"])
    assert set(inventory["focal_targets"].values()).issubset(exact)
    assert {control["target"], control["countertarget"]}.issubset(exact)
    assert set(inventory["contextual_names"].values()).issubset(exact)
    assert inventory["rule_semantics"]["definition"]
    assert inventory["rule_semantics"]["evaluated_evidence"]["C"]
    assert inventory["rule_semantics"]["evaluated_evidence"]["W"]
    expansion = inventory["surface_expansion"]
    assert expansion["exact_semantic_values_and_names_enumerated"] is True
    assert expansion["punctuation_variants_status"].startswith("PENDING_")
    assert expansion["spelled_number_variants_status"].startswith("PENDING_")
    assert expansion["unicode_normalization_variants_status"].startswith("PENDING_")
    assert expansion["production_tokenizer_subsequence_status"].startswith("PENDING_")
    assert expansion["target_aware_semantic_review_required"] is True
    core = {key: value for key, value in inventory.items()
            if key != "semantic_inventory_sha256"}
    assert inventory["semantic_inventory_sha256"] == sha256(
        recipe.canonical_json_bytes(core)).hexdigest()


def test_interval_forbidden_inventory_includes_numeric_and_rendered_time_forms():
    fixture = _dev_fixture("interval_schedule_overlap", explicit=False)
    inventory = fixture["carrier_forbidden_inventory"]
    assert {510, 540, 570, 600, 810, 840, 900, 990}.issubset(
        set(inventory["rule_semantics"]["numeric_values"]))
    assert {"08:30", "09:00", "09:30", "10:00", "13:30", "14:00",
            "15:00", "16:30"}.issubset(
        set(inventory["rule_semantics"]["time_forms"]))
    assert {"10:00–13:30", "09:00–15:00"}.issubset(
        set(inventory["rule_semantics"]["rendered_interval_forms"]))


@pytest.mark.parametrize(
    ("stratum", "expected_forms"),
    (
        ("threshold_eligibility", {"67", "68", "66"}),
        ("sequential_tiebreak", {"14:20", "14:35", "14:05", "73", "87"}),
        ("conjunction_all_conditions", {"9", "12", "65", "69", "61"}),
        ("arithmetic_capacity_budget", {"17", "2", "7", "105", "106"}),
        ("categorical_set_membership",
         {"counted", "wrapped", "current", "deferred", "missing"}),
        ("ordered_priority_exception",
         {"immediate sorting", "ordinary sorting", "Pear Rest", "active", "inactive"}),
        ("referent_alias_resolution",
         {"envelope", "orchard card", "north basket", "south basket",
          "woven tag", "plain tag"}),
    ),
)
def test_carrier_forbidden_exact_forms_cover_family_rule_atoms(
        stratum, expected_forms):
    fixture = _dev_fixture(stratum, explicit=False)
    exact = set(fixture["carrier_forbidden_inventory"]["exact_forbidden_forms"])
    assert expected_forms.issubset(exact)


def test_carrier_forbidden_inventory_mutation_fails_closed():
    fixture = _dev_fixture("referent_alias_resolution", explicit=True)
    fixture["carrier_forbidden_inventory"]["contextual_names"].pop("site")
    with pytest.raises(recipe.RecipeError, match="carrier-forbidden inventory differs"):
        recipe.validate_fixture_schema(fixture)


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("tail", "retained tails differ"),
        ("third_distractor", "exactly two distractor"),
        ("prefix_target", "focal targets are equal/prefixing"),
        ("execution", "incorrectly authorizes execution"),
    ),
)
def test_schema_validator_fails_closed_on_core_contract_mutations(mutation, match):
    fixture = _dev_fixture()
    if mutation == "tail":
        fixture["variants"]["W"]["messages"][7]["content"] += " changed"
    elif mutation == "third_distractor":
        fixture["distractor_chains"].append({"chain_id": "extra"})
    elif mutation == "prefix_target":
        fixture["focal"]["countertarget"] = fixture["focal"]["target"] + "bank"
    else:
        fixture["execution_ready"] = True
    with pytest.raises(recipe.RecipeError, match=match):
        recipe.validate_fixture_schema(fixture)


def test_module_has_no_rng_tokenizer_subject_or_result_dependency():
    source = inspect.getsource(recipe)
    for forbidden in (
        "import random", "import numpy", "import torch", "import transformers",
        "AutoTokenizer", "from_pretrained", "results/",
    ):
        assert forbidden not in source
