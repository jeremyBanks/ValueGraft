"""Out-of-pool-only tests for the v13 tokenizer/mechanical layer."""

from __future__ import annotations

import copy
from hashlib import sha256
import inspect
from pathlib import Path

import pytest

import powered_v13_stimuli as stimuli
import powered_v13_tokens as token_plans
from powered_v13_recipe import (
    STRATA,
    canonical_json_bytes,
    materialize_development_sentinel,
)
from powered_v13_schema import (
    CARRIER_REQUEST, MODEL_ID, MODEL_REVISION, V13SchemaError,
)


EXPECTED_SENTINEL_MATRIX_SHA256_BY_AUTHORIZATION_FIELDS = {
    frozenset({
        "kind", "status", "candidate_id", "permutation_rank",
        "seed_manifest_sha256",
        "seed_git_commit", "literal_permutation_sha256",
        "literal_git_commit",
    }): "f545875979b59bdeb1efea452e2f2818420c07ec2da5d76527ee8951b566ec7f",
}
EXPECTED_TARGET_BANK_QUALIFICATION_SHA256 = (
    "378c078e4ab5862e6a8895a130443dcb4e2d990d1dda69e18598114841b374ab"
)
FROZEN_80_TOKEN_TEST_CARRIER = (
    "The earlier exchange established shared operating context and confirmed "
    "that decision criteria were recorded. This handoff preserves continuity "
    "for the next assistant while withholding the criteria themselves and "
    "every settled detail. Continue from the visible conversation, using the "
    "available context for any later request pseudopseudohypoparathyroidism "
    "pseudopseudohypoparathyroidism antidisestablishmentarianism "
    "electroencephalographically."
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    except OSError:
        pytest.skip("pinned production tokenizer is not cached")


def _fixture(stratum="threshold_eligibility", *, explicit=True):
    # This is the recipe's exact out-of-pool sentinel constructor.  Tests never
    # construct or expand a production-pool CandidateTuple.
    return materialize_development_sentinel(stratum, explicit=explicit)


def _carrier_args(tokenizer, fixture,
                  text=stimuli.FROZEN_TEST_CARRIER_CONTENT, *,
                  render_index=1, attempt_index=1):
    return {
        "attempt_spec": stimuli.carrier_attempt_spec(
            fixture,
            render_index=render_index,
            attempt_index=attempt_index,
        ),
        "generated_content_ids": tokenizer.encode(
            text, add_special_tokens=False),
        "termination_token_id": tokenizer.eos_token_id,
        "hit_token_cap": False,
    }


def test_integer_sequence_hash_uses_protocol_canonical_json_without_coercion():
    values = [0, 1, 151_643, -1]
    assert stimuli.sha256_ints(values) == sha256(
        canonical_json_bytes(values)).hexdigest()
    for invalid in ([True], [1.0], ["1"]):
        with pytest.raises(stimuli.V13StimulusError,
                           match="contains a non-integer"):
            stimuli.sha256_ints(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("stratum", STRATA)
@pytest.mark.parametrize("explicit", (True, False))
def test_all_family_by_subtype_out_of_pool_sentinels_pass_exact_mechanics(
        tokenizer, stratum, explicit):
    fixture = _fixture(stratum, explicit=explicit)
    result = stimuli.validate_development_sentinel(tokenizer, fixture)
    assert result["status"] == \
        "MECHANICAL_SENTINEL_PASS_NO_EXECUTION_AUTHORIZATION"
    assert result["scope"] == "OUT_OF_POOL_DEVELOPMENT_SENTINEL_ONLY"
    assert result["execution_ready"] is False
    assert result["scientific_outcome"] is False
    assert result["pool_member"] is False
    assert result["in_pool_tuple_materialized"] is False
    assert result["permutation_seed_present"] is False
    assert result["literal_permutation_present"] is False
    authorization = fixture["materialization_authorization"]
    assert all(authorization.get(key) is None for key in (
        "candidate_id", "permutation_rank", "seed_manifest_sha256",
        "seed_git_commit",
        "literal_permutation_sha256", "literal_git_commit",
    ))
    assert result["pair"] == {
        "changed_message_allowlist": [3, 6],
        "retained_tail_byte_identical": True,
        "per_message_production_widths_identical": True,
        "canonical_roundtrip_both_variants": True,
        "role_native_geometry_identical": True,
        "turn_aligned_geometry_identical": True,
        "fresh_destination_identical": True,
        "no_call_exceeds_4096": True,
    }
    for variant in ("C", "W"):
        row = result[variant]
        assert 940 <= row["noncarrier_token_count"] <= 1180
        assert 900 <= row["completed"]["canonical_token_count"] <= 1400
        assert row["history"]["decoded_round_trip"] is True
        assert row["completed"]["decoded_round_trip"] is True
        assert row["role_native"]["max_call_width"] <= 4096
        assert row["turn_aligned"]["max_call_width"] <= 4096
        assert row["fresh_destination"]["max_call_width"] <= 4096
    assert result["C"]["history"]["content_token_widths"] == \
        result["W"]["history"]["content_token_widths"]
    assert result["C"]["history"]["canonical_message_widths"] == \
        result["W"]["history"]["canonical_message_widths"]
    assert result["C"]["completed"]["content_token_widths"] == \
        result["W"]["completed"]["content_token_widths"]
    assert result["C"]["fresh_token_ids_sha256"] == \
        result["W"]["fresh_token_ids_sha256"]

    for context in ("C", "W", "F"):
        for probe_kind in ("focal", "nonfocal"):
            probe = result["probe_contexts"][context][probe_kind]
            assert (probe["source_token_count"] ==
                    probe["probe_message_start"] <
                    probe["assistant_header_start"] <
                    probe["answer_position"] ==
                    probe["generation_prefix_token_count"])
            counts = [target["token_count"] for target in probe["targets"]]
            assert all(1 <= count <= 4 for count in counts)
            assert abs(counts[0] - counts[1]) <= 1
            assert probe["target_ids_distinct_nonprefix"] is True
            assert all(target["target_start"] == probe["answer_position"]
                       for target in probe["targets"])
            assert all(target["assistant_close_position"] ==
                       target["target_end"]
                       for target in probe["targets"])


def test_complete_sentinel_matrix_has_stable_compact_audit_and_no_authority(
        tokenizer):
    result = stimuli.validate_all_development_sentinels(tokenizer)
    assert result["status"] == \
        "MECHANICAL_SENTINEL_MATRIX_PASS_NO_EXECUTION_AUTHORIZATION"
    assert result["sentinel_count"] == 16
    assert result["scope"] == \
        "ALL_16_OUT_OF_POOL_FAMILY_BY_SUBTYPE_SENTINELS"
    assert result["execution_ready"] is False
    assert result["scientific_outcome"] is False
    assert result["in_pool_tuple_materialized"] is False
    assert result["permutation_seed_present"] is False
    assert result["literal_permutation_present"] is False
    target_banks = result["production_target_bank_qualification"]
    assert target_banks["qualification_sha256"] == \
        EXPECTED_TARGET_BANK_QUALIFICATION_SHA256
    assert target_banks["focal_bank_count"] == 8
    assert target_banks["nonfocal_bank_count"] == 4
    assert target_banks["target_surface_count"] == 24
    assert target_banks["in_pool_tuple_materialized"] is False
    assert target_banks["permutation_seed_present"] is False
    authorization_fields = frozenset(
        _fixture()["materialization_authorization"])
    assert authorization_fields in \
        EXPECTED_SENTINEL_MATRIX_SHA256_BY_AUTHORIZATION_FIELDS
    assert result["audit_sha256"] == \
        EXPECTED_SENTINEL_MATRIX_SHA256_BY_AUTHORIZATION_FIELDS[
            authorization_fields]
    assert len({row["stable_candidate_id"] for row in result["rows"]}) == 16
    assert all(row["status"].endswith("NO_EXECUTION_AUTHORIZATION")
               for row in result["rows"])
    core = {key: value for key, value in result.items()
            if key != "audit_sha256"}
    assert result["audit_sha256"] == stimuli.sha256_json(core)


def test_module_has_no_production_pool_materializer_or_rng_dependency():
    source = inspect.getsource(stimuli)
    assert "materialize_ranked_candidate" not in source
    assert "enumerate_candidate_tuples" not in source
    assert "numpy" not in source
    assert "secrets" not in source
    assert "random." not in source


def test_tokenizer_binding_covers_backend_files_and_resolved_snapshot(tokenizer):
    binding = stimuli._tokenizer_binding(tokenizer)
    assert binding["model_id"] == MODEL_ID
    assert binding["requested_revision"] == MODEL_REVISION
    assert binding["resolved_snapshot_revision"] == MODEL_REVISION
    assert binding["backend_serialization_sha256"] == \
        stimuli.PINNED_TOKENIZER_BACKEND_SHA256
    assert binding["vocabulary_sha256"] == \
        stimuli.PINNED_TOKENIZER_VOCAB_SHA256
    assert binding["chat_template_sha256"] == \
        stimuli.PINNED_CHAT_TEMPLATE_SHA256
    assert binding["snapshot_files"] == {
        name: dict(record)
        for name, record in stimuli.PINNED_TOKENIZER_FILES.items()
    }
    assert binding["snapshot_files_sha256"] == stimuli.sha256_json(
        binding["snapshot_files"])
    transformers = pytest.importorskip("transformers")
    snapshot = str(Path(tokenizer.init_kwargs["vocab_file"]).parent)
    snapshot_loaded = transformers.AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    assert stimuli._tokenizer_binding(snapshot_loaded) == binding


def test_tokenizer_binding_rejects_changed_backend_serialization(tokenizer):
    normalizers = pytest.importorskip("tokenizers.normalizers")
    altered = copy.deepcopy(tokenizer)
    altered.backend_tokenizer.normalizer = normalizers.Lowercase()
    assert altered.encode("The", add_special_tokens=False) != \
        tokenizer.encode("The", add_special_tokens=False)
    with pytest.raises(stimuli.V13StimulusError,
                       match="backend serialization hash differs"):
        stimuli._tokenizer_binding(altered)


def test_section5_attempt_digest_seed_indices_and_sampler_are_exact():
    fixture = _fixture()
    spec = stimuli.carrier_attempt_spec(
        fixture, render_index=1, attempt_index=1)
    digest_input = [
        "coherent-state-powered-successor-v13",
        fixture["stable_candidate_id"],
        1,
        1,
    ]
    digest = sha256(canonical_json_bytes(digest_input)).digest()
    assert spec["attempt_digest_input"] == digest_input
    assert spec["attempt_digest_sha256"] == digest.hex()
    assert spec["attempt_digest_sha256"] == \
        "8517a6c538e692b037aba58e9e16f5f3ee57477264005325bd8ed77e8e33212b"
    assert spec["torch_generator_seed"] == \
        int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    assert spec["torch_generator_seed"] == 366945260647387824
    assert 0 <= spec["torch_generator_seed"] < 2 ** 63
    assert spec["render_count"] == 2
    assert spec["max_attempts_per_render"] == 3
    assert spec["prompt"] == CARRIER_REQUEST
    assert spec["q_call_width"] == 1
    assert spec["sampling"] is True
    assert spec["temperature"] == 0.7
    assert spec["top_p"] == 0.95
    assert spec["top_k"] is None
    assert spec["top_k_disabled"] is True
    assert spec["content_token_cap"] == 80
    assert spec["maximum_sampling_calls_including_eos"] == 81
    assert spec["normal_eos_required"] is True
    core = {key: value for key, value in spec.items()
            if key != "attempt_spec_sha256"}
    assert spec["attempt_spec_sha256"] == stimuli.sha256_json(core)


@pytest.mark.parametrize(
    ("render_index", "attempt_index", "match"),
    (
        (0, 1, "render index"),
        (3, 1, "render index"),
        (True, 1, "render index"),
        (1, 0, "attempt index"),
        (1, 4, "attempt index"),
        (1, True, "attempt index"),
    ),
)
def test_section5_attempt_indices_fail_outside_exact_one_based_bounds(
        render_index, attempt_index, match):
    with pytest.raises(stimuli.V13StimulusError, match=match):
        stimuli.carrier_attempt_spec(
            _fixture(), render_index=render_index, attempt_index=attempt_index)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("prompt", "A changed prompt."),
        ("temperature", 0.8),
        ("top_p", 0.9),
        ("top_k", 50),
        ("q_call_width", 2),
        ("torch_generator_seed", 17),
        ("content_token_cap", 79),
        ("maximum_sampling_calls_including_eos", 80),
    ),
)
def test_section5_attempt_spec_rejects_any_sampler_prompt_seed_or_cap_change(
        field, replacement):
    fixture = _fixture()
    altered = stimuli.carrier_attempt_spec(
        fixture, render_index=2, attempt_index=3)
    altered[field] = replacement
    with pytest.raises(stimuli.V13StimulusError,
                       match="prompt, sampler, digest, seed, or cap differs"):
        stimuli._validated_attempt_spec(fixture, altered)


def test_all_production_target_banks_have_durable_exact_token_qualification(
        tokenizer):
    result = stimuli.validate_production_target_banks(tokenizer)
    assert result["schema"] == stimuli.TARGET_BANK_SCHEMA
    assert result["status"] == "ALL_PRODUCTION_TARGET_BANKS_TOKENIZER_QUALIFIED"
    assert result["qualification_sha256"] == \
        EXPECTED_TARGET_BANK_QUALIFICATION_SHA256
    assert result["focal_bank_count"] == 8
    assert result["nonfocal_bank_count"] == 4
    assert result["target_surface_count"] == 24
    assert result["all_target_surfaces_mutually_distinct_nonprefix"] is True
    assert result[
        "all_target_token_sequences_mutually_distinct_nonprefix"] is True
    assert result["in_pool_tuple_materialized"] is False
    assert result["permutation_seed_present"] is False
    assert len({row["bank_id"] for row in result["focal_banks"]}) == 8
    assert len({row["bank_id"] for row in result["nonfocal_banks"]}) == 4
    for row in result["focal_banks"] + result["nonfocal_banks"]:
        probe = row["probe_evidence"]
        assert probe["target_ids_distinct_nonprefix"] is True
        assert all(1 <= target["token_count"] <= 4
                   for target in probe["targets"])
        assert all(target["token_ids"] for target in probe["targets"])
        assert all(target["decoded_text_exact"] is True
                   for target in probe["targets"])
        assert all(target["encode_decode_ids_exact"] is True
                   for target in probe["targets"])
        assert all(target["token_ids_sha256"] == stimuli.sha256_ints(
            target["token_ids"]) for target in probe["targets"])
    core = {key: value for key, value in result.items()
            if key != "qualification_sha256"}
    assert result["qualification_sha256"] == stimuli.sha256_json(core)


def test_forbidden_expansion_covers_unicode_case_punctuation_numbers_and_time(
        tokenizer):
    fixture = _fixture("sequential_tiebreak", explicit=False)
    expansion = stimuli.expand_carrier_forbidden_surfaces(tokenizer, fixture)
    assert expansion["schema"] == stimuli.FORBIDDEN_SURFACE_SCHEMA
    assert expansion["unicode_normalization_modes"] == [
        "NFC", "NFD", "NFKC", "NFKD"]
    assert expansion["casefold_applied"] is True
    assert expansion["separator_insensitive_matching_applied"] is True
    assert expansion["punctuation_hyphen_space_expansion_applied"] is True
    assert expansion["slash_camel_split_letter_expansion_applied"] is True
    assert expansion["plain_nonfocal_fact_name_included"] is True
    assert expansion["plain_nonfocal_fact_name"] == "filing token"
    assert expansion["case_declared_forbidden_phrases"] == []
    assert expansion["case_declared_forbidden_phrases_sha256"] == \
        stimuli.sha256_json([])
    assert expansion["all_recorded_numerics_expanded"] is True
    assert expansion["all_recorded_times_expanded"] is True
    assert "seventy-three" in expansion["numeric_form_inventory"]["73"]
    assert "eighty-seven" in expansion["numeric_form_inventory"]["87"]
    assert "14:35" in expansion["time_form_inventory"]
    assert "two thirty-five pm" in expansion["time_form_inventory"]["14:35"]
    normalized = set(expansion["normalized_surface_forms"])
    assert "orchard exchange" in normalized
    assert "orchard-exchange" in normalized
    assert "orchard/exchange" in normalized
    assert "orchardexchange" in normalized
    assert "orchardExchange" in expansion["raw_surface_forms"]
    assert "q.u.a.r.t.z" in normalized
    assert any(record["skeleton"] == "quartz"
               for record in expansion["separator_insensitive_surfaces"])
    assert "filing/token" in normalized
    assert "seventy three" in normalized
    assert "seventy-three" in normalized
    assert expansion["production_token_subsequences"]
    encoded = tuple(tokenizer.encode(" Orchard Exchange", add_special_tokens=False))
    assert any(tuple(record["ids"]) == encoded
               for record in expansion["production_token_subsequences"])
    core = {key: value for key, value in expansion.items()
            if key != "expansion_sha256"}
    assert expansion["expansion_sha256"] == stimuli.sha256_json(core)


@pytest.mark.parametrize(
    ("stratum", "replacement", "match"),
    (
        ("threshold_eligibility", "QUARTZ", "normalized surface"),
        ("threshold_eligibility", "ＱＵＡＲＴＺ", "normalized surface"),
        # Bounded surface matching intentionally does not treat Quartzite as
        # the literal Quartz target; the production-token subsequence gate is
        # the independent mechanism that rejects the shared target token.
        ("threshold_eligibility", "Quartzite", "token subsequence"),
        ("threshold_eligibility", "sixty-seven", "normalized surface"),
        ("threshold_eligibility", "sixty/seven", "normalized surface"),
        ("threshold_eligibility", "sixty—seven", "normalized surface"),
        ("threshold_eligibility", "sixty_seven", "normalized surface"),
        ("threshold_eligibility", "sixty.seven", "normalized surface"),
        ("threshold_eligibility", "Q.u.a.r.t.z", "normalized surface"),
        ("threshold_eligibility", "Q•u•a•r•t•z", "separator-insensitive"),
        ("threshold_eligibility", "Q·u·a·r·t·z", "separator-insensitive"),
        ("threshold_eligibility", "Q\u200bu\u200ba\u200br\u200bt\u200bz",
         "separator-insensitive"),
        ("threshold_eligibility", "Q'u'a'r't'z", "separator-insensitive"),
        ("threshold_eligibility", "Qʼuʼaʼrʼtʼz", "separator-insensitive"),
        ("threshold_eligibility", "Q|u|a|r|t|z", "separator-insensitive"),
        ("threshold_eligibility", "Q\\u\\a\\r\\t\\z",
         "separator-insensitive"),
        ("threshold_eligibility", "Orchard/Exchange", "normalized surface"),
        ("threshold_eligibility", "OrchardExchange", "normalized surface"),
        ("threshold_eligibility", "orchardExchange", "normalized surface"),
        ("threshold_eligibility", "filing token", "normalized surface"),
        ("sequential_tiebreak", "two thirty-five pm", "normalized surface"),
        ("ordered_priority_exception", "Pear-Rest", "normalized surface"),
        ("ordered_priority_exception", "Pear/Rest", "normalized surface"),
        ("ordered_priority_exception", "PearRest", "normalized surface"),
    ),
)
def test_carrier_gate_rejects_expanded_semantic_leaks(
        tokenizer, stratum, replacement, match):
    fixture = _fixture(stratum, explicit=True)
    text = stimuli.FROZEN_TEST_CARRIER_CONTENT.replace(
        "settled detail", replacement)
    with pytest.raises(stimuli.V13StimulusError, match=match):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, text, **_carrier_args(tokenizer, fixture, text))


def test_case_declared_forbidden_phrases_are_accepted_expanded_and_hash_bound(
        tokenizer):
    fixture = copy.deepcopy(_fixture())
    fixture["case_declared_forbidden_phrases"] = [
        "veiled checkpoint",
        "shared-only marker",
    ]
    expansion = stimuli.expand_carrier_forbidden_surfaces(tokenizer, fixture)
    assert expansion["case_declared_forbidden_phrases"] == \
        fixture["case_declared_forbidden_phrases"]
    assert expansion["case_declared_forbidden_phrase_count"] == 2
    assert expansion["case_declared_forbidden_phrases_sha256"] == \
        stimuli.sha256_json(fixture["case_declared_forbidden_phrases"])
    assert "veiled/checkpoint" in expansion["normalized_surface_forms"]
    text = stimuli.FROZEN_TEST_CARRIER_CONTENT.replace(
        "settled detail", "veiled/checkpoint")
    with pytest.raises(stimuli.V13StimulusError,
                       match="forbidden normalized surface"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, text,
            **_carrier_args(tokenizer, fixture, text))
    for obfuscated in (
        "veiled•checkpoint", "veiled\u200bcheckpoint", "veiledʼcheckpoint",
    ):
        text = stimuli.FROZEN_TEST_CARRIER_CONTENT.replace(
            "settled detail", obfuscated)
        with pytest.raises(stimuli.V13StimulusError,
                           match="separator-insensitive"):
            stimuli.validate_carrier_attempt(
                tokenizer, fixture, text,
                **_carrier_args(tokenizer, fixture, text))


@pytest.mark.parametrize(
    "phrases",
    (
        "not-a-list",
        [""],
        [" leading-space"],
        ["same phrase", "SAME PHRASE"],
    ),
)
def test_case_declared_forbidden_phrases_fail_closed_when_malformed(
        tokenizer, phrases):
    fixture = copy.deepcopy(_fixture())
    fixture["case_declared_forbidden_phrases"] = phrases
    with pytest.raises(stimuli.V13StimulusError,
                       match="case-declared forbidden phrase"):
        stimuli.expand_carrier_forbidden_surfaces(tokenizer, fixture)


def test_carrier_gate_accepts_frozen_mechanical_carrier_but_keeps_review_pending(
        tokenizer):
    fixture = _fixture("referent_alias_resolution", explicit=False)
    result = stimuli.validate_carrier_attempt(
        tokenizer, fixture, stimuli.FROZEN_TEST_CARRIER_CONTENT,
        **_carrier_args(tokenizer, fixture))
    assert result["status"] == \
        "MECHANICAL_CARRIER_PASS_SEMANTIC_REVIEW_PENDING"
    assert result["execution_ready"] is False
    assert result["subject_model_called_by_validator"] is False
    assert result["target_aware_semantic_review"] == "PENDING_REQUIRED"
    assert 40 <= result["word_count"] <= 60
    assert 40 <= result["production_content_token_count"] <= 80
    assert result["terminated_with_eos"] is True
    assert result["hit_token_cap"] is False
    assert result["decoded_text_exact"] is True
    assert result["encode_decode_ids_exact"] is True
    assert result["content_ids_exclude_terminal_eos"] is True
    assert result["attempt_spec"]["render_index"] == 1
    assert result["attempt_spec"]["attempt_index"] == 1
    assert result["attempt_spec_sha256"] == \
        result["attempt_spec"]["attempt_spec_sha256"]
    assert result["content_token_cap"] == 80
    assert result["maximum_sampling_calls_including_eos"] == 81
    assert result["sampling_call_count_including_eos"] == \
        result["production_content_token_count"] + 1


def test_carrier_gate_rejects_digit_special_literal_length_cap_and_non_eos(
        tokenizer):
    fixture = _fixture()
    digit = stimuli.FROZEN_TEST_CARRIER_CONTENT.replace("earlier", "2")
    with pytest.raises(stimuli.V13StimulusError, match="digit"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, digit,
            **_carrier_args(tokenizer, fixture, digit))

    special = stimuli.FROZEN_TEST_CARRIER_CONTENT.replace(
        "earlier", "<|im_start|>")
    with pytest.raises(stimuli.V13StimulusError, match="control literal"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, special,
            **_carrier_args(tokenizer, fixture, special))

    short = "This deliberately short handoff omits all specific prior facts."
    with pytest.raises(stimuli.V13StimulusError, match="word count"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, short,
            **_carrier_args(tokenizer, fixture, short))

    token_long = " ".join(
        ["pneumonoultramicroscopicsilicovolcanoconiosis"] * 40)
    with pytest.raises(stimuli.V13StimulusError, match="token count"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, token_long,
            **_carrier_args(tokenizer, fixture, token_long))

    valid_args = _carrier_args(tokenizer, fixture)
    with pytest.raises(stimuli.V13StimulusError, match="hit the token cap"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, stimuli.FROZEN_TEST_CARRIER_CONTENT,
            **(valid_args | {"hit_token_cap": True}))
    with pytest.raises(stimuli.V13StimulusError, match="terminate with EOS"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, stimuli.FROZEN_TEST_CARRIER_CONTENT,
            **(valid_args | {
                "termination_token_id": tokenizer.eos_token_id - 1}))
    for changed_cap in (79, 81, 100):
        with pytest.raises(stimuli.V13StimulusError,
                           match="content-token cap differs from frozen 80"):
            stimuli.validate_carrier_attempt(
                tokenizer, fixture, stimuli.FROZEN_TEST_CARRIER_CONTENT,
                **valid_args, max_new_tokens=changed_cap)


@pytest.mark.parametrize("replacement", ["785", 785.0, True, -1, 10**9])
def test_carrier_gate_rejects_nonplain_or_out_of_range_generated_ids(
        tokenizer, replacement):
    fixture = _fixture()
    args = _carrier_args(tokenizer, fixture)
    changed = list(args["generated_content_ids"])
    changed[0] = replacement
    with pytest.raises(stimuli.V13StimulusError,
                       match="non-plain or out-of-range integer"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, stimuli.FROZEN_TEST_CARRIER_CONTENT,
            **(args | {"generated_content_ids": changed}))


@pytest.mark.parametrize("replacement", ["151645", 151645.0, True])
def test_carrier_gate_rejects_nonplain_eos_witness(tokenizer, replacement):
    fixture = _fixture()
    args = _carrier_args(tokenizer, fixture)
    with pytest.raises(stimuli.V13StimulusError, match="terminate with EOS"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, stimuli.FROZEN_TEST_CARRIER_CONTENT,
            **(args | {"termination_token_id": replacement}))


def test_exactly_80_content_tokens_then_separate_eos_witness_is_accepted(
        tokenizer):
    fixture = _fixture("referent_alias_resolution", explicit=False)
    ids = tokenizer.encode(
        FROZEN_80_TOKEN_TEST_CARRIER, add_special_tokens=False)
    assert len(FROZEN_80_TOKEN_TEST_CARRIER.split()) == 48
    assert len(ids) == 80
    result = stimuli.validate_carrier_attempt(
        tokenizer, fixture, FROZEN_80_TOKEN_TEST_CARRIER,
        **_carrier_args(tokenizer, fixture, FROZEN_80_TOKEN_TEST_CARRIER),
        max_new_tokens=80,
    )
    assert result["production_content_token_count"] == 80
    assert result["sampling_call_count_including_eos"] == 81
    assert result["termination_token_id"] == tokenizer.eos_token_id
    assert result["terminated_with_eos"] is True
    assert result["hit_token_cap"] is False


def test_carrier_gate_rejects_token_text_mismatch(tokenizer):
    fixture = _fixture()
    args = _carrier_args(tokenizer, fixture)
    altered_ids = list(args["generated_content_ids"])
    altered_ids[-1] = tokenizer.encode("!", add_special_tokens=False)[0]
    with pytest.raises(stimuli.V13StimulusError, match="do not decode exactly"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, stimuli.FROZEN_TEST_CARRIER_CONTENT,
            **(args | {"generated_content_ids": altered_ids}))


@pytest.mark.parametrize(
    "boundary_name",
    ("canonical_ids_any", "generation_prefix_ids",
     "rendered_assistant_content_ids"),
)
@pytest.mark.parametrize("replacement_type", ("string", "float", "oov"))
def test_full_sentinel_rejects_type_laundering_at_token_plan_boundaries(
        tokenizer, monkeypatch, boundary_name, replacement_type):
    original = getattr(token_plans, boundary_name)

    def malformed(*args, **kwargs):
        observed = original(*args, **kwargs)
        if replacement_type == "string":
            return [str(value) for value in observed]
        if replacement_type == "float":
            return [float(value) for value in observed]
        result = list(observed)
        result[-1] = len(tokenizer)
        return result

    monkeypatch.setattr(token_plans, boundary_name, malformed)
    with pytest.raises(V13SchemaError, match="non-plain|out-of-vocabulary"):
        stimuli.validate_development_sentinel(tokenizer, _fixture())


def test_carrier_gate_rejects_same_decoded_text_with_noncanonical_exact_ids(
        tokenizer):
    fixture = _fixture()
    args = _carrier_args(tokenizer, fixture)
    canonical = list(args["generated_content_ids"])
    assert canonical[0] == 785
    noncanonical = [51, 383, *canonical[1:]]
    decoded = tokenizer.decode(
        noncanonical,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert decoded == stimuli.FROZEN_TEST_CARRIER_CONTENT
    assert tokenizer.encode(decoded, add_special_tokens=False) == canonical
    assert noncanonical != canonical
    with pytest.raises(stimuli.V13StimulusError,
                       match=r"encode\(decode\(ids\)\).+exact content IDs"):
        stimuli.validate_carrier_attempt(
            tokenizer, fixture, stimuli.FROZEN_TEST_CARRIER_CONTENT,
            **(args | {"generated_content_ids": noncanonical}))


def test_nonallowlisted_tail_corruption_fails_closed_before_geometry(tokenizer):
    fixture = copy.deepcopy(_fixture())
    fixture["variants"]["W"]["messages"][7]["content"] += " Changed."
    with pytest.raises(Exception, match="tail|changed|hash"):
        stimuli.validate_development_sentinel(tokenizer, fixture)


def test_allowlisted_width_corruption_fails_production_width_gate(tokenizer):
    fixture = copy.deepcopy(_fixture())
    fixture["variants"]["W"]["messages"][6]["content"] += (
        " Additional mechanically asymmetric wording.")
    fixture["literal_history_pair_sha256"] = sha256(canonical_json_bytes({
        "C": fixture["variants"]["C"]["messages"],
        "W": fixture["variants"]["W"]["messages"],
    })).hexdigest()
    with pytest.raises(stimuli.V13StimulusError,
                       match="production content widths differ"):
        stimuli.validate_development_sentinel(tokenizer, fixture)
