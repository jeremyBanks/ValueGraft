"""Frozen pre-text randomization records for powered successor v13.

The boundary is intentionally two-step:

1. write and commit eight OS-random stratum seeds at a clean pre-seed commit;
2. from that dedicated seed commit, write and separately commit every literal
   integer permutation.

Only compact tuples and stable IDs are handled here.  This module does not
import or call the conversation materializer.  Later replay validates the
persisted integers structurally and therefore does not depend on the installed
NumPy version.  A separate construction-time check replays PCG64 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from powered_v13_recipe import (
    DESIGN_ID,
    GENERATOR_VERSION,
    POOL_SIZE_PER_STRATUM,
    STRATA,
    TEMPLATE_VERSION,
    CandidateTuple,
    RankedMaterializationAuthorization,
    enumerate_candidate_tuples,
    stable_candidate_id,
)


SEED_SCHEMA = "coherent_state_powered_v13_permutation_seeds_v1"
PERMUTATION_SCHEMA = "coherent_state_powered_v13_literal_permutations_v1"
SEED_STATUS = "SEEDS_FROZEN_NO_PERMUTATION_NO_TEXT"
PERMUTATION_STATUS = "LITERAL_PERMUTATIONS_FROZEN_NO_TEXT"
SEED_MANIFEST_RELATIVE_PATH = (
    "data/coherent_state_powered_v13/permutation-seeds-v1.json")
LITERAL_PERMUTATION_RELATIVE_PATH = (
    "data/coherent_state_powered_v13/literal-permutations-v1.json")

# These are the exact selection/randomization inputs that become immutable at
# the seed boundary.  The later Stage-A static manifest is broader and binds
# every experiment-bearing runtime/release path.
PRESEED_INPUT_PATHS = (
    "COHERENT-STATE-POWERED-SUCCESSOR-V13-PREREGISTRATION.md",
    "data/coherent_state_powered_v13/recipe-foundation-v1.json",
    "pyproject.toml",
    "scripts/write_powered_v13_literal_permutations.py",
    "scripts/write_powered_v13_seed_manifest.py",
    "src/powered_v13_permutation.py",
    "src/powered_v13_recipe.py",
    "src/powered_v13_schema.py",
    "src/powered_v13_stimuli.py",
    "src/powered_v13_tokens.py",
    "uv.lock",
)

RANDOM_SOURCE = (
    "eight independently requested 128-bit operating-system random values; "
    "literal values are recorded below"
)
SEED_ENCODING = "exactly 32 lowercase hex digits; int(seed_hex,16)"
SEED_NUMPY_CONSTRUCTOR = (
    "numpy.random.Generator(numpy.random.PCG64(seed_int))")
PERMUTATION_NUMPY_CONSTRUCTOR = (
    "numpy.random.Generator(numpy.random.PCG64(int(seed_hex,16)))")
SEED_BOUNDARY = {
    "compact_tuple_enumeration_only": True,
    "literal_permutation_created": False,
    "ranked_conversation_text_materialized": False,
    "execution_authorized": False,
}
PERMUTATION_BOUNDARY = {
    "ranked_conversation_text_materialized": False,
    "production_tokenizer_gate_run_on_ranked_text": False,
    "content_review_run": False,
    "execution_authorized": False,
}

_SEED_HEX = re.compile(r"[0-9a-f]{32}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class V13PermutationError(ValueError):
    """A seed/permutation record violates the frozen pre-text law."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13PermutationError(message)


def _numpy_module():
    # Deliberately lazy: durable structural verification must remain usable
    # after NumPy upgrades or in a verifier environment without NumPy.
    try:
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise V13PermutationError(
            "NumPy is required only for seed/permutation construction replay") from exc
    return numpy


def _numpy_version() -> str:
    return str(_numpy_module().__version__)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str],
                        label: str) -> None:
    _require(set(value) == expected,
             f"{label} keys differ: expected {sorted(expected)}, "
             f"observed {sorted(value)}")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V13PermutationError(
            f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def validate_utc(value: object, label: str) -> str:
    _require(isinstance(value, str) and value.endswith("Z"),
             f"{label} must be RFC-3339 UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise V13PermutationError(f"{label} is invalid") from exc
    _require(parsed.utcoffset() == timezone.utc.utcoffset(parsed),
             f"{label} is not UTC")
    return value


def _validated_sha256(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{label} is not lowercase SHA-256")
    return value


def _validated_git_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _GIT_SHA.fullmatch(value) is not None,
             f"{label} is not a full lowercase git SHA")
    return value


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    return _load_json_mapping_bytes(path.read_bytes(), label)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V13PermutationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_mapping_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise V13PermutationError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def parse_json_mapping_bytes(raw: bytes, label: str) -> dict[str, Any]:
    """Strict JSON object loader with duplicate-key rejection."""

    return _load_json_mapping_bytes(raw, label)


def validate_seed_map(value: Mapping[str, object]) -> dict[str, str]:
    _require(isinstance(value, Mapping), "seeds must be a mapping")
    _require(set(value) == set(STRATA), "seed strata differ from frozen STRATA")
    result: dict[str, str] = {}
    for stratum in STRATA:
        seed = value[stratum]
        _require(isinstance(seed, str)
                 and _SEED_HEX.fullmatch(seed) is not None,
                 f"{stratum} seed is not 32 lowercase hex digits")
        result[stratum] = seed
    _require(len(set(result.values())) == len(STRATA),
             "per-stratum seeds are not unique")
    return result


def _ordered_pool_digests() -> dict[str, str]:
    result: dict[str, str] = {}
    for stratum in STRATA:
        ids = [stable_candidate_id(candidate)
               for candidate in enumerate_candidate_tuples(stratum)]
        _require(len(ids) == len(set(ids)) == POOL_SIZE_PER_STRATUM,
                 f"{stratum} compact pool differs")
        result[stratum] = sha256_bytes(canonical_json_bytes(ids))
    return result


def validate_input_inventory(
    value: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    _require(isinstance(value, list), "input inventory is not a list")
    _require(len(value) == len(PRESEED_INPUT_PATHS),
             "input inventory length differs")
    result: list[dict[str, str]] = []
    for index, expected_path in enumerate(PRESEED_INPUT_PATHS):
        record = value[index]
        _require(isinstance(record, Mapping),
                 f"input inventory record {index} is not a mapping")
        _require_exact_keys(record, {"path", "sha256"},
                            f"input inventory record {index}")
        _require(record.get("path") == expected_path,
                 f"input inventory path {index} differs")
        digest = _validated_sha256(
            record.get("sha256"), f"input inventory {expected_path}")
        result.append({"path": expected_path, "sha256": digest})
    return result


@dataclass(frozen=True)
class SeedManifestInput:
    observed_utc: str
    seeds: Mapping[str, object]
    preseed_git_commit: str
    input_inventory: Sequence[Mapping[str, object]]
    git_clean_before_write: bool


def build_seed_manifest(value: SeedManifestInput) -> dict[str, Any]:
    """Build only the first immutable OS-random record."""

    observed = validate_utc(value.observed_utc, "observed_utc")
    seeds = validate_seed_map(value.seeds)
    commit = _validated_git_sha(
        value.preseed_git_commit, "preseed_git_commit")
    inventory = validate_input_inventory(list(value.input_inventory))
    _require(value.git_clean_before_write is True,
             "seed manifest requires a clean pre-write tree")
    return {
        "schema": SEED_SCHEMA,
        "design_id": DESIGN_ID,
        "status": SEED_STATUS,
        "observed_utc": observed,
        "random_source": RANDOM_SOURCE,
        "seed_encoding": SEED_ENCODING,
        "numpy_constructor": SEED_NUMPY_CONSTRUCTOR,
        "numpy_version": _numpy_version(),
        "strata_order": list(STRATA),
        "seeds": seeds,
        "entropy": {
            "api": "secrets.token_bytes",
            "bytes_per_request": 16,
            "requests": len(STRATA),
            "stratum_order": list(STRATA),
        },
        "repository_boundary": {
            "branch": "trunk",
            "git_clean_before_write": True,
            "preseed_git_commit": commit,
        },
        "input_inventory": inventory,
        "pool_binding": {
            "template_version": TEMPLATE_VERSION,
            "generator_version": GENERATOR_VERSION,
            "pool_size_per_stratum": POOL_SIZE_PER_STRATUM,
            "ordered_candidate_ids_sha256": _ordered_pool_digests(),
        },
        "randomization_boundary": dict(SEED_BOUNDARY),
    }


def validate_seed_manifest(value: Mapping[str, Any]) -> dict[str, str]:
    _require(isinstance(value, Mapping), "seed manifest is not a mapping")
    _require_exact_keys(value, {
        "schema", "design_id", "status", "observed_utc", "random_source",
        "seed_encoding", "numpy_constructor", "numpy_version",
        "strata_order", "seeds", "entropy", "repository_boundary",
        "input_inventory", "pool_binding", "randomization_boundary",
    }, "seed manifest")
    _require(value.get("schema") == SEED_SCHEMA, "seed schema differs")
    _require(value.get("design_id") == DESIGN_ID, "seed design ID differs")
    _require(value.get("status") == SEED_STATUS, "seed status differs")
    validate_utc(value.get("observed_utc"), "observed_utc")
    _require(value.get("random_source") == RANDOM_SOURCE,
             "seed random source statement differs")
    _require(value.get("seed_encoding") == SEED_ENCODING,
             "seed encoding differs")
    _require(value.get("numpy_constructor") == SEED_NUMPY_CONSTRUCTOR,
             "seed NumPy constructor differs")
    _require(isinstance(value.get("numpy_version"), str)
             and bool(value["numpy_version"]), "seed NumPy version is missing")
    _require(value.get("strata_order") == list(STRATA),
             "seed stratum order differs")
    seeds = validate_seed_map(value.get("seeds", {}))
    _require(value.get("entropy") == {
        "api": "secrets.token_bytes",
        "bytes_per_request": 16,
        "requests": len(STRATA),
        "stratum_order": list(STRATA),
    }, "seed entropy provenance differs")

    repository = value.get("repository_boundary")
    _require(isinstance(repository, Mapping),
             "seed repository boundary is missing")
    _require_exact_keys(repository, {
        "branch", "git_clean_before_write", "preseed_git_commit",
    }, "seed repository boundary")
    _require(repository.get("branch") == "trunk",
             "seed repository branch differs")
    _require(repository.get("git_clean_before_write") is True,
             "seed clean-tree attestation differs")
    _validated_git_sha(
        repository.get("preseed_git_commit"), "preseed_git_commit")
    validate_input_inventory(value.get("input_inventory", []))

    pool = value.get("pool_binding")
    _require(isinstance(pool, Mapping), "seed pool binding is missing")
    _require_exact_keys(pool, {
        "template_version", "generator_version", "pool_size_per_stratum",
        "ordered_candidate_ids_sha256",
    }, "seed pool binding")
    _require(pool.get("template_version") == TEMPLATE_VERSION,
             "seed template version differs")
    _require(pool.get("generator_version") == GENERATOR_VERSION,
             "seed generator version differs")
    _require(pool.get("pool_size_per_stratum") == POOL_SIZE_PER_STRATUM,
             "seed pool size differs")
    digests = pool.get("ordered_candidate_ids_sha256")
    _require(isinstance(digests, Mapping) and set(digests) == set(STRATA),
             "seed ordered pool digest strata differ")
    for stratum in STRATA:
        _validated_sha256(digests[stratum], f"{stratum} pool digest")
    _require(dict(digests) == _ordered_pool_digests(),
             "seed ordered pool digests differ from compact frame")
    _require(value.get("randomization_boundary") == SEED_BOUNDARY,
             "seed randomization boundary differs")
    return seeds


def _candidate_record(candidate: CandidateTuple, *, compact_index: int,
                      rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "compact_index": compact_index,
        "stable_candidate_id": stable_candidate_id(candidate),
        "canonical_tuple": candidate.canonical_object(),
    }


def _permutation_indices(seed_hex: str) -> list[int]:
    _require(isinstance(seed_hex, str)
             and _SEED_HEX.fullmatch(seed_hex) is not None,
             "seed is not 32 lowercase hex digits")
    np = _numpy_module()
    values = np.random.Generator(np.random.PCG64(
        int(seed_hex, 16))).permutation(POOL_SIZE_PER_STRATUM)
    return [int(value) for value in values.tolist()]


def build_stratum_permutation(stratum: str, seed_hex: str) -> dict[str, Any]:
    _require(stratum in STRATA, f"unknown stratum {stratum!r}")
    candidates = list(enumerate_candidate_tuples(stratum))
    _require(len(candidates) == POOL_SIZE_PER_STRATUM,
             f"{stratum} compact pool size differs")
    compact_indices = _permutation_indices(seed_hex)
    _require(sorted(compact_indices) == list(range(POOL_SIZE_PER_STRATUM)),
             f"{stratum} output is not a literal permutation")
    ids = [stable_candidate_id(candidates[index]) for index in compact_indices]
    _require(len(ids) == len(set(ids)) == POOL_SIZE_PER_STRATUM,
             f"{stratum} stable IDs collide")
    return {
        "stratum": stratum,
        "seed_hex": seed_hex,
        "seed_int_decimal": str(int(seed_hex, 16)),
        "pool_size": POOL_SIZE_PER_STRATUM,
        "compact_indices": compact_indices,
        "compact_indices_sha256": sha256_bytes(
            canonical_json_bytes(compact_indices)),
        "ranked_candidate_ids": ids,
        "ranked_candidate_ids_sha256": sha256_bytes(
            canonical_json_bytes(ids)),
        "first_ten": [
            _candidate_record(
                candidates[index], compact_index=index, rank=rank)
            for rank, index in enumerate(compact_indices[:10], start=1)
        ],
    }


def build_literal_permutations(
    seed_manifest_raw: bytes,
    *,
    seed_git_commit: str,
    observed_utc: str,
) -> dict[str, Any]:
    """Build the second record from the actual committed seed-file bytes."""

    _require(isinstance(seed_manifest_raw, bytes),
             "seed manifest input must be exact raw bytes")
    seed_manifest = _load_json_mapping_bytes(
        seed_manifest_raw, "seed manifest")
    seeds = validate_seed_manifest(seed_manifest)
    _require(seed_manifest.get("numpy_version") == _numpy_version(),
             "seed NumPy version differs from permutation build runtime")
    seed_commit = _validated_git_sha(seed_git_commit, "seed_git_commit")
    observed = validate_utc(observed_utc, "observed_utc")
    strata = {
        stratum: build_stratum_permutation(stratum, seeds[stratum])
        for stratum in STRATA
    }
    all_ids = [
        identifier
        for stratum in STRATA
        for identifier in strata[stratum]["ranked_candidate_ids"]
    ]
    _require(len(all_ids) == len(set(all_ids)) ==
             len(STRATA) * POOL_SIZE_PER_STRATUM,
             "global stable candidate IDs collide")
    return {
        "schema": PERMUTATION_SCHEMA,
        "design_id": DESIGN_ID,
        "status": PERMUTATION_STATUS,
        "observed_utc": observed,
        "numpy_version": _numpy_version(),
        "numpy_constructor": PERMUTATION_NUMPY_CONSTRUCTOR,
        "strata_order": list(STRATA),
        "pool_size_per_stratum": POOL_SIZE_PER_STRATUM,
        "global_candidate_count": len(all_ids),
        "global_ranked_candidate_ids_sha256": sha256_bytes(
            canonical_json_bytes(all_ids)),
        "seed_binding": {
            "path": SEED_MANIFEST_RELATIVE_PATH,
            "sha256": sha256_bytes(seed_manifest_raw),
            "git_commit": seed_commit,
        },
        "strata": strata,
        "randomization_boundary": dict(PERMUTATION_BOUNDARY),
    }


def _candidate_from_record(stratum: str, value: Mapping[str, Any]) -> CandidateTuple:
    canonical = value.get("canonical_tuple")
    _require(isinstance(canonical, Mapping), "canonical tuple is not a mapping")
    _require_exact_keys(canonical, {
        "resolution_mode_id", "domain_skin_id", "label_pair_id",
        "logic_pack_id", "nonfocal_tail_pack_id",
    }, "canonical tuple")
    try:
        return CandidateTuple(stratum_id=stratum, **dict(canonical))
    except TypeError as exc:
        raise V13PermutationError(f"canonical tuple is invalid: {exc}") from exc


def _validate_stratum_literal(
    stratum: str,
    value: Mapping[str, Any],
    seed_hex: str,
) -> list[str]:
    _require(isinstance(value, Mapping), f"{stratum} record is not a mapping")
    _require_exact_keys(value, {
        "stratum", "seed_hex", "seed_int_decimal", "pool_size",
        "compact_indices", "compact_indices_sha256",
        "ranked_candidate_ids", "ranked_candidate_ids_sha256", "first_ten",
    }, f"{stratum} record")
    _require(value.get("stratum") == stratum, f"{stratum} name differs")
    _require(value.get("seed_hex") == seed_hex, f"{stratum} seed differs")
    _require(value.get("seed_int_decimal") == str(int(seed_hex, 16)),
             f"{stratum} decimal seed differs")
    _require(value.get("pool_size") == POOL_SIZE_PER_STRATUM,
             f"{stratum} pool size differs")
    indices = value.get("compact_indices")
    _require(isinstance(indices, list)
             and all(isinstance(item, int) and not isinstance(item, bool)
                     for item in indices),
             f"{stratum} compact indices are invalid")
    _require(len(indices) == POOL_SIZE_PER_STRATUM
             and sorted(indices) == list(range(POOL_SIZE_PER_STRATUM)),
             f"{stratum} compact indices are not a complete permutation")
    _require(value.get("compact_indices_sha256") == sha256_bytes(
        canonical_json_bytes(indices)), f"{stratum} index hash differs")

    candidates = list(enumerate_candidate_tuples(stratum))
    observed_ids = value.get("ranked_candidate_ids")
    _require(isinstance(observed_ids, list)
             and len(observed_ids) == POOL_SIZE_PER_STRATUM
             and all(isinstance(item, str) for item in observed_ids),
             f"{stratum} ranked candidate IDs differ")
    ids = [stable_candidate_id(candidates[index]) for index in indices]
    _require(observed_ids == ids, f"{stratum} ranked candidate IDs differ")
    first_ten = value.get("first_ten")
    _require(isinstance(first_ten, list) and len(first_ten) == 10,
             f"{stratum} first-ten records differ in length")
    for rank, (compact_index, row) in enumerate(
            zip(indices[:10], first_ten, strict=True), start=1):
        _require(isinstance(row, Mapping),
                 f"{stratum} rank {rank} is not a mapping")
        _require_exact_keys(row, {
            "rank", "compact_index", "stable_candidate_id", "canonical_tuple",
        }, f"{stratum} rank {rank}")
        _require(row.get("rank") == rank, f"{stratum} rank field differs")
        _require(row.get("compact_index") == compact_index,
                 f"{stratum} compact index/rank differs")
        candidate = _candidate_from_record(stratum, row)
        _require(candidate == candidates[compact_index],
                 f"{stratum} rank {rank} tuple/index binding differs")
        identifier = stable_candidate_id(candidate)
        _require(row.get("stable_candidate_id") == identifier,
                 f"{stratum} rank {rank} candidate ID differs")
    _require(len(ids) == len(set(ids)) == POOL_SIZE_PER_STRATUM,
             f"{stratum} ranked IDs collide")
    _require(value.get("ranked_candidate_ids_sha256") == sha256_bytes(
        canonical_json_bytes(ids)), f"{stratum} ranked ID hash differs")
    return ids


def validate_literal_permutations(
    value: Mapping[str, Any],
    seed_manifest_raw: bytes,
    *,
    expected_seed_git_commit: str,
) -> None:
    """Validate persisted integers without regenerating RNG output."""

    _require(isinstance(value, Mapping),
             "literal permutation manifest is not a mapping")
    _require_exact_keys(value, {
        "schema", "design_id", "status", "observed_utc", "numpy_version",
        "numpy_constructor", "strata_order", "pool_size_per_stratum",
        "global_candidate_count", "global_ranked_candidate_ids_sha256",
        "seed_binding", "strata", "randomization_boundary",
    }, "literal permutation manifest")
    _require(value.get("schema") == PERMUTATION_SCHEMA,
             "literal permutation schema differs")
    _require(value.get("design_id") == DESIGN_ID,
             "literal permutation design ID differs")
    _require(value.get("status") == PERMUTATION_STATUS,
             "literal permutation status differs")
    validate_utc(value.get("observed_utc"), "observed_utc")
    _require(isinstance(value.get("numpy_version"), str)
             and bool(value["numpy_version"]),
             "literal NumPy version is missing")
    _require(value.get("numpy_constructor") == PERMUTATION_NUMPY_CONSTRUCTOR,
             "literal NumPy constructor differs")
    _require(value.get("strata_order") == list(STRATA),
             "literal stratum order differs")
    _require(value.get("pool_size_per_stratum") == POOL_SIZE_PER_STRATUM,
             "literal pool size differs")

    _require(isinstance(seed_manifest_raw, bytes),
             "seed manifest input must be exact raw bytes")
    seed_manifest = _load_json_mapping_bytes(
        seed_manifest_raw, "seed manifest")
    seeds = validate_seed_manifest(seed_manifest)
    _require(value.get("numpy_version") == seed_manifest.get("numpy_version"),
             "literal and seed NumPy versions differ")
    binding = value.get("seed_binding")
    _require(isinstance(binding, Mapping), "literal seed binding is missing")
    _require_exact_keys(binding, {"path", "sha256", "git_commit"},
                        "literal seed binding")
    _require(binding.get("path") == SEED_MANIFEST_RELATIVE_PATH,
             "literal seed manifest path binding differs")
    _require(binding.get("sha256") == sha256_bytes(seed_manifest_raw),
             "literal seed manifest SHA binding differs")
    seed_commit = _validated_git_sha(
        binding.get("git_commit"), "literal seed binding git_commit")
    _require(seed_commit == _validated_git_sha(
        expected_seed_git_commit, "expected_seed_git_commit"),
        "literal seed git commit binding differs")

    strata = value.get("strata")
    _require(isinstance(strata, Mapping) and set(strata) == set(STRATA),
             "literal permutation strata differ")
    all_ids: list[str] = []
    for stratum in STRATA:
        all_ids.extend(_validate_stratum_literal(
            stratum, strata[stratum], seeds[stratum]))
    _require(value.get("global_candidate_count") == len(all_ids) ==
             len(STRATA) * POOL_SIZE_PER_STRATUM,
             "global candidate count differs")
    _require(len(set(all_ids)) == len(all_ids),
             "global ranked candidate IDs collide")
    _require(value.get("global_ranked_candidate_ids_sha256") == sha256_bytes(
        canonical_json_bytes(all_ids)), "global candidate ID hash differs")
    _require(value.get("randomization_boundary") == PERMUTATION_BOUNDARY,
             "literal permutation boundary differs")


def validate_construction_rng_replay(
    value: Mapping[str, Any],
    seed_manifest_raw: bytes,
    *,
    expected_seed_git_commit: str,
) -> None:
    """Construction-only exact PCG64 replay under the pinned NumPy version."""

    validate_literal_permutations(
        value, seed_manifest_raw,
        expected_seed_git_commit=expected_seed_git_commit)
    seed_manifest = _load_json_mapping_bytes(
        seed_manifest_raw, "seed manifest")
    _require(seed_manifest.get("numpy_version") == _numpy_version(),
             "seed NumPy version differs from RNG replay runtime")
    _require(value.get("numpy_version") == _numpy_version(),
             "literal NumPy version differs from RNG replay runtime")
    seeds = validate_seed_manifest(seed_manifest)
    for stratum in STRATA:
        expected = _permutation_indices(seeds[stratum])
        observed = value["strata"][stratum]["compact_indices"]
        _require(observed == expected,
                 f"{stratum} PCG64 construction replay differs")


def ranked_materialization_authorization(
    literal_permutation_path: Path,
    seed_manifest_path: Path,
    *,
    stratum: str,
    rank: int,
    expected_seed_git_commit: str,
) -> tuple[CandidateTuple, RankedMaterializationAuthorization]:
    """Bind one compact tuple/rank to both actual immutable file hashes."""

    _require(stratum in STRATA, "authorization stratum is invalid")
    _require(isinstance(rank, int) and not isinstance(rank, bool)
             and 1 <= rank <= 10,
             "authorization rank is invalid")
    literal = _load_json_mapping(
        literal_permutation_path, "literal permutations")
    seed_raw = seed_manifest_path.read_bytes()
    validate_literal_permutations(
        literal, seed_raw,
        expected_seed_git_commit=expected_seed_git_commit)
    row = literal["strata"][stratum]["first_ten"][rank - 1]
    candidate = _candidate_from_record(stratum, row)
    identifier = stable_candidate_id(candidate)
    _require(identifier == row["stable_candidate_id"],
             "authorization candidate ID differs")
    return candidate, RankedMaterializationAuthorization(
        status="PERMUTATION_COMMITTED_RANKED_CANDIDATE",
        candidate_id=identifier,
        permutation_rank=rank,
        seed_manifest_sha256=sha256_file(seed_manifest_path),
        seed_git_commit=expected_seed_git_commit,
        literal_permutation_sha256=sha256_file(literal_permutation_path),
    )


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "LITERAL_PERMUTATION_RELATIVE_PATH",
    "PERMUTATION_SCHEMA",
    "PRESEED_INPUT_PATHS",
    "SEED_MANIFEST_RELATIVE_PATH",
    "SEED_SCHEMA",
    "SeedManifestInput",
    "V13PermutationError",
    "build_literal_permutations",
    "build_seed_manifest",
    "build_stratum_permutation",
    "canonical_json_bytes",
    "parse_json_mapping_bytes",
    "ranked_materialization_authorization",
    "sha256_bytes",
    "sha256_file",
    "validate_construction_rng_replay",
    "validate_input_inventory",
    "validate_literal_permutations",
    "validate_seed_manifest",
    "validate_seed_map",
    "write_json_exclusive",
]
