import json
from pathlib import Path

import mlx.core as mx
import pytest

from local_n48_core import (
    LocalN48CoreError,
    arrays_bit_exact,
    build_value_alignment_pairs,
    snapshots_bit_exact,
)
from powered_v13_tokens import (
    build_fresh_destination_plan,
    build_role_native_plan,
)


REPO = Path(__file__).resolve().parents[1]
TOKENIZER_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--mlx-community--Qwen3-30B-A3B-Instruct-2507-4bit"
    / "snapshots/e9675aa3ca5f900ccef55267914466d55ab325fa"
)
FIXTURE_PATH = REPO / (
    "data/coherent_state_local_n48_v1/candidates/"
    "arithmetic_capacity_budget/"
    "rank-01_40863a11e9219da42ad5625ec3103def482b079b33ad1202ea15e81421f10fd3.json"
)
CARRIER_BANK = REPO / "data/coherent_state_local_n48_v2/fixed-carriers-v1.json"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    if not TOKENIZER_SNAPSHOT.is_dir():
        pytest.skip("exact local MLX tokenizer snapshot is not cached")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            TOKENIZER_SNAPSHOT,
            local_files_only=True,
        )
    except OSError:
        pytest.skip("exact local MLX tokenizer snapshot is unavailable")


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def carrier_text():
    bank = json.loads(CARRIER_BANK.read_text())
    carrier = next(row for row in bank["carriers"]
                   if row["carrier_id"] == "fixed_a")
    return carrier["text"]


def test_value_alignment_is_exact_visible_twin_partition(
        tokenizer, fixture, carrier_text):
    alignment = build_value_alignment_pairs(tokenizer, fixture, carrier_text)
    middle = fixture["middle_end_msg"]
    correct = build_role_native_plan(
        tokenizer,
        fixture["variants"]["C"]["messages"],
        middle_end_msg=middle,
        carrier_content=carrier_text,
    )
    wrong = build_role_native_plan(
        tokenizer,
        fixture["variants"]["W"]["messages"],
        middle_end_msg=middle,
        carrier_content=carrier_text,
    )
    fresh = build_fresh_destination_plan(
        tokenizer,
        fixture["variants"]["C"]["messages"],
        middle_end_msg=middle,
        carrier_content=carrier_text,
    )
    special_ids = set(tokenizer.all_special_ids)

    assert alignment.correct_pairs == alignment.wrong_pairs
    assert alignment.content_correct_pairs == alignment.content_wrong_pairs
    assert alignment.structural_correct_pairs == alignment.structural_wrong_pairs
    assert set(alignment.correct_pairs) == (
        set(alignment.content_correct_pairs)
        | set(alignment.structural_correct_pairs)
    )
    assert not (
        set(alignment.content_correct_pairs)
        & set(alignment.structural_correct_pairs)
    )
    assert alignment.content_correct_pairs
    assert alignment.structural_correct_pairs

    content_start, content_end = alignment.content_physical_interval
    expected_content = []
    for destination, source in alignment.correct_pairs:
        token_id = fresh.token_ids[destination]
        assert destination >= fresh.system_width
        assert token_id not in special_ids
        assert source == fresh.logical_positions[destination]
        assert correct.token_ids[source] == wrong.token_ids[source] == token_id
        if content_start <= destination < content_end:
            expected_content.append((destination, source))
    assert alignment.content_correct_pairs == expected_content
    assert len(alignment.content_correct_pairs) == 49
    assert all(row["partition"] in ("content", "structural")
               for row in alignment.rows)
    assert any(source > destination for destination, source
               in alignment.correct_pairs)


def test_value_alignment_records_packed_and_source_geometry(
        tokenizer, fixture, carrier_text):
    alignment = build_value_alignment_pairs(tokenizer, fixture, carrier_text)
    assert alignment.destination_logical_positions != list(range(
        len(alignment.destination_token_ids)))
    assert alignment.source_token_counts["C"] == alignment.source_token_counts["W"]
    assert alignment.source_token_counts["fresh"] == len(
        alignment.destination_token_ids)
    assert alignment.source_token_counts["C"] > alignment.source_token_counts["fresh"]
    assert all(row["destination_physical_position"] == pair[0]
               and row["source_logical_position"] == pair[1]
               for row, pair in zip(alignment.rows, alignment.correct_pairs))


def test_alignment_accepts_real_mlx_wrapper_shape(
        tokenizer, fixture, carrier_text):
    class Wrapper:
        def __init__(self, inner):
            self._tokenizer = inner

    direct = build_value_alignment_pairs(tokenizer, fixture, carrier_text)
    wrapped = build_value_alignment_pairs(Wrapper(tokenizer), fixture,
                                           carrier_text)
    assert wrapped.correct_pairs == direct.correct_pairs
    assert wrapped.destination_token_ids == direct.destination_token_ids


def _snapshot(value: float, *, offset: int = 1):
    keys = mx.array([[[[value]]]], dtype=mx.float32)
    values = mx.array([[[[2.0]]]], dtype=mx.float32)
    return [(keys, values, offset)]


def test_snapshot_comparison_uses_offsets_dtypes_shapes_and_raw_bits():
    assert snapshots_bit_exact(_snapshot(0.0), _snapshot(0.0))
    assert not snapshots_bit_exact(_snapshot(0.0), _snapshot(-0.0))
    assert not snapshots_bit_exact(_snapshot(0.0, offset=1),
                                   _snapshot(0.0, offset=2))

    bf16 = [(
        mx.array([[[[0.0]]]], dtype=mx.bfloat16),
        mx.array([[[[2.0]]]], dtype=mx.bfloat16),
        1,
    )]
    assert not snapshots_bit_exact(_snapshot(0.0), bf16)


def test_array_comparison_uses_dtypes_shapes_and_raw_bits():
    assert arrays_bit_exact(mx.array([[0.0]], dtype=mx.float32),
                            mx.array([[0.0]], dtype=mx.float32))
    assert not arrays_bit_exact(mx.array([[0.0]], dtype=mx.float32),
                                mx.array([[-0.0]], dtype=mx.float32))
    assert not arrays_bit_exact(mx.array([[0.0]], dtype=mx.float32),
                                mx.array([0.0], dtype=mx.float32))


def test_snapshot_comparison_rejects_malformed_offsets():
    keys = mx.zeros((1, 1, 2, 1), dtype=mx.float32)
    values = mx.zeros((1, 1, 2, 1), dtype=mx.float32)
    with pytest.raises(LocalN48CoreError, match="logical offset"):
        snapshots_bit_exact([(keys, values, 1)], [(keys, values, 1)])
