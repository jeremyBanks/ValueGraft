import torch

from diagnose_bf16_attention_schedule import (
    Execution,
    cache_difference,
    causal_4d_mask,
    execution_difference,
    schedule_difference,
)


def _rows(values):
    tensor = torch.tensor(values, dtype=torch.bfloat16).reshape(1, 1, 2, 2)
    return [(tensor, tensor + 1)]


def test_independent_physical_causal_mask_geometry():
    mask = causal_4d_mask(
        q_len=3, kv_len=5, past_len=2,
        dtype=torch.bfloat16, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 3, 5)
    assert mask[0, 0, 0].tolist()[:3] == [0, 0, 0]
    assert mask[0, 0, 1].tolist()[:4] == [0, 0, 0, 0]
    assert mask[0, 0, 2].tolist() == [0, 0, 0, 0, 0]
    assert mask[0, 0, 0, 3] == torch.finfo(torch.bfloat16).min


def test_cache_difference_reports_per_layer_and_global_maxima():
    reference = _rows([0, 1, 2, 3])
    candidate = _rows([0, 1, 4, 3])
    result = cache_difference(reference, candidate)
    assert result["k_max_abs"] == 2.0
    assert result["v_max_abs"] == 2.0
    assert result["per_layer"] == [
        {"layer": 0, "k_max_abs": 2.0, "v_max_abs": 2.0}]


def test_schedule_comparison_uses_only_full_suffix_and_records_margin():
    full = Execution(
        logits=torch.tensor([[[100.0, -100.0], [50.0, -50.0],
                              [2.0, 1.0], [4.0, 1.0], [6.0, 1.0]]]),
        rows=_rows([0, 1, 2, 3]))
    split = Execution(
        logits=torch.tensor([[[2.0, 1.0], [4.0, 1.0], [5.0, 1.0]]]),
        rows=_rows([0, 1, 2, 3]))
    result = schedule_difference(full, split, (0, 1), cut=2)
    assert result["logits_max_abs"] == 1.0
    assert result["last_logits_max_abs"] == 1.0
    assert result["selected_margin_reference"] == 5.0
    assert result["selected_margin_candidate"] == 4.0
    assert result["selected_margin_abs_shift"] == 1.0
    assert result["split_query_lengths"] == [2, 3]


def test_identical_execution_is_exactly_zero():
    run = Execution(
        logits=torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16),
        rows=_rows([0, 1, 2, 3]))
    result = execution_difference(run, run, (0, 1))
    assert result["logits_max_abs"] == 0.0
    assert result["last_logits_max_abs"] == 0.0
    assert result["selected_margin_abs_shift"] == 0.0
    assert result["k_max_abs"] == 0.0
    assert result["v_max_abs"] == 0.0
