from types import SimpleNamespace

import pytest
import torch

from l_coherent_state_hf import (
    _require_summary_boundary,
    _validate_exact_length_wrong,
    _verify_intervention,
)


def _snapshot(offset=0.0):
    key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2) + offset
    value = key + 100
    return [(key, value)]


def test_summary_boundary_rejects_precomputed_tail():
    layout = SimpleNamespace(physical_summary_end=4)
    _require_summary_boundary(_snapshot(), layout)
    tailed = [(torch.zeros(1, 1, 5, 2), torch.zeros(1, 1, 5, 2))]
    with pytest.raises(RuntimeError, match="fork exactly at summary boundary"):
        _require_summary_boundary(tailed, layout)


def test_exact_insert_preserves_every_non_summary_row():
    fresh = _snapshot()
    source = [(torch.full((1, 1, 2, 2), 9.0),
               torch.full((1, 1, 2, 2), 19.0))]
    treated = [(fresh[0][0].clone(), fresh[0][1].clone())]
    treated[0][0][..., 1:3, :] = source[0][0]
    treated[0][1][..., 1:3, :] = source[0][1]
    _verify_intervention(
        fresh, treated, source, 1, use_keys=True, use_values=True)

    treated[0][1][..., 3, :] += 1
    with pytest.raises(RuntimeError, match="non-summary rows changed"):
        _verify_intervention(
            fresh, treated, source, 1, use_keys=True, use_values=True)


def test_exact_length_wrong_rejects_structure_and_special_content():
    correct = [10, 11, 12, 13]
    wrong = [10, 21, 22, 13]
    _validate_exact_length_wrong(correct, wrong, [0, 3], [1, 2], [99])

    with pytest.raises(RuntimeError, match="altered structure"):
        _validate_exact_length_wrong(
            correct, [14, 21, 22, 13], [0, 3], [1, 2], [99])
    with pytest.raises(RuntimeError, match="special token"):
        _validate_exact_length_wrong(
            correct, [10, 99, 22, 13], [0, 3], [1, 2], [99])
