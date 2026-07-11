from types import SimpleNamespace

import pytest
import torch

from coherent_state_hf import CoherentStateError
from coherent_state_runtime import (
    AMENDMENT_ID,
    DESIGN_ID,
    GAPPED_ARM_NAMES,
    eager_backend_fingerprint,
    gapped_arm_boundary,
)
from l_coherent_state_hf import FROZEN_SCHEDULES


class FakeAttention(torch.nn.Module):
    def __init__(self, layer_idx, implementation=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj = torch.nn.Identity()
        self.k_proj = torch.nn.Identity()
        implementation = implementation or "eager"
        self.config = SimpleNamespace(
            _attn_implementation=implementation,
            _attn_implementation_internal=implementation,
        )


class FakeModel(torch.nn.Module):
    def __init__(self, backend="eager", local=None):
        super().__init__()
        self.config = SimpleNamespace(
            _attn_implementation=backend,
            _attn_implementation_internal=backend,
            num_hidden_layers=2,
        )
        self.layers = torch.nn.ModuleList([
            FakeAttention(0, local), FakeAttention(1, local)])


def _snapshot(rows=4):
    x = torch.arange(rows * 2, dtype=torch.float32).reshape(1, 1, rows, 2)
    return [(x.clone(), (x + 10).clone())]


def test_v8_identity_and_exact_six_arms_are_frozen():
    assert AMENDMENT_ID == "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7-8-9-10"
    assert DESIGN_ID == "coherent-state-gapped-v10"
    assert GAPPED_ARM_NAMES == (
        "A_full", "G_fresh", "G_correct", "G_wrong",
        "G_Vcorrect", "G_Kcorrect",
    )
    assert [x[0] for x in FROZEN_SCHEDULES] == [5, 64, 900, 4096, 4097, 8193]


def test_eager_backend_fingerprint_is_complete_stable_and_serializable():
    first = eager_backend_fingerprint(FakeModel())
    second = eager_backend_fingerprint(FakeModel())
    assert first == second
    assert first["expected_layer_count"] == 2
    assert first["model_config"]["scope"] == "model_config"
    assert first["text_config"]["scope"] == "text_config"
    assert [x["layer_index"] for x in first["layers"]] == [0, 1]
    assert {x["resolved_implementation"] for x in first["layers"]} == {"eager"}
    assert len(first["sha256"]) == 64


def test_backend_fingerprint_rejects_non_eager_and_heterogeneous_layers():
    with pytest.raises(CoherentStateError, match="non-eager"):
        eager_backend_fingerprint(FakeModel("sdpa"))
    model = FakeModel("eager")
    model.layers[1].config._attn_implementation = "sdpa"
    with pytest.raises(CoherentStateError, match="non-eager"):
        eager_backend_fingerprint(model)


def test_g_delta_is_explicitly_retired_fail_closed():
    fresh = _snapshot()
    rows = [(fresh[0][0][..., :2, :], fresh[0][1][..., :2, :])]
    with pytest.raises(CoherentStateError, match="retired by Amendment 4"):
        gapped_arm_boundary("G_delta", fresh, rows, rows, 2, 17)
