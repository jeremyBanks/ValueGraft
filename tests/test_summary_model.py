from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "summary_model.py"
    spec = importlib.util.spec_from_file_location("summary_model_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["summary_model_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_is_luna_medium() -> None:
    mod = load_module()

    spec = mod.resolve_spec("codex", None, None)

    assert spec.provider == "codex"
    assert spec.model == "gpt-5.6-luna"
    assert spec.reasoning == "medium"
    assert spec.provenance() == {
        "provider": "codex",
        "model": "gpt-5.6-luna",
        "reasoning": "medium",
    }


def test_claude_defaults_to_sonnet_without_reasoning() -> None:
    mod = load_module()

    spec = mod.resolve_spec("claude", None, "xhigh")

    assert spec.model == "sonnet"
    assert spec.reasoning is None


def test_wrapper_command_round_trips_spaces() -> None:
    mod = load_module()
    spec = mod.SummaryModelSpec("codex", "model with spaces", "high")

    command = mod.wrapper_command(spec)

    assert "'model with spaces'" in command


def test_custom_command_provenance_does_not_store_command() -> None:
    mod = load_module()

    provenance = mod.custom_command_provenance("provider --token secret")

    assert provenance["provider"] == "custom"
    assert set(provenance) == {"provider", "command_sha256"}
