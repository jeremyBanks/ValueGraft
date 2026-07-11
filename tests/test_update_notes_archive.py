from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "update_notes_archive.py"
    spec = importlib.util.spec_from_file_location("update_notes_archive_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["update_notes_archive_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    values = {
        "conversations": True,
        "normalize": True,
        "rollups": True,
        "summary_provider": "codex",
        "summary_model": "gpt-5.6-luna",
        "summary_reasoning": "medium",
        "summary_command": None,
        "force_small_continuations": False,
        "resummarize_all": False,
        "no_subagent_finals": False,
        "force_rollups": False,
        "forbid_regex": [],
        "dotenv": None,
        "no_default_forbid_regex": False,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_unified_command_propagates_provider_to_both_summary_levels() -> None:
    mod = load_module()

    stages = mod.build_stage_commands(args())

    assert [label for label, _cmd in stages] == [
        "conversation summaries",
        "archive filename normalization",
        "hierarchical rollups",
    ]
    conversation = stages[0][1]
    rollups = stages[2][1]
    for command in (conversation, rollups):
        assert "--summary-provider" in command
        assert "gpt-5.6-luna" in command
        assert "--summary-reasoning" in command


def test_stages_can_be_disabled() -> None:
    mod = load_module()

    stages = mod.build_stage_commands(args(conversations=False, rollups=False))

    assert stages == [
        ("archive filename normalization", ["python3", "scripts/normalize_notes_archive_names.py"])
    ]


def test_custom_command_is_split_only_for_conversation_remainder_args() -> None:
    mod = load_module()

    stages = mod.build_stage_commands(
        args(normalize=False, summary_command="python3 custom.py --model 'model name'")
    )

    assert stages[0][1][-5:] == ["--command", "python3", "custom.py", "--model", "model name"]
    assert stages[1][1][-2:] == ["--command", "python3 custom.py --model 'model name'"]


def test_full_regeneration_replaces_incremental_stage() -> None:
    mod = load_module()

    stages = mod.build_stage_commands(args(normalize=False, rollups=False, resummarize_all=True))

    assert [label for label, _command in stages] == ["conversation summaries full rebuild"]
    assert "--resummarize-all" in stages[0][1]


def test_custom_command_comes_after_all_conversation_updater_flags() -> None:
    mod = load_module()

    stages = mod.build_stage_commands(
        args(
            normalize=False,
            rollups=False,
            resummarize_all=True,
            dry_run=True,
            summary_command="python3 custom.py",
        )
    )

    command = stages[0][1]
    command_index = command.index("--command")
    assert command.index("--dry-run") < command_index
    assert command.index("--resummarize-all") < command_index
    assert command[command_index + 1 :] == ["python3", "custom.py"]
