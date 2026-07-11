from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def load_filters():
    path = ROOT / "scripts" / "notes_summary_filters.py"
    spec = importlib.util.spec_from_file_location("notes_filters_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["notes_filters_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dotenv_forbid_regex_is_loaded(tmp_path: Path, monkeypatch) -> None:
    filters = load_filters()
    dotenv = tmp_path / ".env"
    dotenv.write_text("VALUEGRAFT_NOTES_FORBID_REGEX='alpha|beta'\n", encoding="utf-8")
    monkeypatch.delenv("VALUEGRAFT_NOTES_FORBID_REGEX", raising=False)
    monkeypatch.delenv("NOTES_FORBID_REGEX", raising=False)

    patterns = filters.resolve_forbid_patterns([], tmp_path, include_defaults=False, dotenv_paths=[dotenv])

    assert patterns == ["alpha|beta"]


def test_environment_forbid_regex_overrides_dotenv(tmp_path: Path, monkeypatch) -> None:
    filters = load_filters()
    dotenv = tmp_path / ".env"
    dotenv.write_text("VALUEGRAFT_NOTES_FORBID_REGEX=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("VALUEGRAFT_NOTES_FORBID_REGEX", "from-env")

    patterns = filters.resolve_forbid_patterns([], tmp_path, include_defaults=False, dotenv_paths=[dotenv])

    assert patterns == ["from-env"]


def test_forbidden_retry_prompt_uses_vague_rewrite_instruction() -> None:
    filters = load_filters()
    matches = [filters.ForbiddenMatch(pattern="secret", text="secret phrase")]

    prompt = filters.retry_prompt_for_forbidden_matches("Original prompt", matches)

    assert "secret phrase" in prompt
    assert "underlying referent" in prompt
    assert "not a word-ban or synonym substitution exercise" in prompt
    assert "narrative clues" in prompt
    assert "Original prompt" in prompt


def test_source_match_adds_referent_guard_before_first_generation() -> None:
    filters = load_filters()
    seen_prompts: list[str] = []
    filters.run_command = lambda _command, prompt: seen_prompts.append(prompt) or "Safe summary.\n"

    result = filters.run_filtered_summary_command(
        "unused",
        "Summarize discussion of secret phrase.",
        [r"secret phrase"],
        2,
        "test",
    )

    assert result == "Safe summary.\n"
    assert len(seen_prompts) == 1
    assert "underlying referent" in seen_prompts[0]
    assert "Do not play taboo" in seen_prompts[0]
