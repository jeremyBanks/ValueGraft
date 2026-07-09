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
    assert "vague, generic phrasing" in prompt
    assert "Original prompt" in prompt
