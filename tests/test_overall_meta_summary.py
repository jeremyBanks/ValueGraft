from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def load_overall():
    path = ROOT / "scripts" / "update_overall_meta_summary.py"
    spec = importlib.util.spec_from_file_location("overall_meta_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["overall_meta_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def init_repo(repo: Path) -> None:
    subprocess.check_call(["git", "init"], cwd=repo, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo)


def test_daily_summary_paths_selects_only_daily_meta_files(tmp_path: Path) -> None:
    overall = load_overall()
    notes = tmp_path / "notes"
    notes.mkdir()
    daily = notes / "20260708.md"
    ordinary = notes / "2026070801-ordinary.md"
    readme = notes / "README.md"
    for path in (daily, ordinary, readme):
        path.write_text(path.name, encoding="utf-8")

    assert overall.daily_summary_paths(notes) == [daily]


def test_overall_prompt_uses_source_day_headers_but_bans_copying(tmp_path: Path) -> None:
    overall = load_overall()
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    notes = repo / "notes"
    notes.mkdir()
    first = notes / "20260704.md"
    second = notes / "20260705.md"
    first.write_text("# First\n\nAlpha", encoding="utf-8")
    second.write_text("# Second\n\nBeta", encoding="utf-8")
    subprocess.check_call(["git", "add", "--", "notes/20260704.md", "notes/20260705.md"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "add daily"], cwd=repo, stdout=subprocess.DEVNULL)

    summaries = overall.load_daily_summaries(notes, repo)
    prompt = overall.build_prompt(summaries, repo, notes / "README.md")

    assert "## Source Day: 2026-07-04 UTC" in prompt
    assert "## Source Day: 2026-07-05 UTC" in prompt
    assert "do not copy those date headers" in prompt
    assert "do not structure your output as one section per day" in prompt
