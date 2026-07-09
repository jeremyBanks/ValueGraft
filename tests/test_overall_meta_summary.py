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


def test_source_days_ignores_generated_rollups(tmp_path: Path) -> None:
    overall = load_overall()
    notes = tmp_path / "notes"
    notes.mkdir()
    daily = notes / "20260708.md"
    monthly = notes / "202607.md"
    yearly = notes / "2026.md"
    ordinary = notes / "2026070801-ordinary.md"
    readme = notes / "README.md"
    for path in (daily, monthly, yearly, ordinary, readme):
        path.write_text(path.name, encoding="utf-8")

    assert overall.source_days(notes) == ["20260708"]


def test_hierarchy_creates_month_but_promotes_readme_for_single_month(tmp_path: Path) -> None:
    overall = load_overall()
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    notes = repo / "notes"
    notes.mkdir()
    for name in [
        "2026070401-alpha.md",
        "2026070402-beta.md",
        "2026070501-gamma.md",
        "2026070502-delta.md",
    ]:
        (notes / name).write_text(name, encoding="utf-8")
    (notes / "20260704.md").write_text("# Day 1\n\nAlpha", encoding="utf-8")
    (notes / "20260705.md").write_text("# Day 2\n\nBeta", encoding="utf-8")
    subprocess.check_call(["git", "add", "--", "notes"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "add notes"], cwd=repo, stdout=subprocess.DEVNULL)

    plans, expected = overall.build_plans(notes, repo, notes / "README.md")

    assert [plan.path.name for plan in plans] == ["202607.md", "README.md"]
    assert plans[0].mode == "summary"
    assert plans[0].level == "month"
    assert [source.path.name for source in plans[0].sources] == ["20260704.md", "20260705.md"]
    assert plans[1].mode == "promote"
    assert plans[1].sources[0].path.name == "202607.md"
    assert {path.name for path in expected} == {"202607.md", "README.md"}


def test_rollup_prompt_uses_immediate_sources_and_bans_sources_section(tmp_path: Path) -> None:
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
    sources = [
        overall.load_source(first, "20260704", "day", repo),
        overall.load_source(second, "20260705", "day", repo),
    ]
    plan = overall.RollupPlan(notes / "202607.md", "202607", "month", "summary", sources)

    prompt = overall.build_prompt(plan, repo)

    assert "You are writing notes/202607.md" in prompt
    assert "## Source: notes/20260704.md" in prompt
    assert "## Source: notes/20260705.md" in prompt
    assert "Do not include a `Sources` section" in prompt
    assert "Your first non-whitespace character must be `#`" in prompt


def test_rollup_validator_rejects_tool_like_output() -> None:
    overall = load_overall()

    assert overall.invalid_rollup_reason("# Title\n\nBody") is None
    assert "heading" in overall.invalid_rollup_reason("I'll check AGENTS.md first.")
    assert "tool-call JSON" in overall.invalid_rollup_reason(
        '# Title\n\n{"path":"/tmp/file","output_mode":"content"}'
    )


def test_refresh_plan_sources_updates_promoted_readme_source(tmp_path: Path) -> None:
    overall = load_overall()
    repo = tmp_path / "repo"
    repo.mkdir()
    notes = repo / "notes"
    notes.mkdir()
    month = notes / "202607.md"
    month.write_text("# Old\n", encoding="utf-8")
    readme = notes / "README.md"
    plans = [
        overall.RollupPlan(month, "202607", "month", "summary", []),
        overall.RollupPlan(
            readme,
            "overall",
            "readme",
            "promote",
            [overall.RollupSource(month, "202607", "month", "# Old\n", "old")],
        ),
    ]

    month.write_text("# New\n", encoding="utf-8")
    refreshed = overall.refresh_plan_sources(plans, repo)

    assert refreshed[1].sources[0].text == "# New\n"
