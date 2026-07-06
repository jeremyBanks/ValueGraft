"""E1 synthetic coding tasks: small repos with failing tests; the task is to
make the tests pass. Designed so success REQUIRES facts established early in
the session (the agent is told constraints up front, then must apply them
after exploration pushes context past the compaction threshold).

Commands: materialize <task> <dir> | prompt <task> | score <task> <dir> <log>
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Each task: files, a constraints preamble (the compactable early context),
# and tests that check constraint-consistent behavior.
TASKS = {
    "t1": {
        "prompt": (
            "You are working in the repo at the current directory. "
            "IMPORTANT project constraints (memorize; they are checked): "
            "(1) all public function names must be snake_case verbs starting "
            "with 'compute_'; (2) currency amounts are integer CENTS, never "
            "floats; (3) the tax rate is 8.25% and lives ONLY in "
            "config.py as TAX_RATE_BP = 825 (basis points). "
            "First read every file under src/ and tests/ to understand the "
            "codebase. Then implement the missing module src/pricing.py so "
            "that `python -m pytest tests/ -q` passes. Do not modify tests."
        ),
        "files": {
            "config.py": "TAX_RATE_BP = 825\n",
            "src/__init__.py": "",
            "src/cart.py": (
                "from dataclasses import dataclass\n\n"
                "@dataclass\nclass Item:\n"
                "    name: str\n    unit_cents: int\n    qty: int\n"
            ),
            "tests/test_pricing.py": (
                "from src.cart import Item\n"
                "from src.pricing import compute_subtotal, compute_total\n"
                "import config\n\n"
                "def test_subtotal():\n"
                "    items = [Item('a', 199, 3), Item('b', 1050, 1)]\n"
                "    assert compute_subtotal(items) == 1647\n\n"
                "def test_total_uses_config_bp():\n"
                "    items = [Item('a', 10000, 1)]\n"
                "    assert compute_total(items) == 10000 + 825\n\n"
                "def test_int_cents():\n"
                "    items = [Item('a', 333, 3)]\n"
                "    t = compute_total(items)\n"
                "    assert isinstance(t, int)\n"
            ),
        },
        "filler_files": 6,
    },
    "t2": {
        "prompt": (
            "You are working in the repo at the current directory. "
            "IMPORTANT constraints (checked later): (1) the log format is "
            "exactly 'LEVEL|epoch_seconds|message' — pipes, no spaces; "
            "(2) retries use exponential backoff base 250ms doubling, max 3 "
            "attempts total; (3) timeouts must raise our custom FetchTimeout "
            "from src/errors.py, never builtin TimeoutError. "
            "First read every file under src/ and tests/. Then implement "
            "src/fetcher.py so `python -m pytest tests/ -q` passes. "
            "Do not modify tests."
        ),
        "files": {
            "src/__init__.py": "",
            "src/errors.py": "class FetchTimeout(Exception):\n    pass\n",
            "tests/test_fetcher.py": (
                "import src.fetcher as f\n"
                "from src.errors import FetchTimeout\n\n"
                "def test_format_log():\n"
                "    s = f.format_log('INFO', 1700000000, 'ok')\n"
                "    assert s == 'INFO|1700000000|ok'\n\n"
                "def test_backoff_schedule():\n"
                "    assert f.backoff_ms(0) == 250\n"
                "    assert f.backoff_ms(1) == 500\n"
                "    assert f.backoff_ms(2) == 1000\n\n"
                "def test_timeout_type():\n"
                "    import pytest\n"
                "    with pytest.raises(FetchTimeout):\n"
                "        f.fetch_with_retry(lambda: (_ for _ in ()).throw("
                "TimeoutError()), max_attempts=3)\n"
            ),
        },
        "filler_files": 6,
    },
}

FILLER = (
    "\"\"\"Module {i}: utility helpers (read-only context filler).\"\"\"\n"
    + "".join(f"def helper_{{i}}_{j}(x):\n"
              f"    'Helper {j}: transforms x by rule {j}.'\n"
              f"    return x + {j}\n\n" for j in range(30))
)


def materialize(task, d):
    t = TASKS[task]
    root = Path(d)
    for rel, content in t["files"].items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    for i in range(t["filler_files"]):
        p = root / "src" / f"util_{i}.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(FILLER.replace("{i}", str(i)))
    (root / "pytest.ini").write_text("[pytest]\npythonpath = .\n")


def score(task, d, log):
    r = subprocess.run(["python", "-m", "pytest", "tests/", "-q",
                        "--no-header", "-x"],
                       cwd=d, capture_output=True, text=True, timeout=300)
    passed = r.returncode == 0
    logtxt = Path(log).read_text(errors="replace") if os.path.exists(log) \
        else ""
    steps = len(re.findall(r"ACTION", logtxt))
    repeats = len(re.findall(r"same command|already (?:ran|tried)", logtxt,
                             re.I))
    return {"task": task, "tests_pass": passed, "pytest_tail":
            (r.stdout or r.stderr)[-300:], "agent_steps": steps,
            "repeat_signals": repeats}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "materialize":
        materialize(sys.argv[2], sys.argv[3])
    elif cmd == "prompt":
        print(TASKS[sys.argv[2]]["prompt"])
    elif cmd == "score":
        print(json.dumps(score(sys.argv[2], sys.argv[3], sys.argv[4]),
                         indent=1))
