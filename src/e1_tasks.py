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
    "t3": {
        "prompt": (
            "You are working in the repo at the current directory, which "
            "implements a small internal billing microservice used by the "
            "checkout team. The billing service began life as a prototype "
            "and has since accumulated several load-bearing conventions "
            "that the tests enforce, so read carefully before writing "
            "anything. First, every public function exported from this "
            "service's modules must be named with the 'svc_' prefix — "
            "this was mandated after a naming collision with another "
            "team's library, and any public callable lacking that prefix "
            "is treated as an unstable internal helper that other teams "
            "must not depend on. Second, all currency amounts anywhere in "
            "this service are represented as integer CENTS; the team "
            "banned floating point money values after a rounding bug "
            "reached production, so a monetary value returned by this "
            "service must never be a float, only a plain int. Meanwhile, "
            "the loyalty-discount feature finance asked for last quarter "
            "reads its rate from config.py: the line DISCOUNT_RATE_BP = "
            "1250 there encodes a 12.50% discount rate expressed in basis "
            "points, and config.py is the single source of truth for that "
            "rate — nothing else in the service may hardcode a different "
            "discount percentage. On the reliability side, the on-call "
            "rotation required, after a painful outage, that any flaky "
            "network call be retried according to config.py's "
            "RETRY_MAX_ATTEMPTS = 4, a hard cap of 4 total attempts (the "
            "original call plus 3 retries) before giving up and "
            "propagating the underlying failure; retrying fewer or more "
            "times than that cap is considered a regression of the "
            "original incident fix. Finally, every structured log line "
            "this service emits must follow the exact format "
            "LEVEL::epoch_seconds::message, using a double-colon '::' "
            "delimiter between the three fields and no extra whitespace "
            "around the delimiters, because the downstream log parser "
            "splits on that exact token and will silently drop malformed "
            "lines. First read every file under src/ and tests/ to "
            "understand the existing codebase and conventions before "
            "changing anything. Then implement the missing module "
            "src/billing.py so that `python -m pytest tests/ -q` passes. "
            "Do not modify tests."
        ),
        "files": {
            "config.py": "DISCOUNT_RATE_BP = 1250\nRETRY_MAX_ATTEMPTS = 4\n",
            "src/__init__.py": "",
            "tests/test_billing.py": (
                "import config\n"
                "import inspect\n"
                "import pytest\n"
                "import src.billing as b\n\n"
                "def test_public_functions_have_svc_prefix():\n"
                "    for name, obj in vars(b).items():\n"
                "        if name.startswith('_'):\n"
                "            continue\n"
                "        if inspect.isfunction(obj) and "
                "obj.__module__ == b.__name__:\n"
                "            assert name.startswith('svc_'), "
                "f'{name} missing svc_ prefix'\n\n"
                "def test_integer_cents():\n"
                "    result = b.svc_apply_discount(12345)\n"
                "    assert isinstance(result, int)\n\n"
                "def test_discount_uses_config_rate():\n"
                "    assert b.svc_apply_discount(10000) == "
                "10000 - config.DISCOUNT_RATE_BP\n\n"
                "def test_discount_general_amount():\n"
                "    expected = 20000 - (20000 * config.DISCOUNT_RATE_BP "
                "// 10000)\n"
                "    assert b.svc_apply_discount(20000) == expected\n\n"
                "def test_log_format():\n"
                "    s = b.svc_format_log('INFO', 1700000000, 'ok')\n"
                "    assert s == 'INFO::1700000000::ok'\n\n"
                "def test_retry_exhausts_at_cap():\n"
                "    calls = {'n': 0}\n"
                "    def flaky():\n"
                "        calls['n'] += 1\n"
                "        raise ValueError('boom')\n"
                "    with pytest.raises(ValueError):\n"
                "        b.svc_retry(flaky)\n"
                "    assert calls['n'] == config.RETRY_MAX_ATTEMPTS\n\n"
                "def test_retry_succeeds_within_cap():\n"
                "    calls = {'n': 0}\n"
                "    def flaky():\n"
                "        calls['n'] += 1\n"
                "        if calls['n'] < config.RETRY_MAX_ATTEMPTS:\n"
                "            raise ValueError('boom')\n"
                "        return 'ok'\n"
                "    assert b.svc_retry(flaky) == 'ok'\n"
                "    assert calls['n'] == config.RETRY_MAX_ATTEMPTS\n"
            ),
        },
        "filler_files": 8,
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
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
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
    if len(sys.argv) > 2 and ":" in sys.argv[2]:
        base, seed = sys.argv[2].split(":")
        import hashlib
        h = int(hashlib.sha1(seed.encode()).hexdigest()[:6], 16)
        raw = json.dumps(TASKS[base])
        if base == "t1":
            bp = 700 + h % 300
            raw = raw.replace("825", str(bp)).replace("8.25%", f"{bp/100:.2f}%")
        elif base == "t2":
            ms = 200 + (h % 20) * 10
            raw = (raw.replace("1000", str(ms * 4))
                      .replace("500", str(ms * 2))
                      .replace("250", str(ms)))
        elif base == "t3":
            bp = 1000 + h % 500
            n = 3 + h % 4
            raw = (raw.replace(
                        "4 total attempts (the original call plus 3 "
                        "retries)",
                        f"{n} total attempts (the original call plus "
                        f"{n - 1} retries)")
                      .replace("DISCOUNT_RATE_BP = 1250",
                               f"DISCOUNT_RATE_BP = {bp}")
                      .replace("RETRY_MAX_ATTEMPTS = 4",
                               f"RETRY_MAX_ATTEMPTS = {n}")
                      .replace("12.50%", f"{bp / 100:.2f}%"))
        TASKS[sys.argv[2]] = json.loads(raw)
    if cmd == "materialize":
        materialize(sys.argv[2], sys.argv[3])
    elif cmd == "prompt":
        print(TASKS[sys.argv[2]]["prompt"])
    elif cmd == "score":
        print(json.dumps(score(sys.argv[2], sys.argv[3], sys.argv[4]),
                         indent=1))
