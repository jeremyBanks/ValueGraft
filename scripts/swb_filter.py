"""List non-Docker-safe SWE-bench-Lite instance ids, one per line, on stdout.

"Non-Docker-safe" == installable locally via plain `uv pip install -e .`
(pure-Python or wheel-only deps, no compiler toolchain / pinned old
numpy-family C-extension builds needed) AND whose FAIL_TO_PASS/PASS_TO_PASS
entries are real pytest node ids (path.py::test) that our pytest-based
harness (src/swebench_tasks.py) can run directly.

Repos matching both criteria: pytest, requests, flask, pylint, xarray,
seaborn (~41 instances; see standard-tasks-scouting.md for the hands-on
verification of pytest instances).

sympy is deliberately EXCLUDED despite being pure-Python/"light" to install:
its FAIL_TO_PASS/PASS_TO_PASS entries are bare test *function* names (e.g.
"test_ccode_Relational") with no file path, because upstream SWE-bench
identifies sympy tests via sympy's own bin/test runner rather than pytest
node ids. Feeding these to `pytest <id>` doesn't work, and resolving them to
file-qualified node ids via `-k` substring matching risks ambiguous/incorrect
matches across a 700+ file test suite, so they don't satisfy our "run the
exact FAIL_TO_PASS id under pytest" contract. django/sphinx/astropy/
matplotlib/scikit-learn are excluded per the scouting doc (DB stack /
period-pinned C-extension builds).
"""

import sys
import urllib.request
from pathlib import Path

CACHE_ROOT = Path.home() / ".cache" / "swb_tasks"
DATA_CACHE = CACHE_ROOT / "SWE-bench_Lite_test.parquet"
DATASET_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/"
    "resolve/main/data/test-00000-of-00001.parquet"
)

SAFE_REPOS = {
    "pytest-dev/pytest",
    "psf/requests",
    "pallets/flask",
    "pylint-dev/pylint",
    "pydata/xarray",
    "mwaskom/seaborn",
}


def main():
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not DATA_CACHE.exists():
        tmp = DATA_CACHE.with_suffix(".tmp")
        urllib.request.urlretrieve(DATASET_URL, tmp)
        tmp.rename(DATA_CACHE)
    import pandas as pd
    df = pd.read_parquet(DATA_CACHE)
    ids = sorted(df[df["repo"].isin(SAFE_REPOS)]["instance_id"].tolist())
    for i in ids:
        print(i)
    print(f"# {len(ids)} instances", file=sys.stderr)


if __name__ == "__main__":
    main()
