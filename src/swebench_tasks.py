"""SWE-bench-Lite coding tasks: real GitHub issue/PR instances, run locally
via `uv venv` + editable install (no Docker). Task ids: "swb:<instance_id>",
e.g. "swb:pytest-dev__pytest-5227".

Commands: materialize <task> <dir> | prompt <task> | score <task> <dir> <log>

materialize clones the repo at base_commit into <dir>/repo, creates
<dir>/.venv, installs the repo editable (+ "setuptools<81" for old commits
that import pkg_resources, + pytest), applies the instance's test_patch, and
writes <dir>/meta.json with FAIL_TO_PASS/PASS_TO_PASS.

score reads <dir>/meta.json and runs the venv's pytest on the exact
FAIL_TO_PASS + PASS_TO_PASS node ids (parametrize brackets included verbatim).

Dataset (princeton-nlp/SWE-bench_Lite, test split) is downloaded once as a
parquet file and cached under ~/.cache/swb_tasks/ (not in the repo). Repo
clones are cached under ~/.cache/swb_repos/<owner>__<name> and copied
(git clone --local) per instance so arbitrary base_commits can be checked
out cheaply.
"""

import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

CACHE_ROOT = Path.home() / ".cache" / "swb_tasks"
DATA_CACHE = CACHE_ROOT / "SWE-bench_Lite_test.parquet"
REPO_CACHE_ROOT = Path.home() / ".cache" / "swb_repos"
DATASET_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/"
    "resolve/main/data/test-00000-of-00001.parquet"
)

_DF = None


def _load_df():
    """Download (once, cached) and return the SWE-bench-Lite test split."""
    global _DF
    if _DF is not None:
        return _DF
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not DATA_CACHE.exists():
        tmp = DATA_CACHE.with_suffix(".tmp")
        urllib.request.urlretrieve(DATASET_URL, tmp)
        tmp.rename(DATA_CACHE)
    import pandas as pd
    _DF = pd.read_parquet(DATA_CACHE)
    return _DF


def _iid(task):
    assert task.startswith("swb:"), f"not a swb task id: {task!r}"
    return task[len("swb:"):]


def _row(instance_id):
    df = _load_df()
    hits = df[df["instance_id"] == instance_id]
    if len(hits) == 0:
        raise KeyError(f"instance_id not found in SWE-bench-Lite: "
                        f"{instance_id!r}")
    return hits.iloc[0]


def _clean_ids(raw_json):
    """Parse a FAIL_TO_PASS/PASS_TO_PASS JSON-string field into a list of
    valid pytest node ids, dropping the occasional garbage entry (some
    instances' PASS_TO_PASS lists contain stray captured-output fragments
    like "[100%]" instead of real test ids)."""
    if not raw_json:
        return []
    ids = json.loads(raw_json)
    # A handful of instances' FAIL_TO_PASS/PASS_TO_PASS lists were built by
    # naively comma-splitting a repr() of the id list, which truncates any
    # node id whose parametrize id itself contains a literal comma (e.g.
    # "test_x[(AttributeError, TypeError)]" -> "test_x[(AttributeError,").
    # Such truncated ids have unbalanced brackets and pytest's arg parser
    # treats them as a fatal "not found" (aborting the whole invocation,
    # not just that id), so drop them rather than let one bad id block
    # every other id in the same run.
    return [i for i in ids if ".py::" in i and i.count("[") == i.count("]")]


def _repo_cache_dir(repo):
    return REPO_CACHE_ROOT / repo.replace("/", "__")


def _run(cmd, **kw):
    kw.setdefault("check", True)
    return subprocess.run([str(c) for c in cmd], **kw)


def _ensure_repo_cache(repo):
    cache_dir = _repo_cache_dir(repo)
    REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not (cache_dir / ".git").exists():
        url = f"https://github.com/{repo}.git"
        _run(["git", "clone", "--quiet", url, cache_dir])
    return cache_dir


def _commit_present(cache_dir, commit):
    r = subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"],
                        cwd=cache_dir, capture_output=True)
    return r.returncode == 0


def _clone_at_commit(repo, base_commit, dest):
    cache_dir = _ensure_repo_cache(repo)
    if not _commit_present(cache_dir, base_commit):
        _run(["git", "fetch", "--quiet", "origin"], cwd=cache_dir)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--quiet", "--local", cache_dir, dest])
    _run(["git", "checkout", "--quiet", base_commit], cwd=dest)
    return dest


def _venv_python(root):
    return root / ".venv" / "bin" / "python"


def materialize(task, d):
    iid = _iid(task)
    row = _row(iid)
    root = Path(d)
    root.mkdir(parents=True, exist_ok=True)

    repo_dir = root / "repo"
    _clone_at_commit(row["repo"], row["base_commit"], repo_dir)

    venv_dir = root / ".venv"
    r = subprocess.run(["uv", "venv", str(venv_dir), "--python", "3.9"],
                        capture_output=True, text=True)
    if r.returncode != 0:
        _run(["uv", "venv", venv_dir])  # fall back to default python

    py = _venv_python(root)
    _run(["uv", "pip", "install", "--python", py,
          "-e", repo_dir, "setuptools<81", "pytest"])

    test_patch = row.get("test_patch") or ""
    if str(test_patch).strip():
        patch_file = root / "test_patch.diff"
        patch_file.write_text(test_patch)
        _run(["git", "apply", "--whitespace=fix", patch_file.resolve()],
             cwd=repo_dir)

    meta = {
        "task": task,
        "instance_id": iid,
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "FAIL_TO_PASS": _clean_ids(row["FAIL_TO_PASS"]),
        "PASS_TO_PASS": _clean_ids(row["PASS_TO_PASS"]),
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=1))


def prompt(task):
    iid = _iid(task)
    row = _row(iid)
    preamble = (
        "You are working in the repo at the current directory. First "
        "explore the relevant files, then fix the issue described below. "
        "Do not modify test files.\n\n"
    )
    print(preamble + str(row["problem_statement"]))


_OUTCOME_RE = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)\b",
                         re.M)


def _run_pytest(repo_dir, py, ids, timeout=600):
    r = subprocess.run(
        [str(py), "-m", "pytest", "-v", "-p", "no:cacheprovider", *ids],
        cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
    )
    outcomes = dict(_OUTCOME_RE.findall(r.stdout))
    return outcomes, r


def score(task, d, log):
    root = Path(d)
    meta = json.loads((root / "meta.json").read_text())
    repo_dir = root / "repo"
    py = _venv_python(root)
    f2p = meta["FAIL_TO_PASS"]
    p2p = meta["PASS_TO_PASS"]
    all_ids = f2p + p2p

    outcomes, tail = {}, ""
    if all_ids:
        outcomes, r = _run_pytest(repo_dir, py, all_ids)
        tail = (r.stdout or r.stderr)[-3000:]

    f2p_pass = sum(1 for t in f2p if outcomes.get(t) == "PASSED")
    p2p_ok = all(outcomes.get(t) == "PASSED" for t in p2p)
    tests_pass = (f2p_pass == len(f2p)) and p2p_ok

    return {
        "task": task,
        "tests_pass": tests_pass,
        "f2p_pass": f2p_pass,
        "f2p_total": len(f2p),
        "p2p_ok": p2p_ok,
        "pytest_tail": tail,
    }


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "materialize":
        materialize(sys.argv[2], sys.argv[3])
    elif cmd == "prompt":
        prompt(sys.argv[2])
    elif cmd == "score":
        print(json.dumps(score(sys.argv[2], sys.argv[3], sys.argv[4]),
                          indent=1))
    else:
        raise SystemExit(f"unknown command: {cmd!r}")
