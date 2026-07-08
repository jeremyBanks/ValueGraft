# Scouting standard coding-agent task datasets (for context-compaction eval)

Criteria: (a) tests run locally via plain python/pytest/uv, no Docker; (b)
enough file exploration that a session plausibly exceeds ~9000 tokens; (c)
success plausibly depends on info stated early (issue text/constraints, at risk
of being compacted away); (d) dozens+ instances; permissive license.

All hands-on work done in a scratch dir (deleted afterward); nothing under
`scratchpad/e1_runs` was touched, no pods started.

## 1. Aider polyglot benchmark (Aider-AI/polyglot-benchmark)

- **What it is**: 225 hardest Exercism exercises across 6 languages (Python
  subset: 34 exercises). License: Exercism's open-source exercise license
  (permissive, redistributable).
- **Hands-on**: cloned the repo, inspected the Python `affine-cipher` exercise.
  Layout is one stub file + one test file + one instructions.md per exercise,
  e.g. `affine_cipher.py` (6 lines, a `pass`-stub), `affine_cipher_test.py` (82
  lines), `.docs/instructions.md` (74 lines). Total content ~150-300 lines/1-2k
  tokens. No project, no other files to explore — the whole task is legible in a
  single read.
- **Verdict vs criteria**: (a) yes, pure pytest, no docker. (c) yes, in a
  trivial sense (the instructions ARE the whole task). (d) yes, 133 legacy
  Python exercises or 34 in the harder polyglot subset. **(b) FAILS** — there is
  no repo to explore; a solving session is dominated by writing code, not
  reading files, and will not naturally cross 9k tokens. This dataset tests
  code-writing skill, not context management under exploration.
- **Recommendation**: not suitable as primary source. Could be used only as a
  cheap "control" condition (short session, no compaction expected) if we want a
  contrast case, but it doesn't produce the long-exploration sessions we need.

## 2. SWE-bench-Lite, run WITHOUT Docker (venv + pip install -e at commit)

- **What it is**: 300 instances from 11 popular Python repos (django 114, sympy
  77, matplotlib 23, scikit-learn 23, pytest 17, sphinx 16, astropy 6, requests
  6, pylint 6, xarray 5, seaborn 4, flask 3). Each instance ships `repo`,
  `base_commit`, `patch` (gold fix), `test_patch` (test-only diff),
  `problem_statement`, `FAIL_TO_PASS`/`PASS_TO_PASS` test ID lists,
  `environment_setup_commit`. License: MIT (both SWE-bench and SWE-Gym repos).
  Loaded easily via `datasets.load_dataset("princeton-nlp/SWE-bench_Lite")` with
  `uv pip install datasets` (no HF token needed for this dataset).

- **Hands-on trial**: picked 2 pure-Python instances from `pytest-dev/pytest`
  (avoiding numpy/scipy/matplotlib-family repos which need compiled deps and are
  the well-known source of non-Docker flakiness):
  - `pytest-dev__pytest-5227` (base commit `2051e30b`)
  - `pytest-dev__pytest-7432` (base commit `e6e300e7`)

  Procedure per instance: `git clone` pytest once, `cp -r` +
  `git checkout
  <base_commit>` per instance, `uv venv --python 3.9`,
  `uv pip install -e .`, apply `test_patch`, confirm `FAIL_TO_PASS` tests fail,
  apply `patch`, confirm they pass.

  **Timing**: venv create + editable install ≈ **1-2 seconds** per instance
  (uv's wheel cache made this near-instant; pytest itself is pure-Python with a
  handful of small deps — attrs, pluggy, py, more-itertools). Test runs
  (`pytest -q` on the 1-4 targeted test IDs) were **<0.3s** each.

  **What broke** (both fixed within a couple of minutes, exactly the kind of
  friction the task brief warned about):
  1. `ModuleNotFoundError: No module named 'pkg_resources'` on inst1 — modern
     setuptools (uv installed 82.0.1) no longer bundles `pkg_resources` by
     default; this old pytest version imports it directly. Fix:
     `uv pip
     install "setuptools<81"`. This will hit **any** old-repo
     instance that imports `pkg_resources`, `distutils`, or other deprecated
     stdlib/packaging surface — a systemic, not instance-specific, issue for
     pre-2021 commits.
  2. Test-ID mismatches: `FAIL_TO_PASS` test IDs include pytest parametrize
     suffixes (e.g. `[test_input1-expected1]`), and in `pytest-7432` the test
     didn't exist at `base_commit` at all — it's introduced by `test_patch`.
     Trap for a harness: you must apply `test_patch` before checking the
     pre-patch "fails" state, and must pass the exact FAIL_TO_PASS string
     (including brackets) to pytest, not a hand-guessed test name.
  3. Confirmed **fail-before-fix / pass-after-fix holds** once (1) and (2) are
     handled: 3 failed→passed on inst1, 1 failed→passed on inst2.

- **Verdict vs criteria**:
  - (a) Yes for pure-Python, dependency-light repos (pytest, flask, requests,
    pylint, xarray: 37 instances). Likely rough for django (needs a DB stack and
    specific old Django/Python pairings — not disqualifying, just heavier pip
    installs), and painful-to-outright-broken for
    matplotlib/scikit-learn/astropy on modern macOS/arm64 without conda
    (C-extension builds against period-correct numpy/scipy). Recommend filtering
    to a **non-Docker-safe subset** rather than all 300.
  - (b) Yes — pytest repo alone is 270 `.py` files / 52MB; real issues require
    grep/search across multiple modules to locate the fix site, unlike the Aider
    exercises. The `problem_statement` text itself is short (700-1000 chars in
    the two sampled), so exploration, not issue length, is what pushes sessions
    past 9k tokens — good for testing whether an agent still respects the
    constraints from the (short, easily-compacted) issue text after a long
    exploration phase.
  - (c) Yes — the issue text/hints and FAIL_TO_PASS test list state the required
    behavior up front; success requires the patch to satisfy those specifics
    after exploring the (sometimes large) surrounding codebase.
  - (d) Yes — 300 total, ~37+ safely non-Docker Python-light instances by repo
    filter, more if django/sphinx are included with extra install care.

- **Recommendation**: **primary candidate.** Use SWE-bench-Lite filtered to
  repos with cheap, C-extension-free installs: `pytest-dev/pytest`,
  `psf/requests`, `pallets/flask`, `pylint-dev/pylint`, `pydata/xarray`,
  `mwaskom/seaborn` (~64 instances), with django/sphinx as a secondary tier
  after validating a couple of instances each (sphinx and django both pure
  Python + moderate deps, likely fine). Treat matplotlib/scikit-learn/astropy as
  out of scope for the no-Docker path.

  **Integration sketch (e1_driver.sh-shaped)**:
  - `materialize(instance_id, dir)`: `git clone <repo> dir/repo` (or reuse a
    shared bare clone + `git worktree add`), `git checkout <base_commit>`,
    `uv venv dir/repo/.venv --python <pin>`, `uv pip install -e dir/repo` (+
    `uv pip install "setuptools<81"` defensively for older commits), then
    `git apply <test_patch>` so the target tests exist and are in the pre-fix
    failing state.
  - `prompt(instance_id)`: emit `problem_statement` (+ `hints_text` if desired)
    as `task.txt`, plus an instruction not to touch test files.
  - `score(instance_id, dir, log)`: run
    `pytest <FAIL_TO_PASS ids> <PASS_TO_PASS ids> -q` inside the venv; pass iff
    all FAIL_TO_PASS now pass and all PASS_TO_PASS still pass. This maps
    directly onto the existing `score` step contract in `e1_driver.sh`.
  - Cost: setup ~1-3s/instance for the light repos measured (uv cache warm);
    budget more for first-time installs of heavier repos (django migrations,
    sphinx doc deps).

## 3. Other candidates considered (not hands-on tested, desk review only)

- **commit0**: benchmark reconstructs libraries from stubs against pinned
  per-library setup commands; explicitly built around per-repo Docker images for
  reproducibility. Local-without-Docker is possible in principle (it's just
  pinned pip/conda installs) but not documented as a first-class mode, and needs
  per-library environment reverse-engineering. Lower priority given
  SWE-bench-Lite's pure-Python subset already satisfies our criteria with known
  tooling.
- **SWE-smith**: designed to synthesize new SWE-bench-style task instances
  (harder to reproduce faithfully, and Docker-oriented tooling); more relevant
  as a way to generate additional instances later than as an off-the-shelf
  source right now.
- **BigCodeBench**: function-level code generation with test harnesses, but
  tasks are single-function/self-contained like the Aider benchmark — same
  concern as (1), unlikely to induce long exploration sessions since there's no
  surrounding repo to explore.

## Bottom line

SWE-bench-Lite, restricted to pure-Python/light-dependency repos and run via
`uv venv` + `pip install -e` + direct `git apply` of test/gold patches (no
Docker), is the strongest fit. Verified hands-on on 2 `pytest-dev/pytest`
instances: sub-3-second setup, sub-second test runs, and the
fail-before-patch/pass-after-patch invariant holds once (i) setuptools is pinned
below 81 to keep `pkg_resources` available for older code, and (ii) FAIL_TO_PASS
test IDs (including parametrize suffixes) are used verbatim and `test_patch` is
applied before checking the pre-fix failing state.
