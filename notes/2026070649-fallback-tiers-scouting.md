# Scouting fallback (easier) task tiers for the no-Docker SWE-bench-Lite harness

Context: risk that even the easiest SWE-bench-Lite instances (11-line,
single-file gold patches) all time out at 60 min under Qwen3-30B/OpenHands. This
is a search for genuinely _easier_, still-real, still-locally-runnable fallback
tiers, ranked. All hands-on work done under the scratchpad dir; nothing under
`scratchpad/e1_runs` touched, no pods started, no other repo files modified.

## TL;DR ranking

1. **SWE-bench Verified, filtered to `difficulty == "<15 min fix"` AND
   non-Docker-safe repos** — top pick. Real issues, human-labeled easy by
   OpenAI's own annotators, same schema/tooling as the existing
   `swebench_tasks.py` Lite loader. Validated hands-on (see below).
2. **BugsInPy**, lightweight-project subset — viable secondary tier, more bugs
   available, but weaker on license clarity and needs new harness plumbing (not
   HF `datasets`-shaped).
3. **SWE-smith** — deprioritized. Docker-only tooling, no realistic non-Docker
   path documented; not a fit for this constraint.

## 1. SWE-bench Verified difficulty annotations (TOP CANDIDATE)

**What it is**: `SWE-bench/SWE-bench_Verified` on HF, 500 instances, same schema
as Lite (`repo`, `base_commit`, `patch`, `test_patch`, `problem_statement`,
`FAIL_TO_PASS`/`PASS_TO_PASS`, `environment_setup_commit`) **plus a `difficulty`
column**: OpenAI's human-annotator time-estimate label, one of `<15 min fix`
(194/500), `15 min - 1 hour` (261), `1-4 hours` (42), `>4 hours` (3). License:
same MIT-style terms as SWE-bench/SWE-bench_Lite (Princeton/OpenAI release).

**Downloaded and analyzed the full parquet** (`test-00000-of-00001.parquet`,
cached locally during this session, not committed anywhere):

- Repo distribution across all 500: django 231, sympy 75, sphinx 44, matplotlib
  34, scikit-learn 32, astropy 22, xarray 22, pytest 19, pylint 10, requests 8,
  seaborn 2, flask 1.
- Restricting to the non-Docker-safe repos already identified in
  `standard-tasks-scouting.md` (pytest, requests, flask, pylint, xarray,
  seaborn, **adding sympy**, which is pure-Python + only depends on mpmath):
  **137 instances total**, of which **48 are labeled `<15 min fix`** (sympy 25,
  pytest 8, requests 6, xarray 5, pylint 3, flask 1, seaborn 0).
- **Overlap with SWE-bench-Lite**: only **11** of those 48 `<15 min` safe-repo
  instances are also in Lite's 300 (8 sympy, 1 each pytest/requests/xarray). →
  Verified is mostly a _disjoint_ pool, not a re-labeling of Lite — this
  genuinely adds ~37 new safe, human-confirmed-easy instances beyond Lite's
  existing set, plus corroborates the "<15 min" label on 11 instances already in
  the Lite rotation.
- **Caveat that undercuts the "these are simpler" hope**: gold-patch line counts
  for the 48 easy instances are _not_ meaningfully smaller than the Lite
  instances already tried — median 14 lines, min 12, several at 12-13 (e.g.
  `pydata__xarray-4094`, `sympy__sympy-13480/17139/18189/22714` all at 12
  lines). If timeouts are driven by exploration/agent-loop overhead rather than
  patch size, this tier may not resolve them — but it _is_ the best available
  signal for "a human found this fast," and adds sympy as a new, very
  lightly-dependent repo (only mpmath) not previously in the safe set.

**Hands-on validation** (2 instances, both from the 48-instance easy/safe
overlap):

- `sympy__sympy-17139` (base commit `1d3327b8`): `git clone` sympy (~6s),
  `uv venv --python 3.9` + `uv pip install -e .` → **<1s**, only pulls `mpmath`.
  Applied `test_patch`, ran
  `pytest sympy/simplify/tests/test_fu.py::test__TR56
  sympy/simplify/tests/test_simplify.py::test_issue_17139`
  → **2 failed** (`TypeError: Invalid comparison of complex I`), confirming
  pre-fix failure. Applied gold `patch` → **2 passed** in 0.10s.
- `pydata__xarray-4094` (base commit `a64cf2d5`): clone (~2.3s), install
  editable + pytest (~1s). Hit the **same two systemic issues** already logged
  in `standard-tasks-scouting.md`: (i) `ModuleNotFoundError:
  pkg_resources` →
  fixed with `setuptools<81`; (ii) numpy 2.0 (uv's default resolve) breaks
  2020-era xarray (`np.unicode_` removed) → fixed by pinning `numpy<2`. After
  both pins: `test_to_stacked_array_to_unstacked_dataset` **failed** pre-fix,
  **passed** after applying gold `patch` (0.34s).
- **Timing**: total setup+verify per instance is low single-digit seconds (clone
  dominates; installs are sub-second on a warm uv cache) — no material cost
  difference from the existing Lite harness.
- **New wrinkle found for sympy specifically**: sympy's `FAIL_TO_PASS` ids are
  bare function names (e.g. `"test__TR56"`, no `path.py::name` prefix), unlike
  pytest/requests/flask/xarray which do include the file path. **This breaks the
  existing `_clean_ids()` filter in `src/swebench_tasks.py`**, which drops any
  id without `.py::` in it — as written, it would silently discard 100% of
  sympy's FAIL_TO_PASS/PASS_TO_PASS ids. Confirmed the ids still resolve fine as
  plain pytest node ids once the test file is known
  (`pytest sympy/simplify/tests/test_fu.py::test__TR56` works), so the fix is a
  resolution step (map bare name → file path via `test_patch`'s target file, or
  `pytest --collect-only -q | grep <name>`), not a fundamentally different
  runner.

**Integration notes for `swebench_tasks.py`**:

- New task-id namespace, e.g. `"swbv:<instance_id>"`, sourced from
  `princeton-nlp/SWE-bench_Verified` (or `SWE-bench/SWE-bench_Verified`)
  parquet, cached the same way as `SWE-bench_Lite_test.parquet` is now.
- `materialize()` is a near-identical copy of the Lite version: clone at
  `base_commit`, `uv venv --python 3.9`,
  `uv pip install -e . "setuptools<81"
  pytest`, apply `test_patch`. **Add
  `numpy<2` to the defensive pin list** alongside `setuptools<81` for repos with
  a numpy dependency (xarray; probably also needed for
  matplotlib/scikit-learn/astropy if ever added, but those stay out of scope as
  before).
- Filter selection: build the task pool from `difficulty == "<15 min fix"` AND
  `repo in {pytest-dev/pytest, psf/requests, pallets/flask,
  pylint-dev/pylint, pydata/xarray, mwaskom/seaborn, sympy/sympy}`
  → 48 instances (37 net-new beyond what Lite already offers).
- `score()`: needs the bare-name-id fix described above before sympy instances
  will work — either (a) resolve bare FAIL_TO_PASS/PASS_TO_PASS names to
  `file::name` by parsing which test file(s) `test_patch` touches and running
  `pytest <file>::<name>` for each name found there, or (b) run
  `pytest --collect-only -q <file>` and grep-match by suffix. Non-sympy repos in
  this tier need no change to `score()`.
- `prompt()`: identical to Lite (`problem_statement` + "explore first, don't
  touch tests" preamble); Verified additionally carries `hints_text`, same as
  Lite already exposes via the dataframe.

## 2. BugsInPy (secondary candidate, not fully validated)

**What it is**: `soarsmu/BugsInPy`, real historical bugs in 17 Python projects,
curated with reproduction scripts (not HF/`datasets`-shaped — it's a git repo of
per-bug diffs + a shell-script framework: `framework/bin/bugsinpy-checkout`,
`bugsinpy-test`, etc., callable without Docker by adding `framework/bin` to
`PATH`; Docker is offered as an alternative, not a requirement).

**Desk review** (cloned the repo, walked the layout, did not run a full
checkout/test cycle — time-boxed in favor of validating candidate #1):

- **501 bug entries across 17 projects** (`ls projects/*/bugs`): ansible 18,
  black 23, cookiecutter 4, fastapi 16, httpie 5, keras 45, luigi 33, matplotlib
  30, pandas 170, PySnooper 3, sanic 5, scrapy 40, spacy 10, thefuck 32, tornado
  16, tqdm 9, youtube-dl 43.
- Lightweight/pure-Python subset (no heavy C-extension/DB deps): tqdm (9),
  PySnooper (3), httpie (5), cookiecutter (4), sanic (5), thefuck (32), fastapi
  (16, needs pydantic/starlette but pure-Python) — roughly **~74 instances** in
  a plausibly non-Docker-safe subset; the rest (pandas, keras, matplotlib,
  scrapy, spacy, ansible) skew toward the same C-extension/service dependency
  problems flagged for SWE-bench's numpy-family repos.
- Sampled `projects/tqdm/bugs/1/bug_patch.txt`: a real 1-line semantic fix
  (argument-order bug in `tqdm.contrib.tenumerate`), same shape/size as the
  smallest SWE-bench-Lite/Verified patches — no evidence these are
  systematically _smaller_ fixes, just an independently-curated pool.
- **License risk**: searched the full clone for `LICENSE`/`COPYING` files —
  **none exist in the BugsInPy repo itself.** The underlying bug fixes come from
  permissively-licensed upstream projects (tqdm=MPL-2.0, httpie=BSD,
  thefuck=MIT, etc.), but the curation framework/dataset packaging has no stated
  license, which is a real gap against the "permissive license" requirement and
  would need per-project license verification before use.
- No `difficulty` or time-estimate annotation exists in BugsInPy — "easier"
  would have to be operationalized via patch size/file count as a proxy, not a
  human label.

**Verdict**: usable as a _third-choice_ fallback pool if Verified's 48 instances
are exhausted or still prove too hard, but it needs (a) new harness code
(shell-framework wrapper or reimplementing its checkout logic directly against
each project's upstream git repo, similar to what `swebench_tasks.py` already
does), and (b) a license audit per project before committing to it. Not hands-on
validated this session — recommend a quick trial (2 tqdm or PySnooper bugs)
before relying on it.

## 3. SWE-smith (deprioritized)

`SWE-bench/SWE-smith` on HF: 50,137 generated instances across 128 repos, but
the tooling is explicitly Docker-first — 125 Docker images (295GB total), one
per (repo, commit) pair, and the project's own docs state it targets Ubuntu
22.04 and does not support macOS. No documented non-Docker install path was
found. Given the hard "no Docker" constraint, this is not a realistic
off-the-shelf source right now; it's more relevant later as a way to _generate_
new instances in repos we already run natively (e.g. targeting
sympy/pytest/xarray directly) than as a ready-made task pool today.

## Other notes

- Did not find a good-first-issue-style curated benchmark meeting the bar
  (locally runnable, py-only, ≥20 instances, permissive license) beyond the
  three investigated; SWE-bench-family sources remain the strongest fit given
  existing harness investment.
- All scratch clones/venvs (`sympy_repo`, `xarray_repo`, `bugsinpy`, parquet
  files, throwaway `.venv-research`) live under `/private/tmp/...scratchpad` and
  `/tmp` and were not left in the project tree.
