# PIPELINE-AUDIT.md — adversarial validity audit (post SUMMARY-PROMPT-FIX)

Scope: DECISIONS.md (esp. 07-06 entries), INCIDENTS.md, src/serve_shim.py,
scripts/e1_driver.sh, scratchpad/e1_matrix.sh, src/e1_agent.py, src/e1_tasks.py,
src/swebench_tasks.py, src/enrich_runs.py, src/arms_common.py, src/arms_hf.py,
src/kvlib_hf.py, tune_configs.json, scratchpad/swb_filter.py, scripts/launch_pod.sh.

Findings are numbered within severity. Each has file:line evidence, why it
matters, and the minimal fix.

---

## CRITICAL

### C1. Uniform wall-clock timeout hides a real per-arm compute-cost asymmetry
`scripts/e1_driver.sh` wraps the whole episode in a single fixed budget:
`perl -e 'alarm 3600; exec @ARGV' -- ...e1_agent.py` (e1_driver.sh, the
`perl -e 'alarm 3600...'` line). That 3600s is identical across A/B/E.

But the per-turn compute cost is **not** identical across arms. With
`SC_INCR_CACHE` currently withdrawn for all real-task rows (DECISIONS
07-06 13:55: "r-shims restarted WITHOUT SC_INCR_CACHE ... stable no-cache
shims on all 4 lanes"), `src/serve_shim.py`'s uncompacted path
(`_generate`, the `if mode == "A" or len(ids) <= compact_at:` branch,
serve_shim.py ~166-172) does a **fresh full prefill of the entire growing
conversation from scratch on every single request** — no cache reuse
across turns. Mode A never compacts, so its per-turn prefill cost grows
with the *entire* episode history (quadratic total compute over a long
episode). Modes B/E cap context at `compact_at` + a bounded tail once the
first compaction fires, so their per-turn cost plateaus.

Consequence: on real SWE-bench episodes running 17-27K tokens with dozens
of turns (DECISIONS 07-06 13:20/15:30: "51 and 26 compacted calls in live
sessions"), mode A can burn dramatically more wall-clock per turn than B/E
for reasons that have nothing to do with memory/compaction damage — it is
simply re-processing more tokens every time. Under a fixed 3600s budget
this means:
 - A is more likely to be cut off by `timeout.marker` before reaching
   `E1_AGENT_DONE` purely from compute load, independent of whether it
   "remembered" more.
 - The recall probe appended after `E1_AGENT_DONE` (e1_agent.py lines
   24-44) is itself vulnerable to being truncated for A specifically.
 - Any A-vs-B/E comparison of pass-rate, "agent_events", or the recall
   probe on real tasks is confounded by this asymmetry, and it moves in
   the direction that could make A (the reference/oracle arm) look
   *worse* for compute reasons — the opposite direction from the
   compaction-damage story, so it's not merely "conservative," it can
   invert conclusions on borderline items.

This is exactly an "arm asymmetry besides the intended intervention"
(hunt item #1) and isn't mentioned anywhere in DECISIONS/INCIDENTS.

**Minimal fix:** record total tokens prefetched/prefilled per episode (or
wall-time-per-turn) per arm in the enrichment step
(`src/enrich_runs.py`), and report timeout-rate stratified by arm and by
episode token-size; treat "timed_out" as a covariate, not a clean failure
signal, until this is measured. Longer-term: reinstate a *symmetric*
incremental-cache (already built as v2, DECISIONS 07-06 14:40) so A's
per-turn cost stops scaling with full history.

### C2. sc_debug never records which SUMMARY_REQUEST variant or TAIL_KEEP was actually used — the exact bug class that caused the SUMMARY-PROMPT confound is still undetectable from data
`src/serve_shim.py` line ~192: `_sreq = (SUMMARY_REQUEST_PROD if
os.environ.get("SC_SUMMARY", "prod") == "prod" else SUMMARY_REQUEST_BRIEF)`.
This is the fix for the caught confound (DECISIONS 07-06 17:40). But the
`dbg` dict built in `_generate` (serve_shim.py lines 112-115, updated at
208-209 and 229) **never includes which variant was used**, nor the
`TAIL_KEEP` value in effect. Fields present: mode, alpha, compact_at,
full_tokens, n_compactions, alpha_kind, compacted, b_tokens,
summary_tokens, summary_cache, grafted_positions — no `summary_variant`,
no `tail_keep`.

Both `SC_SUMMARY` and `SC_TAIL_KEEP` are read once at process/module load
and never touch the per-request debug payload. Grep across the repo
confirms neither is ever set in any *tracked* launch script — the humane
tier's `TAIL_KEEP=6000` (DECISIONS 07-06 16:20/16:45) is only realized via
an ad hoc `SC_TAIL_KEEP=6000` env var passed to a per-pod `job.sh` that
is generated at deploy time and is **not in version control**
(`scripts/launch_pod.sh` just rsyncs `src/` + uploads whatever `job.sh`
was handed to it). So: if a shim wasn't actually restarted with the
intended env (the exact same class of failure documented four times in
INCIDENTS.md — session leak, cfg-parser 500s, ccache VRAM growth, mis-wired
tunnel — none of which were caught by inspection, all by accident/alarm),
there is currently **no artifact in score.json/sc_debug that would reveal
it**. The only record of "what env a run used" is the DECISIONS.md
narrative, which has itself required a retraction once already (13:00
07-06 entry retracting the 12:50 entry).

**Minimal fix:** add `dbg["summary_variant"]` and `dbg["tail_keep"]` (and
ideally `dbg["incr_cache_on"] = INCR_CACHE`) to every response; assert at
shim startup and print a one-line "CONFIG: tail_keep=X compact_at=Y
summary=Z" banner into job.log for post-hoc verification against pods.list.

### C3. `_clean_ids` silently drops FAIL_TO_PASS/PASS_TO_PASS ids for *any* SWE-bench instance, not just the excluded sympy repo — can silently shrink both the pass bar and the regression-detection set
`src/swebench_tasks.py` lines 70-86 (`_clean_ids`): any id in
FAIL_TO_PASS/PASS_TO_PASS that fails `".py::" in i and i.count("[") ==
i.count("]")` is dropped outright, with the stated rationale that a
handful of ids were corrupted upstream by a naive comma-split on ids
containing literal commas (`test_x[(AttributeError, TypeError)]` →
truncated, unbalanced brackets). This filter runs on **every** instance
scored via `swebench_tasks.py:score` (line 195-197: `f2p =
meta["FAIL_TO_PASS"]; p2p = meta["PASS_TO_PASS"]`, both already
`_clean_ids`-filtered at materialize time, line 160-161).

DECISIONS/INCIDENTS document only that "sympy" is *excluded from the
instance pool* (`scratchpad/swb_filter.py`, SAFE_REPOS excludes sympy)
because of a *different, repo-specific* id-scheme mismatch (sympy uses
its own bin/test runner, not file-qualified pytest ids). That exclusion
does **not** protect against `_clean_ids` dropping a mangled id from one
of the six *included* repos (pytest, requests, flask, pylint, xarray,
seaborn) — the docstring itself says the comma-truncation problem is a
general SWE-bench-Lite dataset artifact, not sympy-specific.

Effect: if a dropped id was in FAIL_TO_PASS, `tests_pass` can read `True`
while a genuinely required fix-verifying test was **never executed at
all** (silently absent from `f2p`, so `f2p_pass == len(f2p)` trivially
holds on a smaller set). If a dropped id was in PASS_TO_PASS, a real
regression the fix introduced in that specific test is never checked
(`p2p_ok` is computed only over the surviving, smaller list) — this is
literally the audit brief's "can score.json disagree with what actually
happened ... P2P regressions" concern, and the blast radius is broader
than the one repo (sympy) that's currently flagged as the known-affected
case.

**Minimal fix:** log the *count and identity* of any dropped id per
instance into `meta.json`/`score.json` (e.g. `"dropped_ids": [...]`).
Any instance with a nonempty `dropped_ids` should be flagged/excluded
from headline numbers rather than silently scored on a reduced test set,
or manually repaired (recover the true id by hand for the ~handful of
affected instances) before being reused across arms.

---

## MAJOR

### M1. `n_compactions` counts compacted *calls*, not compaction *events* — the humane-tier pressure calibration may be validating the wrong number
`src/serve_shim.py` line 233: `sess["n_compactions"] =
sess.get("n_compactions", 0) + 1` sits unconditionally inside the B/E
compaction branch, incrementing on **every** request that is served
above `compact_at` — regardless of whether that request was a
`summary_cache` "HIT" (reusing the frozen boundary's existing summary) or
a "MISS" (a genuinely new summary/boundary generated). The only true
per-request signal of a *new* compaction event is `dbg["summary_cache"] ==
"MISS"` (line 204), which is not accumulated anywhere — it's logged once
per request and never aggregated into a session-level counter.

DECISIONS 07-06 16:45 ("Humane-tier calibration finalized") reasons about
"cycle math" and a target band of "2-6 compactions/episode" in a way that
assumes this counts *recompaction events* (new boundary/new summary), not
calls served under an already-frozen boundary. As coded, `n_compactions`
(and anything derived from the final `dbg` blob per episode) systematically
over-counts relative to that intended meaning, once a boundary freezes and
serves multiple subsequent turns (which it will, by design — "boundary
freezes at first compaction ... only rolls forward when tail regrows").

**Minimal fix:** add a separate `sess["n_recompactions"]` incremented only
on MISS; report both fields; re-check the "2-5 compactions per episode
achieved" claims against the corrected counter before trusting the
pressure calibration.

### M2. Humane-tier `TAIL_KEEP`/`compact_at` values live only in ad hoc, untracked `job.sh`, with no runtime assertion
Same underlying fact as C2 but the actionable half of it: `serve_shim.py`
reads `TAIL_KEEP = int(os.environ.get("SC_TAIL_KEEP", "2500"))` once at
import time (line 54) with no assertion of the value it received, no log
line at startup, and no mechanism to compare "value in effect" against
"value the analysis assumes." `scripts/launch_pod.sh` uploads whatever
`job.sh` is handed to it (line: `rsync -az ... "$JOB" root@$IP:/workspace/exp/job.sh`)
— that file is generated per-session and isn't in this repo, so there is
no committed record of what env values any given pod's shim actually ran
with.

**Minimal fix:** print `"CONFIG tail_keep=... compact_at=... summary=..."`
once at shim boot (goes into job.log, which the watchdog already tails),
and land the same triple in every response's `sc_debug` (ties directly
into C2's fix).

---

## MINOR

### N1. Dead/misleading env exports in `scripts/e1_driver.sh`
`scripts/e1_driver.sh`:
```
export LLM_MODEL="openai/sc-$MODE"  # MODE may carry :aX :cN suffixes LLM_BASE_URL="$BASE" LLM_API_KEY="sc"
```
Everything after the `#` is a bash comment — `LLM_BASE_URL` and
`LLM_API_KEY` are **never actually exported**, despite reading as if they
were. Currently harmless because `src/e1_agent.py` takes `base_url` and
`mode` as explicit argv (line 15-17: `mode, base, ws, taskf =
sys.argv[1:5]; llm = LLM(model=..., base_url=base, ...)`), not from env —
so the intervention is not actually silently broken today. But it is a
live footgun: if anyone later swaps in an OpenHands CLI/entrypoint that
*does* read `LLM_BASE_URL`/`LLM_API_KEY` from env (a very plausible
refactor, since that's the SDK's normal convention), it will silently
route all requests to whatever default base URL litellm falls back to,
which is exactly the shape of the SUMMARY_REQUEST confound. Fix now while
it's free: split onto its own line.

### N2. `e1_tasks.py`'s "repeat_signals" secondary metric is ungrounded (same class as the already-flagged SWE-Gym metric, but a different, not-yet-flagged instance)
`src/e1_tasks.py` `score()` (lines 213-215): `repeats =
len(re.findall(r"same command|already (?:ran|tried)", logtxt, re.I))`
matches the **agent's own natural-language text** in the log (e.g. if the
agent narrates "I already tried this approach"), not a grounded
detector of literally-repeated tool invocations. DECISIONS (07-05 eve,
Stage-2-complete entry) already flags this exact weakness for the
SWE-Gym "repeats-failed" behavioral metric ("fires for oracle too; needs
exit-code-grounded definition") — but that entry is about a different
file/metric; this is the same failure mode recurring, undocumented, in
the synthetic e1 task scorer. Any "kept-own-promises"/repeat-diagnostic
claim built from this field for synthetic rows should carry the same
caveat already applied to the SWE-Gym version.

### N3. `e1_tasks.py` scores with `pytest -x` (stop-at-first-failure); `swebench_tasks.py` does not
`src/e1_tasks.py` line 208: `["... "-q", "--no-header", "-x"]` vs
`src/swebench_tasks.py` `_run_pytest` (no `-x`). Doesn't change the
boolean `tests_pass` verdict, but means the two task-source strata
(synthetic vs standard, which DECISIONS explicitly says are pooled with
"task-source is a reported stratum, never pooled silently") have
systematically different diagnostic granularity in `pytest_tail` — a
synthetic-task failure log tells you about only the first failing
assertion; a standard-task failure log shows every failing id. Harmless
for the primary endpoint, worth normalizing if per-test diagnostics are
ever compared across strata.

---

## Summary

- CRITICAL: 3
- MAJOR: 2
- MINOR: 3

**Single worst finding:** with `SC_INCR_CACHE` withdrawn, mode A re-prefills
the *entire* growing conversation from scratch on every agent turn while
B/E's cost plateaus after the first compaction — so the shared 3600s
wall-clock alarm (`e1_driver.sh`) silently converts into a much smaller
effective turn-budget for A on long real-repo episodes, for reasons that
have nothing to do with memory loss. This can make the oracle arm A time
out or truncate its recall probe more often than B/E purely from compute
cost, which is exactly backwards for a story about compaction *damage*
and isn't measured, logged, or mentioned anywhere in DECISIONS/INCIDENTS.
