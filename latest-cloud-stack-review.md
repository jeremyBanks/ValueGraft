# Latest Cloud Stack Review Notes

Review date: 2026-07-05.

Scope: latest RunPod/HF/cloud-plan work around `cloud-plan.md`, `src/pod.py`,
`src/run_lme_hf.py`, `src/kvlib_hf.py`, `src/arms_hf.py`, and
`src/score_lme.py`. This is a review note only; no experiment code was changed.

## Recommended Changes

### 1. Ignore generated pod state before it gets committed

`.pod_state.json` is generated lifecycle state from `src/pod.py` and is
currently untracked, not ignored. It contains pod/account/runtime metadata
such as pod id, IP, ports, machine id, and env fields. It should not be part
of commits.

Recommendation:

- Add `.pod_state.json` to `.gitignore` or local exclude.
- Consider deleting it automatically after a successful `pod.py terminate`,
  or marking terminated state explicitly so stale pod ids are harder to reuse
  accidentally.

The smoke result directory `results/longmemeval_smoke_hf/` is also currently
untracked. If it is meant only as a local smoke artifact, either ignore it or
commit it intentionally with a note.

### 2. Make LongMemEval HF writes atomic before full cloud runs

`src/run_lme_hf.py` writes each result directly to its final JSON path and
skips any existing file on resume. If the process is interrupted mid-write, a
truncated JSON file can be left behind and future runs will skip it.

Recommendation:

- Write each item to `outfile.tmp`.
- Close/flush the file.
- Atomically rename to the final `outfile`.
- On resume, validate existing JSON before counting it as done.

This matters because the cloud plan depends on long resumable batches and the
repo has already had incidents with killed or duplicated long jobs.

### 3. Strengthen `score_lme.py` for benchmark-scale claims

`src/score_lme.py` is fine for a quick judged summary, but for a 500-question
benchmark run it should preserve more analysis state.

Recommendation:

- Save the scored rows, not just the verdict mapping.
- Report missing verdicts explicitly.
- Provide clean aggregates excluding `s_leak`.
- Stratify by `question_type`, since the LongMemEval results already suggest
  strong frame/type dependence.
- Optionally report conditional metrics on the subset where arm `A` is judged
  `CORRECT`, since that is the clearest test of compaction damage and recovery.

Without this, later write-up numbers are harder to audit and reproduce from
the committed artifacts.

### 4. Audit HF `H-gap` semantics against MLX on one shared item

The HF runner uses `packed_suffix(...)` for `H-gap`, while the original MLX
LongMemEval runner treated `H-gap` as a rendered-message arm. This may be
equivalent because the appended user-turn frame is token-identical, but it is
worth checking explicitly before relying on HF `H-gap` results.

Recommendation:

- For one shared LongMemEval item, compare the token suffix used by MLX
  `H-gap` and HF `H-gap`.
- If they are intended to differ, document the reason in `DECISIONS.md`.
- If they are intended to match, assert it in a small smoke check.

This is lower risk than the atomic-write issue because `H-pack`/`B-min-pack`
are the central deployable arms, but `H-gap` is used as a bridge to earlier
results.

### 5. Make pod setup assumptions explicit in the runbook

`src/pod.py` now creates and terminates pods, but the cloud plan still relies
on several operational assumptions that should be codified before spending
real time on the pod.

Recommendation:

- Record the exact setup commands for installing repo deps on the pod.
- Ensure the pod environment exports or otherwise uses the Hugging Face token
  when needed.
- Verify the `ssh-cmd` path against actual RunPod `portMappings`; RunPod API
  response shapes can vary.
- Log balance before and after each stage, as promised in `cloud-plan.md`.
- Commit or intentionally discard the current uncommitted `src/pod.py`
  changes before relying on it for a longer run.

The recent `src/pod.py` diff adds useful HTTP error reporting and a user-agent
header, and removes `startSsh`; if that reflects what made the live server
work, it should become an intentional commit.

## Overall Read

The cloud refactor is pointed in the right direction. The plan is now focused
on scaling evidence for the known Qwen3-30B bf16 target on standard datasets,
which matches the strongest current story. The HF port has an identity ladder,
the LongMemEval runner has a smoke path, and the RunPod lifecycle tool is the
right level of automation for a prepaid single-pod workflow.

The main things to fix before trusting a long cloud run are operational rather
than conceptual: generated pod state, atomic/resumable writes, auditable
scoring outputs, and a small HF-vs-MLX suffix sanity check.
