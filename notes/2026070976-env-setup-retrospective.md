# Environment / dependency-setup retrospective (2026-07-09)

*Outside read by a read-only mining subagent. Take with a grain of salt — I flag
confidence vs. speculation explicitly at the end. All citations are file:line as
of 2026-07-09.*

## TL;DR

There is a real, recurring "environment fire," and it is almost entirely a
consequence of a **split-brain runtime** (LOCAL = MLX 4-bit / `uv` / py3.12 vs
POD = HF bf16 / **system** python3.11 / pip) whose POD side has **no single
source of truth**. The pod dependency setup is copy-pasted, per-job, and
inconsistent — including a live self-contradiction on the transformers pin. The
project already discovered the meta-lesson (07-08: "written-only rules don't
survive; gates do") and built gates for *launch* failures, but the **dependency
install itself was never turned into one artifact + one gate**. That's the gap.

---

## 1. Recurring environment/dependency failure modes (with evidence)

**A. The split-brain itself is the root cause.** It's documented, not hidden:
- LOCAL: `uv` project, Python 3.12, `mlx-lm`, MLX 4-bit — `pyproject.toml:4`
  (`requires-python = ">=3.12"`), `DECISIONS.md:7`, `AGENTS.md:55-56`.
- POD: system python3.11, bf16 HF/transformers, from the base image
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (`src/pod.py:29`).
  Pods install deps ad-hoc via `python3 -m pip …` inside each job script.
- The precision consequence is a logged provenance hazard: "the local-vs-pod
  'same model' is 4-bit vs bf16" (`DECISIONS.md:50`). So the two halves aren't
  just different setups — they produce non-comparable numbers unless annotated.

**B. transformers version thrash — and a *live contradiction*.** This is the
single clearest "re-solved from scratch" symptom.
- LOCAL pins `transformers==5.0.*` because mlx-lm needs ≥5 (`pyproject.toml:12`,
  rationale `DECISIONS.md:7`).
- On PODS, transformers 5.x **breaks weight loading** ("automatic weight
  conversion" RuntimeError) — so `job_sweep.sh` force-pins 4.57.1 and *verifies
  it took, refusing to run otherwise* (`scripts/job_sweep.sh:58-66`). Good.
- BUT other pod job scripts install the version `job_sweep.sh` explicitly
  forbids: `job_kv_layer_probe.sh:34` and `job_kv_sweep.sh:24` both do
  `pip install -U "transformers==5.0.*"` **on the pod**. Meanwhile
  `job_batchtest.sh:9`, `job_gate.sh:31`, `job_cross_arch.sh:62`,
  `job_trusted.sh:15` use `">=4.57.0,<5"`, and `job_effect_bound.sh:42` uses
  `">=4.57.0,<5"` too. So across seven pod scripts there are **three different
  transformers specs**, one of which is known-broken on the pod. The daily
  summary even lists "an unpinned transformers major-version auto-upgrade" as a
  cause of tearing down a fan-out (`notes/20260708.md:75`).

**C. pandas / pyarrow missing on pods.** `run_swegym_hf.py` needs pandas
(`src/run_swegym_hf.py:67` — `import pandas as pd; pd.read_parquet(...)`), as do
`src/swebench_tasks.py:55` and `scripts/swb_filter.py:52`. Only **one** job
script installs them: `job_effect_bound.sh:42` (`… pandas pyarrow`). None of the
other pod bootstraps install pandas/pyarrow. So any pod provisioned by
`job_sweep.sh` / `job_cross_arch.sh` / etc. that then runs the SWE-Gym path
would `ModuleNotFoundError` on pandas. (Confidence: high that the gap exists;
**medium** on whether it has actually bitten in a run — I didn't find a specific
incident numbered for it, only the structural gap the task described.)

**D. torch / CUDA / torchvision breakage on pods (well-documented, repeated).**
- Incident #27 (`INCIDENTS.md:361-376`): community-pod base image differed →
  `pip install -U … torch` upgraded torch to cu130 vs a cu125 driver →
  `torch.cuda.is_available()=False` → 30B **silently loaded to CPU**; the
  `torchvision::nms` import error was the same mismatch. Rule 24: never
  `pip install -U torch` on a pod. Every pod job now does a defensive
  `pip uninstall -y torchvision` (`job_gate.sh:29`, `job_cross_arch.sh:60`,
  `job_kv_layer_probe.sh:32`, `job_sweep.sh:56`, `job_kv_sweep.sh:23`) — i.e.
  the same workaround copy-pasted five times.

**E. Missing base-image tools / packages surface only at run time.**
- rsync not preinstalled on community pods → deploys silently no-op'd
  (`INCIDENTS.md:364`; now `apt-get install -y rsync` in `launch_pod.sh:51`).
- tau2 needed `rank_bm25` (uninstalled), only visible on RUN not on code-read
  (`INCIDENTS.md:335-338`, incident #24).
- 5/16 candidate models turned out to be multimodal wrapper classes that won't
  load in the text path (`notes/20260708.md:70-71`) → now caught by
  `preflight.py` check B (`scripts/preflight.py:34-77`).

**F. Interpreter/backend confusion (LOCAL side).** Incident #31
(`INCIDENTS.md:418-423`): local 4-bit 4B MLX reflexively reused to generate
corpus text (slowest possible path) — "using local MLX is insane" (user). Not a
missing-dep bug, but the same split-brain confusion about *which runtime is for
what*: MLX is the experimental subject + dev sanity tool only, never the
content/compute backend (`DECISIONS.md:153`, memory `question-the-backend`).

**Pattern:** the same handful of pod facts (uninstall torchvision, pin
transformers <5, install accelerate/safetensors/hf_hub, don't upgrade torch,
add pandas/pyarrow for the SWE path, load HF token) get re-derived and
re-pasted per job script, and drift apart. That drift *is* the recurring fire.

---

## 2. Existing guidance — scattered and partly contradictory

There **is** guidance, but it's spread across ≥5 places and no single canonical
pod-env definition exists (I confirmed **no `ENVIRONMENT.md`** and no remote
`requirements.txt` — `find` returned nothing; the only lock is `uv.lock`, which
governs LOCAL only):

- `AGENTS.md:53-60` — the LOCAL quick technical map (uv, py3.12, pins) and the
  "Pod / RunPod ops" section (`AGENTS.md:345-364`) — but that's about *launch
  mechanics* (detach, ssh flags, degraded pods), **not** the dependency set.
- `DECISIONS.md:7,50` — the transformers-pin rationale and precision provenance.
- `INCIDENTS.md` #27/#30/#24 — the torch/CUDA, disk, and missing-dep lessons,
  encoded as prose "RULES 24/25/26."
- `RELIABILITY.md` PREVENTION MAP + `scripts/preflight.py` — a real fail-closed
  gate, but it checks **model ids, rsync source paths, secrets, git-clean, and
  syntax** (`preflight.py` docstring lines 20-30). It does **not** verify the
  remote python version, the transformers pin that will be installed, or that
  pandas/pyarrow/etc. are covered for the job being launched. The gate is
  local-precondition-only; the *dependency environment on the pod* is outside
  its coverage.
- The seven `scripts/job_*.sh` bootstraps — the *de facto* env spec, but
  duplicated and mutually inconsistent (§1B/§1C/§1D).

**What's missing/contradictory:**
1. No canonical pod dependency manifest — the truth is smeared across 7 scripts.
2. transformers pin is contradictory across those scripts (§1B) — `==5.0.*` on
   two pod scripts vs the `<5` that `job_sweep.sh` says is *required or it
   refuses to run*.
3. pandas/pyarrow coverage is per-script and mostly absent (§1C).
4. The `pip uninstall torchvision` + `don't upgrade torch` lesson lives as
   copy-pasted lines + a prose RULE, not one shared bootstrap.

---

## 3. Did the owner previously ask for a canonical env spec / setup standard?

**Not found** — I did not find an explicit owner request for an `ENVIRONMENT.md`,
a pinned remote requirements file, or a "provisioning standard." I searched the
conversation-user notes, daily summaries, AGENTS/RELIABILITY/DECISIONS for env /
python-version / setup / "why do we keep hitting this" framings; the hits were
either about the *science* ("secondary metric" etc.) or about launch mechanics,
not a standing request for a canonical env doc.

The closest things to a directive are **derived rules after incidents**, not a
pre-emptive ask:
- "NEVER `pip install -U torch` on a pod" (Rule 24, `INCIDENTS.md:374`).
- The 07-08 meta-conclusion (this is the strongest supporting evidence for the
  recommendation, quoted): *"failures recur specifically where guidance exists
  only as prose that must be recalled under pressure — every case converted into
  an automated gate … stopped recurring, while written-only rules did not."*
  (`notes/20260708.md:79-85`; echoed in AGENTS.md's "forms not prose" framing,
  e.g. `AGENTS.md:148-149`).

So the honest answer to the owner's suspicion ("there's an existing request we
dropped"): I found **no dropped explicit request**, but I did find that the
project's *own stated principle* (turn recurring prose-rules into one gate)
was **applied to launch/model checks and never applied to the dependency
install** — which is exactly why this one keeps recurring.

---

## 4. Recommendation (grain of salt — proportionate, no rewrite)

**Single highest-leverage fix: one pod bootstrap artifact + one preflight
assertion.** Concretely, and small:

1. **`scripts/pod_env.sh`** (or `requirements-pod.txt` + a 10-line installer):
   the ONE place that encodes the pod dependency truth — pin transformers to the
   known-good `>=4.57,<5`, `pip uninstall -y torchvision`, never upgrade torch,
   install `accelerate safetensors huggingface_hub pandas pyarrow`, load the HF
   token, and `verify` (the assert-it-took-or-exit pattern already in
   `job_sweep.sh:60-66`). Every `job_*.sh` replaces its bespoke pip block with
   `source pod_env.sh`. This kills the three-way transformers contradiction
   (§1B), the pandas gap (§1C), and the five copies of the torchvision dance
   (§1D) in one move. **This matches the project's own "gate not prose" lesson**
   and is a ~1-file change, not a rewrite.
2. **Extend `preflight.py` with a cheap dep-manifest check (fail-closed):**
   assert the job script's install line resolves to `transformers<5` and that,
   if the job's code path imports pandas/pyarrow, the bootstrap installs them.
   Preflight already parses the job + launcher (`preflight.py` checks C/E) — this
   is an incremental assertion in an existing gate, the cheapest possible place.
3. **A short `ENVIRONMENT.md`** that states the split-brain in one paragraph
   (LOCAL uv/py3.12/MLX-4bit vs POD system-py3.11/HF-bf16, the transformers
   contradiction and *why*, precision provenance) and **points at** `pod_env.sh`
   as the source of truth. Doc for orientation; the *enforcement* is the script +
   gate, per the 07-08 principle. Keep it short so it doesn't rot.

I'd rank #1 as the actual leverage; #2 makes it fail-closed; #3 is orientation.
Do **not** try to unify local and pod runtimes — the split is intrinsic (Apple
Silicon MLX vs CUDA bf16) and already correctly reasoned; the fix is a single
source of truth for the *pod* side, not convergence.

---

## Confidence separation

**Confident (verified from files today):**
- The split-brain, its file locations, and the precision consequence (§1A).
- The three-way transformers-pin inconsistency across job scripts, including two
  pod scripts installing the `==5.0.*` that `job_sweep.sh` refuses to run (§1B).
- pandas imported by `run_swegym_hf.py`/`swebench_tasks.py`/`swb_filter.py`, and
  only `job_effect_bound.sh` installs pandas/pyarrow (§1C).
- torch/CUDA/torchvision incident #27 and the five copy-pasted uninstall lines.
- No `ENVIRONMENT.md`, no remote requirements file; preflight is
  local-precondition-only and does not cover the remote dep environment.
- The 07-08 "gates beat prose" conclusion, quoted verbatim.

**Speculation / lower confidence:**
- Whether the pandas/pyarrow gap has *actually* crashed a real run vs. being a
  latent structural gap — I found the structural gap, not a numbered incident
  for it. The task framing implied it bites; I couldn't confirm the specific
  event.
- Whether `job_kv_layer_probe.sh` / `job_kv_sweep.sh` (the two `==5.0.*` pod
  scripts) are still *actively used*, or are older/abandoned scripts whose pin
  simply never got updated when `job_sweep.sh` learned the 4.57 lesson. If
  they're dead, the contradiction is lower-stakes than it looks — but it's still
  a live foot-gun in the tree.
- I did not exhaustively read all 130+ notes; a buried owner request could
  exist. "Not found" means "not found in a broad targeted search," not "proven
  absent."
