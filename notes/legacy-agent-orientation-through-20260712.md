# ARCHIVE — agent orientation through the pre-ultra handoff

This is the complete historical agent-orientation file that existed before the
2026-07-12 powered-successor takeover. It preserves the operational lessons and
old paper/protocol directives, but its v10/v12 and paper-first state is not
current. Current orientation lives in root `AGENTS.md`.

# AGENTS.md — orientation for any agent working in this repo

> FINDINGS.md = the headline results (read for conclusions). INCIDENTS/DECISIONS
> = process.
>
> `notes/` now contains GENERATED daily + overall SUMMARIES — start at `notes/README.md`.
> Worth reading for trajectory, context, and background on how we got here (esp. before a
> strategic decision or when picking up cold).

## What this repo is

A self-contained research project run by an autonomous coding agent on a 32 GB
Apple-Silicon MacBook: measuring whether KV-cache state written at generation
time carries semantic meaning that re-encoding the same text loses, across LLM
conversation-compaction boundaries.

## Read in this order

0. `INCIDENTS.md` — what went wrong, KNOWN vs THEORY, and which data is VOID.
   Read before trusting ANY result.
1. `STATE.md` — where things stand right now and what's queued. Start here.
2. `DECISIONS.md` — every methodology decision, deviation, and verified runtime
   fact. Non-negotiable reading before touching cache-surgery code.
3. `semantic-continuity-experiment-brief.md` — the original experiment design
   (arms A–F, probe suite, build ladder).
4. `amendments-from-external-review.md` — controls added after review (B-causal,
   negative grafts, leakage classes, metric hierarchy).
5. `COHERENT-STATE-PREREGISTRATION-AMENDMENT-{1..11}.md` — the additive frozen
   contract for the current `coherent-state-gapped-v10` assay. Amendment 10 adds
   independent prior-technical/semantic-launch authorization reconstruction,
   serial-terminal closure, and normal incremental-generation provenance.
   Amendment 11 changes execution scheduling only: the local ladder and one paid
   technical-only attempt may overlap, but semantics require a separately
   machine-enforced `L AND T` release overlay. Also read
   `COHERENT-STATE-AUTHORIZATION-CLARIFICATION-11A.md`.
6. `followup-explorations-arms-GH.md`, `phase2-scaleup-and-coding-extension.md`
   — future work, only after the main analysis.

## House rules (from the user + hard-won)

- **Never skip the build ladder.** Any new model/config must pass L0/L1/L3
  identity tests (`src/l0_identity.py`, `src/l1_l2_surgery.py`,
  `src/l3_l4_identity.py`) before its results count. Silent position bugs
  masquerade as findings.
- **Serial GPU jobs.** Two MLX processes halve each other; run one at a time.
- **Detach long jobs** (`nohup ... & disown`, PID file in scratchpad) —
  harness-tracked background shells have been killed mid-run. Use `python -u`,
  `tee` full output to a scratchpad log, monitor the log.
- **Timestamps around long commands** (`date` before/after) to spot pathological
  runtimes.
- **Commit often** (snapshot-style, fine-grained). No large binaries — model
  weights live in the HF cache, never in the repo.
- **Local subject model generates conversation text only.** All meta-work
  (authoring scenarios, judging answers, analysis review) goes to Claude
  subagents — the local model is the experimental subject, not a collaborator.
- **Subagents: small numbers, bounded batches** (≤2 concurrent).
- **Honest reporting.** Nulls are reportable; contaminated probes get
  reclassified, not deleted; negative results in DECISIONS.md, not buried.
- Temperature 0 for all evaluation; corpus generation is seeded sampling.

## Quick technical map

- Python via `uv run python src/<script>.py`; deps pinned in pyproject (mlx-lm
  0.31.3, transformers 5.0.0 — do not upgrade casually).
- `src/kvlib.py` — cache serialize/rebuild, GappedKVCache (position counter
  decoupled from storage), teacher-forcing (batched; NEVER compare batched
  logits to stepwise logits — different kernels).
- `src/arms.py` + `src/run_arms.py` — the experimental arms and driver.
- Qwen3 chat-template trap: the final assistant message (and the assistant
  message before a generation prompt) grows an empty `<think>` block, so token
  prefixes are unstable across re-renders. Always use the canonical non-final
  rendering (`canonical_ids` — dummy-user trick); locate message boundaries by
  scanning `<|im_start|>` positions, never by re-tokenizing prefixes.
- Data: `data/scenarios.json` (authored plants), `data/synthetic/`,
  `data/natural/` (composed conversations), `results/raw*/` (per-conversation
  arm outputs), `results/scores.json` (probe scoring).
- Conversation-summary archive: run `python3 scripts/update_notes_archive.py`
  from the repo root. This single entry point updates conversation notes,
  normalizes archive filenames, and recursively refreshes daily/month/year/
  archive summaries. It defaults every summary level to Codex
  `gpt-5.6-luna`/medium; use `--summary-provider claude --summary-model sonnet`
  to switch the whole pipeline to Claude. It includes repo-rooted user Codex
  sessions plus the configured Claude source and interleaves final labeled
  subagent responses into parent conversations without including their tool
  traffic. Use `--resummarize-all --force-rollups` for a clean raw-transcript
  rebuild. Small live-tail continuations are deferred by default; use
  `--force-small-continuations` only when you intentionally want to rewrite a
  note for a tiny recent exchange. Conversation notes include a generated
  `**Participants:** ...` paragraph; it includes `User` only when user messages
  are present, then full assistant model identifiers sorted by contributed text
  volume, including models recovered from contributing subagent sessions. If
  reasoning effort is present, append it to the model identifier with a hyphen,
  such as `gpt-5.5-xhigh`. Every model ID found in the raw conversation metadata
  must appear there. Filenames contain only compact user/model identifiers and
  omit effort, task labels, and source IDs. Each conversation note ends with an
  unlinked `## Conversation sources` list of the opaque main/subagent source IDs.
- **Amendment-11 semantic release:** never invoke the inventoried semantic job
  directly. Seal/verify the unique `results/v10_release/preflight_*.json` with
  `uv run python scripts/validate_semantic_release.py`, then launch only through
  `scripts/launch_semantic_release.sh`. A semantic result does not count until a
  separately committed final release attestation also passes.

## Source-control policy (07-05)

- Committed: all docs, src/, small JSON experiment artifacts in results/ (the
  experiment record — keep them versioned). NOT committed: datasets (*.parquet,
  HF caches), model weights, credentials (.gitignored), venv.
- A local pre-commit hook rejects staged files >4MB (.git/hooks/pre-commit —
  hooks do NOT travel with clones; recreate it from this note if absent).

## When stuck: invoke a different model family (user directive, 07-05)

Both Claude and (per user) OpenAI Codex CLIs should be available on this machine
(`claude` is on PATH; `codex` was NOT found on PATH as of 07-05 — check again /
ask the user if needed). If you are hitting repeated obstacles — many attempts,
little progress, or a diagnosis that keeps not paying off — you are STRONGLY
ENCOURAGED to invoke an agent from a different model family for a fresh
perspective:

- Use the highest available model + reasoning-effort settings.
- Point it at the relevant documents (STATE.md, DECISIONS.md, the experiment
  briefs, the failing code/logs) so it builds real context, and ask it to (a)
  explain what might be going wrong and (b) suggest what else to try.
- If the CLI supports session resumption, resume the same session for an
  extended back-and-forth rather than one-shot queries.
- Rationale (user): differently-trained models make different connections; the
  value is the independent perspective, not raw capability. Getting stuck is
  natural — treat cross-model consultation as a normal tool, not a last resort.

## Dual-agent convention (07-06)

A second agent (different model) may read this repo and occasionally create its
OWN new files/folders in non-conflicting paths, committing them directly (rare,
user-requested). Rules for the primary agent: stage with explicit paths (avoid
`git add -A` sweeps); unexpected new files are normal, not anomalies;
commit-lock races are retry-safe. Second agent: never modify existing files,
scripts, specs, state docs, or pods.

## Results-in-repo rule (07-06, from a real gap)

EVERY scored result must land in results/ and be committed, regardless of where
the run executed (pod, scratchpad, local). The repo IS the scientific audit
trail. Scratchpad is for working files (workspaces, logs, transcripts) only — a
score file is never a working file. Contaminated/void results go to
results/_QUARANTINE_* paths with READMEs, never deleted, never left outside the
repo.

## SAVE EVERY RENDER — absolute MUST (user 07-09, after incident #38)
The GENERATION of a render (native conversation replies + self-gen summary, ~16 min/conv) is the single
most expensive thing we do. **SAVE EVERY RENDER to disk as TEXT and COMMIT it — always, for ALL harness
work from here on. NON-NEGOTIABLE.** A harness that generates without persisting the generation is broken
by default. Why:
1. **Cheap re-testing forever:** any future test (per-layer tuning, champion/depth scan, rescue test,
   different alpha/region, a "crazy idea") reuses a saved render via a fast forward-pass — seconds, not
   minutes, near-zero GPU. The generation is done ONCE; everything after is cheap.
2. **Reproducibility:** the render IS the experimental artifact — saving it lets anyone reproduce/re-score.
Renders are small as text (~30-50 KB/conv) so they commit cleanly (well under the 4 MB hook limit).
NOT saving renders is EXACTLY incident #38 — a 6.3h render lost because it lived only in memory. This
extends the Results-in-repo rule from scored results to the (far more expensive) generation that produced
them. Every per-conv checkpoint = {rendered conversation, self-gen summary, traces, raw_EB, config}.

## Stateful-change checklist (07-06, after incident #11 — MANDATORY)

Any change adding/modifying retained state in a serving path must answer, IN THE
COMMIT MESSAGE: (1) What state is retained, keyed by what? (2) What bounds its
size, and where is that bound ASSERTED in code? (3) Who evicts it and when? (4)
What is the correctness/equivalence proof? (5) Which probe gate exercises it
before production traffic? Unanswered = do not deploy. Prose rules do not
survive attention under pressure; forms do.

## PUBLISHING: promote the best report to README.md (user 07-07)

- The repo's landing page is **README.md**. NEVER edit README.md directly.
- STALE-DRAFT NOTE (user 07-08): the prior working draft `REPORT.md` has been
  ARCHIVED to `notes/2026070771-report.md` and REMOVED from root. It is a PRIOR
  draft whose framing/style no longer reflect current requirements — kept for
  reference only. Do NOT treat it as the live paper or the target; reading it as
  "current" would mislead. The next paper is to be (RE)WRITTEN per the CURRENT
  requirements (METHODS-PROVENANCE-REQUIREMENTS.md + the notes `report2-plan` /
  `report-synthesis`) as a FRESH working file — do not resurrect the old draft.
- Workflow once a NEW working report file exists: edit that file (drafting,
  critics, Fable passes); when a major update is finished and CONFIDENT, publish by
  COPYING it over README.md (`cp <working-file> README.md`). README.md = the
  published snapshot; the working file = the live version; re-promote after each
  major confident update.
- **PODS OFF BEFORE THE PAPER (user 07-08):** the moment DATA COLLECTION is done, TERMINATE every
  pod — the writing phase is no-GPU, so pods must never run into it. Shutdown is a data-phase step,
  not a post-paper afterthought.
- **PAPER = HEAVY-FABLE collaboration (user 07-08):** written by owner + Fable + me, with HEAVY
  emphasis on Fable (as valuable drafting the paper as it's been everywhere else), holding the line
  on the stated requirements + the prior draft's good aspects. Written FRESH per current requirements.
- **PUBLISHING BOUNDARY (user 07-08, reaffirmed + sharpened 07-09):** writing the paper, putting it in
  the REPO, promoting to README when GENUINELY confident (after the full meticulous review stack —
  Fable multi-angle passes, readability, focus), committing, and **`git push`ing to the repo remote** =
  AUTONOMOUS and explicitly authorized ("feel free when you're all done to promote it to readme and push").
  The ONLY thing held for the owner is anything OUTSIDE the repository — the HF community blog post / any
  public posting. "Inside the repo" (incl. git push) = go; "outside the repo" = needs the owner.

## FINAL PAPER REVIEW: multi-perspective Fable passes (user 07-07)

For the FINAL review of the FINAL paper (after all work done), IN ADDITION to
the normal pipeline: run ~3 Fable subagent reviews, NO TOOLS (Fable is expensive
— just reasoning over the text), each prompted from a DIFFERENT angle to get
varied perspectives:

- (1) generic: "review this paper — what's good, what's not, structure, flow,
  suggestions to improve."
- (2+3) slightly different framings (e.g. a skeptical-reviewer angle; a
  first-time-reader/accessibility angle; a "what would make this
  stronger/publish- ready" angle). Vary the prompt to surface different angles.
- Synthesize the three into the final revision. Tool-less to keep cost down.

## OUTPUT NAMING: unique, self-announcing (user 07-07, after a 30B run overwrote 27B)

Experiment outputs MUST be uniquely named so a re-run NEVER overwrites a prior
result (a 30B run silently clobbered the 27B summary.json at a shared path).

- Output filename = <experiment>_<model-slug>_<UTC-timestamp>.json (unique by
  model AND time). Never a fixed shared "summary.json".
- At PROCESS START, LOG the resolved model + the exact unique output path it
  will write ("RUN <exp> model=<m> → results/.../<unique>.json"), so we know
  where to look before it finishes.
- Also verify at launch (rule 26 extension): confirm the RIGHT MODEL loaded, not
  just that work is happening — a run using the wrong/default model looks
  healthy but answers the wrong question (the 30B run that was actually 27B).

## PAPER REVIEW — Codex/GPT-5.5 external review (user 07-08)

For EACH MAJOR REVISION we might SHARE, get — AT LEAST ONCE (NOT repeatedly;
it's heavy) — an external review from **Codex with GPT 5.5 on EXTRA-HIGH
effort**. This complements (does not replace) the Fable multi-perspective
passes + adversarial critics + terminology-consistency dimension. So the review
stack for a shareable major revision = Fable ~3 tool-less angle passes +
critics + terminology check + ONE Codex/GPT-5.5/xhigh review. INVOCATION (I run
it myself — do NOT ask the user): codex CLI is installed
(`~/.nvm/.../bin/codex`, add that nvm bin to PATH in the Bash call) and authed
via the user's ChatGPT login. ~/.codex/config.toml ALREADY defaults
model=gpt-5.5 + model_reasoning_effort=xhigh, so: export
PATH="/Users/jeb/.nvm/versions/node/v25.4.0/bin:$PATH" codex exec -s read-only
"Review /Users/jeb/experimentation/REPORT.md as a skeptical peer reviewer:
<focus>. Do not modify files." Use -s read-only for reviews (read the paper,
don't edit). It runs autonomously (approval_policy=never). Capture its output
into the review record. Gate: don't ship a shareable major revision without this
Codex/GPT-5.5/xhigh review on record.

## FABLE MUST HAVE CURRENT FACTS (user 07-08)

The findings have evolved MASSIVELY (estimator bug → robust metric; keys neutral
not hurting; fixed-summary suppresses graft → MECHANISTIC FINDING graft needs
model's OWN summary; cross-arch map). ANY Fable writing/review/gut-check prompt
MUST include or point to the CURRENT FINDINGS.md (not stale memory of earlier
claims). Give Fable the up-to-date facts explicitly — it does NOT see FINDINGS
unless the prompt provides it, and writing from stale facts would reintroduce
corrected errors (the -0.31 stance, the 'keys hurt', the fixed-summary numbers).
Brief Fable on: the robust metric, the CI'd effect (referent significant/sense
underpowered/stance null), keys-neutral, and the own-summary mechanism.

## FABLE CONTEXT MUST BE MINIMAL AND DECISION-SPECIFIC (owner 07-11)

Do **not** keep resuming a huge accumulated Fable session or dump the whole repo / full
project history into routine consultations. That is expensive, can make the model less
independent, and encourages continuation of the team's existing frame instead of a fresh
perspective. Give Fable the smallest structured evidence bundle that is sufficient for
the exact decision: normally the current question, the literal primary artifacts or
short audit notes that bear on it, and only the cited code/spec snippets needed to verify
them. State material current facts explicitly, but omit irrelevant trajectory and prior
argument. Point to `notes/README.md` and the generated daily/overall summaries as optional
orientation when background or project trajectory may help; do not require Fable to read
the full archive. Every focused prompt must state the review boundary and explicitly say
that unrelated settled questions are out of scope and need not be re-litigated. Prefer a
fresh session for a genuinely fresh perspective. If a focused consult
cannot finish for roughly `$4`, stop and narrow the context/question rather than raising
the cap or repeatedly resuming. A large comprehensive context is appropriate for the
final paper synthesis/final review, where integration across the whole record is the task.

## FABLE WRITES TO A NOTES FILE — every serious consult (owner 07-09)

Fable's major outputs (verdicts, assessments, trajectory reviews, experiment
designs) were living only in agent-to-agent replies and GETTING LOST. New standing
practice for EVERY serious Fable consultation: **I create an empty `notes/` file**
(naming convention `YYYYMMDD<counter>-slug.md`, next counter after the latest),
pass Fable the **minimal sufficient decision-specific evidence bundle** defined above,
and instruct it to **write/edit its
assessment directly into that file itself** (it owns the file), returning only the
path + a short topline. This preserves Fable's reasoning for posterity and for the
paper. Combine with the un-anchored rule (don't lead the prompt) and CURRENT FACTS
above. Default this going forward; don't let a serious Fable result be chat-only.

## FRAMING PROVENANCE (owner 07-09): mitigation-first was the owner's intent ALL ALONG

The daily summaries narrate the 07-05 mitigation-first "reframing" as an external-review
redirection. The owner corrects this: mitigation was their intent from the start; the
mechanism-as-finding framing was the AGENTS' misunderstanding, which the owner only later
noticed. Never narrate project history (in the paper or briefings) as "we thought the
mechanism difference was the finding until review corrected us." Full correction:
notes/2026070901-framing-provenance-correction.md.

## FABLE FREEDOM on paper title + intro (user 07-08)

When Fable works on the PAPER writing/review, it has FULL FREEDOM to change the
paper's TITLE and the opening few sentences (the forum-post intro blurb) to
sound better — it's the strongest at making those land. Don't constrain it
there; let it improve the title/opening. (Current title: "Value grafting:
recovering lost semantic continuity when a conversation is compacted" — Fable
may revise.)

## FINAL PAPER: consider letting FABLE do the INITIAL DRAFTING (user 07-07)

For the FINAL version of the paper, we might let Fable write MOST of the initial
draft itself — not just review it — PROVIDED we can give it the RIGHT
INFORMATION (full current facts/findings, the results, the framing decisions,
the honesty guardrails). It's the strongest writer here. Process is otherwise
UNCHANGED: we still do the iterations + the full review stack, and I (main loop)
still make whatever changes I judge necessary. So: Fable-initial-draft (with a
thorough facts brief) → iterate/critique/terminology → Fable readability passes
→ Codex/GPT-5.5 → my edits → ship. Do this IF feasible (i.e. if we can brief it
well enough that its draft is a real starting point, not a re-explain). Requires
the [[fable-must-have-current-facts]] discipline taken to its fullest — a
complete, current, structured brief. Fable also has full freedom on title +
intro.

## HARD RULES & LEARNINGS — every agent (main + subagents) MUST follow

These are IN THE REPO on purpose so all agents can see them (private memory
files can't be read by subagents).

**Git**

- Work ONLY on trunk. NEVER create/use branches. If a branch appears,
  fast-forward it into trunk and delete it (just moving refs, non-disruptive).
- Commit AND push to origin/trunk after every unit of work. Never leave critical
  code/data uncommitted.
- NEVER `rm`/delete/overwrite uncommitted work. Commit the thing before running
  anything that consumes or cleans it. A cleanup step must never run after a
  failed step (no unconditional `rm` after a merge/build).

**Verify the boring things before anything clever or expensive**

- Use the EXACT model id + config the known-good result used — not just the same
  family/size. (Cost us hours: ran thinking Qwen3-30B-A3B vs the non-thinking
  Instruct-2507 the +0.156 was measured on.)
- Positive-control a pipeline on its ACTUAL production config, not a proxy.
  ("Equivalence-verified on fixed summaries" did NOT cover self-gen — twice.)
- Read the actual NUMBER yourself. `status=OK` != correct.
- If two independent apparatuses fail IDENTICALLY, the bug is in SHARED code / a
  shared input — look there first.
- Don't harden/re-engineer correct code to soothe a misdiagnosed alarm; you'll
  introduce real brittleness.

**Subagents & generation**

- SHARD independent multi-item work across parallel subagents from the START
  (quality AND speed): author diversity + fresh attention per item. Don't run N
  items sequentially in one subagent.
- Text/content generation → fast model mix (Fable/Opus/Sonnet/Codex), never a
  local model (MLX/Ollama) except quick sanity checks.
- NEVER kill a subagent off a proxy signal (output-file size/mtime). Check real
  progress (recent activity, its last message) before any destructive action.

**When stuck → consult Fable EARLY**

- **The urge to stop and ask/report to the USER is the signal to consult FABLE
  instead — and keep working.** When stuck or uncertain, do NOT turn to the user
  for direction; that is offloading the thinking. Consult Fable autonomously
  (it's the resource for the thinking) and keep driving. Escalate to the user
  ONLY for decisions genuinely theirs — spend limits, scope, taste — never to
  resolve your own confusion. Fable first, then results; bring the user
  decisions and outcomes, not "here's where I'm confused, what do you think?"
- The moment a fix hasn't converged in ~1-2 attempts, or a subagent is looping,
  or a result is confusing: STOP and consult Fable for the STRATEGIC/diagnostic
  view. Do NOT grind for hours first. Fable advises; it does not implement.
  (Fable caught the wrong-model class of bug and the nativeness confound that
  hours of narrow debugging missed.)

**Record learnings IN THE REPO** (DECISIONS.md / INCIDENTS.md / FINDINGS.md /
here) — not in private memory files agents can't see.

## PAPER: methods/provenance are a BLOCKING requirement

Before the paper ships, it MUST satisfy every item in
METHODS-PROVENANCE-REQUIREMENTS.md (data provenance = who/what generated each
token, exact model ids, procedures, gates, design rationale, reproducibility).
Every prior writeup omitted this; it makes the result un-reproducible. Brief
Fable + critics + Codex to review the paper AGAINST that file. Provenance gaps =
blocking failure.

## Pod / RunPod ops (learned 07-08, hours lost to flaky pods)

- **HOST DRIVER IS PART OF THE RUNTIME (incident #43).** The same Secure A100
  GPU type + container image returned drivers `580.159.03` and `550.90.12`;
  CUDA 13 initialized only on the former. A container does not pin the host
  kernel driver. Provision with `SC_POD_ALLOWED_CUDA` (RunPod
  `allowedCudaVersions`), then independently gate actual GPU name, driver, and
  memory through `src/pod_admission.py` **before bootstrap**. Exact v12 must use
  `scripts/launch_coherent_canary_v12_technical.sh` (CUDA 13.0, driver
  `>=580.65.06`, A100-80GB, three attempts, no unfiltered fallback). An
  AI-recommended provider is not validated until a provider-qualification
  checklist proves it can enforce the experiment's host-level invariants.
- **POD RETRY SAFETY (incident #44).** On exact v12, only exit 85 (explicit
  provider no-allocation) and 86 (rejected host with successful DELETE) may
  retry, within the three-attempt bound. Exit 87 means cleanup failed; every
  other status stops. API calls are timeout-bounded, required credentials are
  deployed fail-closed, and the exact wrapper runs the frozen verifier locally
  before allocation. Never re-run the wrapper after an ambiguous post-launch
  failure; inspect the one registered pod and preserve wanted artifacts first.
- LAUNCH detached jobs the PROVEN way: `scripts/launch_pod.sh <name> <job.sh>`
  (it does `nohup bash job.sh > job.log 2>&1 &` and the ssh RETURNS) — this
  reliably detached all session. Or a run_in_background Bash running an inline
  `nohup python … > log 2>&1 & echo PID` that returns immediately. Do NOT use
  `setsid … &` or a foreground `bash script` held open by the ssh — on a flaky
  pod the connection drop (exit 255) kills the job and no log is ever written.
- A DEGRADED pod (API shows desiredStatus=RUNNING but runtime=None / uptime
  None) answers QUICK commands (nvidia-smi, ls) but DROPS sustained connections
  and won't launch jobs. Do not fight it: terminate + reprovision. Symptom =
  launches silently produce no log.
- macOS has NO `timeout` command — never wrap ssh in `timeout N`; use ssh -o
  ConnectTimeout=15 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 instead.
  (Also: pass ssh -o flags INLINE, not via a shell variable — a `$O="-o …"` var
  expands wrong: "keyword stricthostkeychecking extra arguments".)
- Community RunPod create 500s are TRANSIENT (availability fluctuates) — retry,
  don't conclude it's down.
- Don't grind on infrastructure. If a pod degrades, terminate+reprovision; the
  science isn't the pod.

- **RELIABILITY ([RELIABILITY.md](RELIABILITY.md)):** START at the **PREVENTION MAP**
  at the top — the single table of "if you do X, this mechanism catches you," fail-closed.
  The hard rules it indexes: (1) fail-closed PRE-FLIGHT GATE before any scaled spend
  (`scripts/preflight.sh`; now also **check B2** = wrong-checkpoint interlock, incident #34);
  (2) OBSERVABILITY / SRE so failures self-report; (3) **MONITOR TRUST GATE** — a monitor is
  untrusted until `scripts/monitor_selftest.sh` prints "MONITOR CLEARED" (fault-injection incl.
  happy-path); monitors SOURCE the pure `scripts/classify_pod.sh`, never re-implement state logic
  inline; (4) **PROCESS TRIPWIRES** — concrete conditions (canary-before-fanout, verify-the-number,
  consult-Fable at ≤2 failed attempts, commit/push hygiene, no false confidence) that replace the
  fuzzy behavioral rules. `scripts/launch_pod.sh` now bounds its ssh (a hang → nonzero exit, not a
  starve) and runs a **post-launch real-work check** (crash/never-started → `exit 1`, incident #28/#35).
  These are solved problems; use the established patterns, not hacks.

## HARD RULES added from notes-audit (07-08) — were user directives but not written down
- **NEVER rewrite/amend/rebase git history. Correct ADDITIVELY** (a dated correction note/commit).
  When ~314 commits were found mislabeled (Fable vs Opus 4.8), the user's absolute rule was: never
  edit history — add a dated PROVENANCE-CORRECTION note instead. A "cleanup" rebase would violate it.
- **Keep sensitive/charged terms OUT of all file names, directory names, and job names.**
  Standing user directive. This repo is pushed to GitHub — names are the exposure surface.
  Archive+delete such working dirs when done, per the user.
- **Credential handling risk (B5, flagged not fixed):** secret keys (.huggingface_key, .runpod_key,
  .openrouter_key) are kept out of git via `.git/info/exclude`, which does NOT travel with clones and
  is invisible to other agents. `git add -A` could stage them. Prefer a tracked `.gitignore` entry +
  a pre-flight check that no key file is staged. (Do not `git add -A` — stage explicit paths.)

## Behavioral hard rules (07-08 — trust-critical)

- **Report ONLY what you have OBSERVED (past tense); explicitly name what you have NOT verified.**
  No predictions; no "it works / is fixed / is robust / will work." Those claims were false many
  times and destroyed the user's trust. Success is declared retroactively from an observed number,
  never in advance.
- **Don't kill WANTED WORK / rm / spend by INFERENCE about what the user wants for their experiment.**
  Do not infer "they probably want me to stop" and act on it — if unsure about their INTENT, ASK or WAIT.
  BUT this is NOT a ban on pod lifecycle management: terminating an idle / surplus / failed / degraded
  pod for a clear OPERATIONAL reason (budget, waste) is your judgment — over-rigidity is also a failure.
  Test: guessing they changed their mind about the work (→ ask) vs. an operational fact (→ decide).
  (Full rule in RELIABILITY.md.)
