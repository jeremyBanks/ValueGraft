# Phase 2 — Scale-Up and Coding-Agent Extension

_Third companion document, following the main experiment brief (Arms A–F) and
the follow-up explorations (Arms G–H / ValueGraft naming)._

## Standing caveat — read first

This document was written **without visibility into how the local experiments
have actually gone**. It was drafted in parallel with, not in response to, your
work. Treat everything here as suggestions from a well-informed but
uninformed-about-your-results collaborator: where your empirical findings,
practical discoveries, or better judgment disagree with this plan, **your
judgment wins**. Adapt freely, discard freely, and record what you changed and
why in `DECISIONS.md` as before. The only firm requests are the validity rules
carried over from the main brief (identity tests before results; controls before
claims; honest reporting of nulls).

## Trigger and purpose

This phase activates only if the local results look meaningful — meaning at
least one contrast from the matrix (B-min vs H, B vs E, E vs C, E vs G) shows a
consistent, non-trivial effect that survived the leakage and identity controls.
The purpose of Phase 2 is to convert a local mechanism finding into a
publication-shaped result, which needs two things the MacBook cannot provide:
**realism** (evaluation on the workload where compaction actually hurts — coding
agents) and, budget permitting, **scale** (evidence about whether the effect
grows or shrinks with model size).

## Budget strategy (~$100–200 in cloud credits)

That budget buys roughly 40–100 hours on a single 80 GB-class GPU (A100/H100),
depending on provider and spot pricing — compare AWS against Lambda, RunPod, and
similar; the difference is often 2–3×. It is enough for **one** of the
following, not both. Choose based on what the local results showed:

- **Strategy S (scale):** if the local effect was clear, rerun the core
  contrasts on one substantially larger dense model (70B–120B class) to answer
  the first question reviewers will ask: does graft-take grow, shrink, or hold
  with model size? A single clean scale point plus the local results is a
  stronger paper than many runs at one size.
- **Strategy P (power):** if the local effect was real but marginal, spend the
  hours on statistical power at the local model size — many more trajectories,
  tighter confidence intervals, more probe instances per category — rather than
  a bigger model.

The deliverable of the local phase that feeds this choice is a short decision
memo: which hypothesis survived, which contrast was largest, and therefore which
strategy the credits buy.

## The coding-agent extension (the realism leg)

Coding sessions are where production compaction pain concentrates, and they
solve the synthetic-materials problem: **no synthesis needed.**

### Materials (real, in ascending effort)

1. Long transcripts from real agent sessions (e.g., exported Claude Code /
   similar tool logs, if available).
2. Public agent trajectory datasets — SWE-agent and OpenHands trajectory dumps
   on SWE-bench tasks are downloadable: thousands of genuine multi-turn sessions
   with tool calls, errors, retries, and fixes.
3. Self-generated: run an open-source agent harness (OpenHands, Aider, or
   similar) with the same model under test against SWE-bench-Lite issues,
   recording full trajectories. Real repositories, real errors, and the test
   model produced its own materials.

Compact at tool-call boundaries; keep the most recent tool results raw and
complete in the tail (standard production practice); treat large file reads as
candidate spans for Arm F rather than as ordinary chat turns.

### Probes become behavioral (a strength, not an adaptation cost)

The chat probe categories map onto coding with _more_ objective scoring, because
most reduce to machine-checkable behavior instead of judged answers:

- **Referent:** a helper/function/branch defined mid-session, referenced later
  by description ("the retry wrapper we wrote"). Score: does the continuation
  call the right symbol? Grep-checkable.
- **Sense-disambiguation:** ambiguous names ("the handler", "config", "the
  second endpoint") bound mid-session. Score: which binding does the
  continuation's code use?
- **Stance/conventions:** agreements established mid-session ("no new
  dependencies", naming style, error-handling policy). Score: lintable/greppable
  compliance in generated code.
- **Ruled-out (the big one for agents):** commands or approaches that already
  failed mid-session. Score: does the agent re-run a failed command or
  re-propose a rejected approach? Detectable automatically from the transcript.
  This is the probe most directly tied to real compaction cost.
- **Evicted-fact:** an API signature or constraint discovered mid-session (e.g.,
  from reading a file or an error message). Score: does the continuation's code
  compile/run against it? Ground truth no chat probe can offer.

Where feasible, the headline realism metric is end-to-end: task resolution rate
(or partial-progress proxies) under each arm on a fixed issue set, with the
compaction point standardized. Behavioral probe rates are the mechanism-level
companion to that headline.

### Arm emphasis shifts for code

Code is saturated with verbatim repetition — identifiers, paths, error strings
recur exactly — so Arm E's verbatim-twin alignment covers far more of the
context than in prose, and Arm F (span retention for whole files or diffs,
original KV, CacheBlend-style seam handling) has an obvious natural role. Arm G
matters less here (less paraphrase), unless your local results say otherwise.
Suggested core for Phase 2: **B (baseline), E, F, H**, with C only if the
gapped-cache machinery proved robust locally.

## Reproducibility carry-over

The cloud runs should be able to cite the local results as the
controlled-conditions arm of the same eventual writeup. Preserve now, before
scaling: exact model files and quantization, mlx-lm/library versions, seeds,
conversation manifests, probe manifests, and the surgery code at a pinned
commit. Re-running the local phase later under version drift is the failure mode
to prevent this week, not then.

## Publication shape (so effort lands where it counts)

Three legs: **mechanism** (local phase: controlled chat corpus, contrast matrix,
identity-validated machinery), **realism** (this phase: behavioral probe rates
and task outcomes on real coding trajectories), **scale** (one larger model, if
Strategy S). The contrast matrix supplies the story — where meaning lives across
a compaction boundary: encoding, payloads, addresses, correspondence — and the
coding numbers supply the reason anyone should care. A null on any leg is
reportable; a null caused by machinery that skipped its identity tests is not.
Same rule as always.
