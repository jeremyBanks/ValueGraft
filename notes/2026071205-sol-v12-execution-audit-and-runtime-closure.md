# V12 execution audit and runtime closure status

**Author:** Sol — gpt-5.6-sol-xhigh

**Date:** 2026-07-11

**Subject-model forward performed:** No
**Paid coherent-state compute:** $0

## Why this note exists

After revision-4 stimulus review passed, I did not treat correct tokenizer plans
as an executable experiment. An independent Codex execution audit and my own
line-by-line review found that the repository still lacked every v12 model-facing
component: causal event execution, a differentiable path control, bounded storage,
release modes, a harvester, and budget enforcement. Existing implementations were
v10-bound and prohibited by the v12 implementation boundary.

## Defects found before any forward

1. The stimulus validator wrote `decoded_round_trip: true` without performing a
   decode/re-encode comparison. Current code performs the comparison and fails
   closed.
2. The preregistration required complete token/event/logical/physical/source arrays
   before a forward, while the old artifacts persisted mostly hashes and event
   kind/width summaries. The revision-4 authoritative manifest now contains the
   literal arrays; it is 3.6 MB and remains nonauthorizing.
3. The control-fixture validator accepted equal-width substitutions rather than
   proving the preregistered literal fixture. It now binds exact schema, design,
   model/revision, histories, identity prompt, probe, and targets.
4. Oracle schedule, generated-answer stopping, the natural-calibration recovery
   denominator/stale cell, exact generated/forced comparison scope, and post-bf16
   placebo validity were underspecified. They are now frozen before outcomes.
5. `STATE.md` and `DECISIONS.md` described R2 too narrowly. R2 contains carrier
   content plus the complete carrier-close/anchor-user/anchor-assistant-header
   structural call; R3 adds q=1 `Acknowledged.` content.

## Stimulus gate outcome

Revision 3 repaired e04's false close-then-reopen chronology. A paired reviewer
then caught stale e04 control indices left by that reorder: `[7,8,31,32]` pointed
partly to unrelated edge-bead/pressure turns. Revision 4 changes metadata only and
binds `[7,8,33,34]`; literal histories are byte-identical to revision 3.

Fresh sealed reviews observed:

- blind carrier/history verdict: PASS, 12/12 histories;
- paired causal verdict: PASS, 6/6 cases;
- cross-case diversity: PASS;
- semantic/model outcomes seen by reviewers: no.

These reviews authorize only authored-content eligibility, not execution.

## Frozen ambiguity closures

- Oracles are unsurgered contiguous N-schedule source histories with the carrier
  before the retained tail. P/compact oracles do not exist.
- Greedy generation appends at most 64 non-EOS content tokens, then inspects a
  65th next-token candidate. EOS is never appended and has no cache row.
- Generated/forced identity is byte-exact over IDs, positions, log-probability
  float32 bits, every selected K/V row, and the final EOS witness.
- Natural calibration uses one fresh `FF` stale baseline:
  `rho_green=(m(Tg)-m(F))/(m(Ag)-m(F))` and
  `rho_amber=(m(F)-m(Ta))/(m(F)-m(Aa))`.
- The bf16 placebo uses the first deterministic SHA-counter candidate meeting
  frozen applied norm/cosine tolerances. Exhaustion is reported as
  `PLACEBO_UNAVAILABLE`; it cannot be silently weakened after outcomes.

## Runtime code now observed under fake-model tests

The additive v12 namespace now covers:

- exact ReplayPlan and FreshDestinationPlan event calls;
- explicit logical `position_ids` and contiguous physical `cache_position`;
- a 7,000-token live-cache ceiling and 256-row extraction ceiling;
- K/V, K-only, or V-only insertion with exact selected rows and unchanged
  non-selected rows;
- causal suffix/tail recomputation;
- immutable q=1 target scoring;
- 64-token greedy generation and exact generated/forced comparison;
- differentiable all-layer/all-token K/V gradients at a selected region;
- bit-exact gradient-enabled versus public no-grad baseline comparison;
- deterministic bf16 ULP edits in both directions, detached graph destruction,
  public-path reinsertion, and opposite margin movement.

The focused schema/planner/control/runtime suite most recently passed 48 tests.
Those tests used tokenizer-only reconstruction, pure functions, and deterministic
fake cache models. They do **not** establish Hugging Face/Qwen causal correctness.

## Remaining blockers

1. Complete and audit the bounded store with separate technical, eligibility,
   and treatment release modes.
2. Implement the raw runner schema and complete source/arm orchestration.
3. Implement an independent harvester that reconstructs geometry/formulas without
   importing runner planners or trusting runner aggregates.
4. Implement exact preflight, committed-receipt release, budget forecast/admission,
   and pod lifecycle wrappers.
5. Freeze the preregistration only after those inventories and schemas exist.
6. Run the complete local Qwen3-0.6B apparatus. A pass there is only permission to
   try the exact-stack subset, not scientific evidence.
7. Spend at most the initial $2 on exact provenance/gates/path/natural calibration;
   observe cost before any extension.

The correct current conclusion is: stimulus construction has passed its review
gate and causal runtime primitives have advanced materially, but v12 is not yet a
complete or authorized experiment.
