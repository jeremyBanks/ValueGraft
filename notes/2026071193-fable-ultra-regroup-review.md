Runtime model identifier as exposed to this reviewer: `claude-fable-5`

# Fable adversarial review of the ultra-regroup decision canary

**Reviewer:** Claude Fable 5 (`claude-fable-5`), fresh session, no prior project frame
**Requested by:** Sol — GPT-5.6 Sol, extra-high reasoning effort
**Date:** 2026-07-11
**Inputs:** `notes/2026071189-sol-ultra-regroup-decision.md` (the decision under review);
`notes/2026071187-trajectory-red-team-ultra-regroup.md` and
`notes/2026071188-causal-statistical-blackboard-ultra-regroup.md` (independent inputs);
the 2026-07-11 correction banner at the top of FINDINGS.md (premise check only);
AGENTS.md for house constraints.
**Status:** Advisory. Sol retains final authority. Unrelated settled project history is
out of scope and was not relitigated.

## Verdict

The exploratory exact-model decision canary is the right next step and is better
than every alternative on the table (completing N=12 now, going straight to a
live-agent pilot, or writing the paper with no further evidence). Sol's central
premise checks out against the FINDINGS.md banner: the corrected apparatus has
produced no semantic outcome, the schedule-sensitivity result is pinned only on
Qwen3-0.6B CPU bf16 eager with the 30B kernel path and magnitude explicitly
unmeasured, and v10 is permanently non-authorizing. Confirmation before a large
exact-model signal would be premature; the canary logic is sound.

But the design as written has two defects I would treat as blocking (findings 1
and 2), one structural inefficiency that a simpler sequencing fixes (finding 3),
and several places where an unspecified judgment call will be made on the day —
which is exactly how this project's previous adaptive-analysis problems started.
None of this reverses the decision; all of it should be fixed in the frozen
canary spec before any paid token.

## Severity-ranked findings

### 1. Blocking — every technical gate is specified local-only; the paid 30B stack is exactly where this project's bugs live

The gate list ("Before paid semantic scoring, the **local** end-to-end runner
must observe...") runs entirely on the local stack, and only the downstream-note
positive control is specified on the 30B. The FINDINGS banner says in so many
words that the low-level kernel path and the 30B magnitude of the query-shape
effect are unmeasured. The pod is different hardware, different kernels, likely
a different attention implementation — precisely the class of environment change
behind the project's worst incidents (wrong checkpoint, batched-vs-stepwise
kernels, position bugs that masquerade as findings).

Required fix: a fast gate subset must re-run **on the pod, on the exact 30B,
before any semantic scoring**, and its artifacts must be committed fail-closed
in the Amendment-11 style: exact checkpoint/tokenizer/revision binding (the B2
interlock), generated-versus-forced identity on one case under the new schedule,
self-replacement identity, row/region confinement for K, V, and joint
replacement, and the engineered perturbation sensitivity check. This is minutes
of pod time. Relatedly, the downstream-note positive control must go through the
**same gapped-destination and row-replacement machinery** as the treatment arms
— a positive control on a simpler path validates nothing — and it must run
first, with a prespecified abort-if-fail, so a dead readout path costs cents
rather than the whole cap.

### 2. Blocking — the anchor region of the factorial is confounded with visible text as specified

Regions 1 and 2 reuse row subsets of the same persisted source capture and are
text-matched against the fresh baseline — good, and cheap. Region 3 appends a
new anchor string before eviction, which means its destination visible text
differs from regions 1–2 and from the fresh arm. Stopping rule 5's decision
("content null but closing/anchor selective → pivot") is precisely a
cross-region comparison, and as written it confounds retained-rows with
changed-text. Two fixes, either acceptable: (a) include the fixed target-neutral
anchor text in **every** arm's destination and vary only which rows are
retained — cleanest and nearly free; or (b) give region 3 its own text-matched
fresh baseline and forbid cross-region conclusions. I recommend (a).

Additionally, prespecify **which region carries the stop/go decision** (rule 3's
"no channel"). Three regions × margin/correct-target/selectivity outcomes is a
small garden of forking paths, and the stop decision — not a paper claim — is
the real product of the canary; noise-selected "channel exists" verdicts burn
the confirmatory budget just as surely as a false paper claim would. Given the
prefill-notes prior (conclusions stored on aggregator/delimiter positions),
region 2 (content + closing boundary) is the natural primary; regions 1 and 3
descriptive.

### 3. High — simpler design available: make the two strata sequential, not parallel

The engineered carrier stratum is the maximum-favorability condition: short,
clean, target-neutral carrier, matched geometry, explicit rule flip. If
full-KV correct-versus-counterfactual shows nothing there with a working
positive control, the harder conversation cases are near-hopeless, and their
cost (long stepwise source captures × the 2×2 crossover) is the bulk of the
paid budget. Run the engineered stratum first in the pod session and make the
conversation stratum conditional on an engineered channel appearing. This
matches Sol's own stopping logic, fits the cap far more comfortably, and
concentrates the money on the branch where it changes the decision. One
caveat to record honestly: a 1–2k-token engineered null does not strictly bound
a long-context aggregation effect at 6–8k tokens; that residual risk is being
accepted for budget reasons (or hedged with one mid-length engineered case).

### 4. High — the crossover balances the contradiction confound; it does not remove it

The 2×2 history-by-summary crossover is a real improvement, but every same-text
column pair still has exactly one native and one forced-contradictory cell, so
the history contrast within a column includes consistency/conflict effects that
are themselves focal (the blackboard note's point stands: the contradiction is
about the focal fact, so focal selectivity cannot excuse it). The cheapest clean
cell is the one the engineered stratum already uses and the conversation stratum
omits: force one blind-verified **target-neutral** summary through both
histories. Both cells non-native, geometry and text identical, contradiction
minimized — this is the single cleanest conversation-stratum semantic contrast
available, at the cost of two extra source captures per case. Add it. Keep the
crossover as the "does the native summary carry state" question it actually
answers, and retain the forced-token NLLs and blind leakage labels as specified.

### 5. High — the stopping rules have no ambiguity branch, and the stop is hair-triggered relative to the decision asymmetry

Rules 1–7 carve clean branches, but the likeliest real outcome —
directional-but-not-large, or engineered-positive with mixed conversation cases
— lands nowhere, which means an on-the-day judgment call, which is how adaptive
analysis re-enters. Prespecify exactly one bounded extension (e.g., +2
conversation cases, once, ~$2–3, triggered only by a written ambiguity
criterion) and otherwise stop. Note also the asymmetry: a false "no channel"
stop kills the project's central question; a false "go" costs a bounded few
dollars against a $15 reserve. The blackboard note's minimum decision-useful
canary was six conversations; Sol's is 1–3 plus engineered cases. That is
defensible only with the bounded-decision framing stated loudly and the
extension branch available. Finally, rule 2's single stimulus-redesign cycle
must be gated on **pre-treatment measures only** (competence, damage, leakage,
NLL): sequence the harvester so gate verdicts are computed and committed before
any treatment contrast is computed, so the redesign decision cannot be tuned on
treatment signal even accidentally.

### 6. Medium-high — role-native stepwise replay: right move, three unstated obligations

The identity claim is correct as stated: forcing observed token IDs through the
same one-token calls computes the same cache states as greedy selection of those
IDs, given identical call geometry, positions, and kernels. Three obligations
follow. (a) The claim is stack-relative — "live" means a q=1 incremental loop,
not a production serving engine with chunked prefill and continuous batching;
the claim boundary should name the exact loop. (b) The Qwen3 chat-template trap
(unstable rendered prefixes, the canonical-non-final-rendering machinery) must
be revalidated under the new close/open-token append order; the new schedule
multiplies exposure to exactly that instability. (c) Rule 7's fragility
comparison ("schedule movement comparable to the semantic contrast") is
currently unanchorable: the c10 numbers are 0.6B CPU bf16 eager, and the banner
says the 30B magnitude is unmeasured. The P-vs-stepwise sensitivity condition
must therefore run on the same paid stack and cases as the semantic contrast,
and "comparable" needs a prespecified ratio (e.g., semantic contrast ≥ 3× the
observed same-case schedule movement) rather than a judgment call. Note the
happy corollary: stepwise per-token calls make assistant-span call geometry
match trivially, so exact-width authoring pressure drops (positions still
require matched widths, but the unnatural-prose failure mode shrinks).

### 7. Medium — $5 cap arithmetic is optimistic; prespecify the truncation order

Stepwise-forcing a 30B-A3B through multi-thousand-token histories × 2 histories
× multiple summary variants × 1–3 conversation cases, plus engineered cases,
pod gates, the positive control, and the schedule-sensitivity subset, plus
provisioning/download overhead on a fresh pod, plausibly lands at $5–10, not
comfortably under $5. The one-case forecast is the right mechanism but must
forecast the **full per-case arm set** (crossover, neutral-summary cell, region
captures, sensitivity replay), and provisioning overhead should be budgeted
explicitly. Prespecify the execution priority order (gates → positive control →
engineered stratum → first conversation case → crossover cells → sensitivity)
so a cap hit yields an interpretable truncated canary instead of a scattered
one. Log per-case wall-clock and cost — that observed number is the confirmatory
forecast. And say plainly what the note only implies: with ~$8.8 uncommitted
after the canary and reserve, a GO branch almost certainly means returning to
the owner for a funding decision, not improvising a cut-rate confirmation.

### 8. Medium — two spec gaps that will otherwise be decided on the day

(a) The full-KV-positive / value-only-null branch (rule 4) says stop optimizing
naive value grafts — but does not say whether a full-KV-only channel authorizes
a confirmatory corpus for the **full-KV retention claim**, which is a different
paper claim from the owner's value-only mitigation. Prespecify which claim
family confirmation would confirm in that branch. (b) The engineered stratum
says "several" independently worded paired histories: fix the count, and apply
the author/model-diversity rule (different subagent models per case) — the
red-team's scaffold-monoculture finding applies with equal force to engineered
cases authored by one model voice.

### 9. Minor

Prespecify the norm-matching convention for the nonsemantic perturbation control
(e.g., per-row L2 of the value delta). Rule 1's two bounded technical repairs
should happen with the pod stopped or locally, so repair time cannot silently
consume the cap. The wrong-history arms must be the decoded, plant-specific
minimally counterfactual histories the banner requires — the note already
intends this; the canary spec should cite it as a hard precondition.

## What I endorse without reservation

Pausing the twelve-case corpus while preserving the drafts; the
permanently-exploratory firewall on canary stimuli; two strata analyzed
separately rather than pooled; same-schedule correct-versus-counterfactual as
the semantic contrasts with fresh comparisons demoted to utility; margin
decomposition with the correct-target-must-rise requirement; focal selectivity
against an unchanged control fact; persisted-source row reuse making the region
factorial nearly free at destination time; the 3×3 K/V grid as optional and
non-primary; the exact-30B downstream-note positive control (with finding 1's
strengthening); starting the corrected negative/methodological paper now, not
hostage to the canary; and the explicit no-rescue posture toward P/O schedules.

## Bottom line

Go — but freeze the canary spec first with: pod-side gate re-runs and a
positive-control-first abort rule (finding 1); an anchor-text-matched design or
region-local baselines plus a prespecified decision-carrying region (finding 2);
engineered-stratum-first conditional sequencing (finding 3); the target-neutral
forced-summary cell in the conversation stratum (finding 4); one prespecified
ambiguity extension and gate-then-unblind sequencing (finding 5); a same-stack
schedule yardstick with a fixed ratio (finding 6); and a full-arm-set forecast
with a truncation order (finding 7). None of these are new machinery; they are
one page of spec that closes the specific doors this project has previously
walked through.
