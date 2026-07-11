# Coherent-state decision canary v12 — preregistration

**Design ID:** `coherent-state-decision-canary-v12`  
**Author/decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** **DRAFT UNDER INDEPENDENT AUDIT — NO MODEL FORWARD AUTHORIZED**

This is a new additive exploratory experiment. It does not amend, waive, or
reinterpret v10, and it does not execute the frozen paired-v11 confirmatory
contract. V10 remains permanently non-authorizing. V11 remains paused with its
drafts non-executable.

The strategy record is in `notes/2026071152-sol-ultra-regroup-decision.md`. The
exact Fable review and Sol disposition files are
`notes/2026071153-fable-ultra-regroup-review.md` and
`notes/2026071155-sol-fable-review-disposition-and-canary-closure.md`. The
proportionate validation threat model is recorded in
`notes/2026071169-scientific-validation-threat-model-correction.md`. This
document freezes the literal experiment that follows
from those decisions. Until its status is changed additively to `FROZEN`, it
authorizes tokenizer-only construction and unit tests, but no local or paid
model forward.

## 1. Purpose and claim boundary

The canary decides whether a new independent confirmatory experiment has enough
expected value to propose. It cannot establish an efficacy or population claim.
Its stimuli, outcomes, thresholds, and any implementation choices learned from
them are permanently excluded from a later confirmatory sample.

The questions are:

1. Does a standard untrained instruction checkpoint write history-specific
   information into a fixed visible summary/boundary carrier?
2. Does retaining that state improve a damaged compacted continuation relative
   to a fresh same-token state?
3. Does value-only retention preserve a useful channel, separately from full
   K+V retention?
4. Is the state localized to summary content, the natural assistant boundary,
   or a deliberately appended neutral downstream anchor?

The canary is not evidence about a production continuously batched server, a
live organic coding agent, arbitrary models, arbitrary summaries, or broad
conversation traffic. The primary protocol is this repository's exact q=1
incremental loop over fixed token IDs.

## 2. Exact subjects and environments

### Local apparatus subject

- Model: `Qwen/Qwen3-0.6B` from the already cached local Hugging Face snapshot.
- Device: CPU.
- Dtype: `torch.bfloat16`.
- Attention: resolved eager on every layer.
- Purpose: technical completeness only. No semantic result from this subject
  can authorize or discourage the exact-model canary.
- After an independently validated, technically valid Phase A, the local
  subject may exercise the full treatment/harvest path even if its oracle
  competence is `ESTIMAND_INADEQUATE`; that run remains apparatus-only. The
  exact subject never receives this exception.
- The resolved revision/hash, tokenizer files, Transformers/PyTorch versions,
  model config, layer geometry, and backend enumeration must be persisted.

### Exact scientific subject

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Dtype: bf16; no quantization.
- Attention: eager requested and resolved/attested for every decoder layer.
- Temperature: 0; greedy summary generation.
- Device: one paid NVIDIA GPU environment capable of loading the exact model.
- Transformers and repository commit must be identical between the sealed
  preflight packet and run. No unrecorded dependency upgrade is allowed.
- Expected model geometry, independently checked against the loaded config:
  48 decoder layers, 32 attention heads, 4 KV heads, head dimension 128, and
  RoPE theta 10,000,000. Any mismatch aborts rather than updating this document
  from the runtime.

The pod gate is not inherited from a local pass. A required exact-stack subset
runs and persists before any semantic treatment score.

## 3. Stimulus strata and independence

### 3.1 Engineered carrier stratum

Freeze six complete paired cases before any treatment outcome:

- primary cases `e01`--`e04`;
- ambiguity-reserve cases `e05`--`e06`, run only under Section 15's trigger.

E01--e03 contain roughly 1,000--2,000 production-tokenizer tokens before the
carrier request. E04 contains roughly 4,000--6,000. Reserve cases span different
lengths within 1,000--6,000. All six are frozen before e01 treatment unblinding.

Each case has:

- one correct history `C`;
- one minimally counterfactual history `W`;
- exactly one focal derived decision changed by `W`;
- exactly one non-focal control decision/fact left byte-identical;
- one focal probe with exact correct/counterfactual target phrases;
- one non-focal probe with its own fixed target pair;
- a retained tail after the frozen compaction boundary that is byte-identical
  and coherent under C and W;
- one fixed target-neutral carrier text compatible with C and W.

The authored set deliberately contains two focal-history subtypes, fixed before
outcomes: `e01`, `e02`, and `e05` explicitly resolve the derived focal decision
before the boundary; `e03`, `e04`, and `e06` leave the focal result unstated and
require it to be derived at probe time. Results are reported by subtype. A
positive confined to the explicit subtype is evidence for retained resolved-
decision state, not for latent computation of an unstated result.

At least three independent authoring sessions spanning at least two model
families contribute, with no session authoring more than two cases. Authors do
not see model outcomes. A blind diversity review rejects a shared fill-in-the-
nouns scaffold.

These six fixed cases are not random population draws. Counts, target rows,
regions, schedules, and probes within a case do not increase N.

### 3.2 Conditional conversation stratum

This stratum runs only after a clear engineered pass in at least one primary
claim family. Freeze exactly three
discovery cases before the first conversation treatment score:

- `d01` is an additive discovery derivative of the committed `v11-c01` draft,
  after complete blind and target-aware review;
- `d02` and `d03` are newly authored after the engineered branch opens, under
  this already-frozen contract and without access to arm/region/layer values;
- d02/d03 must be structurally different and independently authored;
- the three may not share one planning-dialogue scaffold or author session;
- each has one focal changed plant and one byte-identical non-focal control;
- each has a correct/minimally-counterfactual history with exact geometry and a
  retained tail.

Conversation authors are told only which already-frozen claim-family branch opened;
they do not see per-case, arm, layer, region, or schedule results. All cases are
exploratory and excluded from confirmation.

## 4. Decoded and mechanical eligibility

For every paired history:

- roles, message count/order, system text, retained tail, and non-allowlisted
  messages are byte-identical;
- every changed message has identical production-tokenizer width;
- canonical full-prefix length, every message start, every q=1 assistant span,
  and every call width are identical;
- the counterfactual establishes exactly the frozen alternative and repairs all
  focal downstream references;
- the non-focal chain is byte-identical;
- no repeated padding, clipped response, fake external action, special-token
  literal, dangling reference, or conspicuous balancing prose is permitted.

Before a case is executable it must pass:

1. production-tokenizer mechanical validation;
2. randomized blind singleton review of both histories;
3. target-aware paired review;
4. cross-case scaffold/diversity review;
5. third-reviewer adjudication for any disagreement.

Review failures produce new additive candidates and hashes. Never edit a
reviewed snapshot in place or delete a failed candidate.

## 5. Carrier texts and maximal visible carrier

### Engineered neutral carrier

Every engineered history appends this exact user request:

> Write the fixed neutral handoff note for the next assistant. Output only that
> note.

Every engineered case uses this exact assistant content as the common carrier:

> The prior discussion established the operating context and relevant decision
> criteria. Continue from this handoff, preserve the existing constraints, and
> answer later questions from the state available here. No unresolved action is
> introduced by this note.

It must be blind-verified not to state, imply, or favor either focal target or
the non-focal target.

### Conversation summaries

For each conversation case, the subject greedily generates summaries under C
and W from this exact request:

> Write a compact handoff note of 180–240 words for another assistant who will
> continue this conversation without seeing it. Identify the participants and
> their roles, current goals, decisions that still affect next steps,
> constraints, unresolved questions, and the next actions. Prioritize what the
> next assistant needs to act; compress settled background details when they are
> not needed. Use plain prose or concise bullets. Do not add commentary before
> or after the note.

Also freeze one externally authored target-neutral summary containing only
shared non-focal information and equally coherent under C and W. Its author sees
the two histories but no subject state or outcome. Blind reviewers label every
summary/plant `target-neutral`, `partial`, or `explicit` before treatment
unblinding.

### Fixed downstream anchor exchange

Every source and destination arm contains this exact visible exchange after the
carrier assistant message closes:

- user: `Acknowledge receipt of this handoff without adding or repeating any factual detail.`
- assistant: `Acknowledged.`

The exchange is present even when its rows are fresh. It must not contain or
favor any target. The maximal source carrier is:

1. assistant summary/carrier content;
2. the exact canonical assistant close suffix;
3. the exact canonical anchor user message, assistant header/content, and close.

## 6. Retained regions

All regions use identical visible tokens and positions. They vary only the rows
replaced by source rows; all subsequent rows are causally recomputed.

- `R1_content`: carrier assistant content only, ending at a q=1 call boundary.
- `R2_boundary`: R1 plus the **complete next structural call**: the canonical
  carrier close, fixed anchor-user message, and anchor-assistant generation
  header. It ends immediately before the q=1 anchor-assistant content. This is
  the smallest production-turn-shaped boundary containing the close without
  cutting an intervention boundary inside one model call.
- `R3_anchor`: R2 plus the fixed anchor-assistant content `Acknowledged.` forced
  q=1. Its canonical close and the retained-tail continuation are visible in
  every arm but causally recomputed after the R3 boundary.

`R2_boundary` is the sole stop/go region. R1 and R3 are descriptive localization
conditions. They cannot rescue a failed R2 decision or become primary from their
outcomes.

## 7. Role-native q=1 source schedule N

The primary source protocol is `N = role_native_q1_replay`:

1. Render the entire alternating transcript canonically using the Qwen dummy-
   user/non-final method; derive boundaries from exact special-token markers,
   never from separately tokenized text concatenation.
2. Prefill the initial system/user material and assistant generation header in
   the exact canonical order.
3. Force each historical assistant content token in a separate one-token model
   call with explicit logical `position_ids` and contiguous physical
   `cache_position`.
4. Prefill the canonical assistant close plus the next user message and next
   assistant header as one complete next-message addition. No retained-region
   boundary may cut inside this call.
5. Continue through the history, carrier request, generated/forced carrier,
   canonical close, and anchor exchange.

The model predicts the Qwen assistant-close/EOS token but the generation loop
does not append its K/V row. The replay plan therefore appends the exact
canonical close once as the next structural block. A duplicate or absent close
is invalid. No implicit `prefill` auto-chunking is permitted: every explicit
call is at most 4,096 tokens or the stimulus fails/splits under a recorded plan.

For fixed IDs, q=1 forcing computes the same state as q=1 greedy selection
conditional on those IDs. It does not make imported authored text subject-native
and does not emulate another serving engine's batching policy.

Correct and wrong N branches must have exactly identical call widths and
positions. Generated-versus-forced identity is a technical gate on a separate
literal fixture, not a requirement that greedy decoding reproduce the externally
fixed engineered carrier. Under that fixture, greedily generate one sequence,
persist the observed IDs, rebuild the identical prefix, force those observed IDs
q=1, and require bit-exact token log probabilities and K/V rows. Exact comparison
also covers prefix IDs, logical/physical positions, event widths, every content
row's dtype/shape/bytes, and the final unappended EOS candidate ID and log-probability
bits. EOS is a stop witness, not a forced row; decoded-text equality is diagnostic
and token-ID equality is authoritative. No tolerance or engineered-carrier equality
enters this gate. The gate fixture
is: system `Answer plainly.`; user `Write one short neutral sentence acknowledging
that a record exists.`; greedy temperature 0; maximum 64 content tokens; normal
EOS required. The engineered carrier remains forced fixed text under every
history. Same-stack repeats must also be bit-exact. Any discrepancy is
`INVALID_TECHNICAL`; no tolerance is introduced after observing it.

## 8. Schedule sensitivity P

`P = turn_aligned_replay` is independently derived over the complete v12 token
stream. Before the carrier request, each complete canonical historical message
(including an assistant message's open/content/close) is one conceptual prefill,
split into a frozen ordered list only above 4,096 tokens. The carrier request
plus assistant header is then one prefill. From the first carrier-content token
onward, P uses the **identical q=1 carrier, complete carrier-to-anchor structural
call, q=1 acknowledgment, and retained-tail events as N**. Thus P varies how the
imported pre-boundary history wrote state; it does not introduce a second
carrier/anchor/tail protocol. F has no imported historical messages and is the
same compact schedule from the carrier onward. P runs on the same exact 30B
stack for every engineered primary case's R2 `CC`, `WW`, `FC`, and `FW` cells.
P is a numerical yardstick, not a replicate or alternative primary. C/W P
plans must have exact token, message-start, region, call-width, logical-position,
and physical-position equality before execution.

If the ambiguity reserve runs, P also runs the same four R2 cells for e05/e06;
the six-case yardstick then uses all six completed cases.

For state family X and cases i, define
`S_X = mean_i(abs(D_i,N^X - D_i,P^X))`. A clear N decision requires
`mean_i(D_i,N^X) > 0` and `mean_i(D_i,N^X) >= 3*S_X`. Compute this separately
for full-KV and value-only. The 3x rule is an exploratory robustness heuristic.
O/ordinary-4096 is deferred and may not be added after outcomes to rescue a
result.

## 9. Fresh gapped destination

For a fixed visible carrier:

- build exactly this compact canonical message list once, with no duplicated
  anchor: `[original system, carrier request, carrier assistant, anchor user,
  anchor assistant, retained-tail messages...]`;
- derive its token IDs by selecting the complete original-system source interval
  plus the full source suffix beginning at the carrier-request start; independently
  rerender the compact message list and require byte-identical token IDs;
- assign every compact token its exact full-source logical position: original
  system positions remain `0..system_end`, while the carrier request through the
  retained-tail end keep the source suffix positions after the evicted gap;
- keep `cache_position` contiguous in compact physical storage;
- force fresh carrier content q=1 and append close/anchor in the same exact call
  pattern used for the source region;
- fork at each exact region boundary, replace only selected rows, then causally
  recompute all following visible tokens and the retained tail;
- append each probe and teacher-force its exact target phrases from separately
  rebuilt immutable arm snapshots.

Before any model forward, persist for every case the complete arrays
`token_ids`, `logical_positions`, `physical_positions`, and
`source_token_indices`; the exact event list; source and physical R1/R2/R3
intervals; system width; and suffix-source start. Assert packed physical
positions, one intentional logical gap, exact source-token reconstruction,
identical C/W fresh plans, whole-event region boundaries, and equal region
widths/IDs/logical positions across F/C/W. N and P share this same fresh plan
from carrier content onward.

The fresh destination comparison is a deployment-utility contrast. Its compact
prefix differs from the long source trajectory by construction; it is not the
sole semantic control.

## 10. Source states and arm grid

For each history/carrier/schedule/region persist source K/V rows from:

- `F`: fresh compacted carrier;
- `C`: correct-history carrier;
- `W`: minimally counterfactual-history carrier.

At destination under N, run the complete 3x3 grid for all three regions, with
key source first and value source
second:

`FF, FC, FW, CF, CC, CW, WF, WC, WW`.

Under P, run R2 `CC`, `WW`, `FC`, and `FW` only. The preregistered primary
families are:

- full-KV: `CC` versus `WW`;
- value-only: `FC` versus `FW`.

`CF/WF` (K-only), crossed `CW/WC`, and interactions are secondary. `FF` is the
fresh utility baseline. Every replacement must be bit-exact in selected rows and
bit-identical to fresh outside them before downstream recomputation.

Add one deterministic nonsemantic V control per case/region. Flatten all heads
and head dimensions for each layer/token row. Let `d = V_C - V_W`. Draw a
deterministic Rademacher vector `r` from a SHA-256 counter stream seeded by the
UTF-8 string `coherent-state-v12-placebo-20260711` plus case/layer/row indices.
Project it orthogonal to d, normalize it to `L2(u)=L2(d)`, reshape it, and set
`V_placebo = V_F + u` with fresh keys. If `L2(d)=0`, use `u=0`; if projection
norm is zero, advance the counter. For nonzero `d`, try deterministic counter
attempts `0..1023` and select the first whose **applied bf16** delta is nonzero,
has relative L2 error at most `0.05`, and absolute cosine with `d` at most `0.02`;
the pre-cast relative norm error and absolute cosine must each be at most `1e-12`.
Zero-`d` rows remain bit-exactly fresh. Check every attempted row in memory and
persist a compact per-region record: row and attempt counts, zero-delta count,
maximum applied norm error and absolute cosine, a canonical hash of the complete
diagnostics, and source/result row hashes. If any
nonzero row has no valid deterministic attempt, do not score that placebo arm:
record `PLACEBO_UNAVAILABLE` for that case/region. This is adverse control
availability but does not invalidate the separately specified full-KV or
value-only primary arms. No target text or label enters the PRNG or selection.

## 11. Oracles, probes, and raw outcomes

Score two full-history oracles with the same visible maximal carrier/anchor:

- `A_C`: correct full history;
- `A_W`: counterfactual full history.

Both oracles are unsurgered contiguous executions under schedule N only. Their
literal order is evicted history prefix, carrier/anchor, then retained tail—the
same order frozen by the source planner. Execute through the final assistant
close, append probe user plus assistant header as one canonical structural call
derived by exact prefix differencing, and fork that immutable probe-prefix state
for generation and each q=1 target forcing. Logical and physical positions are
contiguous. P-schedule or compact oracles are not run and cannot substitute.

For each arm and probe persist:

- exact target token IDs;
- per-token and mean log probability for the C target;
- per-token and mean log probability for the W target;
- margin `Y = mean_lp(C_target) - mean_lp(W_target)`;
- deterministic generated answer and stop reason;
- complete teacher-forcing feed IDs/positions;
- immutable source/destination/arm hashes.

Every generated answer uses greedy argmax with the lowest token ID winning an
exact tie, no sampling or text stop, and a preflight-bound nonempty EOS set that
must agree across model/generation/tokenizer metadata. Append at most 64 non-EOS
content tokens q=1. After the 64th, inspect one additional next-token distribution:
EOS there is a normal stop but is not appended; another token is `max_content_tokens`.
Persist content IDs/log-probabilities, stop candidate ID/log-probability, EOS set,
content count, stop reason, and cap flag. Identity generation requires nonempty
content and normal EOS. An ordinary arm cap hit remains behavioral evidence and
fails any generated-answer oracle-competence requirement but does not erase its
forced-target scores.

Never call a larger margin "less loss" unless the C-target mean log probability
also improves. Never discard the two log-probability components.

## 12. Per-case contrasts

For state family `X` (`R=full-KV`, `V=value-only`) and primary R2/N:

- `D_focal^X = Y_focal(X_C) - Y_focal(X_W)`;
- `D_nonfocal^X = Y_nonfocal(X_C) - Y_nonfocal(X_W)`;
- `SEL^X = D_focal^X - abs(D_nonfocal^X)`;
- `Hplus^X = mean_lp_C(X_C) - mean_lp_C(X_W)`;
- `U^X = Y_focal(X_C) - Y_focal(FF)`;
- `Uplus^X = mean_lp_C(X_C) - mean_lp_C(FF)`.

The semantic claim direction is `D_focal > 0`, `SEL > 0`, and `Hplus > 0`.
Utility requires `U > 0`; favorable utility-side correct-target movement
requires `Uplus > 0`. The wrong-state
arm moving only by generic disruption fails selectivity.

For conversation native-summary crossover, compute these contrasts separately
within `s_C`, `s_W`, and the target-neutral summary. The neutral-summary cell is
the clean semantic comparison; native-summary cells describe state plus
history/text congruence and are never substituted for it.

## 13. Pre-treatment gate and unblinding order

Phase A may compute only:

- mechanical/review eligibility;
- summary text and exact token IDs;
- leakage labels;
- forced-token NLL/support under C and W;
- `A_C`, `A_W`, and `FF` competence/damage outcomes;
- technical identities, bidirectional path-control outcomes, and the natural
  downstream-note calibration.

An independent validator writes and commits one of:

- `PRETREATMENT_PASS`;
- `ESTIMAND_INADEQUATE` with literal reasons;
- `INVALID_TECHNICAL`.

Treatment contrasts cannot be computed or exposed until this artifact passes.
Any single allowed stimulus redesign is triggered only by Phase-A evidence and
creates a new additive hash. No redesign may inspect a treatment source row,
arm score, layer effect, or region effect.

Pre-treatment adequacy requires:

- both full-history oracles favor their own focal target;
- the non-focal oracle remains correct under C/W;
- for every focal and non-focal oracle predicate, the forced margin has the
  target's required sign **and** generated content IDs begin with the complete
  exact target token-ID sequence; empty generation, a cap hit, or explanation
  text before the target fails generated competence;
- fresh compaction shows positive focal damage relative to A_C before that case
  can support a utility interpretation; zero/negative damage does not invalidate
  its fixed-text semantic contrast;
- utility-eligible damage requires both `Y_focal(A_C)>Y_focal(FF)` and
  `mean_lp_C(A_C)>mean_lp_C(FF)`; record both components for every case;
- generated conversation summaries and the generated/forced identity fixture
  end normally with no embedded non-EOS special token or cap hit; the engineered
  carrier is externally fixed and therefore has finite forced-token support, not
  a generation-stop predicate;
- the target-neutral carrier is blind-approved;
- forced carrier NLLs are finite for every token under C and W.

NLL and headroom remain continuous diagnostics. No arbitrary historical
0.30-nat or post-observation support cutoff is inherited. Full-history competence
is literal: A_C must generate/favor the C target, A_W must generate/favor the W
target, and the non-focal oracle must remain correct in both histories.

## 14. Exact-model gates and positive control

Before semantic treatment scoring on the pod, persist and independently validate:

1. exact model/revision/tokenizer/dtype/backend/code/inventory binding;
2. deterministic same-schedule repeat on the gate fixture;
3. generated-versus-forced q=1 identity;
4. fresh self-replacement identity for R1/R2/R3;
5. selected-row exactness and non-selected-row preservation for K, V, and K+V;
6. canonical close/open rendering identity;
7. gapped physical/logical position coverage and tail recomputation;
8. deterministic perturbation sensitivity;
9. durable raw-artifact creation and independent validation.

### 14.1 Technical bidirectional path control (gating)

The technical and natural controls share one exact fixture but produce separate
artifacts and have different interpretations. Its literal source conversation is:

- system: `Apply the stated policy exactly. At the final question answer with exactly approve or deny.`
- user C: `Policy: approve only when the status is green. Current status: green.`
- user W: `Policy: approve only when the status is green. Current status: amber.`
- assistant: `The policy and current record have been processed. Use the recorded state for the decision.`
- frozen compaction boundary after that assistant message;
- retained user: `At the final question, answer with exactly one policy label.`
- retained assistant: `Understood.`

The common revision-2 engineered carrier and anchor are inserted at the boundary
through the exact public v12 planner. The probe is `Decision?`; targets are the
single production-tokenizer tokens `approve` and `deny`; the frozen margin is
`log p(approve) - log p(deny)` from one immutable probe-prefix forward.

Start from this fixture's fresh technical R2 boundary on the exact public gapped
path. Freeze model weights and compute the gradient of the frozen downstream margin
with respect to the selected R2 K/V rows. The edit is explicitly bf16-aware:
for every layer/channel/token row with a nonzero finite gradient, flatten heads
and head dimensions, select the lowest flat index among elements with maximal
absolute gradient, and move that cached bf16 scalar in the signed gradient
direction with `nextafter`. Try the frozen ULP-count sequence
`[1, 2, 4, 8, 16, 32, 64]`; the negative edit moves the same coordinates in the
opposite direction. Stop at the first count for which both persisted directions
differ from fresh and are measurable. Any nonfinite gradient, saturated step,
or unrecorded coordinate aborts. Detach the `+` and `-` edited rows, destroy the
graph and live destination, reconstruct a new fresh gapped destination, reinsert
through the exact public K+V replacement function, recompute the bridge, and
score.

The gate passes only if `+` raises and `-` lowers the frozen margin by at least
`1e-4`, selected insertion/confinement remains exact, and the execution trace,
ULP attempts, direction/coverage counts, canonical hashes of checked row
diagnostics, edited row hashes, and raw log probabilities persist.
It licenses intervention/readout sensitivity only. Failure aborts semantic
scoring.

### 14.2 Natural downstream-note calibration (reported, not a plumbing gate)

Run two label-balanced variants of a downstream-note causal calibration through
the **same gapped destination, R2 interval, row replacement, target scorer,
persistence, and harvester functions**. The literal control texts are:

- system: `Apply the stated policy exactly. At the final question answer with exactly approve or deny.`
- policy/status A: `Policy: approve only when the status is green. Current status: green.`
- policy/status B: `Policy: approve only when the status is green. Current status: amber.`
- source assistant before the frozen boundary: `The policy and current record have been processed. Use the recorded state for the decision.`
- retained user/assistant: `At the final question, answer with exactly one policy label.` / `Understood.`
- post-boundary carrier/anchor: the exact common revision-2 engineered literals;
- probe: `Decision?`
- targets: `approve` and `deny`.

The label-balanced variant swaps which source is named C/W for reporting and
which target is treated as positive; it does not create an independent fixture.
Production-tokenizer IDs and exact matched geometry must be committed before
FROZEN status; if the literal strings do not match geometrically, an additive
revision replaces them before any forward.

Let `m(x)=log p_x(approve)-log p_x(deny)`. Freeze five raw cells: full-history
green oracle `A_g`, full-history amber oracle `A_a`, the unique fresh compact
`F=FF`, R2/N green full-KV transplant `T_g=CC`, and R2/N amber full-KV transplant
`T_a=WW`. The C/W fresh constructions must hash identically; `F` is the sole
stale baseline. Define

- `rho_green = (m(T_g)-m(F)) / (m(A_g)-m(F))`;
- `rho_amber = (m(F)-m(T_a)) / (m(F)-m(A_a))`.

All terms must be finite and both denominators strictly positive. Require
`m(A_g)>0`, `m(A_a)<0`, `rho_green>=0.5`, and `rho_amber>=0.5`; ratios are
unclipped and values above one are reported. The two label-balanced directions
are oriented views of these same five raw cells, not independent fixtures or
duplicated evidence. Oracle generation must begin with/favor its exact target;
for this single-token fixture that means the first generated content token equals
the target token and the forced margin has the required sign. The
natural calibration result is reported before e01.
Its failure is scientifically adverse but does not relabel a passing technical
path control as broken; the frozen engineered decision rules remain terminal.

## 15. Decision rules

All rules refer to R2/N and the four engineered primary cases unless stated.

### Claim-family terminal logic

Apply the following independently to X in {full-KV R, value-only V}. Proceed to
the conversation stratum for family X only if:

- all technical, pre-treatment, and bidirectional path-control gates pass;
- mean `D_focal^X > 0`;
- at least three of four cases have `D_focal^X > 0` and `SEL^X > 0`;
- mean `Hplus^X > 0`;
- X clears its aggregate 3x N-versus-P yardstick.

### Clear engineered stop

Family X is a clear stop after four if the path control passes but either:

- mean `D_focal^X <= 0`; or
- at most one of four cases has both `D_focal^X > 0` and `SEL^X > 0`.

Report the accepted residual risk: engineered short/mid-length nulls do not
logically exclude a channel unique to longer organic histories.

Every other valid four-case result for either family is `AMBIGUOUS`; this
includes schedule failure, nonpositive Hplus, or conflicting signs not already a
clear stop. If both families stop, stop the program. If either is ambiguous, run
the single shared reserve extension.

### Single ambiguity extension

Run frozen reserve e05/e06, once, only if:

- either claim family is `AMBIGUOUS` under the exhaustive rule above.

After e05/e06, family X passes only with mean D>0, at least four of six positive
D+SEL cases, mean Hplus>0, and the 3x yardstick. Every other six-case result is a
terminal stop for X. No second extension, arm change, target change, or threshold
change is allowed. The four-of-six requirement necessarily includes at least one
explicit-resolution and one unstated-result case; report those subtypes rather
than pooling away a qualitative split.

Only a family classified `AMBIGUOUS` at four is reclassified over six. A family
already `PASS4` or `STOP4` remains terminal even when the other family's ambiguity
triggers shared e05/e06 execution; its reserve observations are descriptive and
cannot rescue or overturn the four-case decision.

### Full-KV versus value-only branches

- Full-KV pass/value-only stop: conversation discovery may proceed for the
  narrower full-KV claim. No value-only confirmation or live-agent treatment is
  authorized. Difference language requires the direct paired R-minus-V contrast.
- Value-only pass/full-KV stop: conversation discovery may independently proceed
  for the value-only claim. It cannot be described as a component of an
  established full-KV channel. A mitigation branch additionally requires mean
  U>0 and Uplus>0 among cases with positive observed A_C-minus-FF damage.
- R1/R3-only positivity cannot change the R2 branch.

### Conversation discovery branch

A later independent confirmatory proposal is allowed only after all three
frozen conversation cases complete and at least two show, within the target-
neutral summary cell, positive D, SEL, Hplus, U, and Uplus for the claim family being
considered, with an aggregate 3x schedule yardstick. One or two completed cases
because of a cost stop cannot authorize confirmation.

## 16. Analysis and reporting

Canary reporting is descriptive:

- every case and raw component;
- mean, median, SD, and sign count when defined;
- no confirmatory p-value or confidence-interval rhetoric;
- exact pre-treatment exclusions/invalidities;
- schedule and region results without treating them as independent N;
- direct paired full-KV-minus-value-only contrast;
- all truncation and reserve-trigger decisions.

Use the ordinary median and arithmetic mean. Report sample SD (`n-1` denominator)
for `n>=2`; report SD as undefined/null for a singleton or empty stratum. These
summary conventions never enter a decision threshold except where the rules
explicitly name the arithmetic mean.

If a later N=12 benchmark is proposed, it must be independently authored and
preregistered. It will support at most a fixed purposive-benchmark claim and is
powered only for large standardized effects.

## 17. Persistence, state bounds, and artifact schema

Every generated or forced carrier, assistant token stream, source row set, arm
trace, raw score, and gate result is durable before proceeding. Output names are:

`<experiment>_<model-slug>_<UTC-timestamp>.json`

At process start print the resolved model and unique output path. Per-case
checkpoints include:

- committed stimulus hash and review hashes;
- exact environment/repository fingerprint;
- canonical messages and token IDs;
- every call width, logical position, and physical cache position;
- source/result row hashes and exact layer/region coverage;
- maximal visible carrier text and boundaries;
- full arm grid lineage;
- raw oracle/fresh/treatment scores;
- cost/wall-time and resume state.

Lossless cache tensors are not a release prerequisite: the subject/model bytes,
stimulus, deterministic execution code, token/position traces, and row hashes
make them reproducible, while retaining every tensor would add substantial
storage and handling cost without addressing an ordinary failure mode. Every
scored result, generated text/token stream, trace, and row hash must land under
`results/` and be committed.

Stateful-change checklist for implementation commits:

1. retained state = per-case source K/V rows, keyed by exact case/history/
   carrier/schedule/region/model fingerprint;
2. size bounded by one case, three histories, three regions, and explicit layer/
   token assertions;
3. each process handles one case; live caches are released when that case process
   exits, and no cross-case state is retained;
4. correctness evidence = row hashes, self-replacement, generated/forced
   identity, exact selected/unselected-row checks, and independent formula
   recomputation;
5. probe gate = local full apparatus plus exact-stack gate subset, bidirectional
   technical path control, and natural downstream-note calibration.

## 18. Paid execution budget and truncation

Conservative provider-usage estimates before canary implementation total
`$36.863842`; paid coherent-state compute remains `$0`. This includes two capped
Claude Sonnet 5 reserve-stimulus authoring attempts, the second of which left
recoverable authored text that Sol completed through the tokenizer-only writer.
Against the owner's approximate `$60` ceiling, treat `$23.14` as remaining unless
billing evidence changes it.

Paid order:

1. pod provenance/gates;
2. bidirectional technical path control;
3. natural downstream-note calibration;
4. complete e01 full arm/region/schedule set;
5. observed full-canary cost forecast and committed decision;
6. e02--e04;
7. four-case engineered verdict;
8. e05/e06 only if the frozen ambiguity rule triggers;
9. terminal engineered verdict;
10. conditional d01 neutral cell, then native crossover;
11. conditional d02/d03, case-atomic, in that order.

Initial authorization is `$2` through step 3. The run may extend to a hard `$8`
total canary ceiling only after the observed forecast. The `$8` includes
provisioning, failed starts, e01--e04, the possible e05/e06 extension, and any
conversation work; the extension's `$3` sub-cap is inside, not additional to,
`$8`. Record the provider hourly price and forecast each next complete case from
observed wall time with a 25% safety buffer. Do not begin a case unless its full
forecast fits the remaining cap. Incomplete cases are persisted but do not
contribute to a decision. If all three conversation cases cannot fit, do not
begin that stratum. Stop the pod before repairs, analysis, or writing. At least `$15`
remains reserved for failures and final synthesis/review. A GO branch does not
authorize spending that reserve on a cut-rate confirmation; it returns an
observed forecast to the owner for a funding decision.

## 19. Implementation boundary

Use a new additive namespace and design schema. Stable low-level primitives for
canonical rendering, q=1 append/generation, cache snapshot/rebuild, row hashing,
explicit prefill, and target scoring may be wrapped. Do not import or reinterpret
the v10 arm enums, cyclic wrong-history constructor, summary-only `SourceCapture`,
old gapped layout/arm constructors, deranged-delta control, store schema,
release overlay, or harvester.

Required additive components are:

- independent token/event/region planning;
- role-native runtime and maximal R3 capture;
- separate append-only technical, eligibility, and treatment artifacts;
- machine-enforced technical -> persisted eligibility -> treatment release;
- an independent harvester that does not import runner/layout constructors;
- paid-stack preflight/launch wrappers.

The harvester verifies case/release/runtime bindings and the complete frozen arm
selector set, decodes the authoritative float32 score bits, recomputes all
formulas and decision branches, and ignores runner decimal aggregates and status
labels. Layout reconstruction is covered by the focused planner/runtime tests
and technical gate; duplicating the tokenizer and cache planner inside the
harvester is outside the ordinary scientific-failure threat model.

## 20. Claim boundaries (mandatory wording)

- The canary is exploratory and permanently excluded from confirmation.
- Its authored cases are fixed fixtures, not random population draws.
- Imported assistant histories forced through q=1 are replay, not native
  live-agent trajectories.
- The semantic contrast is a controlled-mediator, fixed-visible-text effect,
  not the natural total effect of changing history.
- Fresh comparisons are utility contrasts and include execution-policy
  differences.
- Full-KV success does not validate value-only copying.
- R2 success is carrier content plus the carrier close and fixed anchor
  prompt/header, not summary-content-only success; R3 success additionally
  retains the engineered acknowledgment content.
- The 3x schedule rule is an exploratory heuristic, not a universal numerical
  law.
- Path-control success validates the tested intervention/readout path but does
  not establish a natural semantic channel.
- Failure is a bounded stopping result for this checkpoint, carrier, schedule,
  and budget, not evidence of universal absence.
- No canary p-value, interval, or sign count supports a confirmatory efficacy or
  live-agent claim.

## 21. Authorship, attribution, and paper use

Record the exact runtime model for every author and reviewer. Do not infer Fable
versus Opus from an intended alias. Collaborative commits use the exact
`Co-authored-by` trailer.

The corrected negative/methodological paper proceeds in parallel. A canary
result may be added with its exploratory boundary. Fable leads the eventual
fresh narrative draft after the canary branch resolves; Sol owns factual,
causal, and methodological truth. No paper claim may describe this canary as a
live-agent evaluation or a confirmatory replication.
