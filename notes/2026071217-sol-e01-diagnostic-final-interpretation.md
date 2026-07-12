# Final interpretation and disposition of the v12 e01 diagnostic

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning  
**Date:** 2026-07-12  
**Scientific status:** post-ambiguity diagnostic only; formal v12 remains stopped

## Bottom line

The e01 measurement is technically trustworthy but scientifically weak. One
preselected cell — value-only transplantation at the R2 boundary under the
q=1 N replay schedule — produced a small, focal-selective movement in the
intended direction. It is evidence that the exact hidden state depends on the
history and execution path. It is not evidence that the intervention recovered
useful semantic continuity, reduced behavioral loss, or would help an agent.

Formal v12 stopped before this outcome under the literal written path-control
rule: the first unsigned-measurable edit was ULP 2 and its negative direction
moved the wrong way. The sealed implementation instead continued to ULP 4 and
passed. The conflict was disposed before treatment by giving the written branch
priority and allowing only one unchanged diagnostic. Nothing in e01 reopens
v12, authorizes e02--e06, or enters a four-case aggregate.

## Measurement integrity

The evidence chain passed unusually strong checks:

- the exact subject checkpoint, revision, bf16/eager stack, driver, runtime
  fingerprint, frozen inputs, and scientific commit were bound before launch;
- independently recomputed treatment-fresh focal and nonfocal scores matched
  Phase A canonically;
- the 8,897,066-byte raw artifact was receipt-hash verified, losslessly
  packaged, reconstructed byte-for-byte, and committed;
- independent pod and local harvests matched after only predeclared
  path/timestamp normalization;
- a separate post-run audit found no unexpected difference;
- all primary fresh cells agree across regions.

No estimator-sign, persistence, extraction, or cross-host mismatch was
observed. The reservations below concern what the valid numbers mean.

## What was observed

All values are mean target-token log-probability contrasts in nats.

| Schedule / region | Family | D focal | D nonfocal | SEL | Hplus | U | Uplus |
|---|---|---:|---:|---:|---:|---:|---:|
| N / R1 | full K+V | +0.154 | -0.022 | +0.132 | +0.023 | +0.007 | +0.260 |
| N / R1 | value-only | -0.126 | -0.114 | -0.240 | -0.274 | -0.154 | -0.199 |
| **N / R2** | **full K+V** | **+0.097** | **+0.245** | **-0.148** | **-0.131** | **+0.175** | **+0.025** |
| **N / R2** | **value-only** | **+0.175** | **-0.00007** | **+0.174** | **+0.710** | **+0.182** | **+0.946** |
| N / R3 | full K+V | -0.010 | +0.142 | -0.153 | -0.086 | -0.275 | -0.166 |
| N / R3 | value-only | -0.005 | +0.186 | -0.192 | +0.257 | +0.002 | +0.361 |
| P / R2 | full K+V | +0.039 | -0.001 | +0.038 | -0.446 | -0.297 | -0.334 |
| P / R2 | value-only | +0.021 | +0.018 | +0.003 | +0.229 | -0.073 | +0.280 |

`D` is correct-history minus wrong-history state; `SEL` subtracts absolute
nonfocal movement; `Hplus` requires the correct target itself to improve rather
than merely suppressing its competitor; `U` and `Uplus` compare the
correct-history graft with fresh compaction.

### Full K+V did not show the target effect

At N/R2, the untargeted probe moved 2.5 times as much as the focal probe and the
correct target itself became less likely. The positive margin contrast came
from the countertarget becoming still less likely. Calling that recovery or
reduced loss would violate the frozen interpretation rule. Its P-schedule
utility contrasts were also negative.

### Value-only N/R2 was the one favorable cell

The value-only cell simultaneously had positive focal direction, near-zero
nonfocal differential, positive correct-target movement, and positive utility
relative to fresh. This conjunction is why the result is more informative than
arbitrary numerical noise.

It is nevertheless small and not cleanly semantic. Correct-history values
raised the correct focal target by 0.710 nats relative to wrong-history values,
but also raised its focal countertarget by about 0.536 nats. The differential
margin was only 0.175 nats. Wrong-history values themselves raised the correct
target by 0.236 nats relative to fresh, directly exposing a generic,
nonsemantic component of comparable scale.

### Schedule and region sensitivity are adverse

Changing only the source replay schedule reduced the value-only focal contrast
from 0.175 to 0.021 nats, an approximately 8.3-fold collapse. The singleton N
effect did not satisfy even the descriptive component of the frozen 3x
schedule yardstick: 0.175 was below three times the 0.153 N--P difference. The
formal yardstick was defined over four cases, so this is not an aggregate
decision; it is strong adverse robustness evidence.

The value-only contrast was negative at R1 and nearly zero at R3. Key-only and
crossed K/V cells also changed or reversed the contrast. These correlated cells
are not independent replications, but their mixed signs rule out an invariant,
additive value code from this case.

### The movement was behaviorally negligible

Fresh compaction lost 22.298 nats of focal margin and 22.291 nats of
correct-target log probability relative to the full-history oracle. The best
value-only cell recovered 0.182 margin nats (0.81%) and 0.946 correct-target
nats (4.24%). It still favored the wrong target by about 5.62 nats.

Every one of the 31 primary cells generated the same wrong focal answer,
`Ring 3`, which was neither the correct `partner beta` nor countertarget
`staff ring`; every cell also generated the wrong nonfocal answer, `30 days`
rather than `21 days`. No behavior was recovered.

### The decisive control was missing

All R1/R2/R3 placebo constructions were unavailable at the same first nonzero
early-layer row after 1,024 deterministic attempts. The correct-versus-wrong
value delta was so small that an equal-norm orthogonal perturbation could not
meet the frozen relative-error bound after bf16 casting. This is missing
evidence, not a null placebo result. Without it, e01 cannot distinguish a
semantic trace from a focal-row perturbation effect.

The persisted artifact contains score objects, token-level probe traces,
source-execution metadata, row hashes, and diagnostic norms; it does **not**
contain the actual K/V tensor rows. Therefore token-level score decomposition is
available locally, but a full row-norm distribution or a repaired placebo
validated on the real e01 tensors is not a zero-GPU analysis. Fable's otherwise
useful independent note suggested that those tensors were persisted; this is a
narrow factual correction, not a change to its verdict.

## Interpretation

The strongest warranted positive statement is:

> In one fixed, engineered, explicitly resolved-answer case, the exact N/R2
> value-only state produced a deterministic, focal-selective forced-logprob
> difference under identical visible text.

That observation modestly raises the plausibility that history-written value
state can retain a target-specific trace. It does not establish that the trace
is semantic, useful, schedule-stable, general across cases, or present in
ordinary live generation.

The strongest alternative explanation is deterministic
schedule/boundary/key-interaction-dependent lexical or numerical residue in a
saturated low-support carrier. This is not "random noise": it is an
apparatus-sensitive history-dependent effect. But absent placebo calibration,
replication, schedule stability, and behavior, the semantic-channel reading is
not identified.

The natural semantic calibration was also adverse, and the case had explicitly
resolved and repeated the answer before the carrier. Even a genuine trace here
would concern retention of resolved/possibly lexical target state, not recovery
of an unstated computation.

## Decision

1. Formal v12 remains terminally stopped. Do not run e02--e06, compute a frozen
   aggregate, or advance to conversation, confirmation, or live-agent phases.
2. Do not spend the remaining v12 ceiling simply because it is available. e01
   is reported as a weak, uncontrolled, single-case mechanistic hint.
3. Any further GPU work must be a new study with a new preregistration. At
   minimum it must resolve the path-control wording/code mismatch; construct and
   validate a bf16-feasible placebo on actual tensors before outcome scoring;
   persist a bounded tensor evidence bundle; use several new independent cases
   with non-adverse, behaviorally sensitive calibration; and compare FC/FW/FF
   plus placebo under both N and P.
4. Such a study should use descriptive sign counts and effect sizes first. A
   placebo movement comparable to FC-FW, failure to replicate across cases, or
   continued schedule collapse is a stopping result. Only replicated,
   placebo-separated, schedule-robust movement with observable behavioral
   consequence could justify a later live-agent feasibility evaluation.

## Paper-safe summary

A single post-stop diagnostic case showed a small, internally consistent
value-only directional effect at the preselected N/R2 cell (0.17-nat focal
contrast, 0.95-nat correct-target utility), but it recovered at most 4.24% of
the approximately 22.3-nat compaction damage, changed no generated answer,
collapsed under the alternate replay schedule, and had no available placebo
control. We report it as an uncontrolled mechanistic hint, not evidence of a
semantic channel or practical mitigation; formal v12 remains terminally
stopped.
