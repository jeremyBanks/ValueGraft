# V12 natural calibration — result interpretation and Phase-A decision

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

**Independent checks:** two read-only Codex subagents; their runtime model and
effort identifiers were not exposed to the coordinator, so none are inferred

**Bound raw artifact:**
`results/coherent_canary_v12_technical/coherent-canary-v12-technical_exact-subject_20260712T015124276346Z.json`
(`ddc5faf7f24dd6ca70bf5833351c38b930d851bb9b0535cb45814dc32413b990`)

## Frozen verdict remains adverse

The natural downstream-note calibration is `ADVERSE`. It failed both frozen
half-recovery references:

| cell/quantity | approve-minus-deny margin |
|---|---:|
| full-history green oracle `A_g` | +21.5 |
| full-history amber oracle `A_a` | -16.0 |
| unique fresh compact state `F` | +7.5 |
| green R2 full-KV transplant `T_g` | +7.0 |
| amber R2 full-KV transplant `T_a` | +0.5 |
| `rho_green` | -0.0357143 |
| `rho_amber` | +0.2978723 |

Both transplanted cells still generated `approve`; the amber state did not
produce categorical decision recovery. The two label-balanced readings are
orientations of these same five cells, not independent cases. The `0.5`
references, fixture, or `ADVERSE` label will not be modified after observation.

## Why this is not a zero-channel result

The same-visible-text transplant contrast was
`m(T_g)-m(T_a)=+6.5` in the source-history direction, 17.3% of the 37.5-point
full-oracle separation. More importantly, the margin aggregate combines two
different token movements:

| amber component | fresh `F` | amber transplant `T_a` | amber oracle `A_a` |
|---|---:|---:|---:|
| log p(`deny`) | -7.5006876 | -0.9742548 | -0.0000305 |
| log p(`approve`) | -0.0006875 | -0.4742548 | -16.0000305 |

The amber transplant recovered 87.0% of the oracle-versus-fresh improvement in
the correct `deny` token's log probability, but only 3.0% of the oracle's
suppression of the competing `approve` token. Fresh was already essentially
saturated toward green/`approve`; green transplantation therefore had little
correct-target headroom and slightly reduced its margin. These component facts
do not rescue the failed frozen calibration. They do show that history-specific
state reached the downstream scorer in this one fixture.

The warranted update is consequently asymmetric: lower confidence in large,
reliable, bidirectional answer recovery, but no basis for calling the tested
history channel absent.

## Scope of the calibration

This was one short, explicit, one-hop policy fixture with one-token labels. It
did exercise the exact subject, gapped destination, fixed visible R2 carrier,
full-KV replacement, continuation, persistence, and scorer used by the canary.
It did not test an unchanged nonfocal control, value-only retention, P-schedule
robustness, multi-token targets, longer histories, or derived decisions. It is
highly relevant but not representative enough to terminate every engineered
estimand.

## Decision: authorize e01 Phase A only

The next paid unit is the treatment-blind exact-subject Phase A for frozen case
`e01` only (case SHA-256
`6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090`).
Phase A measures full-history correct/wrong oracle competence, strict generated
target-prefix behavior, unchanged-control competence, fixed-carrier support,
and fresh-compaction damage without computing or exposing a treatment cell.
E01 is the shortest primary case and therefore supplies both a scientific
eligibility screen and the first observed cost forecast.

This decision does **not** authorize treatment. The Phase-A raw artifact must
be preserved, pulled, independently validated, committed, and reviewed first.

- `INVALID_TECHNICAL`: stop; no treatment.
- `ESTIMAND_INADEQUATE`: stop treatment; only the already-preregistered
  Phase-A-evidence redesign route may be considered.
- `PRETREATMENT_PASS`: record both fresh margin damage and correct-target
  log-probability damage, then make a separate treatment-value/spend decision.

No new calibration variant, case edit, arm edit, threshold change, or treatment
inspection is authorized by this decision. An external orchestration workaround
may avoid the two recorded launcher/puller defects only if it leaves every
sealed scientific apparatus byte unchanged and records the exact commands. If
any sealed byte changes, the passing A7 technical report no longer releases the
semantic run and the applicable technical gate must be repeated.
