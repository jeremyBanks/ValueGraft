# V12 e01 Phase A — acceptance, cost, and pod stop

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

## Accepted artifact

The treatment-blind exact-subject e01 Phase-A process completed at
`2026-07-12T03:14:37Z` on the same requalified pod/GPU as its technical gate.
The runner returned `PASS`; the independent validator returned
`PRETREATMENT_PASS`, `semantic_release_eligible=true`, with no invalidity or
estimand-inadequacy reasons. Both the raw document and receipt report
`treatment_scores_present=false`; the execution set is exactly `C_N`, `W_N`,
and `F` and the score set is exactly the six preregistered oracle/fresh cells.

- result commit:
  `34361aed8630a4852e16ccb22d254e68b82d0c7c`
- raw artifact SHA-256:
  `2cd7f6f190b9f4e9598844e45e5488779b72cd54a1fd51dd5b8a5b24154d672d`
- pod validation SHA-256:
  `aabac5f0aa64db8f0db49110583be9a17f8453e1a523f1cbc0e0312ec3c15410`
- receipt SHA-256:
  `23cec74067266d90464dbf2edb3c26f5db1af9fb8ee0829b249d6c42182135a5`
- job log SHA-256:
  `18f8545bcd39ba947927089b7c8e056acf0497863ff906c62eecd948e9a81d5e`

Every receipt-listed file matched its recorded byte size and SHA-256 after
pull. A fresh local validator rerun independently returned
`PRETREATMENT_PASS` and the same release decision and scientific checks. Its
only differences from the pod report were completion time and absolute versus
repository-relative source paths; after normalizing those provenance-display
fields, the reports were equal.

## Pre-treatment numbers

- correct-history focal margin `A_C_focal`: `+16.50018310546875`
- wrong-history focal margin `A_W_focal`: `-19.694557189941406`
- correct-history nonfocal margin: `+11.792686462402344`
- wrong-history nonfocal margin: `+11.667718887329102`
- fresh focal margin `FF_focal`: `-5.798249244689941`
- fresh nonfocal margin `FF_nonfocal`: `-3.699748992919922`
- fresh correct-target log-probability damage: `+22.290991485144332`
- fresh margin damage: `+22.298429489135742`

These establish the preregistered competence, wrong-history sensitivity,
carrier-support, and positive-fresh-damage preconditions. They are not a
treatment effect and do not authorize one without a separate decision.

## Observed cost and shutdown

The provider balance immediately before the same-host pod was created was
`$62.7190692504`. Immediately before deletion it was `$62.0625339393`, an
observed balance delta of `$0.6565353111` for pod `27irwplre7fmm6`. Across the
whole e01 provisioning/requalification/Phase-A unit, the pre-unit balance was
`$62.8738965671`, yielding an immediate observed delta of `$0.8113626278`.
Both are below the additive `$1.10` unit ceiling. Provider settlement may
adjust these immediate deltas; the final billing-endpoint audit remains
authoritative.

The raw runner's `estimated_cost_usd` used the wrapper's deliberately
conservative `$2/hour` argument, not the observed provider rate of `$1.39/hour`,
so it is not a provider charge and must not be summed into billing.

After artifacts were hash-checked, independently revalidated, committed, and
pushed, the pod was deleted at `03:16:14Z`. Direct lookup returned HTTP 404 and
the provider active-pod list contained zero pods at `03:16:15Z`. No pod remains
for analysis or writing.
