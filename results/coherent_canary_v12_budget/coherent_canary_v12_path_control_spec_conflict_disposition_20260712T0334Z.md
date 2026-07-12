# V12 path-control stop-rule conflict — pre-treatment disposition

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

**Timing:** committed before any e01 treatment source row or score exists

## Observed conflict

The frozen §14.1 prose says to try ULP counts `[1, 2, 4, 8, 16, 32, 64]` and
“stop at the first count for which both persisted directions differ from fresh
and are measurable.” It then says the gate passes only if `+` raises and `-`
lowers the margin by at least `1e-4`.

The exact technical raw cells are:

| ULP | Baseline | Plus margin | Minus margin | Oriented plus movement | Oriented minus movement | Code verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 7.5 | 7.5 | 7.0 | 0.0 | +0.5 | fail |
| 2 | 7.5 | 7.75 | 7.75 | +0.25 | -0.25 | fail |
| 4 | 7.5 | 8.0 | 7.25 | +0.5 | +0.25 | pass |

At ULP 2 both downstream margins differ from fresh by `0.25`, so the most
literal unsigned reading of the prose stops there; the minus edit moved in the
wrong direction and the gate fails. The sealed implementation instead defines
“measurable direction” as positive movement in each specified orientation and
continues until both oriented movements clear `1e-4`, first at ULP 4. The
independent validator enforces the implementation.

This is an ex-ante specification/code conflict, not a post-treatment edit. The
pre-freeze Claude Fable 5 review at commit
`96619d3bf1e79e6e5001388f3e3de8699be884cf`, before the final freeze commit
`2e85dee0bee917b8ddf40b59e6da7d043b79aeae`, explicitly described
“first-passing-ULP stop” as consistent with §14.1. The runner, validator, and
tests also all predate exact outcomes. That is strong evidence for intended
directional-pass semantics, but it does not erase the stricter literal reading
after the ULP observations are known.

## Independent review and decision

A fresh Claude Fable 5 review identified the conflict and judged the directional
implementation the better construction, while requiring a pre-treatment
disposition. An independent GPT-5.6 Sol audit recommended treating the written
branch as failed and, if still useful, running e01 only as an explicitly
post-ambiguity diagnostic. Sol adopts the latter, more conservative treatment.

1. **Written-preregistration branch:** classify the v12 technical gate as failed
   at ULP 2. The formal v12 claim-family decision tree is terminal and cannot
   produce `PASS4`, `AMBIGUOUS4`, `PASS6`, or conversation/confirmation/live-agent
   authorization.
2. **Sealed-code branch:** preserve the official validator's observed ULP-4
   `PASS` as implementation-defined evidence that the intervention/readout path
   is bidirectionally sensitive at a larger bf16 edit.
3. **Exploratory salvage:** authorize consideration of exactly one unchanged,
   still-blinded e01 treatment execution as a post-ambiguity diagnostic, subject
   to separate host, runtime, persistence, and cost gates. Its outcome cannot
   choose between the two interpretations or retroactively rescue formal v12.
4. Do not run e02--e06 or apply the frozen four-/six-case aggregate rules. A
   favorable e01 diagnostic may motivate a newly preregistered study; an adverse
   result remains useful bounded evidence.

The paper and final research record must prominently report all three ULP
attempts, this conflict, both interpretations, the adverse natural calibration,
and the diagnostic-only status of any e01 treatment result. Nothing here changes
a model byte, stimulus, selected row, ULP, arm, region, schedule, threshold, or
treatment implementation.
