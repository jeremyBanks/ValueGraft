# Problem Shape Notes

This note records a small local design search for better synthetic
referent-recovery problems. It is exploratory only. The model was
`Qwen/Qwen3-0.6B` on local MPS, so the numbers should be treated as a cheap
screen, not as evidence about the final 30B/27B behavior.

## Why the first AST/Z85 idea was weak

The first harness asked the graft to recover an exact hidden code replacement
and arbitrary sentinel strings from a summary that explicitly omitted them. That
is a useful extreme stress test, but it is not a good first benchmark shape. It
collapses into "can summary-token K/V carry a high-entropy payload?" rather than
"can context-conditioned state reduce reinterpretation error?"

The better target is a missing relation that is low-entropy but still genuinely
evicted: a private label should point to a familiar action, policy, sense, or
small transform. The summary should retain the label and enough local context to
make grafting meaningful, while omitting the relation itself.

## Search Setup

Runner: `explore_problem_shapes.py`

Protocol:

- Full context A contains the payload and a distraction.
- Compacted baseline B contains a sparse summary and recent tail.
- The summary is teacher-forced after the full context to get write-time summary
  K/V.
- Graft policies are tested by teacher-forced logprob of the gold answer.
- A family is promising if `A > B` and the best graft policy moves above B.

Policies tested:

- V-only: `alpha_V` in `{0.05, 0.10, 0.25, 0.50}`
- K-only: `alpha_K` in `{0.05, 0.10, 0.25}`
- coupled K/V: shared alpha in `{0.05, 0.10, 0.25, 0.50}`

## Results Summary

From `outputs/problem_shape_exploration.md`:

| family | cases | A>B | mean best E-B | mean best gap closure | positive cases |
|---|---:|---:|---:|---:|---:|
| policy choice | 4 | 4 | +0.462 | +0.064 | 4 |
| low-entropy transform | 4 | 4 | +0.157 | +0.058 | 3 |
| format order | 2 | 2 | +0.039 | +0.006 | 2 |
| sense label | 2 | 2 | +0.003 | +0.001 | 1 |
| bug fix | 2 | 2 | -0.048 | -0.010 | 1 |

## Read

The best shape so far is **private policy label -> familiar action phrase**.
It created a strong full-vs-compacted gap and grafting helped in all four small
cases. The useful policy was V-only, usually at `alpha_V = 0.25`, with one case
preferring `alpha_V = 0.05`. This looks like the cleanest candidate for a larger
synthetic harness if we want a low-cost behavioral proxy for value-graft
semantic recovery.

The second useful shape is **named low-entropy transformation -> familiar
operation**. It was positive in three of four cases. Unlike policy choice, this
family produced some K-specific winners: `Slate` preferred `k010`, and `Pine`
preferred `k005`. This is the more interesting family if the next question is
specifically about K vs V, not just "can grafting help at all?"

The exact field-order format task is almost neutral. It is not as bad as the
first AST/Z85 task, but the effects are tiny. Keep it as a possible structured
control, not the main benchmark.

The bug-fix and pure sense-label families are weaker than expected in this tiny
screen. They create large A>B gaps, but the graft signal is mostly absent or
negative. They may need a less sparse summary, a bigger model, or better probes
before being worth scaling.

## Recommended Next Harness Shape

Use a **Private Policy Registry**:

1. Generate 30-50 private labels.
2. Map each label to a familiar action phrase from a constrained taxonomy:
   routing policy, UI rendering policy, logging/privacy policy, export policy,
   cache policy, review policy.
3. The sparse summary retains all labels and the fact that they are policy
   labels, but omits label-to-action relations.
4. Probes ask for the action phrase, preferably with a short fixed answer.
5. Score by teacher-forced gap closure first; use generated answers/judging only
   after the teacher-forced gate passes.
6. Sweep low-dose V-only from `0.05` to `0.35`; include K-only and coupled K/V
   as diagnostic arms, but do not expect K to win on policy-choice cases.

For a K-focused companion, use a **Named Transform Registry**:

1. Labels map to simple code/data transformations, not exact code snippets.
2. Gold answers should be short: `timestamp descending`, `userName`,
   `deduplicate by account id`, `redact email addresses`.
3. Include K-only low-dose arms (`0.05`, `0.10`) because this was the only shape
   where K-only showed local positive winners.

## Avoid For Now

- Arbitrary sentinel strings.
- Full transformed-code outputs as the primary score.
- Payloads where the correct answer is effectively random text.
- Summaries that say "the exact mechanics were omitted" but provide no useful
  semantic anchor beyond a bare label.
- High-alpha sweeps as the first pass; `0.50` was often harmful, and `0.05` was
  often the least bad or best low-dose setting.

