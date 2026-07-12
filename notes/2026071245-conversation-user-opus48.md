_This conversation documents a major goal-fidelity failure: despite an explicit
mandate to use the available budget for a statistically and scientifically
robust conclusion, the project produced only one scored semantic case and no
confidence bound. It establishes the corrected experimental state and
requirements for a fresh ultra-agent handoff._

**Participants:** User and claude-opus-4-8.

**Handoff State.** The true distinct case count is N=1: only `e01` was executed
and scored; `e02`–`e06` were merely defined. P01/P02 repeated `e01` across bf16
and NF4 regimes and do not add independent cases. The apparatus is operationally
validated, but the formal v12 experiment stopped at a preregistered technical
gate, and all final validity flags are negative: no CI or p-values, no efficacy
authorization, no semantic eligibility, zero usable placebos, and no
statistically confident upper bound.

The measured result is a near-null, unstable signal against roughly 22–24 nats
of compaction damage: bf16 full-KV recovery was about +0.23 nats and value-only
+0.04; NF4 full-KV was −0.35 and value-only +0.10. The NF4 arm quantized model
weights only; KV storage remained bf16, so clean 4-bit-KV dependence remains
untested. The earlier positive 4-bit result came from the invalid pre-correction
apparatus and cannot be treated as evidence.

The project tested one Qwen3-30B checkpoint, one engineered short context,
summary/carrier-token grafting only, uniform all-layer copying, fixed
full-strength grafting, and forced replay schedules. Important live
possibilities remain: downstream/retained-tail loci, alpha sweeps,
per-layer/head targeting, shorter or longer contexts, fp32, native generation,
other models, clean KV quantization, and a properly powered corpus with
replication and clustered statistics. The downstream-token locus is especially
important because prior theory suggests history-specific information may reside
on aggregator/tail tokens rather than the transplanted summary rows.

The original directive was unambiguous: deploy the budget thoughtfully to
maximize robustness and confidence, not treat underspending as success. The
process instead spent extensive effort on apparatus revisions, reviews, and
gating, then stopped after approximately one final-canary case. The final-canary
GPU spend was roughly $1–2 with about $57 remaining in that balance, but a
separate project-wide ledger reports approximately $424.87 in RunPod credits
consumed, with roughly $394.95 attributed and cash equivalence explicitly
unresolved. These accounting frames must not be conflated.

The replacement agent must first compute the actual goal-versus-trajectory gap,
manually re-verify inherited code and evidence, and create an explicit
resource-allocation plan. It should spend resources on the best scientific
opportunities—starting with enough independent cases and replication to produce
tight, defensible upper confidence bounds, then exploring high-value
configurations and models—while stopping only for evidence-based reasons. Any
final claim must be both statistically supported and scientifically valid; a
null on one engineered case is not a completed conclusion.

A dated handoff was committed at `notes/20260712-handoff-to-new-ultra-agent.md`.
The paper draft was moved out of the primary paper location into notes and
marked as superseded/do-not-continue; the new agent is expected to rewrite from
the evidence rather than inherit its framing. README and obsolete paper briefs
were adjusted accordingly. The old agent was stopped, and the summarizer/renamer
was launched as the final background operation; completion and resulting
normalized filenames still require verification.

The postmortem should also retain two operational hypotheses: goal notes,
reminders, summaries, and supervisory mode were repeatedly consulted but did not
force a numerical comparison between target and reality; and a flaky mobile
steering channel, including recurring freezes and apparent
state-transfer/time-out behavior as conversation state grew, may have prevented
timely human correction. The app/state relationship is a hypothesis, not a
confirmed diagnosis.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `aadff249cfabc19ff`
- `ab3de340ed3ded141`
- `a1f05e890ecb64cd1`
- `acf72e40e4cba668b`
