_The project pivoted from broad cross-architecture/QK-norm ablation work to
validating whether the headline value-grafting effect generalizes beyond the
original 12 conversations, while preserving architecture-specificity as the main
secondary result. A preventable timeout exposed a deeper checkpointing and
capacity-planning defect; the active work is now a deliberate, resumable
recovery._

**Participants:** User and claude-opus-4-8.

**Scientific state.** The validated core remains Qwen3-30B-A3B’s approximately
+0.10 referent gap-closure effect and the referent/sense/stance dissociation,
but its generalization is unresolved. Mistral’s no-QK-norm referent effect was
modestly positive (+0.035, CI excluding zero), while Qwen2.5 was negative and
phi-4 null, falsifying the strong “QK-norm predicts sign” framing and supporting
architecture-dependent, potentially reversing effects. Full QK-norm removal
broke generation and competence scoring, so the ablation was stopped and H1 will
be reported as a preregistered unsupported/null claim; the proposed graded λ
dose-response was also abandoned as lower-value and causally confounded by model
collapse.

The highest-priority test is a matched native-render block experiment: c01–c12
as a positive-control baseline and c13 onward as fresh conversations under one
frozen configuration. A fresh-only run would be uninterpretable because
configuration changes could explain a null. The canary initially rendered
c01–c24 but hit a hard timeout during scoring after roughly 6.3 hours; all
in-memory renders were lost. The timeout formula has been increased, but the
more important defect is that the harness wrote no durable intermediate state.
The recovery must add per-conversation checkpointing/resumption, centrally pool
traces for one relative competence floor and conversation-clustered CI, report
headroom and morphology by cell/category, and verify the baseline before
interpreting fresh data. The stopping rule is fixed: use 24 fresh conversations
first and extend to c37–c54 only if the fresh referent CI still spans zero.

**Current implementation state.** The relative competence floor was
preregistered and implemented as `median(lp_A) − 3·MADN(lp_A)`, with absolute
mode remaining the byte-identical default. It is inert on existing Qwen/Mistral
results and adapts only the lower-scale OLMo case; OLMo is no longer
prioritized. `SC_CONV_START`/`SC_CONV_LIMIT` selection was committed and
CPU-validated to select exact conversation IDs without overlap or regression.
`scripts/block_analysis.py` was committed and validated against the Mistral
fixture; it pools floors centrally, computes clustered CIs, headroom,
morphology, positive-control status, and the precommitted extension decision.
Judged-metric bootstrap work found sense +12.0pp, CI [+2.2,+22.9]pp,
significant; referent +9.7pp, CI [−6.9,+26.2]pp, not significant; stance +3.3pp,
null. Thus significance flips by metric: referent is stronger on logprob, sense
on judged outcomes.

**Operational state and lessons.** Four pods had been reduced to the canary,
cached Qwen3-30B-A3B extension capacity, Qwen3-32B pair-B, and Qwen2.5-32B;
after the timeout and repeated SSH ambiguity, the canary and extension pods were
terminated and only the two architecture poles remained active at about
$2.78/hour. Incident #38 records the lost-render failure and identifies the
generalized lesson: probe-derived per-unit timing must be checked against every
bounded resource before full launch, fail closed on insufficient headroom, and
multi-hour work must checkpoint incrementally. The monitor correctly detected
the failure but cannot prevent waste without capacity planning. Idle capacity
should be filled proactively with useful, preemptible work, while never delaying
higher-priority experiments; wanted work must not be terminated by inferred
intent, but idle, failed, or surplus pods may be stopped for budget.

**Handoff plan.** First complete and CPU-validate checkpointing, obtain an
open-ended Fable review, and commit it only after confirming no numerical
change. Then provision fresh clean pods, run the block experiment with
probe-based timing/headroom checks, harvest and pool results, and stop all pods
immediately after data collection. Run an un-anchored trajectory review before
fixing the narrative. Keep FINDINGS.md, STATE.md, DECISIONS.md, and related live
documentation current; obsolete historical material was trimmed from STATE.md,
and AGENTS.md now points agents to generated daily/overall summaries beginning
at `notes/README.md`. After GPU work ends, write the paper collaboratively with
substantial Fable review, place it in the repository, and promote it to the
README once the evidence is settled.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a946ba511a2e7dc83`
- `a918930cbba748648`
- `abf17e46f8ea33c87`
- `a38a2ae86f70296f6`
- `a46fad7566d9fc073`
- `a824237994b367577`
- `a3505b1916e10d194`
- `a7e0257a67cfc6ccb`
- `a2bc1ff7aff2cd660`
