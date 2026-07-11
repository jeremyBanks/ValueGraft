_This conversation covers an independent audit that invalidated key provenance
and methodology claims in the paper, followed by a jointly designed, tightly
gated $60 mechanistic follow-up experiment whose apparatus exposed numerical
problems and was redesigned before an apparent position-preserving launch._

**Participants:** User and claude-opus-4-8.

**Audit and paper status.** The final paper remains held for one holistic
correction pass; no paper edits were made during the audit. The held-out
synthetic bodies are mixed-source rather than native test-model generations:
c07–c12 use Qwen3-4B-Instruct-2507-4bit output, while c13–c24 are
Claude/Fable-authored. Independent conversation-clustered recomputation
confirmed opposite effects: +0.1197 [+.056, +.180] for the Qwen-rendered subset,
−0.0503 [−.075, −.028] for the Claude-authored subset, and −0.0026 [−.041,
+.042] pooled. The reported synthetic null is therefore a source-conditional
sign reversal, not a homogeneous null.

Additional verified defects are that the SWE-Gym graft covers both retained-tail
and summary-token regions, generated summary text and identifiers were not
saved, the purported text hash hashes the request prompt, and SWE-Gym manifests
contain null commit metadata. The compression-sweep reconstruction concern
remains plausible but unverified. The flagship synthetic harness also discarded
the generation-time snapshot and substituted a reconstructed `force_prefill`
state. These findings narrow or invalidate the paper’s strongest
interpretations. The speculative note records six hypotheses, with H2’s
gradient-projection test as the most decisive falsification experiment.
Note-regeneration and renaming tooling remain descoped and untouched.

**Joint research direction.** A dialogue with the other agent converged on
separating two questions previously conflated: whether compaction leaves a
history-conditioned KV channel at all, and whether the old-value/fresh-key graft
actually exploits it. The agreed design uses fresh, coherent, wrong-history,
old-value/fresh-key, old-key/fresh-value, and matched-placebo states, with
downstream probe margins rather than summary-token likelihood. Native
incremental generation snapshots, tokenwise identity gates, same-text
wrong-history controls, exact provenance, and fail-closed aborts are mandatory.
The 4-bit canary is excluded from scientific conclusions; the prior SWE champion
is diagnostic only.

The budget decision was a $60 ceiling, with the $50 bf16 mechanistic tier
recommended as the highest-value experiment. The staged plan was $0 apparatus
validation, then the bf16 30B experiment; source/nativeness factorials and
completion of missing SWE confirmation were downstream. A $200 expansion would
be considered only if both a channel and a usable treatment cleared their gates,
with a capability/damage canary before any live coding-agent study. Sol owns
implementation, GPU execution, monitoring, harvesting, and spend control; the
Claude scientific lead owns preregistration, launch-gate review, estimand
challenges, and the post-freeze paper.

**Apparatus and preregistration.** The preregistration was reviewed and signed.
Its primary downstream metric is the correct-versus-counterfactual target margin
after the summary/tail/probe, with contrasts designed so target-pair baseline
differences cancel. Twenty-four matched counterfactual targets were authored and
committed. The apparatus includes true generation-time cache capture, stepwise
replay identity checks, RoPE-aware key handling, untransformed values,
fixed-point-free delta permutation placebos, exact-span checks, engineered
downstream positive controls, and negative controls.

The 0.6B build ladder passed: generation and replay caches were bit-identical
across 28 layers, key-rotation round-trip error was about 3e-5 in float32, the
engineered downstream control moved the target margin, and the placebo preserved
the exact delta multiset with no fixed points. These machinery results are not
scientific evidence about the channel. A formal bf16 gate froze tolerances
before launch, including 1e-4 for generation/replay, 0.02 for rotation/native
movement, and placebo limits. One independent review found no launch blockers,
and a conditional GO authorized only the kernel gate before semantic scoring.

**Numerical correction and design pivot.** The first paid attempt cost
approximately $0.066 and correctly aborted before semantic outcomes because the
native-shift gate compared two independent full bf16 prefills at different
absolute positions. Its large value discrepancy measured accumulated
forward-pass numerical divergence, not the actual surgery, which copies stored
values and algebraically rotates stored keys. Further testing showed that even
float32 rotation of stored post-RoPE keys could shift downstream margins by
roughly 0.10–0.19 nats, comparable to the target effect, so higher-precision
arithmetic alone was insufficient.

The preregistration was amended to retire lossy key re-rotation from the primary
assay. The replacement uses position-preserving, gapped logical positions:
summary K/V are copied bit-for-bit at their natural positions, and the
wrong-history donor occupies an exactly matched slot. This removes the rotation
and position confounds but narrows the claim to position-preserving compaction
rather than packed or repositioned serving.

**Current handoff state.** Sol is the execution lead. The apparatus correction
and gapped-v3 redesign were endorsed, but the latest state is unresolved:
commits indicate that a position-preserving v3 pod may have launched and
produced 30B results, apparently after a separate Claude session recorded a
launch GO. The designated gate-holder session had not independently confirmed
that authorization or checked whether the pod was still spending when the
transcript ended with a connection error. The immediate next action is therefore
to verify the v3 launch authorization, live pod status, spend ledger,
kernel-gate artifact, and whether any semantic results were produced before
accepting or interpreting them.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `ac1d16212826b3eb1`
- `a3958878b33fc617d`
