_This conversation covers the transition from an apparent reproducibility crisis
to a corrected value-grafting apparatus, followed by discovery of a
summary-source mechanism and a redesigned, statistically powered architecture
study. The current blocking issue is a self-generated-summary alignment bug
caught by the positive-control gate before the 16-model sweep._

**Participants:** User, claude-opus-4-8, and claude-sonnet-5.

The original F1 reproduction scare was resolved: the effect is stable within an
environment, while the reported mean ratio `(E−B)/(A−B)` is unstable because of
small denominators. Raw `E−B`, %-helped, and conditioned median ratios reproduce
the qualitative ordering referent > sense > stance≈0. Existing bootstrap results
on the live run give referent mean +0.125 with 95% CI [+0.030,+0.218], sense
+0.047 with CI [−0.038,+0.131] (suggestive but underpowered), and stance +0.002
with CI [−0.036,+0.049]. The paper needs a numbers/methods correction and error
bars, not a retraction. The judged metric remains a separate load-bearing result
requiring its own careful audit.

The robust standard is raw logprob lift `lp_E−lp_B`, %-helped, and bootstrap
intervals; bare mean gap-closure ratios are deprecated. This correction also
changes the K/V interpretation: value grafting remains positive, while keys are
approximately neutral rather than actively harmful. Keys are settled and must
not be reopened experimentally. The retired effect-bound probe is not the
apparatus of record.

The alignment implementation was simplified from `difflib.SequenceMatcher` to a
direct exact contiguous span map with loud failure on mismatch. Pair-for-pair
equivalence was verified on all 12 original conversations (1162–2312 pairs each,
12/12 identical), and the new map did not raise on the initial
model-generated-summary check. However, the later gate exposed that this
validation covered fixed summaries but not the production self-generation path:
on c01, the write-time span was 899 tokens versus a 538-token compacted span,
causing a deliberate failure rather than silent sparse grafting. This is
incident 32 and the current blocker. The fix must identify the structural
boundary error, pass a tokenizer-only alignment smoke, and reproduce the known
positive control; no difflib fallback or skipped convolution is acceptable.

A major mechanism result is established: Qwen3-30B-A3B reproduces the positive
effect with its own generated summary (+0.136 referent, CI [+0.034,+0.23], 81%
helped; sense +0.045, 64%), while the same model with a fixed foreign Sonnet
summary was near-null/negative. Qwen2.5-32B is significantly negative on both
fixed-summary and self-generated-summary paths, and the trusted
`gap_closure_cat.py` confirms referent, sense, and stance all negative. Thus
fixed summaries suppress the graft, Qwen2.5’s reversal is genuine, and the
cross-architecture harness agreed with the trusted apparatus on the negative
case. The positive-control failure under fixed summaries means fixed-summary
sweep results (including the earlier Mistral result) are not trustworthy and
must be rerun self-generated after the alignment fix.

The experiment was redesigned around attention geometry rather than the invalid
“MoE causes positivity” story. The corpus was augmented from 12 to 27
conversations, with 297 plants across six categories: sense, referent, stance,
ruled_out, evicted_fact, and strong_prior. New conversations are longer, contain
distractors, vary near/far plant-to-boundary distance, and provide 2–4
paraphrased probes per plant. Cluster bootstrap treats conversation—not probe—as
the inferential unit. c13–c27 were generated through a mixed
Sonnet/Opus/Fable/Codex process, validated, and frozen; c01–c12 remain
unchanged. A second doubling to c28–c54 (~100 plants/category) is in progress to
improve power and domain breadth, motivated specifically by Fable’s warning that
12 clusters were too few.

The harness now includes multi-probe averaging; strong-prior signed mass-shift
and baseline-prior-strength covariates; raw traces; model hyperparameters;
placebo grafts; identity-graft integrity checks; alpha dose-response; cluster
bootstrap; and an exploratory fractional-depth champion scan with per-layer
value alignment, all-region sanity check, arbitrary rescue-region masks, and
timing. Champion scans are explicitly non-exhaustive, weak, tentative,
non-comparable across models, and not optimization claims. Fable’s higher-value
interpretation is that depth-wise graft effects compose nonlinearly: a rescue
test on a negative dense model, grafting only positive-scanning regions, could
turn the scan into causal evidence for a depth-composition mechanism. Any result
must be framed as exploratory unless the rescue and geometry evidence support
stronger language.

The planned model set expanded to 16 models across nine vendors, with Qwen and
Gemma dense/MoE pairs as anchor-quality de-confounds, Gemma-3 as the
sliding-window comparison, Mixtral/Mistral replication, OLMo, GLM, Qwen3.6
variants, GPT-OSS, Phi-4, Yi-1.5, and two Nemotron checkpoints. Llama 2 was
deliberately excluded because of scale and era confounds. The wide sweep remains
held until the self-generation alignment bug is fixed and the positive-control
gate passes. The intended sequence is: tokenizer-only smoke;
tiny-model/full-path smoke; Qwen3 positive control with
identity/placebo/champion checks; then wide parallel execution, with lower-risk
models first and architectural gambles labeled as such. Gemma template handling
was fixed locally, but Gemma may still produce a legitimate
HybridCache/sliding-window unsupported result.

Operationally, pods are monitored by per-job watchers plus a standing all-pod
watchdog checking every five minutes and alerting when GPU usage is below 10%
with no active job; downloads are exempted. Pods are terminated after use. The
prior 200G community pod died after disk exhaustion; the replacement process
uses 400G disks, headroom checks, download verification, eviction, and smoke
gates. The “fail faster” rule is now standing: run cheap CPU/tokenizer checks
before downloads, then a tiny-model smoke, then the large positive control.
Earlier estimates of ~3–5 hours for the full sweep and ~$10–15 were superseded
by corpus doubling and the 16-model plan; exact cost should be recalculated from
the repaired gate’s timings before wide launch.

For writing, the target is a mechanism-focused paper with truthful examples,
broad evidence, and restrained claims. Fable may produce the initial full draft
once the evidence is frozen, followed by iterative edits, conceptual and
adversarial reviews, terminology/readability passes, and one Codex CLI review
using GPT-5.5 at extra-high effort for each major shareable revision. The
reproduction entry point should eventually be filterable and idempotent, but
this is deliberately deferred as yak-shaving until the science is settled.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a0a6271afcf59686a`
- `ae18ba97c7588e8bd`
- `ad729eb2adfc83065`
- `a64126a2bbc850dd1`
- `a4eeaf1e99232493e`
- `a73e35a89e2cd9130`
- `aed0faf00ee844890`
- `a2f67dd5fb42d022f`
- `a679b4e611ad535d1`
- `a4c35335d04b21ad4`
- `aed4ef78d687fcec5`
- `ab2b4d6a24ae3c808`
- `a3d8de07fc51675d8`
- `add61fd34c9818984`
- `a42fa6b853fe4d946`
- `ae8c098774921afd0`
- `a80548718ff0533c7`
- `a7b1feab366d3a26f`
- `a4bc686e08d601133`
- `a5926b479514d7bb7`
