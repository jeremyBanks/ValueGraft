_This conversation covers autonomous implementation and validation of a KV-cache
semantic-continuity experiment, including corpus construction, experimental
arms, early results, operational workflow, and the current 4B batch run._

**Participants:** User and claude-fable-5.

The experiment tests whether replacing old conversation history with a summary
loses meaning that was encoded in the original KV cache, and whether retaining
recent cache state or transplanting value vectors preserves it. Arms include A
(full context), B (summary plus freshly re-encoded tail), C (gapped retained KV
cache), D (tail without summary), E-post/E-inter (value transplantation), and
the later H-gap/B-min matched contrast. Measurements are held-out continuation
prediction and planted probe questions covering referents, word senses, stance,
ruled-out options, and evicted facts, with calibration and leakage controls. The
intended final report is paper-style.

The implementation ladder L0–L4 passed identity and cache-surgery checks. It
caught and corrected batched/single-token kernel differences, unstable Qwen
`<think>` template behavior, and tail-alignment failure when summary and tail
order changed. An isolated test showed transplanted values carrying sense
information (+0.8 nats toward the planted sense at α=1; full oracle +3.8; K+V
+1.9), with a monotone α curve. The 12-scenario synthetic corpus contains 120
validated plants; after repair, all five categories were clean of tail
contamination (24/24 each), with only harmless early-section leaks remaining.
Natural conversations were generated separately. Local subject-model generation
is intentional because held-out continuations must be natural continuations of
that same model; Claude-class agents handle scenario authoring, judging, spot
checks, and analysis.

Operational rules adopted: bracket long commands with timestamps; serialize
GPU-bound jobs when parallelism only increases memory pressure; preserve
complete command output via `tee` before filtering; use unbuffered output for
detached jobs; monitor process health and file counts; commit useful milestones
frequently while keeping model weights and large binaries outside the
repository. Several follow-up documents were committed but deferred until the
main A–E analysis; H-gap and B-min were incorporated immediately because they
reuse existing machinery. A later phase document requires a scale-vs-power
decision memo and reproducibility pinning before external scaling.

Early full-pipeline results are directional only. On c01, D closed 63% of the
A–B continuation gap, E-inter α=1 closed 33%, C matched B, and E-post degraded
with α; H-gap slightly beat B-min. At n=4, referent probes favored C and H-gap
(6/8 each) over B (4/8), while sense probes modestly favored H-gap (3/8 vs. B’s
2/8); E did not yet beat B. Continuation results were noisy and generally below
B, so continuation is now secondary. Evicted-fact scores showed summary leakage,
to be quarantined by the six-class leakage classifier. Stance keyword scoring is
invalid before judging because compliant answers may mention the rejected
option; Claude judging and paraphrase checks are required.

The 4B development batch is the first complete run; the 30B model is intended as
the primary robustness/production-scale result. Larger models improve external
validity and test scale dependence, but statistical power comes mainly from more
conversations, probes, and seeds; a future rental budget should prioritize a
second model family and higher n, with a 70B run as an external-validity anchor.
Latest handoff state: the full batch is alive and healthy at 10/19
conversations, with steady roughly 9–10 minute synthetic runs, no errors,
naturals expected to run faster, and supplement arms queued after the main
batch.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a0b7c1c1d6246bfe5`
