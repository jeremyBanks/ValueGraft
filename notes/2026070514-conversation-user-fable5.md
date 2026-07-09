_This conversation is an unbroken execution log of the 30B-scale
compaction-mitigation experiment chain running through to full completion, plus
one process correction from the user refocusing effort from claim-framing onto
substantive improvement work._

**Participants:** User and claude-fable-5.

At 30B, with full n=12 on the synthetic corpus, both headline contrasts
strengthened: H-gap (write-time-encoded summary) beat B-min (fresh-encoded text)
by +0.128 nats in 12/12 conversations, CI [0.097, 0.158]; the deployable
one-shot value graft (E-post, α=1) beat baseline by +0.052 nats in 10/12, CI
[0.021, 0.086], closing about 29% of the full A−B compaction gap — a reversal
from 4B, where the same graft was harmful. A smaller, hyper-consistent α=0.25
effect (+0.018, 11/12) also held; the C arm stayed null.

Running the same synthetic-derived contrasts against natural conversations at
30B produced a negative result: none of the synthetic effects replicated
(H-gap−B-min went to −0.020, 2/8; both E effects near zero; C below baseline).
The explanation settled on was that natural conversations barely suffer
compaction damage at 30B (A−B gap only 0.07 nats vs 0.18 synthetic), so there is
little for the interventions to recover — meaning intervention benefit is
conditional on how much a continuation actually depends on evicted context, not
a blanket effect. The user then explicitly redirected effort: stop refining the
claim's framing and treat completion of the current run, followed by the
improvement pipeline (Phase 2, α sweep, gating), as the priority; framing/claim
language is deferred entirely to the eventual write-up.

The brief pass (fabrication/honesty probes, where content dependence on evicted
context is guaranteed by construction) completed fully at 30B (12/12), closing
Phase 1. Phase 1 was committed with an addendum; its headline: write-time cache
state reliably suppresses post-compaction fabrication (near-total at 30B), the
deployable value-graft is net-positive at scale, and effects are conditional on
evictable-content dependence.

Phase 2 (packed write-time summary vs fabrication, deployable form) then ran at
both scales. At 4B: 12/12 conversations completed, judged in 4 batches, merged
and committed. At 30B: also completed 12/12 (with one anomalous run, c11, taking
~1133s vs a typical ~155s — flagged as possible memory pressure or thermal
throttling, to be checked if it recurred; it did not recur on c12), HP-0
identity check passed at scale, judged in 4 batches. Judges independently
converged on the same pattern noted earlier: H-family arms admit uncertainty
about evicted content while B and B-min-pack fabricate; at 30B specifically, B
fabricates while all H-family arms admit.

A 30B negative-controls pass (E-shuffled, E-wrongconv) was run via a newly
parameterized supplement runner (committed) to certify the 30B positive effects,
given that meaningful positive effects at 30B newly warranted this certification
(previously "only if needed" at 4B).

The α sweep (grid search over blend strength and layer-gating variants, e.g.
mid-band vs late-band gates) ran at both scales with a held-out validation split
for honest reporting. Two bugs were caught and fixed pre-launch: natural-corpus
conversations needed multi-message holdout scoring (their first held-out turn is
a user message, not an assistant reply), and a key-format mismatch (`a0.0` vs
`a0`) that would have crashed the summary printer. Results diverged sharply by
scale: at 4B, the winning configuration was mid-band gate α=0.25 (middle third
of layers only), holdout-validated at +0.017 nats, 10/10 conversations, CI
[0.012, 0.024], roughly 10% gap closure. At 30B, the grid behaved very
differently — late-band gating led early, and extrapolative α=1.25 (blending
past the original values, not just toward them) dominated the synthetic set by a
wide margin, a qualitatively different optimum from 4B's interpolative mid-band
solution.

A deployment-constraints point was raised and recorded as a limitation for the
eventual write-up: both interventions require the pre-compaction KV cache to
still exist at the time of use, which is not true if the cache has already been
evicted/lost (e.g., server restart, cross-machine migration). H-pack/SelfGist
only needs to retain the compact summary state going forward (~10 MB at 30B)
once generated, making it cheap enough for most local/session use; ValueGraft
(E-post) needs the old cache resident at compaction time for its one-shot blend,
then can discard it — compatible with in-process window-fitting compaction but
not with state-loss recovery scenarios.

With the 30B sweep's holdout analysis complete, the experimental program was
declared complete: mechanism certified at both scales with negative controls,
mitigation quantified with holdout-validated numbers (packed write-time
summaries near-eliminate post-compaction confabulation; value graft closes ~24%
of the continuation gap at 30B, ~10% at 4B), and boundaries explicitly stated
(effects require compaction-dependent content, recall of evicted referents
remains unsolved, part of the honesty gain is layout-induced rather than
content-induced, and all results are n=12+8 pilot scale on one model family at
4-bit quantization).

Two forward-looking, not-yet-started tasks were queued for after the
experimental program: (1) an HF blog-post draft including an honest literature
pass, and (2) benchmark scoping — evaluating whether LongMemEval (closest fit,
long-conversation memory QA), SCBench/RULER (more synthetic,
long-context/KV-reuse focused), or a slice of LongBench v2 could support a
standard benchmark comparison of the compaction interventions, contingent on
whether their conversations fit within the ≤16K-token harness and whether the
4B/30B baselines are strong enough to show measurable differences — explicitly
framed as uncertain and requiring a scoping pass rather than a committed plan.
