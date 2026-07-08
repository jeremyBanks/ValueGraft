_This shard covers the shipping of the ValueGraft synthesis paper (REPORT.md)
through a multi-pass adversarial-critic and prose pipeline, the pivot into Phase
2's K/V-axis exploration (key-grafting with RoPE re-rotation), a clean negative
result on key-grafting, the design and hardening of a free-generation
lens-divergence probe, and the start of a cross-check against another agent's
conflicting referent-recovery findings._

## Model-routing correction and standing practice

The user corrected a persistent misunderstanding: Sonnet should not be treated
as a cost-safe default. **Why:** Opus has ample budget headroom; downgrading to
Sonnet for cost reasons is backwards. **How to apply:** default subagents to
Opus; use Sonnet only when a genuinely independent second perspective is wanted
(e.g., critic panels, bake-offs); use Fable where its prose strength matters,
fed clean context. This was saved to persistent memory
(`model-selection-by-fitness.md`) and enforced throughout the rest of the shard
— notably as a **hard, standing requirement** that Fable must review both final
prose (readability) _and_ high-level conceptual framing/claims via subagent,
specifically because the main loop kept getting filter-flipped back to Opus even
when the user set Fable as the session model. This surfaced twice more late in
the shard as an explicit reminder ("you're opus again").

## Cross-method validation against the other agent's J-lens work

Before this shard's writing effort, the assistant evaluated a separate agent's
`intervention_probe_findings.md`, which used the _real_ Jacobian lens (J-lens,
Gurnee et al., Qwen3.6-27B weights) rather than the assistant's own crude
logit-lens readout. Their controls (alpha-zero, shifted-values) were judged
rigorous. Three independent-method convergences emerged: full-strength (α=1.0)
grafting is harmful and low α is best; alignment of grafted values to position
matters; the effect is mitigation, not full reconstruction. Critically, their
sparse-challenge negative exposed that the assistant's own "referent +10pp,
recovers evicted decisions" framing was overstated — raw numbers (17%→26%) show
referent recovery is weak on a heavily damaged baseline, not a genuine rescue.
The assistant recommended reframing referent as "reduces reinterpretation error
around facts still present in the summary" rather than "recovers omitted facts,"
and flagged this as motivation for a future K/V-axis experiment.

## REPORT.md rewrite: process and outcome

Responding to an earlier (pre-shard) rejection of a vague draft, the user gave a
detailed, wide-ranging brief: grafting research must be the star (≈70% academic
/ 30% blog), the J-lens is a secondary tool used to illuminate the research (not
vice versa), real verbatim transcript exhibits are required, "compelling"
examples should provide insight without requiring full technical understanding,
and the coding-benchmark thread should be minimized to one honest paragraph
since it never produced usable data. The assistant produced a written master
plan (`REPORT2-PLAN.md`), inventoried the other agent's existing lens
infrastructure (finding it already implemented the user's "four key boundary
points" sampling design), and identified a genuine gap: the other agent's own
lens exhibits were weak because they used examples where facts were already
explicit in the tail; the fix was to run the same infrastructure on the
project's own sense/referent corpus (label preserved, role lost), which the
other agent had explicitly recommended.

A model-mismatch was flagged before any spend: the J-lens exists only for
Qwen3.6-27B, but the project's main behavioral results were on Qwen3-30B-A3B.
The user decided (via three locked decisions) to: re-run behavioral evaluation
on the same 27B model to align both instruments, run a larger ($30–50) lens
sweep, and keep the coding-detail cut but stated honestly.

A cheap ~$5 gating smoke test was run first (per the project's now-standard cost
discipline) using three strong sense examples (Nimbus/Hydra/sandbox) with roles
deliberately stripped from summaries. Result: grafting worked correctly on the
hybrid 27B (unblocking the same-model behavioral rerun), but the dramatic lens
exhibits did not materialize — aligned-graft readouts were nearly
indistinguishable from shifted controls at the hinge token. The assistant
diagnosed this as structural, not unlucky: making an example "strong" (role only
recoverable via graft) necessarily puts it in the same sparse regime where
value-only grafting is known to fail; there is no sweet spot for vivid
single-example internal visuals with V-only grafting.

Given this, the user directed a combined mandate: finish the paper properly as a
real stopping point (Phase 1), then immediately proceed to a thorough K/V-axis
exploration (Phase 2) — the long-deferred idea of separately tunable α_K/α_V,
potentially per-layer or per-head.

Phase 1 execution: same-model 27B behavioral gap-closure across 67
sense/referent/stance plants confirmed the sense/stance dissociation replicates
on the lens's own model (sense +0.050/59% helped, stance −0.201/21% helped),
while referent came back essentially flat (+0.004/48% helped) — a much weaker
recovery than on 30B (+10pp/81% helped). This flat result became a load-bearing
motivation for Phase 2. A parallel ordinary-regime lens probe showed the J-lens
is a low-resolution, aggregate-level corroborator only — it confirms
alignment-sensitivity and the omitted-fact negative when averaged over many
tokens/cases, but individual-hinge-token readouts are noise-level, so no vivid
single-example figure could honestly be produced.

The synthesis draft (P1c) ran ~7,566 words. Five parallel adversarial critics
were then run: over-justification/redundancy (verdict 6/10 — the "two
instruments agree" refrain repeated ~6 times, needed tightening),
through-line/coherence (7/10 — lens volume overwhelms its secondary role in the
middle third; unstated model-identity seam between §5–6 and §7; a genuine
unflagged contradiction about whether the honesty effect replicates at 4B),
accessibility (6/10 — "closure" ambiguously names two non-comparable metrics),
and an accuracy critic that surfaced the most consequential issue: the draft's
claim that grafting "does not recover omitted facts" directly contradicted its
own §3 exhibits (Ruben/drivetrain/cover-art), where the summary genuinely
dropped facts and grafting recovered them (+10pp at 30B, 81% helped). The
critic's proposed fix (frame referent recovery as
"scale/architecture-dependent") was itself rejected by the user as an
unsupported overreach — the two models differ by only ~3B parameters (not a real
scale gap) and are confounded on architecture _and_ generation simultaneously
(n=2, cannot isolate a cause). The corrected, more conservative framing adopted
instead: the _core dissociation_ (sense positive, stance null) replicates
robustly across both models; the _referent_ divergence (strong at 30B, flat at
27B) is reported as a single cross-model non-replication with cause left
explicitly open — not claimed to follow any theorized dependency, and not
allowed to overwrite the demonstrated 30B recovery as if it were universally
false.

A Fable subagent was then used proactively (not just for readability) as a
conceptual gut-check on this exact reconciliation, and it independently
validated the correction while adding five refinements: flag one clearly-labeled
speculative hypothesis for the divergence rather than staying fully silent; keep
the pattern two-part (capability-threshold explains the 4B failure; the A-vs-B
referent split is separately unresolved — don't fuse them into one story);
downgrade "robust across two models" to "replicates across both tested models"
(n=2 caution); foreground the 4B honesty-effect control as evidence the 4B
result isn't just noise; and apply the same evidentiary caution to positive
replications (n=3 categories is suggestive correlation, not proof of mechanism).
All five were folded into a v2 synthesis, which grew (not shrank) to ~7,725
words as pure honesty content replaced redundancy, consistent with the user's
earlier "long is fine if it flows" guidance.

A final prose bake-off ran Fable and Opus in parallel on the same v2 source
(both barred from touching numbers, quotes, or caveats). The two versions were
nearly identical; Fable's was used as the base since it correctly cut one
redundant tag the earlier critic had flagged. After a final read confirming the
lens-middle section (§5–6) did not sag and the closing K/V motivation read
honestly, the paper was promoted to `REPORT.md` and pushed to `trunk` (~7,700
words, 28 exhibits, 12 citations).

## Phase 2: K/V-axis exploration

The technical core — key-grafting with RoPE re-rotation (re-rotating a stored
write-time key by the position delta so it matches a freshly-encoded key at its
new position) — was built and validated locally to fp32 precision (cosine
similarity 0.99999982 vs. 0.9964 for an unrotated control), confirming the
approach is sound before any pod spend.

Infrastructure friction consumed significant wall-clock but no scientific
validity: secure-cloud RunPod A100 capacity was briefly exhausted (500 errors),
forcing a community-cloud fallback; a subsequent deploy silently no-op'd because
rsync wasn't preinstalled and stderr had been suppressed; and a
`pip install -U torch` inside the job script upgraded to a CUDA-13 build that
didn't match the pod's CUDA-12.5 driver, causing `device_map="auto"` to silently
fall back to CPU (GPU showing ~1 MiB used) rather than erroring — a genuinely
dangerous silent-failure mode since it looks like a slow run rather than a
broken one. This was logged as incident #27 with two new standing rules: verify
`cuda_available` before trusting any new pod, and never suppress deploy stderr
or let job scripts upgrade torch.

Once fixed, the coarse K/V policy sweep (six global, untuned α_K/α_V
configurations: v_only, k_only, coupled, and three independent points) ran on
30B and passed its sanity gate (V-only exactly reproduced the paper's positive
30B referent value, +0.057, confirming the K-graft path doesn't corrupt the
validated V-only baseline). The result was an honest negative: K-only actively
hurt every category (referent −0.096, sense −0.283, stance −0.211); no coupled
or independent policy beat V-only on referent or sense. The RoPE-addressing
hypothesis — that keys carry retrieval/addressing information relevant to
recovering evicted decisions — was not supported by uniform key-grafting.

Because this was a coarse, untuned test, the user asked directly whether any
tuning (layer- or head-level) was in play; the assistant confirmed the coarse
sweep used a single global α pair and that the K-graft module already supports
per-layer and per-head α maps for future use. A follow-up per-layer probe (α_K
applied at one layer at a time, referent-only, 12 sampled layers across the
48-layer model) also came back a clean negative: K-only was negative at every
sampled layer, and the best single layer's lift over the V-only baseline was
+0.011 — statistically a coin flip (10/21 plants positive). This thorough
negative was recorded as strengthening rather than weakening the paper's earlier
claim: value is _the_ operative axis, checked uniformly and per-layer, not
merely "the axis we happened to test."

## Free-generation lens-divergence probe

Responding to the user's question about whether the lens could be used to make
the graft's (admittedly modest) effect more visible, the assistant identified a
methodological flaw in all prior lens work: teacher-forcing pins the token
trajectory identically across arms, mechanically suppressing the divergence
being measured. The proposed fix — free-generate from the grafted vs.
fresh-compacted states and locate the first token where their argmaxes fork,
then apply the lens at that fork point — targets the paper's weakest section (no
vivid lens exhibit) without claiming a larger underlying effect.

Given the "you're Opus again" reminder, a Fable subagent reviewed this design
conceptually before implementation and caught issues the assistant had not: (1)
a decoder-amplification artifact, where a near-tied (e.g., 51/49) argmax fork
can look dramatic while reflecting a negligible internal difference — requiring
the probe to report fork margin and check robustness across multiple decoding
temperatures; (2) an availability-heuristic risk that one vivid hero exhibit
would misrepresent the modest aggregate effect to readers — requiring the
exhibit to be shown as one point within a full-N distribution, not standalone;
(3) confirmation of a self-flagged "heads I win" concern — that a non-fork or
null result could be laundered into "the effect is just subtle" without being
falsifiable — requiring a pre-registered three-bucket classification
(FORK_TOWARD_A / SUBTLE_LEAN / DISCONFIRMING) where DISCONFIRMING genuinely
counts against the hypothesis. All four fixes were implemented in
`free_divergence_probe.py` (N=43 cases built from the existing corpus, not
hand-picked) and the probe was launched on the 27B pod under a newly quiet,
debounced monitor.

## Monitor debounce policy change

The user explicitly requested that monitors stop notifying on every routine
progress tick and instead debounce, starting at 2 minutes and backing off up to
a 24-minute maximum delay, applying to monitors in general going forward. This
was saved as a standing practice (memory: `quiet-monitors.md`) — silent through
routine progress, speaking only on completion, error, or genuine state change —
and applied immediately to the in-flight K/V and lens-probe watchers, which
subsequently reported single clean completion pings instead of per-tick spam.

## Handoff state at shard boundary

Phase 2's coarse and per-layer K/V results are both complete, negative, and
recorded in FINDINGS.md as a strengthening conclusion ("value is the operative
axis; keys don't help uniformly or at any single layer"). The free-generation
lens probe is running on the 27B pod. At the very end of the shard, the user
surfaced a new complication: another agent's
`referent_recovery_microtest_followup.md` (or similarly named file, outside this
project's own files) reports prospecting evidence that key-grafting (or
K/V-graft) _does_ help for a different task shape — short,
identifier/addressing-style label recovery (e.g., `userName`, `UTC`) — while
confirming V-only dominance for semantic policy-phrase recovery, mirroring the
project's own sense/referent split but on different data. That other agent's
data is explicitly caveated as 0.6B-scale, MPS-run, "prospecting only, not for
claims," creating a scale/rigor mismatch rather than a clean contradiction with
the project's rigorous 30B negative. The user's final instruction was to have
Fable read both the assistant's K/V conclusion and the other agent's microtest
claims together and give its assessment — this cross-check was in progress, not
yet resolved, at the shard's end.

---
