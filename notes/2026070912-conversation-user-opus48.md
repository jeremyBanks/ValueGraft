_The project retained per-layer champion work while narrowing near-term science
to fresh-conversation reproduction, architecture-specificity, and one gated
rescue test. Checkpoint/render persistence is now validated and operationally
central, but the reproduction remains unresolved: the baseline completed weakly
positive and fresh cells are still running._

**Participants:** User and claude-opus-4-8.

**Handoff State.** Per-layer VALUE tuning is banked and load-bearing: a
mid-layer hump around L12–L22 (peak ~L17), champion configurations at 4B/30B,
alpha sweeps, and guard validation. Per-head tuning was already diagnostically
run at 4B and tied rather than beat manual tuning; independent K/V tuning is
deferred because keys are robust-metric near-null. The only retained champion
extension is a depth-profile scan plus rescue test: after reproduction succeeds,
scan positive and negative architecture poles, then graft only the positive
model’s band onto the negative model. This is allowed only if it reuses saved
renders; it must not receive fresh-render budget ahead of reproduction.

**Checkpointing and render archive.** The harness now saves each completed
conversation as an atomic, reusable artifact containing full generated
conversation text, summary, and traces; this enables resume, splitting, and
near-free forward-pass re-scoring for champion scans, alpha/per-layer tuning,
rescues, and future analyses. The prior silent resume bug was fixed in commit
`2989820`: relative-mode checkpoints now preserve pre-scan `lp_A` values even
for empty-alignment conversations. A committed validation script exercises
fresh/resume/kill-simulation equivalence and a negative control; fresh, resumed,
and kill-sim floors all equal −9.499, while the corrupted negative control
drifts to −10.126. Fable’s follow-up review confirmed fresh-path byte identity,
resume safety, and no new gap. Residual caveat: render fingerprints do not hash
scaffold content, safe only while the scaffold remains frozen.

**Launch correction and active run.** `launch_pod.sh` was corrected to forward
`SC_CONV_START` and relative-floor settings; verification confirmed b0 =
c01–c12, b1 = c13–c24, and b2 = c25–c36. A sequential launcher hang initially
prevented b1/b2 from starting, so they were relaunched independently in
parallel. At the transcript endpoint, b0 was complete and b1/b2 were still
rendering; the full pooled baseline-versus-fresh analysis was expected in
roughly 30–60 minutes, revised from earlier overnight estimates as progress
changed.

**Results banked.** Qwen3-32B dense produced referent CI [−0.063, +0.010] (n=48;
null), with negative effects for sense, stance, ruled-out, and evicted-fact
categories. Qwen2.5-32B produced referent [−0.303, −0.157], approximately −0.23,
with sense approximately −0.42 and stance approximately −0.18. These provide a
strong within-Qwen MoE-versus-dense contrast against the original positive
Qwen3-30B-A3B MoE anchor, but they must not be oversold until the fresh anchor
reproduction is resolved. The completed runs used an older harness and did not
bank reusable renders.

**Current reproduction interpretation.** The b0 baseline positive control
returned referent CI [−0.068, +0.090], midpoint about +0.01, rather than the
original approximately +0.10. Initial concern about a broken graft was
corrected: `graft_direction_ok` is informational and aggregates across
categories; it is false because negative sense/stance dilute the mean. The
actual identity and alpha-zero smoke checks pass. The result therefore indicates
possible noise/fragility in the underpowered original n=12 estimate, not
apparatus failure; the pooled n=24 fresh comparison is required before
concluding replication failure or scope reduction.

**Enrichment plan.** Scientific quality and fresh reproduction gate all
additional spending. Once the core is sound, attempt many architectures cheaply
with pre-flight plus a one-conversation smoke test, rendering only loadable
winners. Prioritize standard text-only, high-confidence new vendors—Llama, OLMo,
Gemma-2, Granite, Command-R, Falcon, DeepSeek distill, then GLM and other
medium-risk families. Check for text-only siblings before attempting multimodal
backbone extraction; transformers 5.x and multimodal-only loaders are deferred
end-tier work. The user authorized approximately $50 additional budget, bringing
the planning total near $100: roughly $15–20 for reproduction, ~$10 for
architecture poles, and ~$60–70 for about 8–12 enrichment renders, subject to
the rigor gate. Qwen2.5-Coder-32B is explicitly queued after most new-vendor
diversity as a base-versus-coder training-domain contrast.

A small cross-model render-transfer matrix is an end-of-list exploratory
analysis: transfer generated text, never KV tensors, and measure whether another
model’s graft survives on foreign but coherent text. A few cells may test
native-context specificity; it is strictly after core, diversity, and
champion/rescue work. Major scientific milestones, especially the reproduction
verdict and architecture results, should trigger user notifications as a
standing operating practice.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a6c7a1cd7b3551e3a`
- `a3009cfb847249188`
- `a9ad442007b6ce5c6`
