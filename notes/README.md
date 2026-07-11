# ValueGraft Notes: 2026-07

_July progressed from controlled demonstrations of cache-state differences to a
provenance-audited, mostly null assessment of value grafting as a
compaction-mitigation technique. The strongest surviving evidence is
conditional: grafted values can be less damaging than corrupted or shuffled
alternatives and may alter semantic behavior, but reliable improvement over
ordinary compaction or real agent success was not established._

**Participants/contributors:** Jeremy Banks (user); `claude-fable-5` / Anthropic
Claude Fable 5; `claude-sonnet-5` / Anthropic Claude Sonnet 5; `claude-opus-4-8`
/ Anthropic Claude Opus 4.8; `gpt-5.5-high` (high reasoning effort);
`gpt-5.5-xhigh` / OpenAI GPT-5.5-xhigh (xhigh reasoning effort);
`gpt-5.5-medium`; `gpt-5.5-low`; `gpt-5.5` (low effort); Google Gemini Pro 3.1;
`gpt-5.6-sol-xhigh` (source-labeled, extra-high reasoning); and OpenAI Codex,
GPT-5 family, extra-high reasoning (minor version not verified).

## From mechanism demonstration to mitigation study

The project began by testing whether preserving write-time KV state across text
compaction retained information that fresh recomputation lost. The design
separated summary encoding, value payloads, address/key policies, and
correspondence mechanisms. Terminology settled on ValueGraft for the
intervention family, V-Graft for fresh keys blended with write-time values, and
separate names for K-only, coupled KV, packed-layout, and retrieval-weighted
variants.

The central claim was narrowed repeatedly. ValueGraft is not evidence that
“meaning lives in values,” nor is it a general novelty claim about opaque
compaction handles or KV-state reuse. The defensible question is whether
preserving aligned write-time state reduces behavioral damage after compaction.
By month end, even that claim was restricted to source- and condition-dependent
effects.

Early 4B and 30B work established useful controls and failure corrections:

- Cache-surgery identity ladders, re-rotation checks, token alignment, and
  alpha-zero equality checks passed after correcting batched-versus-single-token
  kernels, Qwen templates, and summary/tail alignment.
- A sense-level probe showed transplanted values moving behavior toward the
  planted sense, but continuation results were noisy and generally below
  baseline.
- Packed historical arms reduced fabrication, especially on decoy and unknowable
  questions, but the benefit was largely attributable to packed layout and
  caution behavior. These results are auxiliary calibration evidence, not clean
  value-only evidence.
- Wrong-conversation, shuffled, and corrupted-value controls caused substantial
  degradation, supporting content and alignment specificity. Later review noted
  that the controls were not treatment-delta matched, so they establish
  compatibility and harm avoidance rather than hidden-history recovery.

The project adopted separate key and value policies, with the operative
production axis being V-only grafting: fresh keys and blended write-time values.
Keys require positional treatment and correct RoPE re-rotation; values do not.
Moderate scalar doses were safer than alpha one, and later per-layer tuning was
treated as exploratory rather than a general solution.

## Behavioral evidence and its reversal

Independent harnesses reproduced a referent-oriented effect on the exact
Qwen3-30B-A3B-Instruct-2507 checkpoint, with the expected referent > sense >
stance pattern. This supported a possible reduction in semantic reinterpretation
for information retained in the summary. J-lens readouts provided only
low-resolution corroboration: moderate aligned interventions sometimes shifted
internal readouts toward full context, while sparse challenges rarely produced
decisive behavioral forks.

The apparent headline did not survive provenance and holdout review:

- The cleanest bf16 value-only `effect_bound` comparison was approximately
  +0.017 nats/token on Qwen3.6-27B with a confidence interval crossing zero, and
  −0.049 on Qwen3-30B-A3B with a confidence interval also crossing zero.
- The primary held-out 30B per-layer champion was −0.0026, with a confidence
  interval spanning zero. Aligned values outperformed position-shuffled values,
  but did not outperform ordinary compaction.
- Per-layer, per-head, intersection, and union champion families were null on
  held-out validation. The compression sweep was flat to slightly negative from
  ultra-brief through realistic summaries.
- The original brief-SWE-Gym pool showed a modest positive teacher-forced
  next-action log-probability change, roughly +0.013 to +0.016 nats/token.
  Independent disjoint trajectories reversed this to approximately −0.0035 with
  an interval spanning zero; no tested alpha improved reliably over baseline.
- A later pooled estimate across the original 75 and independent 98 trajectories
  was about +0.0053 nats/token with uncertainty spanning zero. An internally
  tuned champion was positive on its own held-out split but did not clearly
  outperform the fixed scalar graft or demonstrate executed patches, tests, or
  task resolution.
- The synthetic held-out corpus was heterogeneous: c07–c12 used Qwen3-4B 4-bit
  outputs and c13–c24 used Claude/Fable-authored material. Recomputed effects
  were approximately +0.120 nats/token for the first block and −0.050 for the
  second, producing a pooled value near −0.003. The synthetic result is
  therefore a source-conditional sign reversal, not a homogeneous null or
  universal positive effect.

The surviving interpretation is that aligned values can be content-sensitive and
less harmful than mismatched state, but the intervention is non-additive or
otherwise insufficient for broad downstream improvement. Teacher-forced log
probability remains a proxy, not behavioral success, resolution rate, or
test-pass rate. Action-match scores are coarse tool/path agreement unless actual
edits and tests are inspected.

## Evaluation, infrastructure, and validity corrections

The live-agent program was reset after cross-arm session leakage, duplicate
runs, dead shims, broken probes, ambiguous configurations, and cache-memory
failures. SWE-bench was retired for the tested model after repeated full-context
failures suggested a capability floor. Tau2 banking integration was also retired
as confirmatory evidence because sessions did not reach meaningful compaction
and all arms scored zero. A future benchmark must combine genuine eviction,
cross-exercise dependencies, and sufficient task capability.

The native per-model design became preferred: hold scenarios, plants, prompts,
compaction structure, and gold continuations fixed while each model generates
its own replies and summaries. Reply length, headroom, task competence, and
meaningful eviction must be measured as gating variables. Native rendering
itself was not proven mandatory; independent draws varied from positive to null,
so render variance requires an explicit envelope.

Several operational rules became durable:

- Verify the exact checkpoint, precision, template, model configuration, summary
  request, cache, tokens, alpha, metric, split, and code commit before
  interpreting a result.
- Require positive-control validation before any new pipeline or architecture
  sweep; agreement on a negative case is insufficient.
- Keep layout, summary text, retained tail, positions, prompt shape, decoding,
  and provenance fixed when isolating K or V policy.
- Use conversation-clustered bootstrap intervals and report raw `E − B`,
  percentage helped, and uncertainty; do not headline unstable gap-closure
  ratios.
- Require manifests, completion markers, `score.json`, timeout status, and
  provenance. Process liveness is not evidence of useful progress.
- Use unique per-run directories and checkpoint each conversation. A filename
  collision destroyed the persisted production-champion arm and five
  brief-champion trajectories; earlier claims that no data were lost must not be
  repeated.
- Keep packed-layout, 4-bit, foreign-reply, illustrative, and interpretability
  results visibly separate from bf16 value-only evidence.

The archive itself was consolidated into hierarchical day → month → year →
README rollups. Extraction, filtering, sharding, provenance retention, filename
normalization, and manifest-aware updates were tested, with 57 repository tests
passing after the latest correction. The final Luna rebuild had validated only
part of the replacement archive; the old archive remained untouched pending full
validation and atomic swap.

## Current state / handoff

The paper and README were finalized around an evidence-bounded result, but
external publication is paused pending one authorized correction pass. The
current scientific position is:

- Value grafting has not demonstrated reliable net recovery over ordinary
  compaction, semantic-continuity improvement across models, or agent task
  success.
- Content-specific effects and source-conditional synthetic behavior remain
  plausible and worth a small causal test.
- The only justified positive lead is a heterogeneous, domain-specific
  brief-summary coding proxy, not a reusable technique.
- Packed-layout honesty/fabrication reductions remain supporting calibration
  evidence and must not be presented as value-only ValueGraft results.

The next work should be narrow: audit the final paper, verify
compression-request traces and action-match scoring, rescore surviving
artifacts, and run a small local causal canary comparing coherent original
state, identical-text fresh restart, matched wrong history, aligned
old-value/fresh-key graft, and a treatment-delta-matched placebo. Only a
successful canary should authorize a capability-matched model-native experiment
and eventual agent pilot with environment forks and tests. Broad architecture,
alpha, layer, imported single-action, and further synthetic or quantization
sweeps should receive no additional budget.

## Sources

- [202607.md](202607.md)
