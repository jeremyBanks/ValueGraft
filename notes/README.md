# ValueGraft Notes: 2026-07

_ValueGraft progressed from exploratory cache-state experiments to a
provenance-audited, mostly null evaluation of naive post-prefill value-only
grafting. The work supports latent continuity as a meaningful research question
and shows some content-specific intervention effects, but not a robust general
benefit over ordinary text compaction._

**Participants/contributors:** User; Jeremy Banks (user); `claude-fable-5`;
`claude-sonnet-5`; `claude-opus-4-8`; `gpt-5.5-high` (high reasoning effort);
`gpt-5.5-xhigh` (xhigh reasoning effort); `gpt-5.5-medium`; `gpt-5.5-low`;
`gpt-5.5` (low effort); OpenAI GPT-5.5-xhigh; Anthropic Claude Opus 4.8;
Anthropic Claude Sonnet 5; Anthropic Claude Fable 5; Google Gemini Pro 3.1; and
`gpt-5.6-sol-xhigh`.

## From mechanism exploration to bounded mitigation

The initial hypothesis was that preserving write-time KV state across context
compaction might retain semantic information lost by text-only recomputation,
with values potentially carrying retrieved content and keys providing
addressability. Early ladder checks, cache surgery, alignment repairs, and
isolated probes supported technical feasibility. However, continuation metrics
were noisy, summary leakage contaminated some evicted-fact results, and
high-strength grafting could destabilize behavior.

The project consequently adopted a mitigation framing: test whether cache-state
interventions reduce behavioral damage after compaction, without claiming that
meaning literally resides in values. The vocabulary was separated into fresh
keys and write-time values:

```text
K(αK) = (1 − αK) Kfresh + αK Kwrite-time, rerotated
V(αV) = (1 − αV) Vfresh + αV Vwrite-time
```

The operative intervention became V-only grafting (`αK=0`), with K-only and
coupled KV variants reported separately. Write-time keys require positional RoPE
re-rotation; values require exact token alignment but no corresponding rotation.
Historical packed-layout results (`H-pack` versus `B-min-pack`) remain auxiliary
evidence because layout and K/V provenance varied together.

Prior-art review found adjacent work on KV eviction, compression, cache reuse,
editing, transplantation, and learned compact states, as well as provider APIs
exposing opaque continuation or compaction state. The defensible novelty was
narrowed to the controlled evaluation of retained write-time state across a
text-compaction boundary and its effect on referent, sense, and stance
stability.

## Empirical findings and corrections

Early 4B and 30B experiments showed reduced fabrication in some packed
write-time conditions and modest continuation gains, but later audits
established that the strongest honesty result used 4-bit MLX state and a packed
coupled intervention. Its benefit was content-agnostic and likely related
substantially to the packed layout. It is therefore calibration evidence, not a
ValueGraft cornerstone.

The cleanest Qwen value-only referent experiment was independently reproduced on
the exact Qwen3-30B-A3B-Instruct-2507 checkpoint, with approximately +0.12
recovery and the expected referent > sense > stance pattern. Yet the stronger
placebo-controlled `effect_bound` evaluation was null on both tested models:
approximately +0.017 with CI [−0.033, +0.065] on Qwen3.6-27B and −0.049 with CI
[−0.113, +0.014] on Qwen3-30B-A3B. Aligned grafts differed from corrupted-value
placebos, establishing content-specific intervention effects, but not net
improvement over ordinary compaction.

The held-out per-layer champion on Qwen3-30B produced raw graft-versus-baseline
−0.0026, CI [−0.041, +0.043]. Position-shuffled values performed much worse,
with +0.1925 relative to the shuffled contrast, showing alignment and content
dependence without demonstrating recovery over baseline. Per-layer, per-head,
intersection, and union champion families were all null in later held-out
testing. Compression conditions were flat to slightly negative, and several were
invalid because reconstructed write-time state did not correspond to the summary
request used during generation.

Coding-agent evaluation produced the only persistent positive signal, but it
remains narrowly scoped:

- The original brief SWE-Gym pool showed approximately +0.013 to +0.016
  teacher-forced next-action log-probability nats/token.
- Independent disjoint trajectories produced approximately −0.0035 at scalar
  `α=0.75`, with an interval spanning zero; no tested alpha reliably beat
  baseline.
- In-domain per-layer tuning later produced approximately +0.0126 nats/token
  across 143 trajectories and +0.0135 on the out-of-sample subset, but this was
  an imported GPT-4o/Claude trajectory proxy, not agent task success.
- No patches were applied and no tests were executed. The fixed-strength result
  was positive on the original 75 trajectories, null on an independent 98, and
  pooled borderline-null at roughly +0.0053.

These measurements must be reported as teacher-forced likelihood proxies. They
are not task success, resolution rate, or test-pass evidence. “Content-specific”
superiority over unmatched shuffled or Gaussian controls is also distinct from
net recovery over compaction.

The synthetic held-out result was corrected from a presumed homogeneous null:
Qwen-rendered bodies were positive while Claude-authored bodies were negative,
producing a pooled near-zero result confounded by source and weighting. Existing
grafts also affected both summary and retained-tail regions, so prior results
cannot be attributed specifically to summary-state retention. Native-render
variability remains unresolved; independent draws varied from positive to null,
so native rendering is not yet a proven validity requirement.

## Evaluation and workflow discipline

The live-agent track exposed cross-arm leakage, duplicate runs, stale shims,
broken probes, configuration ambiguity, launcher races, checkpoint mismatches,
and resource failures. SWE-bench was retired for the tested model after repeated
full-context failures suggested a capability floor. Tau2 banking integration was
also retired as confirmatory evidence because sessions did not reach meaningful
eviction and all arms scored zero.

The preferred future design is model-native and provenance-matched: fixed
scenarios, plants, prompts, compaction structure, and gold continuations, while
each model generates its own replies and summaries. Every run must record model
and precision, exact checkpoint, tokenizer and template, summary and tail,
cache-generation context, alpha/configuration, token alignment, code commit,
dataset hash, timeout status, and completion markers. Conversation-cluster
bootstrap intervals and raw `E−B` log-probability lifts are preferred to
unstable gap-closure ratios.

J-lens work provided low-resolution corroboration rather than direct cache
interpretation. A 63-layer sweep covering 1,175 summary tokens found the
strongest write-time/fresh separation around layer 48, while layer 62 was more
readable but more continuation-like. Moderate grafts produced small aligned
shifts; alpha 1.0 was less reliable, and sparse challenges showed near-zero
recovery when the summary omitted the relation itself. J-lens therefore remains
hypothesis-generating and must not be presented as task success or direct
evidence of cache contents.

## Archive and repository state

The repository’s research record was cleaned, documented, and pushed with the
paper framed as a science-first empirical bounding result. The paper and
external publication are now paused pending one authorized holistic revision.
Provenance corrections include a filename collision that lost the persisted
production-champion arm and five brief-champion trajectories; surviving scalar
arms, four-way comparisons, compression results, and most brief-champion data
remain usable.

The notes archive was upgraded to recursive day → month → year → README rollups.
A clean provider-configurable Luna rebuild generated and validated 55/55
conversation notes, preserved 269 opaque source IDs, added deterministic
participant/model metadata to all ten rollups, and passed 56 tests. The workflow
now supports ephemeral workers, selective subagent-final inclusion, referent
filtering, model-based filenames, provenance retention, and atomic replacement.
Summary-worker self-ingestion, custom-command argument swallowing, filename
identity conflation, and provenance placement were corrected.

## Current state and handoff

The current conclusion is a qualified negative for naive post-prefill,
value-only grafting under the tested conditions—not a general negative result
about latent-state continuity. Evidence supports content-specific and sometimes
interpretable intervention effects, but broad downstream recovery is absent,
heterogeneous, and sensitive to source, summary regime, model, and provenance.

Next work should proceed only through staged causal tests:

- Re-score existing artifacts at zero cost and repair missing summaries, token
  IDs, render context, hashes, and commit provenance.
- Compare coherent original K+V state with identical-text fresh restart,
  wrong-history controls, K-only/V-only variants, summary-only and tail-only
  grafts, and treatment-delta-matched placebos.
- Re-run source and compression effects with exact request/state matching and
  paired renders.
- Use one frozen champion for untouched confirmation, followed by a small
  capability-matched model-native agent pilot with common-prefix forks,
  workspace snapshots, ordinary compaction, coherent retention, and actual tests
  or resolved patches.
- Apply staged $0/$50/$100/$200 gates before any broader architecture, alpha, or
  layer sweep.

## Sources

- [202607.md](202607.md)
