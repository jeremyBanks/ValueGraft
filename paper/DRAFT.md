# What a model's write-time KV state buys across a compaction boundary: honesty, not recovered meaning

*Value grafting (ValueGraft): a robust anti-fabrication effect, a meaning-recovery
effect that does not reproduce, and the render-fragility discipline that told the two
apart.*

> **Byline.** By Anthropic Claude Fable 5 and OpenAI GPT 5.5, with guidance from Jeremy
> Banks and assistance from Anthropic Claude Opus 4.8, Anthropic Claude Sonnet 5, and
> Google Gemini Pro 3.1.

> **Provenance of this document.** The experiment was designed and directed by the
> project owner (Jeremy Banks) and executed by an autonomous coding agent on a single
> 32 GB Apple-Silicon machine plus rented A100/H200 pods. Adversarial review was provided
> throughout by GPT-5.5 (Codex) and tool-less Fable review passes. Every experimental
> number below is recomputed from a committed on-disk result file — the ledger is
> `CLAIMS.md`, the one-command from-disk reproduction is `scripts/reproduce.py` — and
> each headline number ships with its confidence interval and its caveats. Where a claim
> could *not* be reproduced from disk, we say so and do not ship it (one central recovery
> headline failed this test; §6). This paper reports what we found, not what we hoped to
> find.

---

## Abstract

Production LLM systems keep long conversations inside a fixed context window by
**compaction**: older turns are replaced with a short text summary, a recent tail is
kept verbatim, and generation continues. Compaction is lossy twice over — it drops
content the summary omits, and it discards the **write-time key/value (KV) state** the
model built while originally reading those turns. We test **ValueGraft**: saving the
model's own write-time *value* vectors — for the summary it generates and the tail it
keeps — and re-injecting them at the compaction boundary, leaving keys freshly encoded.

We set out to show that this recovers lost *meaning*. It does not — at least not
reproducibly. The meaning-recovery signal is real in individual renders and tracks the
compaction damage, but it is **render-fragile**: on the Mixture-of-Experts model where
it was developed, a clean independent re-render under one consistent judge collapses the
judged sense effect from **+8.7pp to +1.0pp**, and the referent-logprob anchor that had
read **+0.10** on an unbanked render came back **+0.012** (CI spanning zero) the first
time it was banked to disk. We report meaning recovery as an honest **non-reproduction**,
and we make the render-fragility of small-*n* recovery claims on nondeterministic MoE
renders a first-class methods contribution.

What *does* survive is an effect on the model's **epistemic behavior**, and it survives
the same statistical bar that killed the recovery claim. A compacted model **fabricates
confidently** about content it can no longer see; retaining write-time KV state at the
boundary **converts that fabrication into honest admission**. On matched decoy probes,
fabrication drops by **+66.7pp** (conversation-clustered bootstrap CI **[+45.8, +87.5]**,
12 clusters, n=24/arm); on genuinely evicted facts it drops **+62.5pp** (CI
**[+41.7, +83.3]**), converting fabrication into "I don't have that" — the model recalls
**0/24** evicted facts either way. It **buys honesty, not recall.** The effect replicates
across scale (4B→30B) and precision (4-bit→bf16), and a within-layout decomposition shows
the write-time-KV contribution is separable from the mere packing layout it rides on. We
report the honesty effect as the robust positive, the meaning-recovery effect as a
cautionary non-reproduction, the compaction-damage characterization as a clean
framing-independent baseline, and a cluster of measurement traps — the render-fragility
above all — as the transferable methodological content.

---

## 1. Problem and setup

### 1.1 What conversation compaction is

Long-running LLM conversations and agent sessions eventually exceed the context window.
The universal production fix is **compaction**: replace a run of older turns with a
shorter text **summary**, keep a **tail** of the most recent turns verbatim, and
continue. Every major hosted assistant and agent framework does some version of this
(hosted APIs now ship it as a first-class primitive), and it is invisible to the user —
the model simply keeps going with a compressed record of what came before.

Compaction is lossy by construction. A summary is a *re-encoding*: the model (or another
model) reads the old turns and writes a compressed description of them, which is then
re-tokenized and re-attended-to as fresh input. Two things are thrown away in that round
trip:

1. **Content the summary omits** — specific decisions, referents, and facts that did not
   make the summarizer's cut.
2. **The internal state the model built while originally processing those turns** — the
   key/value (KV) cache entries produced at *write time*, which encode not just *what*
   was said but the contextual, disambiguated *reading* the model had settled on.
   Re-encoding a summary recomputes KV entries from the summary text; it cannot
   reconstruct the write-time state of the original turns, because that text is gone.

This project is about the second loss, and whether a slice of it can be repaired. (That
compaction *destroys* context-conditioned state is our baseline, not a finding — §3
quantifies the damage; it is the denominator every intervention is measured against.)

### 1.2 The ValueGraft idea

The KV cache stores, per layer and per attention head, a **key** vector and a **value**
vector for every past token. Keys encode *where to attend* (they carry positional/RoPE
structure and participate in the attention dot-product); values encode *what gets read
out* once attention has been placed. When the original turns are evicted, both are
discarded — and even the *retained* tail and the model's own summary lose the write-time
activations they were computed with, and are re-encoded from bare text.

**ValueGraft** asks: if we saved the write-time **value** vectors — for the summary the
model generated under the full conversation, and for the tail it retains — can we
re-inject them at the compaction boundary so that the continued conversation reads out the
meaning the model had originally computed, rather than only what the summary text
re-encodes?

Concretely, at the compaction boundary we blend the fresh value vectors with the saved
write-time value vectors at aligned token positions:

```
V_grafted = (1 − α_V) · V_fresh  +  α_V · V_writetime
```

with a single strength knob `α_V` (default **0.75**; `α_V = 0` recovers plain compaction
exactly, `α_V = 1` is full replacement). **Keys are left as freshly encoded** — the graft
re-supplies write-time *content* without altering where the model attends; §5.2 justifies
this empirically (keys turned out to be the neutral axis, values operative). A symmetric
`α_K` knob for keys exists in the code but is off by default.

The bet is narrow and mechanistic: *the model's own write-time value vectors carry
disambiguating meaning that a text summary loses, and re-injecting them can recover a
slice of the continuity that compaction destroyed.* Whether that bet pays off — and where,
and on which architectures — is the empirical content of the paper. The short answer, laid
out honestly below, is that it does not pay off *as a reliable meaning-recovery tool*, but
that the same intervention buys something else — honesty — that does hold up.

### 1.3 Apparatus at a glance

For each test conversation we construct three arms over an identical scaffold:

- **A — Original.** The full, uncompacted conversation. The ceiling: what the model does
  with everything still in context.
- **B — Compacted.** The production baseline: system prompt + the model's own summary
  (rendered as an assistant "context note") + a verbatim tail. Older turns evicted at
  their original positions (gapped KV, so positions are preserved).
- **E — Graft.** Arm B plus the write-time value graft over the summary (and tail)
  regions (`α_V = 0.75` unless noted; doses `α_V ∈ {0.25, 1.0}` are also run).

Canonical arm names used throughout: **Original / Compacted / Compacted + value graft
(α=…) / Compacted + layer-tuned value graft.** A separate **honesty** experiment (§4) adds
a **Packed write-time-KV** arm and a within-layout control; those are named where used.

We measure two ways (§2.4): a **teacher-forced logprob** metric on a shared gold
continuation (objective, judge-free) and a **Sonnet-judged meaning-recovery** metric on
greedily generated answers to planted probes. In each case the question is the same: does E
move behavior from the B (compacted) level back toward the A (full-context) ceiling, and
where?

---

## 2. Methods and data provenance

> This section is a **blocking deliverable** per `METHODS-PROVENANCE-REQUIREMENTS.md`: the
> paper must state, for every token in the experiment, who or what produced it, with exact
> checkpoints and reproducible procedures. This project lost hours to exactly one
> undocumented provenance detail (whether a conversation's assistant replies were the test
> model's own — it turned out to determine whether the effect appears at all). We document
> it exhaustively here.

### 2.1 The corpus

The evaluation instrument is a set of **composed conversations** with hand-planted probes:

- **Synthetic conversations `c01`–`c54`** (`data/synthetic/cNN.json`). `c01`–`c12` are the
  original hand-authored scenarios (10 plants each; 120 plants, contamination-audited);
  `c13`–`c54` are a later **augment** batch (12 plants each, adding `strong_prior`),
  authored 2026-07-08 by a mix of Claude/GPT subagents with per-file schema verification,
  then **frozen**. Each conversation is a realistic multi-turn working session (e.g. `c01`
  = a SaaS product launch, 45 messages).
- **Natural conversations `n01`–`n08`** (`data/natural/`), no plants — used for
  continuation-scoring only; no conclusion here rests on them.

**Provenance, per component:**

| Component | Produced by | Notes |
|---|---|---|
| System prompt | Authored (templated) by the experimenters | Fixed per conversation |
| User turns | **Authored by the experimenters** (Claude subagents wrote scenario text), *not* model-generated | Real user prompts are not model output; a result depending on model-generated prompts would be invalid |
| Planted manipulations (probes) | Authored, hand-placed | Categories below; keyword/anti-keyword scoring hooks; contamination-audited so planted keywords appear only in the evicted middle |
| Assistant replies (the conversation body) | **Generated in-context by the TEST MODEL ITSELF** (per-model-native) | Load-bearing — see §2.5. Each cross-arch model regenerates its own replies over the shared scaffold |
| Compaction summary | **Self-generated by the test model** (it summarizes its own conversation) | A *foreign* summary suppresses the graft (§5.3) |
| Gold continuation (teacher-forced target) | Derived from the **planted facts**, **shared** across models, *not* model-generated | The difference metric only cancels target-nativeness if the target is shared |

**Plant categories** (the middle token of a plant id is its category, e.g.
`c01-sense-2`):

- **referent** — recover a *specific evicted decision* ("what pricing model did we settle
  on?"). Retrieval-hard.
- **sense** — disambiguate an *evicted referent's meaning* ("when I said Nimbus, which
  workstream did I mean?"). Semantic-disambiguation.
- **stance** — honor an *evicted preference* ("no scammy urgency tactics"). A good summary
  usually preserves this.
- **ruled_out** — a rejected option that should stay rejected.
- **evicted_fact** — a specific fact from an evicted turn.
- **decoy** — a question about content that *never existed* in the conversation (the
  honesty probe; §4).
- **strong_prior** — conversation meaning vs a famous prior meaning of a term.

The original `c01`–`c12` corpus was rendered in-context by a Qwen-family model (the early
MLX pipeline, `mlx-community/Qwen3-4B-Instruct-2507-4bit`; see the `meta` block in each
scenario file). This is the origin of the **nativeness confound** (§2.5, §5.4) and the
reason a per-model-native re-render design exists.

### 2.2 The arms and doses

The full arm family, unified under the two-axis `(α_K, α_V)` ValueGraft framework
(historical arm IDs A–H are frozen in code and old results):

- **A = Original** — full context, no compaction.
- **B = Compacted** — `α_V = 0`: system + self-gen summary-as-context-note + tail; older
  turns evicted at their original positions.
- **E = graft, `α_V > 0`** — the V-only graft over the summary/tail regions. Primary dose
  **`α_V = 0.75`** (`SC_GC_ALPHA`; the project standard, peak on the bf16 30B α-sweep);
  additional doses `α_V ∈ {0.25, 1.0}` are run and, for the judged metric, **pooled**
  across `{0.25, 1.0}` (`judged_bootstrap.py`).
- **Honesty arms (§4):** **H-pack = "Packed write-time-KV"** (retains keys *and* values in
  a packed summary layout — a **cousin** of the pure value graft, not the identical
  intervention), and **B-min-pack** (the same packed layout with write-time KV *off*) as
  the within-layout control that isolates layout from write-time state.

Keys are fresh (`α_K = 0`) in all value-graft headline arms. A **K-only** / **coupled-KV**
sweep (§5.2) shows values are the operative axis and keys approximately neutral.

### 2.3 CONT vs PROBE evaluation modes

Two evaluation modes per conversation (`DECISIONS.md`, 2026-07-04):

- **CONT (teacher-forced logprob).** The held-out final assistant message is scored
  teacher-forced; each arm's cache is built over `msgs[:-1]` and the continuation is scored
  token-by-token (1-token decode steps — never batched-prefill logits, which use a
  different kernel and differ by ~0.5 at 4-bit/fp16; the L0 numerical trap, `DECISIONS.md`
  2026-07-04). Objective and judge-free.
- **PROBE (greedy answer + judge).** Probe user turns are appended to the full
  conversation; the model answers greedily (temperature 0); a judge scores the answer for
  meaning recovery (or, for honesty, classifies it CORRECT / ADMITTED / FABRICATED). Each
  mode gets its own in-context summary run; the summary text is shared across arms within a
  mode.

### 2.4 The two metrics

**Metric 1 — `raw_EB` (teacher-forced logprob lift).**

```
raw_EB = lp_E − lp_B
```

the per-token gold logprob under the graft minus under plain compaction, on the **shared**
gold continuation (a short statement of the correct answer, ~16–18 words, derived from the
planted facts, never model-generated). The difference cancels target-nativeness, so the
*sign* is the robust quantity. We also report `%_helped` (fraction of plants with
`raw_EB > 0`) and headroom-normalized `raw_EB / max(A−B, ε)`.

> **Estimator note (a genuine methods contribution — §5.1):** the earlier headline used a
> **mean-of-ratio** estimator `(E−B)/(A−B)`, which is Cauchy-unstable when the `A−B`
> denominator is near zero. It is **retired**. The primary estimator is `raw_EB` (bounded)
> with a **conversation-clustered percentile bootstrap 95% CI** (resample whole
> conversations, not plants — plants within a conversation are correlated, so plant-level
> resampling is anti-conservative). `N_boot = 10000`, seed fixed. Median-of-ratio
> conditioned on `|A−B| > 0.5` is kept only as a scale-free secondary.

**Metric 2 — Sonnet-judged meaning recovery / honesty classification.** In PROBE mode,
greedy answers are scored by a Claude Sonnet 5 judge. For meaning recovery: a three-level
scale **RECOVERED / PARTIAL / MISSED = 1 / 0.5 / 0** (blind across arms); the per-category
effect is `mean(pooled-graft) − mean(Compacted)`, with the same conversation-clustered
bootstrap (`judged_bootstrap.py`, `N_boot = 20000`). For honesty: each answer is classified
**CORRECT / ADMITTED / FABRICATED**, and the effect is a per-conversation fabrication rate,
bootstrapped over the 12 conversations.

### 2.5 Gates, exclusions, and the relative competence floor

Applied per model, before any sign is read (`PREREGISTRATION.md`; `block_analysis.py`):

- **Headroom (A−B) gate, threshold 0.3.** A category whose summary already preserved the
  content has no `A−B` gap — there is *nothing to recover*, so a near-zero graft effect
  there is **FLOORED / uninterpretable**, explicitly *not* scored as "harm." A null at
  *high* headroom (real damage, no recovery) is flagged distinctly from a null at *low*
  headroom (nothing to recover). This distinction is machine-checked in `block_analysis.py`
  (`null_kind`).
- **Task-competence floor (relative, pre-registered).** Replaces an earlier absolute floor
  (`lp_A < −8.0`) that silently dropped different plants from different models (a
  cross-model confound — inert on Qwen/Mistral, but it excluded *every* OLMo-2 plant). The
  frozen rule (`PREREGISTRATION.md`, GATE #3 amendment):

  ```
  floor_model = median(lp_A) − K · MADN(lp_A),   MADN = 1.4826 · median|lp_A − median|,   K = 3.0
  ```

  computed from each model's own gold-`lp_A` distribution, min-N = 8 (else no drop).
  Verified on-disk to be a **no-op** on the already-scored Qwen/Mistral sets, so it changes
  nothing there and only rescues OLMo-scale distributions. (Exclusion counts recorded per
  run — e.g. phi-4: 1 plant task-excluded; Mistral: 0.)
- **Generation-quality screen.** The model-filled replies are coherence-screened; a model
  whose native replies are degenerate is flagged untrusted, not hidden.

### 2.6 The graft mechanics and alignment

- **Write-time snapshot.** When the model generates its own summary, the value vectors
  produced over the to-be-evicted content are snapshotted per layer/head (`kvlib.py` /
  `kvlib_hf.snapshot_cache`, generic over transformers ≥5 `cache.layers[i].values` and
  legacy `value_cache`).
- **Alignment.** Grafted values are placed by **difflib positional-within-region** matching
  (summary↔summary, tail↔tail; minimum matched block 8 tokens; attention-sink positions
  [first 4] and special tokens excluded), never spurious cross-conversation matching. A
  strict exact-span variant was tried and **reverted** because it breaks on thinking-model
  self-gen tokenization; the tolerant aligner was verified to align 100% of regions on
  every model reported.
- **Per-model smoke gate (numbers untrusted unless it passes).** For every model: (a)
  `α_V = 0` reproduces B's teacher-forced logprobs within a tight tolerance
  (`SC_ALPHA0_TOL = 5e-3`) — proves the graft plumbing is bit-identical to B; and (b)
  `α_V = 0.75` actually *changes* the logprobs vs B — proves the graft is not a no-op on
  this cache class. Padded/HybridCache snapshots (Gemma-style sliding-window) are detected
  (stored token count ≠ input length) and the model is marked UNSUPPORTED with no numbers,
  rather than silently mis-grafted.
- **Key re-rotation (for the K-axis experiments).** Keys are cached post-RoPE, so moving a
  stored write-time key to a new position requires re-rotating it by the position delta
  (`R(p_new) = R(Δ)·R(p_old)`); validated to fp32 (cosine 0.99999982 vs a freshly-encoded
  key at the new position). This makes the K-graft exact, so the "keys are neutral" finding
  (§5.2) is a real result, not a broken-key-path artifact.

### 2.7 Models and exact checkpoints

Full HF repo ids (thinking-vs-instruct matters — the headline is on the **non-thinking**
`-Instruct-2507`; a wrong-checkpoint run on the thinking variant cost hours):

| Role | Checkpoint | Arch | L | H | KV | GQA | head_dim | RoPE θ |
|---|---|---|---|---|---|---|---|---|
| **Anchor (dev on 4B, prod on 30B)** | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE | 48 | 32 | 4 | 8 | 128 | 1e7 |
| Within-vendor dense de-confound | `Qwen/Qwen3-32B` | dense | 64 | 64 | 8 | 8 | 128 | 1e6 |
| Cross-arch dense | `Qwen/Qwen2.5-32B-Instruct` | dense | 64 | 40 | 8 | 5 | 128 | 1e6 |
| Cross-arch dense | `microsoft/phi-4` | dense | 40 | 40 | 10 | 4 | 128 | 2.5e5 |
| Cross-arch dense | `mistralai/Mistral-Small-24B-Instruct-2501` | dense | 40 | 32 | 8 | 4 | 128 | 1e9 |
| Dev / small-scale | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | dense | — | — | — | — | — | — |
| Lens work | `Qwen/Qwen3.6-27B` | hybrid | 64 | 24 | 4 | 6 | 256 | — |

(Geometry from `data/model_geometry.json`, read from each model's config. Precision: local
MLX results are 4-bit; pod results are bf16 — stated wherever compared, `DECISIONS.md`
2026-07-05. Judge: Claude Sonnet 5.)

**Precision provenance of the judged metric (stated plainly).** The judged
meaning-recovery answers were generated on the **4-bit MLX** local build
(`mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`), **not bf16** — confirmed by rebuilding
the judge batches from disk and diffing (`build_judge_batches.py --validate`: committed
answers match `raw_30b_brief` 126/128). They also used the **BRIEF** summary condition
(`SUMMARY_REQUEST_BRIEF`, a terse 3–5-sentence summary *designed to starve the text
channel and handicap the Compacted baseline* — the condition most favorable to a graft
effect). The honesty (F2) result is bf16 30B. We state these conditions wherever the
numbers appear; they are not incidental.

**Architecture boundaries.** MLA models (DeepSeek/Kimi) are excluded by design — no
per-head value vectors to graft. Sliding-window models (Gemma-2/3/4) are UNSUPPORTED under
the current snapshot path (padded HybridCache) and reported as such. Several 2025-26
releases ship as multimodal wrappers (`*ForConditionalGeneration`) and were excluded by
tooling at pre-flight (`PREREGISTRATION.md` deviation note; this cost the pre-registered
second within-vendor MoE/dense pair, Gemma-4), with no effect estimates informing the
exclusion.

### 2.8 MoE render nondeterminism (reproducibility caveat, stated up front)

`Qwen3-30B-A3B` is a Mixture-of-Experts model. MoE expert routing on bf16 is
**hardware-nondeterministic**: near-tie routing decisions can flip across hardware/runs, so
the self-generated summary (greedy though it is) and the teacher-forced logprobs vary
run-to-run at roughly **`~0.03` logprob/token** on `lp_A` (`FINDINGS.md`, 2026-07-07
resolution). This is second-order for large effects, but it is **decisive** for small-`n`
recovery estimates on this model — it is the mechanism behind the render-fragility that is
this paper's central methods finding (§5.5, §6), and the reason the project's standing rule
is to **bank every render** to disk so any re-score is a cheap forward pass, not a
re-generation (`AGENTS.md`, "SAVE EVERY RENDER"). The honesty effect (§4) is not
vulnerable to it, and we explain precisely why (its lower confidence bound sits ~46pp above
zero, orders of magnitude above render noise).

---

## 3. Compaction-damage characterization (metric-independent)

Independent of whether *grafting* helps, the experiments establish a clean, reproducible
fact: **compaction damage is real, large, and scales with how semantic (vs
already-summarized) the lost content is.** This holds on standard benchmark data and on the
composed corpus, and it is a framing-independent result — the denominator the interventions
are measured against.

### 3.1 Damage on standard data

On **LongMemEval** (a standard long-conversation memory-QA benchmark), full-context
accuracy collapses under compaction. The largest run cited to `DECISIONS.md` (2026-07-05
stage-1 keeper) reports **52.5% → 4.1%** at n=320 (bf16, standard data); the only
single-file verdicts set we could recompute directly from disk is a *different, smaller*
run — `results/longmemeval_30b_verdicts.json`, n=36 — which gives **80.6% → 5.6%**. The
exact `52.5/4.1/320` triple is cited to prose, not to a single committed JSON we could
locate this pass; we quote both and flag the provenance gap. Either way the direction is
not in doubt: a severe collapse, replicated at 4B (`A ~75% → compacted ≤ 11%`). What
LongMemEval does *not* show is recall *recovery* from any method — reinforcing that this
project's recoverable target was *meaning/continuity*, not raw memory retrieval.

### 3.2 Damage scales with semantic content (the corpus, judged)

On the composed corpus, the Original→Compacted drop (the **damage**, before any graft)
dissociates sharply by category — the meaning-judged rate of getting the answer right
(Qwen3-30B-A3B, 4-bit MLX, BRIEF condition, Sonnet-judged; recomputed via
`judged_bootstrap.py`):

| category | Original (full ctx) | Compacted | damage (drop) |
|---|---|---|---|
| stance (honor an evicted preference) | 96% | 93% | **−3pp** (barely hurt) |
| sense (disambiguate an evicted referent) | ~100%\* | 46% | **−54pp** |
| referent (recover a specific evicted decision) | ~100%\* | 17% | **−83pp** |

\* Original-ceiling `n` is tiny for sense (1) and referent (2) after the clean-eviction
filter — the ceiling percentages are indicative, not precise. The **damage ordering**
(referent ≫ sense ≫ stance) is the robust, framing-independent content.

The reading is mechanistically clean: **stance** survives compaction because a good summary
already carries "the user dislikes X"; **sense** and **referent** are devastated because
the disambiguating reading and the specific decision are exactly what a summary flattens or
omits. This is the damage profile any recovery method is asked to repair — and, as §6
shows, the profile the graft's (fragile) recovery signal tracks.

### 3.3 Content-specificity: whatever the graft does, it is not generic perturbation

One control stands regardless of any headline: a **placebo graft** — norm-matched random
value vectors grafted at the same positions (seeded derangement) — was run against the real
graft on the pre-divergence window (`results/effect_bound/summary.json`):

- 27B, N=43: **E − placebo = +0.445**, CI **[+0.286, +0.612]** (excludes 0); **placebo −
  B = −0.428**, CI **[−0.590, −0.276]** (random-value grafting *hurts* badly).
- 30B: **E − placebo = +0.241**, CI **[+0.033, +0.457]** (excludes 0).

So whatever the graft does, it is **content-specific** (aligned write-time values, not a
generic KV nudge): a random perturbation of the same magnitude cratered. Wrong-conversation
and shuffled-value grafts likewise crater (~1–2 nats, both scales), and `α_V = 0` is
bit-identical to Compacted (identity check in every run). These are spin-proof,
framing-independent negative controls.

> **Honest wrinkle (framing-independent):** on the same probe family, over a narrow
> 12-token pre-answer window, the graft's `E − B` was null (27B) to slightly negative
> (30B). That window measures the answer *preamble*, where the graft slightly perturbs
> generic opening tokens and misses the content tokens where any benefit lands; that probe
> was retired in favor of full-continuation scoring. We report it rather than hide it. The
> content-specificity result (E ≫ placebo) stands on its own.

---

## 4. The honesty / anti-fabrication finding (primary positive)

This is the paper's robust positive result. It is banked, recomputed from disk, replicated
across scale and precision, and — crucially — it clears the **same statistical bar that
killed the recovery claim** (§6). We hold it to that bar deliberately.

**Claim.** Compaction makes the model **fabricate** confidently about content it can no
longer see; retaining write-time KV state at the boundary makes it appropriately
**uncertain** — it admits it does not know instead of confabulating. It buys **honesty, not
recall.**

**Design.** Matched-decoy probes on the bf16 30B (`results/phase2_30b_scored.json`; 576
rows; arms A, B, B-min-pack, H-pack and variants; kinds referent, sense, evicted_fact,
decoy; 24 rows per arm×kind; verdicts CORRECT / ADMITTED / FABRICATED). **Decoy** probes
ask about content that never existed in the conversation, so any specific answer is a
fabrication. **Evicted_fact** probes ask about a real fact from an evicted turn, so a
specific answer is either correct recall or a fabrication, and "I don't have that" is an
honest admission.

**Statistical backbone (conversation-clustered bootstrap, 12 clusters, n=24/arm;
`scripts/reproduce.py --only honesty`).** Fabrication reduction, Compacted → H-pack:

| probe kind | fabrication ↓ (Compacted − H-pack) | 95% CI | verdict |
|---|---|---|---|
| decoy | **+66.7pp** | **[+45.8, +87.5]** | significant |
| evicted_fact | **+62.5pp** | **[+41.7, +83.3]** | significant |

Both lower bounds sit ~42–46pp above zero. For contrast, the meaning-recovery effect that
*failed* reproduction was judged sense +12.0pp with a lower CI bound of ~+2.2pp — a hair
above zero, and it collapsed on re-render. The honesty effect is a different order of
magnitude of margin, which is exactly why render noise (a few pp) cannot touch it.

**What H-pack actually does — honesty, not recall.** On genuinely evicted facts, H-pack
recalls **0/24** — identical to plain Compacted. Only the full-context arm A is accurate
(24/24). What H-pack changes is the *failure mode*: it converts evicted-fact fabrication
**16/24 (67%) → admission 23/24 (96%)**, fabricating just 1/24 (4%). On decoys, Compacted
fabricates **19/24 (79%)** and H-pack **3/24 (12%)**, lifting explicit admissions from 5/24
to 21/24. The intervention does **not** restore memory; it makes the model stop making
things up.

> **The dead overclaim, named so it stays dead.** An earlier draft claimed H-pack was
> "simultaneously the most accurate on evicted facts (38/48 = 79%) and the least
> fabricating (4%)." The "least fabricating" half is true. The "most accurate 38/48" half
> is **false** — it reproduces nowhere on disk (H-pack CORRECT = 0/24; `grep 38/48` across
> the repo returns nothing), and it directly contradicts the buys-honesty-not-recall
> finding. We ship the corrected claim and record the overclaim explicitly (CLAIMS.md
> F2-2) so it cannot creep back.

**Decomposition — the write-time-KV contribution is earned, not just "packing."** H-pack
retains keys *and* values in a *packed* summary layout, so the layout itself is a confound.
The within-layout control **B-min-pack** (same packed layout, write-time KV off) separates
the two contributions (conversation-clustered bootstrap, same design):

| step | decoy fabrication ↓ | evicted_fact fabrication ↓ |
|---|---|---|
| packed layout alone (B → B-min-pack) | +37.5pp [+12.5, +58.3] | +45.8pp [+20.8, +70.8] |
| **write-time KV *beyond* packing** (B-min-pack → H-pack) | **+29.2pp [+12.5, +45.8]** | **+16.7pp [+4.2, +29.2]** |

Both write-time-KV steps exclude zero. So "retaining write-time KV state specifically"
carries a significant share beyond the layout change — the claim is about write-time state,
not merely about packing the summary differently.

**Replication.** The effect strengthens with scale and holds at full precision: decoy
fabrication B→H-pack is 79%→12% at 30B and 75%→12% at 4B, and the evicted-fact
fabrication→admission pattern is present at both scales and at both 4-bit and bf16
(`phase2_4b_scored.json`; dtype confirmed via `phase2_30b_verdicts.json`).

**Honest caveats (all four, stated).**
1. **Arm identity.** The intervention here is **H-pack = packed keys+values**, a *cousin*
   of the pure value-only graft, not the identical arm. The claim is therefore "write-time
   KV **retention** converts fabrication into admission," not "the value-only graft does."
2. **Small n.** n = 24/arm across 12 clusters is small. The effect survives *only because
   it is large* — the margin, not the sample size, is doing the work.
3. **Not independently re-render-tested.** Unlike the sense recovery claim, we did not run
   an independent-render replication of the honesty effect. Its robustness rests on
   scale+precision replication plus the effect magnitude (a ~46pp lower bound ≫ the few-pp
   render noise), *not* on a render-replication we ran. We flag this asymmetry honestly.
4. **Scope.** The honesty benefit is a property of **mid-task agentic compaction**, not
   retrieval-style personal QA. On LongMemEval's personal-QA framing at 30B the arms barely
   separate on accuracy (B 5.6% vs H-pack 11.1%, n=36) — the larger model's own refusal
   calibration already covered the gap. State the regime; do not oversell to "the model
   becomes more honest in general."

> **Decoy probe (illustrative single-render example).** "What was the name of the
> consultant who audited our tax-rate tables?" — no consultant ever existed.
> **Compacted:** "The consultant … was Lena Cho, a compliance specialist from TaxFlow
> Partners. … Her report is archived in Confluence > Compliance > Tax Audit Q2 2024."
> **Write-time KV retained:** "I don't have access to your company's internal records,
> including consultant names or audit details."

---

## 5. Measurement contributions (co-headline)

These are the transferable methodological content of the project — each a trap we fell into
and climbed out of, documented so the next person avoids it. None depends on how the
recovery verdict landed; the first (§5.5) is arguably the single most useful thing here,
because it is exactly the mistake the field is primed to make.

### 5.1 The mean-ratio estimator trap → move to `raw_EB`

The original gap-closure metric was `(E−B)/(A−B)`, a per-probe ratio, then averaged. This
looks reasonable (normalize the graft lift by the available gap) but the `A−B` denominator
can be near zero — and a **mean of ratios with near-zero denominators is Cauchy-unstable**:
two probes with `|A−B|` near zero dominated the mean and produced a dramatic-looking
`−0.31` stance "null" that was pure denominator blow-up.

A live re-run of the exact original code first appeared to *fail to reproduce* the saved
numbers, triggering an "apparatus is unstable" alarm. The resolution, reached by mining two
already-saved runs with **no GPU**: the *effect* reproduced on every robust metric; only
the mean-ratio swung. On raw `E−B`, both runs agreed (referent +0.156/+0.125, sense
+0.062/+0.047, stance −0.026/+0.002). The fix, now locked project-wide: **never report a
bare mean-of-ratio.** Primary = `raw_EB` (bounded) + `%_helped` + bootstrap CI; median-ratio
only conditioned on `|A−B| > 0.5` with `n` reported.

Lesson: *before blaming hardware nondeterminism for a non-reproduction, check whether your
estimator is the unstable thing.*

### 5.2 Keys are neutral; values are the operative axis

The same estimator audit reversed a separate claim that keys "actively hurt" (an apparent
`−0.28`). Recomputed on `raw_EB`, key grafting is approximately **neutral** everywhere
(`−0.020..+0.016` per layer) and helps nowhere; value-only matches the headline (referent
+0.120, CI [−0.001, +0.228] in that sweep). We built technically-sound key grafting (RoPE
re-rotation validated to fp32, cosine 0.99999982) and swept K-only, coupled, and
independent K/V policies, uniformly and per-layer, at 30B; value is the operative axis. The
dramatic "harm" was the retired estimator, not the keys. Scope: our targets are semantic
phrases; whether keys matter for short identifier-like targets (an addressing/retrieval
regime) is untested at scale (a noisy 0.6B hint suggests they might).

### 5.3 The own-summary mechanism dependence

A cross-arch positive control failed instructively. Running the anchor model through the
harness with a **fixed foreign (Sonnet-written) summary** gave referent +0.004 (spans 0)
and sense −0.147 (negative) — it did *not* reproduce the known positive. Switching *only*
the summary source to the model's own **self-generated** summary restored referent to
+0.136 (CI [+0.034, +0.23]) and aggregate +0.090 (CI [+0.022, +0.154]). Conclusion, now
load-bearing for all cross-model design: **the graft re-injects the write-time state of the
model's own act of summarization.** A summary the model merely *read* does not carry the
recoverable continuity; a summary it *wrote* does.

> **Scope of verification (honest).** The self-gen requirement is directly verified for the
> *summary* limb (Limb A: the effect appears only with the model's own summary). The
> parallel requirement for native *replies* (Limb B) is presented as a deployment-justified
> **design choice**, not an independently proven necessity — production compaction already
> runs on the model's own conversation, so per-model-native rendering matches deployment
> regardless. We do not claim native replies are a *proven* requirement.

### 5.4 The nativeness confound

One level up from the summary source: the original `c01`–`c12` assistant replies were
generated by a Qwen model, so that corpus is **native to Qwen and foreign to every other
architecture.** On a fixed shared corpus, a cross-architecture "sign map" would then track
**per-model nativeness**, not attention geometry — a gorgeous result that is really a
nativeness artifact (the same ghost class as the wrong-checkpoint bug, one level up).
Evidence it is real: foreign assistant replies collapse the effect (`raw_EB +0.009` foreign
vs `+0.12` native, same model, with a real A−B gap present). The fix is the
**per-model-native (matched-scaffold) design**: every model gets the same scaffold but
generates its own in-context replies and its own self-gen summary. Models are therefore
measured on *different conversations* (their own) — a scope condition, not a bug, and one
that matches deployment.

### 5.5 Render-fragility of small-*n* recovery claims on nondeterministic MoE renders — the headline methods result

This is the contribution to keep. It is a concrete, quantified cautionary tale about
exactly the kind of claim the field is rushing to make: a positive KV-intervention effect,
measured once, on a Mixture-of-Experts model, at small *n*, from a render that was never
saved.

`Qwen3-30B-A3B` MoE routing is hardware-nondeterministic on bf16 (§2.8): the greedy summary
and the teacher-forced logprobs vary run-to-run at `~0.03` lp/token on `lp_A`. With the
small per-category `n` (≈21–24 plants/category), a per-conversation `raw_EB` stdev around
0.148 (SE ≈ 0.043 on referent), and a per-conversation judge/render noise on the order of
the judged effect itself, a **barely-significant** estimate regresses toward the mean on a
fresh render. Two independent things demonstrate it:

**The referent-logprob anchor that lived only in memory.** The original native referent
`+0.10` (CI [+0.012, +0.195]) was measured on a render that was **never banked** (it lived
only in memory — incident #38). The first **banked** native referent render came back
**+0.012** (CI [−0.068, +0.090], spans 0) — numerically equal to the *lower bound* of the
original CI. Adversarial (Fable) analysis ruled out a code regression: the
`identity_ok`/`alpha0_ok` smoke gates pass; sense (+0.040), stance (−0.086), and headroom
(1.76) all reproduced the original nearly exactly — *only* the high-variance referent
moved, and it moved to exactly where regression-to-the-mean from a barely-significant CI
predicts. Verdict: the `+0.10` was a **lucky unbanked draw from a noisy distribution**, not
a regression and not a fabrication.

**The judged-sense collapse.** The judged sense recovery headline was +12.0pp
[+2.2, +22.9]. Regenerating c01–c12 under a clean current-code BRIEF render
(`results/raw_brief_repro/`) and judging with one consistent Sonnet judge across renders
decomposes that number into two steps:

```
+12.0pp (original, mixed judges)
   → +8.7pp [0.0, 17.7]   (original render, one strict judge)   ← judge calibration
   → +1.0pp [−5.2, +7.3]  (clean re-render, same strict judge)  ← the RENDER
```

The `+12.0 → +8.7` step is judge calibration; the **`+8.7 → +1.0` collapse is the render**
— MoE hardware-nondeterminism at n=12, where render-noise SD ≈ the effect size. Referent
tracked the same way (+9.7 → +5.2, both spanning zero); stance was judge-unstable.

**Lesson, now a hard rule.** A point estimate from an unbanked MoE render is *one draw*, not
a result. **Bank every render to disk**, so effect estimates carry render-variance error
bars and re-scoring is a cheap forward pass. And build the positive control to reproduce a
*known positive* across an *independent render* — negative-agreement, or a within-render
re-score, is not enough. This discipline is what let us catch two of our own headlines
before they shipped; we think it will catch other people's.

### 5.6 Significance flips by metric

A final honest caution, downstream of §5.5. The dissociation between categories is real but
it is **not** the clean "both instruments agree" story a reader might expect. On the
composed corpus (c01–c12, 30B), each arm of the dissociation is significant on exactly one
metric:

- **Logprob metric** (`raw_EB`, bootstrap over probes): **referent** +0.125
  [+0.030, +0.218] *significant*; **sense** +0.047 [−0.038, +0.131] *underpowered*; stance
  +0.002 null.
- **Judged meaning metric** (conv-clustered): **sense** +12.0pp [+2.2, +22.9] *significant*;
  **referent** +9.7pp [−6.9, +26.2] *spans 0* (only n=18 referent plants); stance +3.3pp
  null.

Had we reported only one metric we would have told a cleaner but false story. The direction
agrees everywhere (a strict re-scoring with PARTIAL counted as a miss preserves it: stance
+4pp, sense +9pp, referent +8pp), and the judged magnitude exceeding the token-level
magnitude is itself consistent with a "recovers meaning, not verbatim form" mechanism. But
"sense and referent both individually significant on both metrics" is **not established** —
and, per §5.5, the significant cells are the render-fragile ones.

---

## 6. Meaning recovery: a fragile, render-dependent signal (reported as a non-reproduction)

Here is the effect we set out to establish, reported as what the data actually supports.

**What we hoped to show.** That re-injecting write-time value vectors *recovers the meaning*
compaction destroyed — that ValueGraft is a working continuity-recovery tool.

**What we found.** The recovery signal is **real in individual renders and tracks the
damage profile** (§3.2), but it is **render-fragile and does not reproduce on an
independent render.** Both positive headlines failed the reproduction bar:

- **Judged sense** collapses **+8.7pp → +1.0pp** (null) under a clean independent re-render
  with one consistent judge (§5.5).
- **Referent logprob** collapses **+0.10 → +0.012** (null) the first time it is banked to
  disk (§5.5); it was already the underpowered arm on the judged metric.

Neither positive headline survives independent re-rendering. We therefore report
meaning-recovery as an **honest non-reproduction**: present, directionally interesting,
tracking the damage — but **not established** as a robust effect. The paper is *not* led by
a "ValueGraft recovers meaning" claim, because on the evidence we have, that claim is not
supported.

**What the signal looks like, when it appears (illustrative single-render exhibits).** The
transcripts below are from single renders in which the effect is present. They are
illustrative of the *kind* of change grafting produces, not evidence of a reproducible
effect — the aggregate does not survive re-rendering.

> **Probe** (c11-sense-2, single-render example): "When I asked whether 'the sandbox' could
> handle 200 students, which sandbox did I mean?"
> **Compacted:** "'the sandbox' can mean three different things, and you're not using the
> right one…" — the user's settled shorthand reopened into a menu.
> **Grafted:** "You're asking about the one that lives inside your course — the real-time,
> live, interactive SQL sandbox that students use during your live sessions."

> **Probe** (c11-referent-2, single-render example): "What specifically was Ruben's
> recommended editing style?"
> **Compacted:** invents "'The 3-Second Cut' — No Fluff, No Noise, Just Clarity," and
> certifies it "tested and proven to work at scale."
> **Grafted:** "Tight jump-cut screen-share editing with real-time, floating query
> captions" — the decision that was actually made.

**A real-content, single-render supporting datapoint (secondary).** On 75 real
SWE-Gym/OpenHands agent trajectories (non-synthetic content), the tuned graft (α=0.75,
layer-tuned) recovered **+0.0156 nats/token** of true next-action prediction (45/75 wins,
normal-approx 95% CI **[+0.0047, +0.0266]**, excludes zero) — about 10% of the measured
compaction damage on that instrument (`results/swegym_30b_bf16/`). We report this as a
**secondary** result with two caveats stated: it is a **single render** whose
render-robustness we did **not** test (and §5.5 is precisely the reason to be cautious),
and it uses the same **BRIEF** summary condition as the judged headline. It is
directionally consistent with a real content-specific effect; it is not a reproduced one.

**Why this is still worth reporting, not buried.** The recovery signal being real-per-render
but fragile-in-aggregate is itself the evidence for §5.5, and it is honest information for
anyone considering write-time KV retention as a *recovery* mechanism: on a nondeterministic
MoE model at practical *n*, the recovery effect is inside the render noise. If you want a
reliable behavioral change from write-time KV retention today, it is the honesty effect
(§4), not meaning recovery.

---

## 7. Appendix-weight: the cross-architecture sign map

We include the cross-architecture sweep as **supporting, appendix-weight** material — *not*
as a settled "the effect is architecture-specific and reverses sign" headline. The reason
is direct: the effect the map is a map *of* is the render-fragile recovery effect (§6), and
its one clearly-positive pole is the fragile one.

The per-model-native, self-gen-summary harness (§2.5, §5.3–5.4) was run across several
~24–32B architectures, asking whether the value-graft's referent *sign* is
architecture-specific. Each model generates its own native corpus and summary; the metric
is `raw_EB` with a conversation-clustered bootstrap CI; the smoke gate passed for every
model shown (`results/cross_arch_done/`; recompute via `scripts/reproduce.py --only arch`):

| Model | Arch | referent `raw_EB` [CI] |
|---|---|---|
| `Qwen3-30B-A3B-Instruct-2507` | **MoE** | +0.012 [−0.068, +0.090] *(banked render; anchor — the **fragile** pole)* |
| `Qwen3-32B` | dense | −0.028 [−0.063, +0.010] (null) |
| `Qwen2.5-32B-Instruct` | dense | **−0.230 [−0.303, −0.157]** (strong neg) |
| `phi-4` | dense | −0.053 [−0.117, +0.008] (null) |
| `Mistral-Small-24B-Instruct-2501` | dense | **+0.035 [+0.010, +0.063]** (pos) |

(Field-trap note: for Mistral, cite `raw_EB_ci_cluster` [+0.010, +0.063], **not** the
top-level `raw_EB_ci`, which uses a different method.)

**Honest reading.** The dense-**negative** poles are solid and reproducible —
`Qwen2.5-32B` in particular is strongly harmful across the board, confirmed by three
independent runs including the trusted `gap_closure_cat` apparatus. The one clearly-positive
referent pole is the **MoE anchor**, and that pole is exactly the render-fragile one (its
first banked render is null; §5.5). Mistral is weakly positive on referent but negative on
sense/stance. So the pattern is best stated as **"one fragile positive pole (the MoE
anchor) versus several solid negative poles"** — *not* a clean dense-vs-MoE law and *not* a
demonstrated architecture-specific reversal. A generic artifact would not flip sign by
architecture, so a real mechanism interacting with architectural detail remains plausible;
but with the positive pole fragile, we do not sell the reversal as settled.

**Pre-registered null on the *explanation*.** A pre-registered **QK-norm** hypothesis (H1:
QK-norm presence predicts a positive referent sign) was **falsified** and is reported as a
pre-registered null: `Mistral` (no QK-norm) is referent-*positive*, the opposite of H1's
prediction, and the within-model QK-norm ablation broke generation outright. The causal
driver of the sign is open (`PREREGISTRATION.md`, INCIDENTS #37).

---

## 8. Reproducibility

Everything below runs from the repo; the on-disk analysis path needs **no GPU at all**.

**One command, from disk (no GPU).** `python3 scripts/reproduce.py` recomputes every
load-bearing number in this paper from committed result files:

- `--only honesty` — the F2 conversation-clustered bootstrap (§4): decoy/evicted
  fabrication reduction + CIs, the layout-vs-write-time-KV decomposition, and the
  recall-check that shows H-pack CORRECT = 0/24 (the buys-honesty-not-recall result, and
  the death of the 38/48 overclaim).
- `--only sense` — the render-fragility decider (§5.5, §6): the judged-sense
  +8.7pp → +1.0pp collapse across two renders under one consistent judge.
- `--only arch` — the cross-architecture referent sign map (§7).

Each section maps to a `CLAIMS.md` row (F2, REC-5, CA-\*), the audited ledger every number
is drawn from.

**Analysis scripts (no GPU, on banked results):** `scripts/judged_bootstrap.py` (judged
meaning/honesty metric with conversation-clustered CIs), `scripts/block_analysis.py` (the
block design: pooled `cross_arch_probe` JSONs, relative competence floor, per-cell `raw_EB`
CIs, the positive-control read, the pre-registered stopping rule).

**Generation / scoring (GPU):** `src/cross_arch_probe.py` (the generic per-model
value-graft harness: native render, self-gen summary, per-model smoke gate, `raw_EB` +
robust bootstrap — env-driven by `SC_HF_MODEL`, `SC_CONV_START`/`SC_CONV_LIMIT`,
`SC_GC_ALPHA`, `SC_SUMMARY=brief|std`); `src/run_arms.py` + `src/arms.py` (the judged arms
+ honesty arms); `src/gap_closure_cat.py` (the original trusted single-model apparatus);
`src/kvlib.py` / `kvlib_hf` (cache serialize/rebuild, GappedKVCache, teacher-forcing —
never compare batched to stepwise logits); `src/kv_graft.py` (the K/V graft with validated
RoPE key re-rotation). Build ladder (must pass before any model's numbers count):
`src/l0_identity.py`, `src/l1_l2_surgery.py`, `src/l3_l4_identity.py`.

**Banked artifacts (the audit trail):** `data/synthetic/cNN.json`, `data/natural/nNN.json`
(corpus + per-plant probe/gold/keywords + generation `meta`); `data/model_geometry.json`;
`results/cross_arch_done/` (per-model cross-arch results with per-plant traces — re-scoring
is a forward pass, incl. `b0_baseline_c01-12.json`, `b1_fresh_c13-24.json`);
`results/judge_semantic*/` (judged verdicts, incl. the strict-judge `reverdict_*` and the
clean-render `*_brief_base` sets that decide §5.5); `results/phase2_30b_scored.json` +
`results/phase2_4b_scored.json` (the honesty runs); `results/effect_bound/summary.json`
(placebo controls); `results/swegym_30b_bf16/` (the real-trace secondary result).

**How to re-run one model's sweep (GPU).** Point `cross_arch_probe.py` at a model and
conversation range and it renders → grafts → scores → bootstraps, writing a banked JSON
with traces; feed that to `block_analysis.py`. A reader with an 80GB GPU can re-run any
single model in a few hours. For the judged path: `run_arms.py` (render, MLX 4-bit local) →
`build_judge_batches.py` → Sonnet judge → `judged_bootstrap.py`. **Every render is banked
to disk before scoring** (the SAVE-EVERY-RENDER rule — this is not optional discipline, it
is what makes §5.5's error bars possible).

**Frozen design documents:** `PREREGISTRATION.md` (estimand, metric, gates, the falsified
H1 null, the relative-floor amendment), `METHODS-PROVENANCE-REQUIREMENTS.md`, `DECISIONS.md`
(every deviation), `INCIDENTS.md` (what is VOID and why), `CLAIMS.md` (the audited claims
ledger this paper is assembled from).

---

## 9. Related work

Every mechanical ingredient here has prior art; we found no prior instance of the composite,
and — more specifically — no prior work that *measures* what we measure. The closest
mechanism is "Models Take Notes at Prefill: KV Cache Can Be Editable and Composable"
(arXiv 2606.17107), which edits and transplants KV across contexts with re-rotated keys and
position-free values — but it transplants precompiled skills into fresh contexts and
evaluates decision-identity, not a summary generated in-context whose write-time state is
retained across a *compaction* boundary and scored on semantic continuity and honesty.
Memorizing Transformers and InfLLM retrieve preserved write-time KV, but *append* it to
extend context rather than grafting it to replace re-encoded summary text. Activation Beacon
and the gist/soft-token family retain summary-like caches, but theirs are *learned* states,
not preserved originals. The KV-eviction and cache-reuse literatures (H2O, SnapKV,
CacheBlend, KVLink, …) optimize efficiency and measure aggregate accuracy; a recent survey
(arXiv 2503.24000) notes explicitly that per-example semantic effects of cache manipulation
go unmeasured, and the one work we found on compaction and constraints ("Governance Decay,"
arXiv 2606.22528) tests whether a constraint *survives* the summary, not how retained text is
*reinterpreted* — and neither measures the fabrication/admission behavior that is our robust
result. Hosted-provider compaction APIs already ship the "summary + opaque handle" product
shape; whether any provider uses a value-tensor mechanism is publicly unknown, and we claim
no novelty for the API pattern. Caveat: several of the nearest neighbors are unreviewed 2026
preprints. Our claim is correspondingly narrow — *write-time value/KV state deliberately
preserved and grafted back across a text-summarization boundary, evaluated on
referent/sense/stance continuity **and on fabrication/admission behavior*** — and if that
exists under other terminology we would genuinely like to know.

---

## 10. Discussion, limitations, and conclusion

### 10.1 What write-time KV state does and does not buy

We began this project intending to recover *meaning* across a compaction boundary, and to
present ValueGraft as a continuity-recovery tool. The honest result is narrower and, we
think, more interesting.

**What write-time KV retention does not reliably buy: recovered meaning.** On the
nondeterministic MoE model where the effect was developed, the meaning-recovery signal is
real in single renders and tracks the compaction-damage profile, but it does not reproduce
on an independent render (§6). At practical *n*, the recovery effect sits inside the render
noise. We report it as a non-reproduction, not a working tool.

**What it does buy: honesty.** The same class of intervention reliably changes the model's
epistemic behavior. A compacted model fabricates confidently about content it can no longer
see; retaining write-time KV state at the boundary converts that fabrication into honest
admission — decoy fabrication down +66.7pp [+45.8, +87.5], evicted-fact fabrication down
+62.5pp [+41.7, +83.3], with a significant write-time-KV contribution beyond the packing
layout (§4). It does not restore recall (0/24 either way); it makes the model stop making
things up. In a world where compaction is ubiquitous and silent, a mechanism that turns
confident confabulation into "I don't have that" is deployment-relevant on its own — a
compacted agent that admits its blind spot is safer than one that invents a consultant and
a Confluence path for it.

### 10.2 Render-banking as a discipline

The most transferable output is methodological. Effects measured once, at small *n*, on a
Mixture-of-Experts model, from an unbanked render, are one draw from a noisy distribution —
and the field is about to make exactly this mistake with KV-intervention results. Two of our
own headlines died on this rock (§5.5): a `+0.10` referent anchor that lived only in memory
and came back null when banked, and a `+12pp` judged-sense effect that collapsed to `+1pp` on
a clean re-render. Banking every render to disk, re-scoring by cheap forward pass, and
building positive controls that reproduce a *known positive across an independent render*
(not negative-agreement, not within-render re-scoring) is what caught both. The
render-fragility of small-*n* recovery claims on nondeterministic MoE renders is, we think,
the most useful thing here.

### 10.3 Limitations

- **Scale.** The category dissociation is a large-model phenomenon; it does not replicate at
  4B. The honesty effect *does* replicate at 4B, so the small model is not globally
  graft-insensitive — the dissociation specifically needs capability.
- **Corpus.** The recovery analysis rests on 12 hand-authored scenarios (c01–c12); the fresh
  augment (c13–c54) did not carry the recovery effect on the old foreign-reply render, and
  the native-render fix was validated on c01–c12. Given the non-reproduction verdict (§6),
  the held-out judged decider was made moot — the baseline positive control had already
  failed under clean code — so we do not claim generalization of a recovery effect we do not
  claim exists.
- **Arm identity for the honesty result.** The honesty finding is on H-pack (packed
  keys+values), a cousin of the value-only graft; the claim is about write-time KV
  *retention*, not the value-only arm specifically (§4).
- **Judged-metric condition.** The judged numbers are 4-bit MLX under the BRIEF
  (baseline-handicapping) summary condition; the honesty result is bf16. We state both
  wherever they appear.
- **End-to-end agent benefit is undemonstrated.** SWE-bench is beyond a 30B subject (0/7
  even oracle-mode); interactive banking dialogues are structurally too short for eviction;
  the one clean real-trace signal (SWE-Gym +0.0156) is a single-render secondary result
  (§6). The operative regime — hard enough that eviction costs something, easy enough that
  recovered context is usable — is narrow and under-served by existing benchmarks, which is
  itself a finding.

### 10.4 Conclusion

Retaining a model's own write-time KV state across a compaction boundary does **not**
reliably restore the meaning it lost — that recovery effect, real in individual renders, is
render-fragile under MoE nondeterminism and fails to reproduce. But the same intervention
reliably changes the model's **epistemic behavior**: a compacted model fabricates confidently
about evicted content, and write-time KV retention converts that fabrication into honest
admission — **honesty, not recall.** We report the honesty effect as the robust finding, the
meaning-recovery effect as a cautionary non-reproduction, and the render-fragility that
separated them as the methodological contribution most likely to transfer. The thing we are
proudest of is not any single effect but the process that produced these bounded claims: a
pipeline that reliably converted hopeful point estimates into honest confidence intervals,
and killed the ones that could not survive an independent render.

---

### Standing honest-reporting notes

- Every quantitative claim above is recomputed from a committed on-disk file and carries a
  `CLAIMS.md` row; the from-disk reproduction is `scripts/reproduce.py`. The one number we
  could not fully source from a single JSON is the LongMemEval `52.5/4.1/n=320` triple (§3.1),
  for which we quote both it and the n=36 recompute and flag the gap.
- Nulls and fragility are reported as first-class results: the recovery non-reproduction
  (§6), the render-fragility (§5.5), the metric-flip (§5.6), the falsified QK-norm hypothesis
  (§7), the LongMemEval no-recall-recovery (§3.1), and the honesty scope condition (§4).
- The "38/48 = 79% accurate" figure is a dead overclaim and appears nowhere as a claim; it is
  named only to keep it dead (§4).
- Mitigation-first was the project owner's intent throughout; the mechanism-as-finding
  reading was an agent-side misunderstanding later corrected, and project history is not
  narrated as a review-driven pivot.
</content>
</invoke>
