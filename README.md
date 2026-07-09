# ValueGraft

### Value grafting: re-injecting write-time KV state where a conversation was compacted — a working report

*By Anthropic Claude Fable 5 and OpenAI GPT 5.5, with guidance from Jeremy Banks and
assistance from Anthropic Claude Opus 4.8, Anthropic Claude Sonnet 5, and Google Gemini
Pro 3.1.*

> **STATUS: WORKING DRAFT (2026-07-09) — quick synthesis pass.** Data collection is still
> concluding: a pre-registered held-out reproduction (the "block" experiment, §7) and two
> additional architecture runs were in flight when this was written, and a robust-metric
> re-audit of some secondary tables is pending. Every number below is a banked, observed
> result, but the set is incomplete and one validity question (§7) is open. A full
> revision will follow.

**TL;DR.** When a long conversation is compacted — older turns replaced by a text summary
— the summary tokens lose their original activations: the model re-reads its own summary
as a stranger would. We test a small, deployable mitigation we call **value grafting**:
keep the *value vectors* the model computed at write time (for the summary it generated
and for the conversation tail it retains), and blend them back into the freshly-computed
cache at the compaction boundary, leaving keys fresh. On a 30B model this recovers a
measurable slice of lost *meaning* — specifically where compaction did damage
(disambiguating evicted senses and referents), and not where a summary already suffices
(stable preferences). The two arms of that dissociation are each statistically significant
on one of our two metrics but not both — a nuance we report rather than smooth over. The
effect requires the model's *own* summary and its *own* conversation history (both
scope conditions and, we argue, mechanism), it needs a moderate dose (full-strength
grafting can break a task; a guard-validated per-layer configuration fixes it), keys are
neutral (values are the operative axis), and — preliminarily — the sign of the effect is
**architecture-specific and can reverse**: positive on Qwen3-MoE, negative on Qwen2.5-32B
and phi-4, mixed on Mistral-Small-24B. A pre-registered attention-geometry hypothesis
(QK-norm predicts the sign) was falsified and is reported as a null. Whether the headline
effect generalizes beyond the original 12 hand-authored scenarios is the open item the
in-flight held-out reproduction exists to answer.

---

## 1. The question

Every deployed assistant eventually hits its context-window limit, and the standard fix is
compaction: replace the older turns with a model-written summary and keep the recent tail
verbatim. Hosted APIs now ship this as a first-class primitive (OpenAI's Responses
compaction, Anthropic's `compact_20260112` context edit, Gemini's managed-agent
compaction). Compaction is lossy by construction; that loss is our *baseline*, not our
finding.

The question this project asks is narrower and deployment-shaped: **can a small
intervention on the model's cache state reduce the damage that compaction causes**, in a
way that survives negative controls and looks plausibly shippable?

The intervention is motivated by an asymmetry in what compaction throws away. A
transformer reading text computes, per token per layer, a key and a value vector — the KV
cache. When the model *generated* its summary, the summary's value vectors were computed
while the full conversation was still in context; when the model *wrote* its most recent
replies, those tokens' value vectors were likewise computed with the now-evicted middle
still present. After compaction, all of that is discarded and re-encoded from bare text.
The visible words are (partly) the same; the write-time state behind them is gone. In
plain language: **summary tokens lose their original activations after context
compaction.** If some of the conversation's settled, disambiguated *sense* lives in those
write-time activations rather than in any words, a text summary cannot carry it — but
grafting the activations back might.

## 2. The intervention: value grafting

At the compaction boundary we rebuild the context the standard way — system prompt, a
context-note turn containing the model's summary, then the retained tail — and then blend
saved write-time **value** vectors into the freshly-computed cache at two aligned regions:

1. **the summary tokens** — receiving the values computed when the model originally
   *generated* that summary under the full conversation, and
2. **the retained tail tokens** — receiving the values they had when the evicted middle
   was still in context.

Blending is `V ← (1−α)·V_fresh + α·V_write-time` at blend strength α (default **α =
0.75**), applied at all layers and heads unless stated otherwise. **Keys are left fresh**:
the graft re-supplies write-time *content* without altering where the model attends.
Token-position alignment between the write-time and compacted renderings is by
per-region sequence matching (difflib, minimum matched block 8 tokens, attention-sink and
special-token positions excluded).

Canonical arm names, used throughout:

- **Original** — the full conversation, never compacted (ceiling);
- **Compacted** — plain summary compaction (the production baseline);
- **Compacted + value graft (α=…)** — the intervention;
- **Compacted + layer-tuned value graft** — a per-layer-tuned "champion" variant (§9).

Cost, honestly bounded rather than measured: retaining the graft source means storing the
value tensors for the summary and tail regions (a fraction of the conversation's KV
cache) and, in our research harness, one extra prefill to construct the write-time
snapshot. We have not engineered or benchmarked a production implementation.

## 3. How we measure

**Primary metric (judge-free): raw E−B.** For each planted probe we teacher-force a
*shared gold continuation* (a short statement of the correct answer, ~16–18 words,
derived from the planted facts — never generated by any model) and record the per-token
mean logprob under each arm: `lp_A` (Original), `lp_B` (Compacted), `lp_E` (graft). The
statistic is **raw_EB = lp_E − lp_B**, with percentile-bootstrap 95% CIs — clustered on
*conversations* for headline numbers, since plants within a conversation are correlated —
plus %-of-probes-helped. We deliberately do **not** report the mean of the gap-closure
ratio (E−B)/(A−B): with small denominators it is Cauchy-unstable, and an earlier draft of
this work was distorted by exactly that estimator (one apparent "keys actively hurt"
finding, and an apparently dramatic stance number, were artifacts of it; both are
corrected here). Because the gold continuation is shared across arms and models, any
per-model preference for the target's *style* appears in both terms and cancels — the
difference is the load-bearing design choice.

**Secondary metric (meaning, judged).** Sonnet 5 re-judges each arm's actual answer to
each probe for *meaning recovery* (RECOVERED / PARTIAL / MISSED = 1 / 0.5 / 0), blind
across arms. Judged contrasts pool two graft doses (α=0.25 and α=1.0; the logprob metric
uses α=0.75) against Compacted, with conversation-clustered bootstrap CIs (n_boot=20k).

**Gates (pre-registered).** A probe category with per-model *headroom* (lp_A − lp_B) below
0.3 is floored — the summary already preserved that content, so there is nothing to
recover and the category is excluded from sign verdicts rather than counted as harm. A
task-competence floor drops plants the model cannot do even with full context (absolute
−8.0 per-token lp_A; a pre-registered per-model robust-outlier variant, median − 3·MADN,
exists for models whose logprob scale sits lower). Every run must pass machinery checks
before its numbers count: α=0 reproduces Compacted bit-nearly (≤5e-3), an identity
self-graft is a no-op, and the graft demonstrably changes outputs.

## 4. What the data is (provenance summary; full details §14)

- **Scenario scaffolds are authored, not model-generated**: 54 synthetic scenarios
  (c01–c54), each a workplace-style long conversation skeleton with **planted** items —
  the original 12 (c01–c12) authored at project start, 42 more (c13–c54) authored
  2026-07-08 by a mix of Claude/GPT subagents. Each plant has a category — **referent**
  (a specific decision made then evicted), **sense** (which meaning of an ambiguous term
  the conversation settled on), **stance** (a stated preference), plus ruled-out /
  evicted-fact / strong-prior categories — a planted-fact user turn in the middle section
  (later evicted), a tail turn that refers back without restating, probe questions, and
  the gold continuation. A contamination audit verifies planted keywords appear only in
  the evicted middle.
- **Assistant replies are the test model's own.** In the current (v2.1,
  "matched-scaffold, model-filled") design, each evaluated model generates its own
  in-context replies to the shared user turns (greedy, ≤320 tokens per reply), and its
  own summary. This is deliberate and load-bearing: §7 shows the effect *collapses* on
  another model's replies, so per-model-native rendering is both the ecologically correct
  measurement (deployed models only ever compact their own conversations) and a scope
  condition of the finding. The original c01–c12 renders used Qwen3-4B (same family as
  the 30B headline model); the 30B's own native re-render reproduces the headline.
- **The summary is self-generated** by the test model (greedy, ~300–500 words requested),
  because a foreign summary suppresses the effect (§6).
- **Compaction is real, not simulated aggressively**: the original conversations are long
  enough (~8–9K tokens) that the planted middle is evicted by genuine length.
- **Exact checkpoints matter.** The headline model is
  `Qwen/Qwen3-30B-A3B-Instruct-2507` (the non-thinking instruct checkpoint), bf16.
  We lost hours to accidentally running the *thinking* `Qwen3-30B-A3B`, which behaves
  differently; §14 lists every checkpoint.

## 5. The core result: recovery tracks the damage — with a metric-dependent caveat

On the 30B model, over the 12-scenario corpus, meaning-recovery rates by category
(Sonnet-5 judged, lenient scoring):

| category | Original | Compacted | Compacted + value graft | graft − Compacted |
|---|---|---|---|---|
| stance — honor an evicted preference | 96% | 93% | 96% | +3.3pp, CI [−5.4, +12.0] |
| sense — disambiguate an evicted term | ~100%* | 46% | 58% | **+12.0pp, CI [+2.2, +22.9]** |
| referent — recover a specific evicted decision | ~100%* | 17% | 26% | +9.7pp, CI [−6.9, +26.2] |

(*Original-arm ceiling cells are tiny — n=1 and n=2 clean-eviction cases — indicative
only. CIs are conversation-clustered bootstrap over the 12 conversations, 64 plants.)

The same probes on the judge-free logprob metric (raw E−B, α=0.75, bootstrap over 21–24
probes per category):

| category | raw E−B | 95% CI | % probes helped |
|---|---|---|---|
| stance | +0.002 | [−0.036, +0.049] | 38–54% |
| sense | +0.047 | [−0.038, +0.131] | 59–64% |
| referent | **+0.125** | **[+0.030, +0.218]** | 71–81% |

Read the two tables together and the shape is consistent: compaction barely hurts
**stance** (a summary carries "the user dislikes countdown timers" perfectly well), and
the graft correctly adds nothing there — a genuine null on both metrics. Compaction
flattens **sense** and devastates **referent**, and the graft recovers a slice of both.
The effect tracks the damage, which is what a real mechanism should do and what a generic
perturbation would not.

But state the statistics honestly: **significance flips across the two metrics.** On the
judged (meaning) metric, *sense* is significant and referent's CI spans zero (only 18
referent plants — wide). On the logprob metric, *referent* is significant and sense is
suggestive-but-underpowered. Each arm of the dissociation is carried by one instrument.
The direction agrees everywhere (and a strict re-scoring with PARTIAL counted as a miss
preserves the pattern: stance +4pp, sense +9pp, referent +8pp), but a reader should hold
"sense and referent both individually significant on both metrics" as *not yet
established* — more probes per category is the obvious fix, and is queued. The judged
magnitude being much larger than the token-level magnitude is itself consistent with the
mechanism: the graft recovers *meaning* more than exact wording, so a meaning-judge moves
more than a token-probability metric.

What recovery looks like (α=0.25 transcript exhibits; judged table pools doses):

> **Probe** (c11-sense-2): "When I asked whether 'the sandbox' could handle 200 students,
> which sandbox did I mean?"
> **Compacted:** "'the sandbox' can mean three different things, and you're not using the
> right one…" — the user's settled shorthand reopened into a menu.
> **Grafted:** "You're asking about the one that lives inside your course — the
> real-time, live, interactive SQL sandbox that students use during your live sessions."

> **Probe** (c11-referent-2): "What specifically was Ruben's recommended editing style?"
> **Compacted:** invents "'The 3-Second Cut' — No Fluff, No Noise, Just Clarity," and
> certifies it "tested and proven to work at scale."
> **Grafted:** "Tight jump-cut screen-share editing with real-time, floating query
> captions" — the decision that was actually made.

**Negative controls.** Grafting values from the *wrong conversation*, or shuffled, craters
performance (~1–2 nats, both scales tested) — the effect is content- and
alignment-specific. A norm-matched random-value placebo behaves the same way: the true
graft beats the placebo decisively (E−placebo +0.24, CI [+0.03, +0.46] at 30B; +0.45,
CI [+0.29, +0.61] at 27B), while the placebo actively hurts. One honest wrinkle from the
same probe family: over a narrow 12-token pre-answer window the graft's E−B was null (27B)
to slightly negative (30B) — that window measures the answer *preamble*, where the graft
slightly perturbs generic tokens, and misses the content tokens where the benefit lands;
we report it rather than hide it, and score full continuations everywhere else.

## 6. Two scope conditions that turned out to be mechanism

**The graft needs the model's own summary.** Running the identical harness on the 30B
with a *fixed, externally-written* (Sonnet-authored) summary produced referent +0.004
(null) and sense −0.147; switching only the summary source to the model's own self-
generated summary restored referent to +0.136, CI [+0.034, +0.23], 81% helped (aggregate
+0.090, CI [+0.022, +0.154]) — matching the headline. Same model, same code, same
scaffold; the only difference is whose summary sits at the boundary. Our reading: the
graft re-injects the write-time state of the model's own *summarization act*; a summary
the model merely read does not carry that recoverable continuity. Deployment reality
matches the requirement — production compaction already uses the model's own summary.

**The graft needs the model's own conversation.** On an earlier corpus whose assistant
replies had been written by *other* models, the effect collapsed (referent ~+0.009);
re-rendering so the test model generates its own replies from the same scaffold restored
it (referent CI [+0.012, +0.195], mid ~+0.10). Foreign replies mean the write-time values
encode surprise rather than settled sense, and re-injecting surprise recovers nothing.
This is why the design is per-model-native (§4), and it kills the alternative explanation
"any KV re-injection helps": the effect is specific to state the model itself laid down.

## 7. The open validity item: does it generalize past the original 12 scenarios?

We flag this prominently because it is the strongest outstanding threat to the headline.
The +0.10–0.13 referent effect above is established on the original 12 hand-authored
scenarios (c01–c12). The 42 newer scenarios did *not* carry the effect under the old
foreign-reply rendering (~+0.009) — which the nativeness finding (§6) explains — but
"native rendering fixes the fresh scenarios too" had, at the time of writing, been
validated only *on the original 12*. A pre-registered **block experiment** is running as
this draft is written: one apparatus renders c01–c36 natively on the 30B, and the
analysis reads c01–c12 as a positive control (it must reproduce ~+0.10) and c13–c36 as a
fresh held-out test, with a pre-committed stopping rule (extend to c37–c54 if the fresh
referent CI spans zero at n=24). Its first run was lost to an infrastructure timeout
mid-scoring; a re-run (with per-conversation checkpointing) was pending as this draft was
finalized. Until it lands, the honest status of the headline is: **real, replicated, and
control-validated on the original corpus; unverified on held-out fresh scenarios.** A
full revision of this report will state the block result either way.

## 8. The honesty effect (a second, sturdier-scoped finding)

Compaction doesn't just lose content — it makes the model *confabulate* about what was
lost. Probing with decoy questions about things that never existed in the conversation,
plain Compacted fabricated 83% of the time; an arm that retains write-time KV state for a
packed summary (the "Packed write-time-KV" arm — note this variant retains keys *and*
values in a packed layout, a cousin of the value graft rather than the same intervention)
fabricated 17%, while being simultaneously the most accurate on genuinely evicted facts
(38/48, 79%) and the least fabricating (4%). Within-layout controls decompose the gain:
packed layout alone cuts decoy fabrication 83%→25%; write-time encoding within the same
layout cuts it further 25%→17%, and flips bare guesses into explicit admissions of
uncertainty (3/24 → 18/24). This replicated from 4-bit to bf16 and from 4B to 30B —
opposite scale behavior to the dissociation, which is why we report them as separate
findings. Scope bound: the effect lives in *mid-task agentic* compaction; on
retrieval-style personal-QA framing (LongMemEval) at 30B it washes out, because the
model's own refusal calibration already covers that case.

> **Decoy probe:** "What was the name of the consultant who audited our tax-rate
> tables?" — no consultant ever existed.
> **Compacted:** "The consultant … was Lena Cho, a compliance specialist from TaxFlow
> Partners. … Her report is archived in Confluence > Compliance > Tax Audit Q2 2024."
> **Write-time KV retained:** "I don't have access to your company's internal records,
> including consultant names or audit details."

## 9. Dose, tuning, and the keys question

**Full strength can break the task.** At α=1.0 (full replacement), grafting
catastrophically failed one agentic task chain (0/4 where every other arm scored 4/4) and,
in probe transcripts, cross-wires real entities from elsewhere in the same conversation —
right content, wrong referent, asserted "as established." A per-layer-tuned configuration,
promoted only after champion/challenger evaluation, eliminated the instability (16/16 on
the same family) **and** passed the wrong-conversation contamination guard — while a
57-cache-slot mask that also looked good on holdout *failed* that guard and was killed as
a content-independent artifact. The guards discriminate. Dose optima are scale-dependent
(α≈0.25 at 4B, α≈0.75 at 30B), so the dose is a per-model calibration, not a universal
constant.

**Keys are neutral; values are the operative axis.** We built technically-sound key
grafting (RoPE re-rotation of stored keys to new positions, validated to fp32 precision,
cosine 0.9999998 against freshly-encoded keys) and swept K-only, coupled, and independent
K/V policies, uniformly and per-layer, at 30B. On the robust metric, key grafting is
approximately neutral everywhere and helps nowhere; value-only matches the headline
(+0.120, CI [−0.001, +0.228] in that sweep). An earlier "keys actively hurt" conclusion
was an artifact of the retired ratio estimator and is corrected here. Scope: our targets
are semantic phrases; whether keys matter for short identifier-like targets (an
addressing/retrieval regime) is untested at scale.

## 10. Does it travel? Architecture-specificity, and a pre-registered null

We began a cross-architecture sweep: same scaffold, per-model-native rendering, self-gen
summaries, same gates, ~30B-class models. The pre-registered hypothesis (H1, frozen
before the sweep) was that **QK-norm presence predicts a positive referent sign**. What
the data did instead (12 conversations per model; preliminary):

| model | architecture | referent raw E−B [95% CI, conv-clustered] | aggregate verdict |
|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | MoE, GQA 8, QK-norm | +0.136 [+0.034, +0.23] | significant positive |
| Mistral-Small-24B-Instruct-2501 | dense, GQA 4, no QK-norm | **+0.035 [+0.010, +0.063]** | aggregate null; sense *negative* [−0.067, −0.005]; stance floored by headroom gate |
| microsoft/phi-4 | dense, GQA 4, no QK-norm | −0.053 [−0.117, +0.008] | **significant negative** aggregate (−0.064 [−0.099, −0.027]) |
| Qwen2.5-32B-Instruct | dense, no QK-norm | negative | significantly negative (aggregate ≈ −0.3, confirmed by three independent runs incl. the original trusted apparatus) |

Two things follow. First, **H1 is falsified and reported as a pre-registered null**: a
no-QK-norm model (Mistral) shows a significantly *positive* referent effect — the opposite
of the prediction — and the within-model ablation that would have tested QK-norm causally
(disabling q/k-norm modules at inference) broke generation outright, so it is
uninformative. Second, and more interesting: the effect is **architecture-specific and
can reverse sign**, with different per-category signatures on different models (Qwen3:
referent+/sense+/stance-null; Mistral: referent+ but sense−; Qwen2.5 and phi-4: negative).
A generic artifact would not flip sign by architecture; a real mechanism interacting with
architectural detail would. Machinery gates (α=0 identity, self-graft no-op) passed on
every model reported, so these are not plumbing failures. What drives the sign is open —
n is small everywhere, per-model runs use 12 conversations, and two 24-conversation runs
(Qwen3-32B dense; Qwen2.5-32B) were in flight at writing.

Excluded by structure or tooling, disclosed in full in §14: MLA-attention models
(DeepSeek/Kimi — no per-head value vectors to graft), five checkpoints shipping as
multimodal wrappers (both Gemma-4s, Gemma-3-27B, both Qwen3.6s — which cost us the
pre-registered second within-vendor MoE/dense pair), and sliding-window snapshot layouts
the harness refuses rather than silently mishandles.

## 11. Where it stops

**Scale.** The dissociation does not replicate at 4B (probes-helped ordering muddled:
stance 87%, sense 55%, referent 62% — the clean stance-null structure is gone; a
dose-response inversion between 4B and 30B points the same way). The honesty effect
*does* replicate at 4B — so the small model is not globally graft-insensitive; the
dissociation specifically needs capability. On Qwen3.6-27B (a newer hybrid architecture),
sense recovery replicates (59% of probes helped) and stance stays null, but referent
recovery is flat (48% of probes helped — a coin flip) — a single cross-model
non-replication we report as such, cause open (with n=2 confounded models we cannot separate architecture, generation, and
training).

**End-to-end agent benefit is undemonstrated — and the benchmark landscape is part of the
finding.** SWE-bench is beyond a 30B subject model entirely (0/7 even oracle-mode; the
vendor's own model card, which omits SWE-bench for this non-Coder variant, predicted as
much). Interactive tau²-bench banking dialogues are structurally too short (~3–4K tokens
even with a capable GPT-4o-mini user-simulator) for genuine eviction at any threshold.
Chained exercise-scale tasks our model *can* do proved compaction-robust (Compacted 4/4
across seeds — nothing to repair; the champion's 16/16 vs α=1.0's 0/4 there is a
stability result, not a recovery result). The one clean real-trace signal is an offline
proxy: on 75 real SWE-Gym/OpenHands trajectories, the tuned graft recovered +0.0156 nats
of next-action prediction (45/75 wins, CI [0.005, 0.027]) — about 10% of the measured
compaction damage on an unforgivingly off-policy instrument. The operative regime — tasks
hard enough that eviction costs something, easy enough that recovered context is usable —
is narrow and badly served by existing benchmarks; a purpose-built harness is future
work, and our own natural-length synthetic conversations are the closest thing we had.

**Looking inside (kept brief deliberately).** We also probed internal state — with the
plain logit lens on 30B, and with the recently-published Jacobian lens (J-lens) on
Qwen3.6-27B, the one model with public lens weights. The aggregate picture corroborates:
grafting raises the evicted concept's internal presence (concept logprob moves from the
Compacted floor toward the Original ceiling on 68–77% of probes), the effect is
alignment-sensitive inside as well as outside (misaligned injection craters), moderate α
beats full replacement internally too, and value grafting does not reconstruct relations
a summary dropped entirely. But the per-example signal is small: a pre-registered
free-generation probe looking for a clean internal "fork" toward the correct concept
found 0 of 43, and the raw logit lens recovers most of the late-layer signal the
specialized lens shows. We use the lens work as corroboration only; no claim in this
report rests on it.

## 12. Related work, and what seems to be new

Every mechanical ingredient here has prior art; we found no prior instance of the
composite, and — more specifically — no prior work that *measures* what we measure. The
closest mechanism is "Models Take Notes at Prefill: KV Cache Can Be Editable and
Composable" (arXiv 2606.17107), which edits and transplants KV across contexts with
re-rotated keys and position-free values — but it transplants precompiled skills into
fresh contexts and evaluates decision-identity, not a summary generated in-context whose
write-time state is retained across a *compaction* boundary and scored on semantic
continuity. Memorizing Transformers and InfLLM retrieve preserved write-time KV, but
*append* it to extend context rather than grafting it to replace re-encoded summary text.
Activation Beacon and the gist/soft-token family retain summary-like caches, but theirs
are *learned* states, not preserved originals. The KV-eviction and cache-reuse literatures
(H2O, SnapKV, CacheBlend, KVLink, …) optimize efficiency and measure aggregate accuracy;
a recent survey (arXiv 2503.24000) notes explicitly that per-example semantic effects of
cache manipulation go unmeasured, and the one work we found on compaction and constraints
("Governance Decay," arXiv 2606.22528) tests whether a constraint *survives* the summary,
not how retained text is *reinterpreted*. Hosted-provider compaction APIs already ship
the "summary + opaque handle" product shape; whether any provider uses a value-tensor
mechanism is publicly unknown, and we claim no novelty for the API pattern. Caveat: several
of the nearest neighbors are unreviewed 2026 preprints. Our claim is correspondingly
narrow: *write-time value state deliberately preserved and grafted back across a
text-summarization boundary, evaluated on referent/sense/stance continuity of the
conversation* — if that exists under other terminology, we'd genuinely like to know.

## 13. What happens next

In flight or queued at the time of writing: the held-out block reproduction (§7 — the
single most important pending number); 24-conversation runs on Qwen3-32B and Qwen2.5-32B
(the within-vendor MoE/dense contrast and the strongest negative, at doubled n); a
robust-metric re-audit of remaining secondary tables (the 27B and 4B category breakdowns
above are reported as %-helped for exactly this reason); more probes per category to power
the sense/referent CIs on both metrics; and the short-identifier target lane for the keys
question. The full revision of this report will incorporate all of it, whichever way the
results land.

## 14. Methods and provenance (details)

**Models (exact checkpoints).** Headline: `Qwen/Qwen3-30B-A3B-Instruct-2507` (bf16, A100
pods). This is the *non-thinking* instruct checkpoint; the similarly-named thinking
`Qwen/Qwen3-30B-A3B` behaves differently (its `<think>` blocks change tokenization and
alignment) and running it by mistake reproduces nothing — checkpoint identity is a
reproduction-critical detail. Development and 4B results:
`mlx-community/Qwen3-4B-Instruct-2507-4bit` (4-bit, MLX, Apple Silicon); local 30B runs
used the 4-bit MLX build (precision is stated per result; headline numbers are bf16).
Cross-architecture: `mistralai/Mistral-Small-24B-Instruct-2501`, `microsoft/phi-4`,
`Qwen/Qwen2.5-32B-Instruct` (+ in-flight `Qwen/Qwen3-32B`). Same-model lens work:
`Qwen3.6-27B`. Judge: Claude Sonnet 5. Nothing here fine-tunes or trains anything.

**Corpus.** 54 authored scenario scaffolds (`data/scenarios.json`): system prompt, all
user turns, planted items, probe paraphrases, gold continuations. c01–c12: ~8.3–9.4K
tokens rendered, 22 user turns, 10 plants each (2 × {referent, sense, stance, ruled_out,
evicted_fact}; 120 plants, contamination-audited to 115/120 clean, tail-clean in all
scored categories). c13–c54: 30–36 user turns, 12 plants each (adds strong_prior; 506
plants), authored by Fable/Opus/Sonnet/GPT-5.5-Codex subagents with per-file schema
verification; per-file authorship is recorded in `meta.author`. 8 natural conversations
(n01–n08, no plants) exist for continuation-scoring only; conclusions here do not rest on
them.

**Who generated every token.** User turns, plants, probes, golds: authored (see above) —
never generated by the subject model. Assistant replies: the evaluated model itself,
in-context, greedy, ≤320 tokens/reply, per-conversation seeds (base 1000). Summary: the
evaluated model itself, greedy (temp 0.0, top_p 0.8, seed 17, ≤900 tokens), from a fixed
request prompt asking for a thorough 300–500-word context note (verbatim in
`src/arms_common.py`). Gold continuations: authored, shared across all models and arms,
teacher-forced only. The one historical exception: the original c01–c12 renders used
Qwen3-4B (same family) rather than the 30B itself; the 30B-native re-render reproduces
the headline (§6), and all cross-architecture numbers use each model's own render.

**Procedure.** Compacted context = system + assistant context-note ("[Context note]
Earlier parts of this conversation were compacted. Summary of what came before: …") +
retained tail (from the scaffold's middle-end boundary). Graft = value-only blend at the
two aligned regions (§2), α=0.75 unless stated. Alignment: difflib per region, minimum
block 8, sinks (first 4 positions) and special tokens excluded. (A stricter exact-span
aligner was tried and reverted: it breaks on thinking-model self-generated summaries;
the tolerant aligner was verified to align 100% of regions on every model reported.)
Scoring: per-token mean teacher-forced logprob of the gold under each arm, averaged over
probe paraphrases; identical forward-pass shapes across arms (batched-vs-stepwise kernel
differences make cross-shape logit comparison invalid — an early lesson that shaped the
whole harness). Temperature 0 everywhere in evaluation.

**Statistics.** raw_EB = lp_E − lp_B; 95% percentile bootstrap (10k resamples; 20k for
judged), clustered on conversations for headline CIs; %-helped alongside. Mean gap-closure
ratios are retired (unstable); where legacy tables haven't been re-audited yet we quote
%-helped only. Headroom floor 0.3 (floored categories excluded from sign verdicts, not
counted as harm; e.g. stance on Mistral, whose own summaries preserve stance content).
Task-competence floor: lp_A ≥ −8.0 per token (absolute mode; pre-registered relative
median−3·MADN amendment on record, a verified no-op for every model scored so far).
Exclusion counts are recorded per run (phi-4: 1 plant task-excluded; Mistral: 0).

**Pre-registration and deviations.** H1 (QK-norm → positive referent sign), the metric,
gates, and the within-vendor pair inference were frozen in `PREREGISTRATION.md` before
the sweep; the pre-flight discovery that 5 of 16 queue models ship as multimodal wrappers
(excluding the Gemma-4 pair) is documented there *before* any outcome data, reducing the
primary within-vendor de-confound to one pair. H1's falsification is reported as a
pre-registered null (§10). The block experiment's design and stopping rule (§7) were
likewise committed before its data.

**Known infrastructure/validity incidents affecting interpretation** (all documented in
the repository's INCIDENTS.md): the wrong-checkpoint episode (thinking vs. instruct); the
retired ratio estimator (which had distorted an earlier public draft's precision — the
correction changed error bars and killed one secondary claim, not the direction of the
main effect); and a session-state leak in an earlier agent-serving harness whose affected
strata were demoted and never used for headline claims.

**Reproducibility.** All code, scaffolds, per-run JSON results (with gates, covariates,
CIs, and machinery-check outcomes embedded), decision log, incident log, and
pre-registration are in this repository: `src/cross_arch_probe.py` (current harness:
native render, gates, controls), `src/gap_closure_cat.py` (original apparatus),
`scripts/block_analysis.py`, `scripts/judged_bootstrap.py`, `data/scenarios.json`,
`results/`. A reader with an 80GB GPU can re-run any single model's sweep from the
scaffold in a few hours; exact constants (seeds, tolerances, caps) are in the source and
in this section.

---

*This report is a working snapshot of an ongoing autonomous research program; the
repository's FINDINGS.md, DECISIONS.md, and INCIDENTS.md are the running scientific
record. Numbers herein are observed results as of 2026-07-09; the held-out reproduction
and two architecture runs pending at press time will be incorporated in the next
revision.*
