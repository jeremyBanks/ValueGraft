# ValueGraft: Preserving Write-Time Attention State Across Conversation Compaction

**Status:** focused second draft. The broader synthesis remains preserved in
`valuegraft-paper-draft.md`.  
**Date:** 2026-07-07  
**Authors:** Jeremy Banks; Anthropic Claude Fable 5; OpenAI GPT-5.5

## Abstract

Long-running LLM agents often compact their conversation history by replacing
old turns with a summary and re-encoding the shortened transcript. This keeps
visible text but discards the attention state written when the model originally
interpreted that text in full context. We test whether preserving part of that
write-time state can reduce the impact of compaction. We use **ValueGraft** as
the name for a family of training-free compaction-state interventions and study
two variants on Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507:
**KV-Graft**, which carries both key and value tensors for summary tokens from
their write-time state into a compact packed context, and **V-Graft**, which
uses the ordinary compacted transcript with fresh keys but blends aligned
write-time value tensors into the fresh cache. The interventions do not recover
evicted factual recall. They produce two narrower mitigation effects: KV-Graft
reduces fabrication on unknowable questions in agentic compaction frames, and
tuned V-Graft recovers a small but consistent fraction of continuation and
coding-agent next-action likelihood lost to compaction. We use full-context vs
text-compacted performance to normalize effect sizes. On 75 OpenHands SWE-Gym
traces, V-Graft improves true next-action likelihood by +0.0156 nats/token,
about 10% of that gap. We also add qualitative Jacobian-lens readouts on
Qwen3.6-27B: the same summary tokens often decode as more situated under their
write-time state than when freshly re-encoded from the summary alone. The
evidence supports a limited mitigation claim: cached write-time attention state
can preserve some behavioral continuity across compaction boundaries, but it is
not a general memory-recovery mechanism.

## 1. Introduction

Conversation compaction is a practical necessity for long-running LLM agents.
When the context window fills, systems commonly replace old history with a text
summary and continue from `summary + recent tail`. The routine is cheap,
auditable, and compatible with ordinary stateless API calls.

But the operation changes more than the visible transcript. The retained tail
and the new summary are encoded after the old context has been removed. Any
context-conditioned computation previously written into the model's cached
attention state is discarded. A phrase like "the second approach" may still be
present in the tail, but its original interpretation was computed when the
earlier discussion was attendable.

We ask whether carrying forward a small amount of write-time attention state can
reduce the impact of text-only compaction.

We use **ValueGraft** as the umbrella name for two closely related
interventions.

**KV-Graft** generates a summary while the full conversation is still available,
then carries the summary tokens' write-time key/value entries into a compact
packed context. Keys are re-rotated to their packed positions; values are
unchanged.

**V-Graft** builds the ordinary compacted context and keeps its fresh keys, but
blends old value tensors into exact-aligned summary and tail positions:

```text
V_final = (1 - alpha_V) * V_fresh + alpha_V * V_old
```

Here `V_old` is the value tensor written when the matching token was processed
with the full context still available. The implemented arms use literal token
alignment only; they do not perform fuzzy semantic retrieval.

This naming is intentionally about state source, not layout. Earlier internal
arm names emphasized "pack" and "blend"; those identifiers remain in code and
logs for provenance, but the public taxonomy is simpler:

```text
Fresh-KV Control: fresh K, fresh V
V-Graft:          fresh K, write-time/blended V
KV-Graft:         write-time K, write-time V
```

The natural parameterization is `(alpha_K, alpha_V)`, where each value controls
how much of the write-time key or value side is used at aligned positions. In
the current V-Graft experiments, `alpha_K = 0` and `alpha_V` is tuned. In the
current KV-Graft experiments, the packed summary branch uses write-time keys
and values, with key re-rotation to account for the new packed positions. A
future controlled K-only branch would set `alpha_K = 1, alpha_V = 0` under the
same summary and layout controls.

Across the current evidence, these interventions do not restore facts that were
only present in the evicted history. Their measurable benefits are concentrated
in two failure modes: fabrication after compaction and loss of continuation or
next-action likelihood.

## 2. Terminology and Outcomes

We use **cached attention state** for the per-layer, per-token key and value
tensors stored during transformer prefill or generation. These are the tensors
usually held in a KV cache. This work is not about caching as a pure speed
optimization; it is about which attention state is present after compaction.

In a transformer attention layer, keys and values are separate projections of
the residual stream at each token position (Vaswani et al., 2017). Later tokens
compare their query against previous keys to decide where to attend, then read
from the corresponding values. In RoPE models, cached keys are position-rotated;
values are not. This is why moving a write-time key to a new packed position
requires re-rotation, while copying a value does not.

We distinguish three outcomes:

- **Recall:** answering questions whose evidence was in the evicted context.
- **Honesty:** admitting missing information rather than fabricating.
- **Continuity:** assigning higher likelihood to the true next continuation or
  next agent action.

The evidence so far supports honesty and continuity effects, not recall
recovery.

## 3. Related Work

Several nearby literatures already establish that cached attention state can be
useful beyond speed. Li (2026) argues that prefill writes conclusions into
downstream cached state and that those cached blocks can be edited or composed.
Zweiger et al. (2026) study latent-space KV compaction directly by constructing
smaller keys and values that reproduce attention behavior. KV reuse systems
such as KVLink (Yang et al., 2025), CacheBlend (Yao et al., 2025), LMCache
(LMCache contributors, 2024), and SamKV (Cao et al., 2025) reuse or mix cache
state for independently encoded chunks. Learned compression methods such as
gist tokens (Mu et al., 2023), AutoCompressor (Chevalier et al., 2023), ICAE
(Ge et al., 2024), Activation Beacon (Zhang et al., 2024), Compressed Context
Memory (Kim et al., 2023), and Cartridges (Eyuboglu et al., 2025) explore
non-textual ways to carry context.

The narrower setting here is conversation compaction: old dialogue is replaced
by a generated text summary and the system continues from a compacted
transcript. Text compression and memory systems are the baseline condition. The
intervention is to preserve write-time state associated with the generated
summary or retained tail and compare it to re-encoding the same visible
compacted text from scratch.

Hosted provider APIs now expose related product surfaces. OpenAI Responses
supports server-side compaction through `context_management` with
`compact_threshold`, plus a standalone `/responses/compact` endpoint; the
public docs describe an encrypted, opaque compaction item that carries prior
state and reasoning forward (OpenAI, 2026a). OpenAI prompt caching also
explicitly discusses persisted key/value tensors as attention-layer prefill
representations (OpenAI, 2026b). Anthropic exposes beta server-side compaction
as a typed `compaction` block containing a summary that must be passed back on
later requests (Anthropic, 2026). Gemini exposes context caching and encrypted
thought signatures for carrying reasoning context across API calls (Google AI
for Developers, 2026a, 2026b). These interfaces show that production APIs can
carry non-textual or semi-opaque continuation state. They do not show whether
providers use a ValueGraft-like mechanism internally.

Our qualitative readout work uses the Jacobian lens introduced by Gurnee et al.
(2026), a refinement of logit-lens-style residual-stream readouts
(nostalgebraist, 2020; Belrose et al., 2023). The J-lens maps residual-stream
states at intermediate layers to ranked vocabulary readouts, intended to expose
verbalizable concepts represented by the model. We use it only as an
interpretability aid: it helps describe what information may be present in
write-time versus freshly re-encoded summary states, but it is not itself the
main behavioral metric.

## 4. Methods

### 4.1 Shared Compaction Setup

Each example is first rendered with the model's chat template in a canonical
non-final form, so token spans can be compared across arms without Qwen's final
assistant-message template instability. We keep the first four tokens as
attention sinks. The retained tail begins at a message boundary near the final
quarter of the tokenized conversation; the material between the sinks and that
tail is the evicted region.

A summary is generated greedily by the same subject model while the full
pre-compaction conversation is still available. The summary request is appended
as a normal user turn, the model is prefixed through the full conversation plus
that request, and the generated summary tokens are decoded from the resulting
continuation. The cache snapshot from this generation run is saved. It contains
the key/value tensors written for the original conversation, the summary
request, and the summary tokens as they were generated under full context.

The production text-compaction baseline then constructs a new transcript:
system message, assistant context note containing the summary, and the retained
tail messages. That transcript is freshly encoded from scratch. The
interventions differ only in which cached attention state is carried across
this boundary.

### 4.2 Arms, Controls, and Public Names

The main evaluated arms are:

- **A:** full context, no compaction.
- **B:** production-style text compaction, `summary + tail`, freshly encoded
  from the shortened transcript.
- **Fresh-KV Control:** the first four sink tokens plus the exact generated
  summary token ids, packed contiguously and freshly encoded.
- **KV-Graft:** the same packed token sequence as the Fresh-KV Control, but
  using the summary's write-time key and value entries.
- **V-Graft:** the normal B transcript with fresh keys and blended old value
  tensors at exact-aligned summary and tail positions.
- **Negative controls:** shuffled-value and wrong-conversation grafts.

KV-Graft vs the Fresh-KV Control isolates the effect of write-time summary
encoding in a packed layout: identical tokens, identical packed positions,
different cache state. V-Graft vs B tests whether old value payloads improve an
otherwise ordinary compacted transcript with fresh keys.

Historical implementation names are preserved in code and result directories:
`B-min-pack` corresponds to the Fresh-KV Control, `H-pack` to KV-Graft, and
`E`/`E-tuned` to V-Graft. We do not use those names as the public taxonomy
because they mix layout details with state-source details.

The subject models for the quantitative arms are Qwen3-4B-Instruct-2507 and
Qwen3-30B-A3B-Instruct-2507 (Qwen Team, 2025a, 2025b).

### 4.3 KV-Graft Construction

KV-Graft extracts the summary token span from the saved summary-generation
cache. For Qwen3 in this stack, keys are stored after RoPE rotation and values
are unrotated. To make the retained summary cache contiguous and prefix-shaped,
we move the summary span immediately after the four sink tokens. The summary
keys are re-rotated by the positional offset between their write-time positions
and their packed positions; the values are copied unchanged. The resulting
cache contains only the sinks and summary entries, with the cache offset set to
the end of the packed summary.

The Fresh-KV Control is the matched text-only control for this operation. It
uses the same sink tokens and the same summary token ids at the same packed
positions, but obtains their key/value tensors by an ordinary fresh prefill.
Any KV-Graft vs Fresh-KV Control difference is therefore not due to summary
wording, token count, or packed position layout.

### 4.4 V-Graft Construction

V-Graft starts from the production text-compaction baseline B. We first
freshly prefill the compacted transcript, preserving its ordinary contiguous
positions and its fresh keys. We then align tokens from the compacted transcript
to tokens from the old full-context summary-generation run.

Alignment is exact-token matching, not semantic retrieval. The summary region
and tail region are matched separately because B places the summary before the
tail, while the old generation run places the summary after the original
conversation and summary request. Matching blocks shorter than eight tokens are
discarded to avoid common-token coincidences, and special tokens and sink
positions are excluded. For each accepted pair `(new_pos, old_pos)`, each
layer's value tensor is replaced by a linear blend:

```text
V_final[layer, new_pos] =
    (1 - alpha_V) * V_fresh[layer, new_pos]
  + alpha_V       * V_old[layer, old_pos]
```

The reported V-Graft arm is the post-prefill version: blending happens after
B's compacted transcript has been encoded. Earlier exploratory arms also tried
interleaving the blend during prefill, but the current headline comparison uses
fresh keys and post-prefill blended values. The 4B setting used a mid-layer-band
gate with `alpha_V = 0.25`; the 30B setting used a global `alpha_V = 0.75`,
both selected on validation continuation likelihood before holdout evaluation.

This V-Graft branch should not be described as "the replace arm." A constant
`alpha_V = 1` is one point in the V-Graft family; `alpha_V = 0` is equivalent
to no value grafting. Current results primarily evaluate tuned intermediate
settings.

### 4.5 Validation

Because cache surgery is sensitive to position, template, and kernel details,
every reported configuration had to pass identity tests before its results
counted. The ladder verifies cache
serialization, null surgery, V-Graft `alpha_V = 0` equivalence to B,
old-context-equals-new-context equivalence to A, tokenization stability, and
key re-rotation for packed KV-Graft. Runtime traps found during the work
include Qwen chat-template instability, batched-vs-stepwise logit differences,
and sequence-length-dependent 4-bit kernel behavior.

Negative controls check whether gains can be explained by generic smoothing or
odd cache perturbations. Shuffled-value grafts use the correct conversation's
old values but attach them to the wrong aligned positions. Wrong-conversation
grafts use old values from another conversation. The wrong-source packed-summary
control uses summary state from another conversation. These controls test
whether an effect survives after content/state alignment is broken.

### 4.6 Data and Scoring

The evidence comes from four sources:

- **Synthetic planted conversations:** 12 approximately 9K-token conversations
  with planted referents, senses, stances, ruled-out approaches, and evicted
  facts.
- **Natural/free-form conversations:** 8 conversations with held-out
  continuations.
- **LongMemEval-S:** standard benchmark material restructured so the evidence
  session is evicted (Wu et al., 2024). Early runs cover n=48 at 4B and n=36 at
  30B; a later 30B-bf16 aggregate covers n=320.
- **SWE-Gym/OpenHands traces:** 75 real coding-agent trajectories scored by
  teacher-forced likelihood of the true next assistant action (Pan et al.,
  2025).

Probe-style tasks append a user question to each arm and greedily generate an
answer at temperature 0. Continuation and coding-trajectory tasks teacher-force
the held-out continuation or next assistant action and report mean per-token
log likelihood. Where possible, results are paired by conversation and reported
with wins, bootstrap confidence intervals, and gap closure `(arm - B) / (A -
B)`. Local 4B results are 4-bit MLX; cloud 30B results are bf16
HuggingFace/Transformers unless otherwise stated.

## 5. Results

### 5.1 Baseline Gap Used for Evaluation

We compare each intervention against text-only compaction by measuring how much
of the full-context vs compacted gap it reduces. The gap is a normalization
denominator for mitigation scores. Removing the evidence is expected to hurt
recall.

For calibration, the early LongMemEval-S runs show full context at 71-81%
correct while compacted variants fall to at most 11%. The later 30B-bf16
aggregate gives a larger standard-data estimate: 52.5% correct with full
context vs 4.1% with text compaction over n=320. On SWE-Gym/OpenHands
next-action prediction, the full-context vs compacted gap is 0.164 nats/token.
The reported intervention effects are measured against these gaps.

### 5.2 Same Text, Different State

The same summary text behaves differently depending on whether its cache entries
were written under full context. In an earlier gapped KV-Graft pilot, the
write-time cache variant beats a fresh-encoded minimal summary control by
+0.093 nats on the 4B pilot, winning 10/12 conversations. At 30B the contrast
grows to +0.128 nats, winning 12/12 conversations, CI [0.097, 0.158].

A separate micro-sense experiment isolates the mechanism. A sentence with
identical tokens and positions is evaluated with and without a
disambiguating context. The bare-context sense margin is -0.23 nats.
Transplanting only value vectors moves it to +0.84; transplanting K+V moves it
to +1.95; full context is +3.81. Cached value tensors therefore carry
context-conditioned interpretation in this controlled setting.

Wrong-conversation and shuffled-value grafts degrade performance rather than
improve it, arguing against a generic smoothing explanation.

### 5.3 KV-Graft Reduces Fabrication in Agentic Frames

KV-Graft's clearest benefit is honesty rather than recall. On unknowable
questions, production-style compaction often fabricates. Packed summary contexts
make the model more cautious, and write-time summary state adds a further
component, especially at 30B.

Fabricated:admitted counts on Phase 2 synthetic/decoy probes:

| Arm | 30B decoys | 30B evicted facts | 4B decoys | 4B evicted facts |
| --- | --- | --- | --- | --- |
| B, production compaction | 19:5 | 16:8 | 18:6 | 15:9 |
| Fresh-KV Control, fresh packed summary | 10:14 | 5:19 | 4:20 | 6:18 |
| KV-Graft, write-time packed summary | **3:21** | **1:23** | **3:21** | **2:22** |

The matched comparison is KV-Graft vs the Fresh-KV Control. At 30B, write-time
state reduces decoy fabrication from 10 to 3. At 4B the matched difference is
small; most of the effect is already produced by the packed minimal layout.

The same distinction explains why the standard benchmark result is weaker. In 30B
LongMemEval personal-QA framing, the compacted baseline already tends to admit
missing history. The larger n=320 run confirms that honesty is flat across
arms, and grafting does not increase fabrication. The honesty effect therefore
appears frame-dependent: it matters most when the compacted context invites the
model to continue acting as a task participant.

### 5.4 V-Graft Recovers a Small Continuation Signal

V-Graft improves continuation likelihood by small but consistent
amounts when tuned on a validation split and evaluated once on holdout.

| Scale | Setting | Holdout gain vs B | Wins | CI | Gap closure |
| --- | --- | --- | --- | --- | --- |
| 4B | mid-band `alpha_V = 0.25` | +0.017 nats | 10/10 | [0.012, 0.024] | ~10% |
| 30B | global `alpha_V = 0.75` | +0.033 nats | 9/10 | [0.014, 0.057] | ~24% |

The dose response changes with scale. At 4B, high `alpha_V` is harmful and
layer gating helps. At 30B, a global `alpha_V` around 0.75 works best, and
`alpha_V` above 1.0 declines smoothly rather than failing immediately. Per-slot
tuning produced a promising 30B holdout result, but it is not yet robust enough
to be a headline method.

In the closest coding-domain check so far, on 75 OpenHands SWE-Gym traces,
tuned V-Graft improves true next-action likelihood by +0.0156 nats/token, wins
45/75, CI [0.005, 0.027], and recovers about 10% of the full-context vs
compacted gap. This remains an offline proxy rather than an end-to-end
task-success result, but it places the effect on real agent trajectories.

### 5.5 Qualitative J-Lens Readouts

To make the phrase "lost context-conditioned state" less abstract, we ran
Jacobian-lens readouts on Qwen3.6-27B residual-stream states using the public
Neuronpedia/Qwen J-lens tooling and weights (Gurnee et al., 2026; Neuronpedia,
2026; Qwen Team, 2026). These probes are not scored as behavioral evidence.
They are a qualitative view into the state ValueGraft tries to preserve.

The readout compares the same literal summary tokens in two paths:

1. **Write-time path:** the model reads the original conversation and then
   writes or sees the summary while the old context is still available.
2. **Fresh-summary path:** the model sees only the compacted summary in a clean
   local wrapper.

For each aligned summary token, we decode selected residual-stream layers into
top-k vocabulary readouts. The useful pattern is consistent across several
formats: in the write-time path, compact labels often decode toward their
situated role in the original conversation; in the fresh-summary path, the same
visible token often decodes toward ordinary lexical priors.

Examples:

The tables below show the intended reading format. The visible text is the text
available to both paths. The two readout columns are top vocabulary items from
the same anchored token under write-time versus fresh encoding. They should be
read as a noisy lens view, not as generated answers.

**Pokemon planning summary.** The old conversation establishes several private
labels: `Vacuum` is a Zigzagoon with Pickup; `Dex` is a person/trade obligation,
not a Pokedex progress tracker. The summary text says:

```text
- Vacuum is the Zigzagoon with Pickup, a utility slot only; Vacuum never
  battles gyms or major fights such as Drake.
- Dex trade: spare Makuhita for Dex's Castform from the Ruby save; Castform
  was for rain support.
```

Matched-wrapper J-lens readouts from
`jlens_boundary_probe/pokemon_readout_notes.md`:

| Anchor token | Write-time readout | Fresh-summary readout | Human reading |
| --- | --- | --- | --- |
| `Vacuum` | layer 62: `is`, `nickname`, `nick`, `Zig`, `nicknamed`, `/Z` | layer 62: `Cleaner`, `cleaner`, `Clean`, `cleaned`, `cleaners` | With old context, the token points at the run-specific nickname; fresh encoding falls back toward the ordinary appliance sense. |
| `Dex` | layer 62: `owes`, `owed`, `owe`, `trade`, `traded`, `trades` | layer 62: `Nav`, `nav`, `completion`, `dex`, `navigation`, `entry` | With old context, `Dex` is an agent in an owed trade; fresh encoding drifts toward Pokedex/DexNav/progress semantics. |

This is the phenomenon ValueGraft is meant to preserve. The summary already
contains the right words. The difference is that write-time state still exposes
the private interpretation those words had when the original conversation was
attendable.

**Ordinary block-party summary.** The old conversation defines local meanings
for otherwise generic labels: `Maple` is a room, `B-410` is stale, `P-771` is
current, and `Crane` is a stage-rental company. The summary text says:

```text
- Maple means the library's Maple Room for storage and volunteer check-in,
  not the tree grove or syrup sponsor.
- Permit: B-410 is stale. Use P-771 on the insurance form.
- Crane is the stage rental company, not equipment. Crane delivers risers at 9
  on Saturday; their driver calls Robin, but Mateo and Jules unload.
```

Matched-wrapper J-lens readouts from
`jlens_boundary_probe/plain_conversation_readout_notes.md`:

| Anchor token | Write-time readout | Fresh-summary readout | Human reading |
| --- | --- | --- | --- |
| `Maple` | layer 48: `refers`, `=`, `referring`, `denotes`; layer 62: `=`, `refers`, `is`, `means` | layer 48: `Street`, `street`, `neighborhood`, `park`, `town`, `City`; layer 62: `Street`, `Ave`, `Avenue`, `St`, `Streets` | With old context, `Maple` behaves like a locally defined label; fresh encoding treats it like a generic place/street name. |
| `B-410` | layer 48: `obsolete`, `outdated`, `deprecated`, `expired` | layer 48: `municipal`, `City`, `city`, `Civic`, `Town` | With old context, the stale-number warning is prominent; fresh encoding mostly sees a civic permit-like identifier. |
| `Crane` | layer 62: `is`, `refers`, `means`, `Stage`, `stage`, `delivers` | layer 62: `rental`, `operator`, `schedule`, `lease`, `license` | Fresh encoding is not nonsensical, but it is more generic; write-time state better preserves the local company/stage referent. |

**Coding next-action readout.** The SWE-Gym examples are less clean as semantic
demos because tool-call syntax, file paths, and subword fragments dominate raw
token ranks. The useful unit is the phrase span. In one getmoto trajectory, the
true next action is:

```text
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/workspace/getmoto__moto__4.1/moto/rds/responses.py</parameter>
<parameter=view_range>[584, 600]</parameter>
</function>
```

The span-level probe finds the operational objects rather than only punctuation:
`str_replace_editor` has mean span divergence 0.651, the full
`/workspace/getmoto__moto__4.1/moto/rds/responses.py` path has mean divergence
0.561, `responses.py` has mean divergence 0.568, and `[584, 600]` has mean
divergence 0.407. This is not as readable as `Vacuum` or `Maple`, but it shows
that the same readout method can be aimed at the practical tokens in a coding
agent trajectory: tool name, command, file path, and line range.

The J-lens evidence has a narrower role than the likelihood and probe metrics.
It supplies a mechanistic readout for one local substrate: the same summary text
can have different verbalizable residual-stream content depending on whether it
is read fresh or preserved from the state in which it was written.

### 5.6 Boundary: No Recall Recovery

The interventions do not recover evicted facts. On LongMemEval, all compacted
variants remain at or below 11% correct in the early runs, and the larger
Stage-1 aggregate gives the expected full-context vs compacted recall gap. In
synthetic probe cuts, referent recovery remains poor. KV-Graft mainly changes
whether the model fabricates or admits missing information; V-Graft mainly
shifts likelihood toward the full-context continuation.

The current evidence therefore supports "less damaging compaction," not "latent
recall of deleted context."

## 6. Discussion

The findings separate two losses from compaction: visible information and
computation state. Summary text can only preserve what it says. Write-time
cached attention state appears to preserve some context-conditioned
interpretation of the text that remains or of the summary generated under full
context. That state is not enough to answer arbitrary questions about evicted
facts, but it can affect uncertainty and continuation behavior.

A compacted coding agent may not need to recall every old line verbatim, but it
does need to avoid confidently repeating failed approaches, hallucinating
settled details, or drifting away from the trajectory implied by the previous
work. The current OpenHands result is only a next-action likelihood proxy, but
it is pointed at a relevant failure surface.

## 7. Limitations

The evidence is narrow. Most results are from one model family. The 4B local
and 30B cloud runs differ in scale and precision. Synthetic probes are clean but
not representative of all agent failures. LongMemEval shows that prompt frame
matters: in personal-QA framing, 30B already admits missing history and the
honesty intervention has little room to help. The coding evidence is offline
next-action prediction, not end-to-end task completion.

Several controls remain. A stronger text-only summary baseline is needed.
KV-Graft's honesty effect is partly layout-driven and partly write-time
state-driven; the matched pair isolates some of the latter, but not every
possible caution mechanism. Per-slot calibration should remain exploratory until
it passes wrong-conversation guards and larger holdout tests. Single examples
are explanatory rather than statistical; the sense-level evidence is the
controlled micro-sense experiment, and the J-lens readouts are qualitative
interpretability evidence rather than task-performance evidence.

## 8. Next Work

The next experimental step is to make the K/V taxonomy fully controlled. Using
the same summary text, same compact layout, and same scoring set, compare
`(alpha_K, alpha_V)` settings that isolate value-only, key-only, and K+V
effects. In public terms, this means keeping V-Graft, adding a controlled
K-only branch, and re-running KV-Graft only where its summary and layout
controls match. This avoids treating unrelated implementation choices as if
they were scientific variables.

The strongest practical test remains coding. The current SWE-Gym result is a
next-action likelihood proxy; an end-to-end agent run would test whether the
small likelihood gain becomes meaningful task behavior. The J-lens probe should
also become span-first for coding traces: report tool names, file paths,
commands, symbols, and line ranges as phrase spans, with raw token rows kept as
drill-down data.

## 9. Conclusion

The current evidence supports the following conclusion:

KV-Graft and V-Graft do not make compacted models remember deleted facts. They
show that write-time cached attention state can reduce some behavioral harm from
compaction. KV-Graft can make post-compaction models less likely to fabricate
unsupported details in agentic frames. V-Graft can recover a small but
consistent fraction of continuation and next-action likelihood lost to
compaction.

This supports treating conversation compaction as a text-plus-state problem. The
summary is the visible artifact, but the computation that produced and
interpreted it may also be worth preserving.

## Code Availability

Code, experiment scripts, draft analysis, and reproducibility notes are available
at <https://github.com/jeremyBanks/ValueGraft>.

## References

- Anthropic. 2026. [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction). Claude Platform Docs. Accessed 2026-07-07.
- Belrose, Nora, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, and Jacob Steinhardt. 2023. [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112). arXiv:2303.08112. DOI: [10.48550/arXiv.2303.08112](https://doi.org/10.48550/arXiv.2303.08112).
- Cao, Ziyi, Qingyi Si, Jingbin Zhang, and Bingquan Liu. 2025. [Sparse Attention across Multiple-context KV Cache](https://arxiv.org/abs/2508.11661). arXiv:2508.11661. DOI: [10.48550/arXiv.2508.11661](https://doi.org/10.48550/arXiv.2508.11661).
- Chevalier, Alexis, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. [Adapting Language Models to Compress Contexts](https://aclanthology.org/2023.emnlp-main.232/). In *Proceedings of EMNLP 2023*, pages 3829-3846. DOI: [10.18653/v1/2023.emnlp-main.232](https://doi.org/10.18653/v1/2023.emnlp-main.232).
- Eyuboglu, Sabri, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, and Christopher Re. 2025. [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266). arXiv:2506.06266. DOI: [10.48550/arXiv.2506.06266](https://doi.org/10.48550/arXiv.2506.06266).
- Ge, Tao, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. [In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945). ICLR 2024; arXiv:2307.06945. DOI: [10.48550/arXiv.2307.06945](https://doi.org/10.48550/arXiv.2307.06945).
- Google AI for Developers. 2026a. [Gemini 3 Developer Guide](https://ai.google.dev/gemini-api/docs/gemini-3). Accessed 2026-07-07.
- Google AI for Developers. 2026b. [Context caching](https://ai.google.dev/gemini-api/docs/caching). Accessed 2026-07-07.
- Gurnee, Wes, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey. 2026. [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html). Transformer Circuits Thread.
- Kim, Jang-Hyun, Junyoung Yeom, Sangdoo Yun, and Hyun Oh Song. 2023. [Compressed Context Memory For Online Language Model Interaction](https://arxiv.org/abs/2312.03414). ICLR 2024; arXiv:2312.03414. DOI: [10.48550/arXiv.2312.03414](https://doi.org/10.48550/arXiv.2312.03414).
- Li, Bojie. 2026. [Models Take Notes at Prefill: KV Cache Can Be Editable and Composable](https://arxiv.org/abs/2606.17107). arXiv:2606.17107. DOI: [10.48550/arXiv.2606.17107](https://doi.org/10.48550/arXiv.2606.17107).
- LMCache contributors. 2024. [LMCache](https://github.com/LMCache/LMCache) and [CacheBlend documentation](https://docs.lmcache.ai/kv_cache_optimizations/blending.html). Accessed 2026-07-07.
- Mu, Jesse, Xiang Lisa Li, and Noah Goodman. 2023. [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467). NeurIPS 2023; arXiv:2304.08467. DOI: [10.48550/arXiv.2304.08467](https://doi.org/10.48550/arXiv.2304.08467).
- Neuronpedia. 2026. [Jacobian Lens - Qwen3.6-27B](https://www.neuronpedia.org/qwen3.6-27b/jlens). Accessed 2026-07-07.
- nostalgebraist. 2020. [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). LessWrong.
- OpenAI. 2026a. [Compaction](https://developers.openai.com/api/docs/guides/compaction) and [Compact a response](https://developers.openai.com/api/reference/resources/responses/methods/compact). OpenAI API documentation. Accessed 2026-07-07.
- OpenAI. 2026b. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching). OpenAI API documentation. Accessed 2026-07-07.
- Pan, Jiayi, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. 2025. [Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139). arXiv:2412.21139. DOI: [10.48550/arXiv.2412.21139](https://doi.org/10.48550/arXiv.2412.21139).
- Qwen Team. 2025a. [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Hugging Face. Accessed 2026-07-07.
- Qwen Team. 2025b. [Qwen3-30B-A3B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507). Hugging Face. Accessed 2026-07-07.
- Qwen Team. 2026. [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B). Hugging Face. Accessed 2026-07-07.
- Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017; arXiv:1706.03762. DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762).
- Wu, Di, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2024. [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813). arXiv:2410.10813. DOI: [10.48550/arXiv.2410.10813](https://doi.org/10.48550/arXiv.2410.10813).
- Yang, Jingbo, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2025. [KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse](https://arxiv.org/abs/2502.16002). arXiv:2502.16002. DOI: [10.48550/arXiv.2502.16002](https://doi.org/10.48550/arXiv.2502.16002).
- Yao, Jiayi, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2025. [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444). *EuroSys 2025*; arXiv:2405.16444. DOI: [10.48550/arXiv.2405.16444](https://doi.org/10.48550/arXiv.2405.16444).
- Zhang, Peitian, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. 2024. [Long Context Compression with Activation Beacon](https://arxiv.org/abs/2401.03462). arXiv:2401.03462. DOI: [10.48550/arXiv.2401.03462](https://doi.org/10.48550/arXiv.2401.03462).
- Zweiger, Adam, Xinghong Fu, Han Guo, and Yoon Kim. 2026. [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284). arXiv:2602.16284. DOI: [10.48550/arXiv.2602.16284](https://doi.org/10.48550/arXiv.2602.16284).

## Author Contributions and Provenance

Jeremy Banks directed the project and is the first author of record. Anthropic
Claude Fable 5 designed and implemented much of the experimental machinery and
executed substantial autonomous coding-agent work. OpenAI GPT-5.5 contributed
adversarial review, reframing, prior-art analysis, and this synthesis. Claude
Sonnet subagents performed logged answer judging. This should be read as
AI-assisted research directed by the human first author.

## Notes for a Later Full Paper

Material intentionally minimized here but preserved in the wider draft:

- Detailed hosted-provider API analysis.
- Full value-steering calibration discussion, including signed vs clamped fits.
- Deployment cost accounting.
- Extended related-work taxonomy.
- Process notes and collaboration history.
