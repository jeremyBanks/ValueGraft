# ValueGraft: Mitigating Conversation Compaction with Write-Time Attention State

**Status:** tentative 2026-07-06 draft. This is a separate draft, not a
replacement for `valuegraft-focused-draft.md`. It incorporates the newer
`alpha_K` / `alpha_V` terminology, the clean-run data-hygiene changes, and the
early end-to-end coding-agent evidence available at the time of writing.
Standard-task SWE-bench-Lite rows had been appended to the live queue, but no
standard-task score files were yet available.

**Authors:** Jeremy Banks; Anthropic Claude Fable 5; OpenAI GPT-5.5

## Abstract

Long-running LLM agents commonly compact their conversation history by
replacing old turns with a text summary and re-encoding the shortened
transcript. This preserves visible text while discarding the cached attention
state written when the model originally interpreted that text in full context.
We study whether preserving or blending that write-time state can reduce the
damage caused by compaction.

We use **ValueGraft** for a family of training-free interventions that operate
on cached key/value state at a compaction boundary. The clean conceptual
parameterization is:

```text
K_final = (1 - alpha_K) * K_fresh
        + alpha_K       * K_write_time_rerotated

V_final = (1 - alpha_V) * V_fresh
        + alpha_V       * V_write_time
```

The experiments completed so far mostly evaluate V-only Graft: fresh keys in
the compacted context, plus blended write-time value tensors at exact-aligned
summary and tail tokens. An older summary-only comparison also tested whether
the same generated summary behaves differently when freshly encoded versus when
carried forward with its write-time key/value state. That comparison is
informative auxiliary evidence outside the main ValueGraft method taxonomy.

Across Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507, the current
evidence supports a limited mitigation claim. ValueGraft does not recover
evicted factual recall. In the current production-shaped experiments, tuned
V-only Graft recovers a small but consistent fraction of continuation and
coding-trajectory likelihood lost to compaction. On 75 OpenHands SWE-Gym
trajectories, V-only Graft improves true next-action likelihood by +0.0156
nats/token, about 10% of the full-context vs text-compacted gap. Separately,
the historical summary-only comparison reduces fabrication in agentic frames;
that result is evidence about write-time summary encoding, not a current
ValueGraft arm. Early end-to-end coding-agent task runs remain unbalanced,
actively running, and not yet confirmatory.

The evidence points to a text-plus-state view of conversation compaction.
Summary text carries explicit information; write-time attention state can carry
part of the context-conditioned interpretation and uncertainty that text-only
re-encoding loses.

## 1. Introduction

Conversation compaction is a normal part of long-running LLM systems. When an
agent's transcript grows too large, the client or serving layer summarizes an
older prefix, keeps a recent tail, and continues from a shorter transcript.
This is operationally attractive: the new context is visible, auditable, and
compatible with ordinary API calls.

But compaction changes more than the transcript. A retained phrase such as
"the second approach" may still appear after compaction, but the cached
attention state that originally represented what "second" referred to was
computed when the relevant earlier discussion was still available. Re-encoding
the compacted text computes a fresh representation in a different context.

The central question in this work is mitigation-first:

> Given that text-only compaction discards context-conditioned computation, can
> preserving a small amount of write-time attention state reduce the behavioral
> damage?

The question is not whether cached state contains information. Prior work and
the transformer architecture already make that unsurprising. The practical
question is whether the state available at compaction time can be reused to
improve downstream behavior.

We study this with a family of interventions called **ValueGraft**. In its
general form, ValueGraft treats key and value state as separate experimental
axes. The current live coding experiments mostly use the V-only region of this
space: keys are freshly computed in the compacted context, while values are
blended from write-time state:

```text
alpha_K = 0
alpha_V = constant or tuned
```

Earlier summary-only experiments carried both key and value state for summary
tokens into a minimal context without the recent tail. Those experiments remain
relevant as historical evidence about write-time summary encoding and honesty.
They changed context shape and tail retention as well as state source, so they
are not part of the main key/value taxonomy.

## 2. Terminology

We use **cached attention state** for the per-layer key and value tensors stored
for each token during transformer prefill or generation. We use **write-time
state** for tensors computed while the full pre-compaction context was still
available, and **fresh state** for tensors computed by re-encoding the compacted
context from scratch.

The standard implementation object is called a KV cache, but this project is
not about caching as a speed optimization. In ordinary inference, caching is
supposed to preserve behavior while saving compute. Here, the cache matters
because compaction normally discards state and then recomputes a different
state from a shorter transcript.

For an aligned token in the compacted context, the clean ValueGraft family is:

```text
K(alpha_K) = (1 - alpha_K) * K_fresh
             + alpha_K * K_write_time_rerotated

V(alpha_V) = (1 - alpha_V) * V_fresh
             + alpha_V * V_write_time
```

`K_write_time_rerotated` is necessary because Qwen3 stores keys after RoPE
rotation. If a write-time key is moved to a different position in the compacted
context, it must be re-rotated into that position. Values are unrotated in the
tested Qwen3 stacks and can be blended directly once tokens are aligned.

The main named regions are:

| Name | Key policy | Value policy |
| --- | --- | --- |
| Plain Summary Compaction | `alpha_K = 0` | `alpha_V = 0` |
| V-only Graft | `alpha_K = 0` | `alpha_V` varied or tuned |
| K-only Graft | `alpha_K` varied or tuned | `alpha_V = 0` |
| KV-Graft | `alpha_K` varied or tuned | `alpha_V` varied or tuned |
| Coupled KV-Graft | `alpha_K = alpha_V = alpha` | same shared alpha |

`alpha = 1` is not a separate method. It is the parameter point where that side
uses write-time state exactly. `alpha = 0` is the ordinary fresh-state
baseline for that side. Values above 1 are extrapolation settings, not merely
stronger blending.

Main result labels translate as follows:

| Historical label | Paper-facing name |
| --- | --- |
| `A` | Full Context |
| `B` | Plain Summary Compaction |
| `E`, `E:a0.75`, `E:a1.0` | V-only Graft with stated `alpha_V` |
| `E:cfg=layers` | Layer-tuned V-only Graft |

The naming follows the controlled variable. A key-side claim requires fixed
summary text, token ids, layout, tail retention, positions, prompt, decoding,
and value policy. A value-side claim requires fixed key policy and fixed
non-state context. The in-flight coding arms satisfy this requirement for
value-side experiments. Future K-only or full KV comparisons should use the
same production-shaped compacted context rather than the older summary-only
setup.

Two older labels appear in result files as provenance names, not method names.
`B-min-pack` means "summary-only fresh encoding." `H-pack` means "summary-only
write-time KV." Their comparison is an auxiliary historical result, not a
current experiment arm family.

Inside that older pair, the controlled variable is state source: fresh KV or
write-time KV for the same generated summary tokens. Compared with the current
ValueGraft design, it also has a different context shape, no retained tail, and
coupled changes to K and V. It supports a narrow statement about write-time
summary encoding; it does not support claims about the current `alpha_K` /
`alpha_V` taxonomy.

## 3. Outcomes

We distinguish three outcomes:

**Recall:** answering questions whose evidence was present only in the evicted
context.

**Honesty:** admitting missing information rather than fabricating an answer.

**Continuity:** assigning higher likelihood to the true next continuation or
next agent action after compaction.

The current evidence supports honesty and continuity effects, not recovery of
evicted factual details.

The contribution is not the baseline fact that full context beats text-only
compaction. That gap measures the damage. The contribution, if any, is reducing
that damage in controlled comparisons.

## 4. Related Work and Product Context

There is substantial nearby work. Li (2026) argues that prefill writes
conclusions into downstream KV state and demonstrates editable, composable KV
blocks. Zweiger et al. (2026) study latent KV compaction through attention
matching. KVLink (Yang et al., 2025), CacheBlend (Yao et al., 2025), and SamKV
(Cao et al., 2025) reuse or fuse cached state for independently encoded
chunks, often for serving efficiency or retrieval. These systems are close in
mechanism because they reuse or combine cache state, but they are not framed
around a dialogue-summary compaction boundary.

Another line of work trains or adapts models to carry context through compact
latent or memory-like forms: gist tokens (Mu et al., 2023), AutoCompressor
(Chevalier et al., 2023), ICAE (Ge et al., 2024), Activation Beacon (Zhang et
al., 2024), Compressed Context Memory (Kim et al., 2024), and Cartridges
(Eyuboglu et al., 2025). Text compression and memory systems such as LLMLingua
(Jiang et al., 2023), RECOMP (Xu et al., 2024), and MemGPT (Packer et al.,
2023) are also relevant, but they operate primarily through text or learned
memory policies rather than by reusing the write-time attention state of the
summary or retained tail.

Parallel Context Compaction (Cim et al., 2026) is close in deployment setting:
it studies compaction for long-horizon agent serving. It remains a text-summary
serving technique, so it is best treated as a neighboring baseline rather than
the same intervention.

The narrower setting here is conversation compaction: old dialogue is replaced
by a generated visible summary plus recent tail, and the system continues as an
agent. The baseline is the ordinary text-only compacted transcript. The
intervention is to preserve or blend the write-time attention state associated
with summary and retained-tail tokens, then compare against re-encoding the
same visible text from scratch.

Hosted APIs now expose adjacent product surfaces. OpenAI Responses exposes
explicit compaction and conversation-state mechanisms; Anthropic exposes
server-side compaction blocks and context-management controls; Gemini exposes
context caching and opaque thought-signature-like artifacts. These interfaces
show that production systems can carry non-textual continuation state. They do
not publicly reveal whether providers use raw KV tensors, value-state blends,
learned summaries, reasoning-state locks, or something unrelated. ValueGraft is
an open, white-box mechanism compatible with these opaque state and compaction
handles, not a claim about provider internals.

## 5. Methods

### 5.1 Models and Runtime

The main model family is Qwen3-Instruct-2507:

| Setting | Model/runtime |
| --- | --- |
| Local pilot | Qwen3-4B-Instruct-2507, 4-bit MLX weights |
| Scale and cloud runs | Qwen3-30B-A3B-Instruct-2507, primarily bf16 HuggingFace/Transformers |

All evaluation uses temperature 0. Local 4B results and cloud 30B-bf16 results
are not treated as identical precision regimes. Cross-scale comparisons state
that distinction explicitly.

Cache surgery is only counted after identity tests. The build ladder verifies
cache reconstruction, null surgery, tokenization stability, summary/tail
alignment, alpha=0 equivalence to Plain Summary Compaction, old-context equals
new-context equivalence to Full Context, and key re-rotation for the older
summary-only comparison. Several implementation traps were discovered and
documented: Qwen chat-template
instability around final assistant messages, batched-prefill vs decode-step
logit differences, sequence-length-dependent 4-bit kernels, and session-state
leaks in the first scaled end-to-end agent harness.

### 5.2 Shared Compaction Setup

Each example begins as a full conversation. The first four tokens are retained
as attention sinks. A tail boundary is chosen at a message boundary; tokens
between the sinks and the retained tail are the evicted region.

A summary is generated greedily by the subject model while the full
pre-compaction conversation is still available. The cache snapshot from that
summary-generation run contains the write-time key/value state for the
conversation, summary request, and summary tokens.

Plain Summary Compaction then constructs a production-shaped compacted
transcript: system message, summary/context note, and retained recent tail.
That transcript is freshly encoded from scratch. ValueGraft arms differ only in
which cached attention state is inserted or blended after that baseline is
constructed.

### 5.3 V-only Graft

The main completed ValueGraft arm is V-only Graft. It keeps the compacted
transcript and its fresh keys. It then aligns summary and retained-tail tokens
from the compacted context to the same literal token spans in the write-time
trace. Alignment is exact-token matching, not semantic retrieval. Summary and
tail regions are aligned separately because their order changes across the old
and compacted contexts.

For each accepted aligned token pair, the value tensor is blended:

```text
V_final[layer, kv_head, new_pos, :] =
    (1 - alpha_V) * V_fresh[layer, kv_head, new_pos, :]
  + alpha_V       * V_write_time[layer, kv_head, old_pos, :]
```

The implemented `E`, `E:a0.75`, `E:a1.0`, and `E:cfg=layers` arms are V-only:

```text
alpha_K = 0
alpha_V = scalar or layer-tuned policy
```

The 4B continuation-tuning result used a mid-layer-band setting with
`alpha_V = 0.25`. The 30B continuation-tuning result used a global
`alpha_V = 0.75`. The live coding-agent matrix currently compares scalar
`alpha_V` settings and a layer-tuned V-only policy.

### 5.4 Historical Summary-Only Comparison

One earlier experiment used a summary-only context. It is auxiliary historical
evidence, not part of the current experiment design.

Both arms in that pair use the same four sink tokens and the same generated
summary tokens in the same summary-only context. The fresh control, historically
`B-min-pack`, obtains K and V by ordinary fresh encoding. The write-time arm,
historically `H-pack`, extracts the generated-summary span from the write-time
cache, re-rotates its keys into the summary-only positions, and copies its
values unchanged.

Within that pair, the controlled contrast is fresh summary KV versus write-time
summary KV. Compared with the current ValueGraft experiments, it also changes
major design axes: no retained tail, a different context shape from the
production-shaped compacted transcript, and coupled K/V changes rather than
separate `alpha_K` and `alpha_V` variation. The result is evidence about
write-time summary encoding and honesty, not evidence about the current
key/value graft taxonomy.

### 5.5 Coding-Agent Harness

The end-to-end coding-agent harness places a coding task behind an OpenAI-style
server shim. The shim implements the compaction policy and serves the subject
model while a coding agent explores and edits a repository. Success is scored
by objective tests.

The synthetic tasks `t1`, `t2`, and `t3` are small repositories whose prompts
state load-bearing constraints early. The agent is instructed to read files
before implementing missing modules, which pushes the session across the
compaction threshold. `t1` is a pricing/cart task, `t2` is a fetcher/API task,
and `t3` is a harder billing task with five constraints.

Because the first scaled matrix exposed a session-key leak, all affected
night-matrix arm comparisons were quarantined. The clean rerun uses
mode/config-specific session keys, probe-gated endpoints, and disjoint
task:seed ownership per lane. The pre-registered sensitivity gate continued
when clean Plain Summary Compaction was 2/7 on `t1`/`t2`, showing that the
instrument was not ceilinged.

A standard-task pivot is now in progress. A SWE-bench-Lite adapter has been
validated on two fail-before/pass-after pytest instances and appended 40 rows
to the live queue: 8 instances by the same five arms. These rows are to be
analyzed as a separate task-source stratum, not silently pooled with synthetic
tasks. At this draft cutoff, no standard-task score files were available yet.

## 6. Results

### 6.1 Text-Only Compaction Produces the Baseline Gap

Full-context performance exceeds text-compacted performance when the task
depends on evicted context. This gap is the damage measurement against which
mitigation is evaluated.

In LongMemEval-S, full context answered 71-81% of questions in early runs,
while compacted variants were at or below 11%. A later 30B-bf16 aggregate
measured 52.5% correct with full context vs 4.1% with text compaction over
320 standard-data questions. In SWE-Gym/OpenHands next-action prediction, the
full-context vs text-compacted gap is 0.164 nats/token.

These numbers establish headroom for mitigation and mark a boundary: when the
evidence is truly evicted, the methods tested here do not recover ordinary
factual recall.

### 6.2 Same Text, Different Write-Time State

Several tests show that identical visible text can behave differently depending
on the cached attention state associated with it. They motivate the method, but
not all of them are current method arms.

In the historical summary-only pilot, the same generated summary predicted
future continuation better when its cached state had been written under full
context than when the same summary tokens were freshly encoded: +0.093 nats,
winning 10/12 conversations. At 30B, the contrast grew to +0.128 nats, winning
12/12 conversations. This pilot is the auxiliary comparison described in
Section 5.4: evidence about write-time summary KV outside the current
`alpha_K` / `alpha_V` taxonomy.

The micro-sense experiment isolates the mechanism. A carrier sentence with
identical tokens and positions is evaluated with and without a disambiguating
context. The fresh bare-context sense margin is -0.23 nats. Transplanting only
write-time values moves it to +0.84. Transplanting K+V moves it to +1.95.
Full context is +3.81. These margins indicate that write-time values carry
some context-conditioned interpretation, while keys also matter.

Wrong-conversation and shuffled-value controls degrade performance rather than
helping. This argues against a generic smoothing or cache-perturbation account.

### 6.3 Historical Summary-Only Comparison: Fabrication

The older summary-only comparison contributes one result: fabrication on
unknowable questions. It separates fresh summary KV from write-time summary KV
inside a summary-only context, under the design caveats in Section 5.4.

On unknowable questions, text-only compaction often fabricates. Summary-only
contexts make the model more cautious, and write-time summary KV adds a further
component, especially at 30B.

Fabricated:admitted counts on Phase 2 synthetic/decoy probes:

| Arm | 30B decoys | 30B evicted facts | 4B decoys | 4B evicted facts |
| --- | --- | --- | --- | --- |
| Plain Summary Compaction | 19:5 | 16:8 | 18:6 | 15:9 |
| Summary-only fresh encoding | 10:14 | 5:19 | 4:20 | 6:18 |
| Summary-only write-time KV | **3:21** | **1:23** | **3:21** | **2:22** |

Within the summary-only pair, the matched comparison is write-time summary KV
vs fresh summary KV. At 30B, write-time KV reduces decoy fabrication from 10 to
3. At 4B, most of the honesty improvement is already produced by the
summary-only context; the write-time component is small.

LongMemEval adds a boundary. At 4B, the summary-only write-time KV arm
reduces fabrication relative to Plain Summary Compaction. At 30B in personal-QA
framing, the production compacted baseline already tends to admit missing
personal history, so the honesty advantage largely disappears. The effect is
frame-dependent: it matters most when compaction happens inside an ongoing
working conversation, where the model is tempted to keep acting as if it knows
the missing details.

### 6.4 V-only Graft Recovers a Small Continuation Signal

V-only Graft improves continuation likelihood by small but consistent amounts
when tuned on validation data and evaluated on holdout.

| Scale | V-only policy | Holdout gain vs B | Wins | CI | Gap closure |
| --- | --- | --- | --- | --- | --- |
| 4B | mid-band `alpha_V = 0.25` | +0.017 nats | 10/10 | [0.012, 0.024] | ~10% |
| 30B | global `alpha_V = 0.75` | +0.033 nats | 9/10 | [0.014, 0.057] | ~24% |

The dose response is model-dependent. At 4B, high `alpha_V` is harmful and
layer gating helps. At 30B, a global setting around 0.75 works best, and values
above 1 decline smoothly rather than failing immediately. Per-model calibration
is required.

Per-head and per-slot tuning remain exploratory. A 57-slot profile looked
strong on one 30B holdout, but its wrong-conversation guard indicated that much
of the gain was content-independent. The simpler global or layer-level
policies are the appropriate headline methods until slot-level policies pass
stronger contamination controls.

### 6.5 Offline Coding-Trajectory Evidence

On 75 OpenHands SWE-Gym trajectories, tuned V-only Graft improves the
teacher-forced likelihood of the true next assistant action by +0.0156
nats/token. It wins 45/75 traces, with CI [0.005, 0.027], recovering about 10%
of the full-context vs text-compacted gap.

This domain is closer to real agent work than the synthetic planted probes.
The metric is still offline next-action likelihood, not end-to-end task
success.

### 6.6 Early End-to-End Coding-Agent Runs

The end-to-end coding-agent evidence contains one promising fragment, but it is
not yet confirmatory. The initial clean round produced a suggestive pattern:
Plain Summary Compaction failed `t1` and `t2`, while V-only Graft passed both.
That was only n=2 per condition, and later rows are more mixed.

The later scaled matrix uncovered a session-state leak, so its arm comparisons
were quarantined. A clean rerun is now active. At the current draft cutoff, the
committed clean snapshot contained 13 synthetic score files:

| Arm | Passes / scored |
| --- | --- |
| Full Context | 3/3 |
| Plain Summary Compaction | 2/4 |
| V-only Graft, `alpha_V = 0.75` | 4/5 |
| V-only Graft, `alpha_V = 1.0` | 1/1 |

The live scratch table had grown to 41 synthetic score files:

| Arm | Passes / scored |
| --- | --- |
| Full Context | 5/9 |
| Plain Summary Compaction | 2/10 |
| V-only Graft, `alpha_V = 0.75` | 4/7 |
| V-only Graft, `alpha_V = 1.0` | 3/9 |
| Layer-tuned V-only Graft | 0/6 |

These numbers should be read cautiously. They are unbalanced, still running,
and include early `t3` rows where Full Context had not yet passed. Two rows had
lane/runtime `TASKMOD` errors rather than valid scores. The clean data so far
show that the instrument is not ceilinged and that `alpha_V = 0.75` remains
encouraging on `t1`; the aggregate live synthetic table is not yet enough to
choose an arm. The decisive question remains the pre-registered paired
comparison after the clean synthetic and standard task strata finish.

### 6.7 Boundary: No Evicted Recall Recovery

Across the current evidence, ValueGraft does not restore facts that exist only
in the evicted context. LongMemEval compacted variants remain near floor for
correctness. Synthetic evicted-fact probes show the same pattern. The
historical summary-only comparison changes whether the model fabricates or
admits missing information. V-only Graft shifts likelihood toward the
full-context continuation. Neither turns a compacted transcript into a
hidden-memory store.

## 7. Discussion

Text-only compaction loses visible information and context-conditioned
computation. Summary text can only preserve what it says. Write-time attention
state can preserve some of the model's previous interpretation of what the text
meant, how confident it was, or what trajectory it was following.

That state is not a retrieval mechanism. It does not recover arbitrary deleted
facts. It matters most when the compacted context still contains the relevant
surface text or summary tokens, but re-encoding them from scratch loses how
they were understood in the original context.

The observed effects split by outcome. In the historical summary-only
comparison, write-time summary KV mostly affects honesty: summary tokens written
under full context may carry a signal about the extent and uncertainty of what
was actually discussed. In the current production-shaped experiments, V-only
Graft mostly affects continuation likelihood: values at aligned tokens can
nudge the compacted trajectory back toward the full-context trajectory without
changing the visible transcript.

The coding-agent setting is the strongest motivation for this work. Agents do
not merely answer one fact question after compaction; they continue a process:
avoid repeating failed approaches, preserve constraints, remember what kind of
work is in progress, and make the next action coherent with a long exploration
history. The offline SWE-Gym result and early end-to-end runs are fragile
proxies, but they test the right failure surface.

The current standard-task pivot strengthens the design. Synthetic tasks make
the dependency structure controllable; standard tasks test whether the same
mechanism survives contact with real repositories and real issue statements.
The pre-stated fit criteria are essential: a standard task only tests
compaction if the run crosses the compaction boundary and the answer-relevant
problem statement falls outside the retained tail.

## 8. Limitations

The evidence is narrow. Most results come from one model family. Local 4B runs
and cloud 30B runs use different precision regimes. The strongest coding result
so far is offline next-action likelihood, not end-to-end task success. The
end-to-end coding-agent matrix is still in progress and has already required
strict quarantine of contaminated runs.

The historical summary-only result has a narrow interpretation. Within that
old pair, context shape is controlled. Relative to the current design, context
shape is different: no retained tail, different token positions, and coupled K
and V changes. Its paper-facing role is auxiliary evidence about write-time
summary encoding, not evidence for K-only Graft, V-only Graft, or KV-Graft in
the production-shaped compacted context. A clean K-only or KV-Graft comparison
should hold summary text, tail, context shape, positions, values, prompts, and
decoding fixed while varying only `alpha_K`, or only the intended pair of
parameters.

The alpha policies are not universal. The best 4B setting is not the best 30B
setting, and per-slot tuning is vulnerable to selection artifacts. The safer
claim is that model-specific calibration is required, not that one scalar or
one head pattern transfers.

Finally, provider API parallels should be handled carefully. Opaque compaction
items, reasoning-state artifacts, and server-managed context are natural
deployment surfaces for a technique like this. They are not evidence that any
provider uses this mechanism internally.

## 9. Conclusion

If the project stopped at this evidence frontier, the defensible conclusion
would be:

ValueGraft does not make compacted models remember deleted facts. It shows that
write-time cached attention state can reduce some behavioral harm from
compaction. The historical summary-only comparison reduces fabrication in
agentic frames, partly through context shape and partly through write-time
summary KV. V-only Graft recovers a small but consistent fraction of
continuation and coding-trajectory likelihood lost to text-only compaction.
Early end-to-end coding-agent results are encouraging enough to continue, but
not yet enough to claim task-success improvement.

The practical lesson is to treat compaction as more than a text summarization
problem. The visible summary is important, but the state written while
generating and interpreting that summary may also be worth preserving.

## Code Availability

Code, experiment scripts, draft analysis, and reproducibility notes are
available at <https://github.com/jeremyBanks/ValueGraft>.

## References

- Anthropic. n.d. [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction), [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows), and [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). Claude Platform Docs. Accessed 2026-07-06.
- Cao, Ziyi, Qingyi Si, Jingbin Zhang, and Bingquan Liu. 2025. [Sparse Attention across Multiple-context KV Cache](https://arxiv.org/abs/2508.11661). arXiv:2508.11661. DOI: [10.48550/arXiv.2508.11661](https://doi.org/10.48550/arXiv.2508.11661).
- Chevalier, Alexis, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. [Adapting Language Models to Compress Contexts](https://aclanthology.org/2023.emnlp-main.232/). In *Proceedings of EMNLP 2023*, pages 3829-3846. DOI: [10.18653/v1/2023.emnlp-main.232](https://doi.org/10.18653/v1/2023.emnlp-main.232).
- Cim, Musa, Burak Topcu, Chita Das, and Mahmut Taylan Kandemir. 2026. [Parallel Context Compaction for Long-Horizon LLM Agent Serving](https://arxiv.org/abs/2605.23296). arXiv:2605.23296. DOI: [10.48550/arXiv.2605.23296](https://doi.org/10.48550/arXiv.2605.23296).
- Eyuboglu, Sabri, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, and Christopher Re. 2025. [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266). arXiv:2506.06266. DOI: [10.48550/arXiv.2506.06266](https://doi.org/10.48550/arXiv.2506.06266).
- Ge, Tao, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. [In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945). ICLR 2024; arXiv:2307.06945. DOI: [10.48550/arXiv.2307.06945](https://doi.org/10.48550/arXiv.2307.06945).
- Google AI for Developers. n.d. [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking), [Context caching](https://ai.google.dev/gemini-api/docs/caching), and [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview). Accessed 2026-07-06.
- Jiang, Huiqiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736). EMNLP 2023; arXiv:2310.05736. DOI: [10.48550/arXiv.2310.05736](https://doi.org/10.48550/arXiv.2310.05736).
- Kim, Jang-Hyun, Junyoung Yeom, Sangdoo Yun, and Hyun Oh Song. 2024. [Compressed Context Memory For Online Language Model Interaction](https://arxiv.org/abs/2312.03414). ICLR 2024; arXiv:2312.03414. DOI: [10.48550/arXiv.2312.03414](https://doi.org/10.48550/arXiv.2312.03414).
- Li, Bojie. 2026. [Models Take Notes at Prefill: KV Cache Can Be Editable and Composable](https://arxiv.org/abs/2606.17107). arXiv:2606.17107. DOI: [10.48550/arXiv.2606.17107](https://doi.org/10.48550/arXiv.2606.17107).
- Mu, Jesse, Xiang Lisa Li, and Noah Goodman. 2023. [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467). NeurIPS 2023; arXiv:2304.08467. DOI: [10.48550/arXiv.2304.08467](https://doi.org/10.48550/arXiv.2304.08467).
- OpenAI. n.d. [Compact a response](https://platform.openai.com/docs/api-reference/responses/compact), [Conversation state](https://platform.openai.com/docs/guides/conversation-state), and [Prompt caching](https://platform.openai.com/docs/guides/prompt-caching). OpenAI API documentation. Accessed 2026-07-06.
- Packer, Charles, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560). arXiv:2310.08560. DOI: [10.48550/arXiv.2310.08560](https://doi.org/10.48550/arXiv.2310.08560).
- Xu, Fangyuan, Weijia Shi, and Eunsol Choi. 2024. [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408). ICLR 2024; arXiv:2310.04408. DOI: [10.48550/arXiv.2310.04408](https://doi.org/10.48550/arXiv.2310.04408).
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

## Notes for Later Revision

- Replace the live synthetic coding table once the clean run and standard-task
  strata finish.
- Add the SWE-bench-Lite fit verdict once `sc_debug` compaction criteria and
  first standard B/A rows are available.
- If K-only or full KV-Graft runs land, revise the method taxonomy from
  "planned" to "evaluated" and report them separately from historical
  summary-only arms.
- Keep quarantined night-matrix data out of headline results except as a data
  hygiene incident.
