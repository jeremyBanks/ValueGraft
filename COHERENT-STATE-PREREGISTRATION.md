# Preregistration: coherent summary state across a compaction boundary

**Frozen:** 2026-07-11, before implementation produced treatment results and before paid execution.

**Scientific leads:** Sol (OpenAI GPT-5.6, extra-high) and Claude Opus 4.8. The design synthesizes the independently reviewed plan in [notes/2026071154-sol-claude-research-plan-dialogue.md](notes/2026071154-sol-claude-research-plan-dialogue.md), the final apparatus audit in [notes/2026071155-sol-final-paper-review.md](notes/2026071155-sol-final-paper-review.md), and Opus's outcome matrix in [notes/2026071157-fable-outcome-interpretation-matrix.md](notes/2026071157-fable-outcome-interpretation-matrix.md).

This document is additive-only after the freeze. Any correction or deviation must be dated, committed before the affected outcome is inspected, and reported in the final paper.

## 1. Question and scope

This is a bounded mechanistic experiment, not a deployment or task-success study.

It asks:

1. Do the K/V states created while a model produces a summary carry downstream-useful information about the evicted history that is absent when the identical summary text is freshly encoded?
2. Is any such effect specific to the correct history rather than generic cache coherence?
3. If a history-specific channel exists, does the repository's training-free old-value/fresh-key intervention exploit it, or does separating the learned K/V representation destroy it?

The primary intervention region is the **summary only**. Retained-tail transplantation is outside this preregistration.

## 2. Model and runtime

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Required resolved Hugging Face revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Required dtype: live parameter tensors must read back as `torch.bfloat16`.
- Required geometry: 48 layers, 32 attention heads, 4 KV heads, head dimension 128, RoPE theta 10,000,000.
- No quantization, CPU/disk offload, `device_map=auto`, alternate checkpoint, or thinking-family substitute is allowed.
- Summary and conversation generation are greedy/temperature 0. Conversation replies are capped at 320 tokens; summaries are capped at 900 tokens. Reaching the cap without EOS is a technical failure, not a truncated valid render.

The exact code commit, clean/dirty status, libraries, GPU, tokenizer/chat-template hashes, and resolved model metadata must be recorded at launch. A null code commit or wrong revision fails closed.

## 3. Corpus and frozen order

The first twelve scenarios in `data/scenarios.json` are used. Conversation bodies are generated anew by the pinned bf16 checkpoint so body provenance is exact. Previously banked Qwen-native bodies may be used for local development but are not the primary paid result because their resolved model revision was not recorded.

Order was frozen by ascending SHA-256 of `20260711:<scenario-id>`:

| Position | Scenario | Stage | Wrong-history donor |
|---:|---|---|---|
| 1 | c10 | first six | c02 |
| 2 | c02 | first six | c01 |
| 3 | c01 | first six | c04 |
| 4 | c04 | first six | c07 |
| 5 | c07 | first six | c11 |
| 6 | c11 | first six | c10 |
| 7 | c05 | extension | c09 |
| 8 | c09 | extension | c06 |
| 9 | c06 | extension | c12 |
| 10 | c12 | extension | c08 |
| 11 | c08 | extension | c03 |
| 12 | c03 | extension | c05 |

The donor mapping is a fixed cyclic derangement within each six-conversation block. It is not optimized after renders or outcomes are observed.

For each scenario, the first `referent` plant and first `sense` plant in the frozen scaffold are the primary natural probes. Their correct targets come from the committed scaffold. Matched counterfactual targets must be authored by the non-subject scientific collaborator and committed before treatment scoring. No target may be revised after an arm result is inspected.

No unfavorable conversation or plant is replaced. Infrastructure may resume the same scenario from a saved checkpoint; it may not substitute a new scenario.

## 4. Exact visible text and compaction assay

The subject model generates one summary token sequence `S` under the correct full target history and the single realistic `SUMMARY_REQUEST`. The exact generated token IDs are the only summary IDs used in every arm.

The fresh destination context is a deliberately clean **mechanistic compaction layout**:

```text
[target system]
[user: exact SUMMARY_REQUEST]
[assistant: exact S, no added preamble]
[verbatim target retained tail]
[user: downstream probe]
```

The legacy `[Context note] ...` preamble is not used because it can change the first summary token through BPE boundary merging. The exact generated `S` IDs must appear as one contiguous assistant-content span in the destination. No difflib or fuzzy alignment is permitted.

This layout is chosen to hold the request wording, role boundary, visible summary text, destination positions, target tail, and downstream probe fixed. It is a mechanistic assay, not a claim that this wrapper exactly reproduces a particular production client.

## 5. Source-state construction

All summary tokens are appended one token at a time through one common incremental function. Generated, replayed, fresh, and wrong-history paths may differ in conditioning prefix but not in the summary-token forward kernel.

### Correct source

The pinned model processes the full target native conversation plus the exact summary request, then greedily generates `S`. The cache actually mutated during this generation is the source of record. Only the summary rows are sliced and retained.

An independent correct-prefix replay teacher-forces the exact `S` IDs through the same one-token path. Actual generation and replay must match under the technical tolerances in section 9.

### Fresh source

The target system plus exact summary request is processed without the evicted history. The exact `S` IDs are counterfactually teacher-forced one token at a time. The target tail is appended only after the summary, as it is in the frozen destination layout.

### Wrong-history source

The target system and target retained tail are kept, while the target evicted-history block is replaced by the frozen donor's corresponding evicted block. The exact target `S` IDs are counterfactually teacher-forced after that wrong history.

This arm is intentionally counterfactual: the wrong history would not naturally produce the target summary text. It exists to isolate whether the retained summary state is tied to the correct history. Its summary-token negative log-likelihood is recorded so teacher-forced surprisal remains visible as an alternative explanation.

### Position handling

Correct- and wrong-source keys are post-RoPE at their source positions. Before insertion, each key is re-rotated by the exact destination-minus-source position delta. K re-rotation must pass the production ladder for every layer. Values are copied without positional transformation.

## 6. Cache states

Every scored compacted arm has the same destination token IDs, summary positions, system, tail, probe, and next-token positions. Only the summary K/V rows differ.

- `A_full`: full target history remains available through the same summary exchange and downstream probe. It is the competence/headroom reference, not a treatment.
- `F_fresh`: fresh destination cache, with exact `S` stepwise-encoded without evicted history.
- `C_coherent`: `F_fresh` with both summary K and V replaced by the correct incremental source rows; K is re-rotated to destination positions.
- `W_wrong`: `F_fresh` with both summary K and V replaced by wrong-history incremental source rows; K is re-rotated.
- `V_only`: correct-source V at summary positions with fresh K, alpha 1.
- `K_only`: correct-source re-rotated K at summary positions with fresh V, alpha 1.
- `D_delta`: fresh K and `V_fresh + permute(V_correct - V_fresh)` at summary positions.

The `D_delta` permutation is fixed-seed, fixed-point-free whenever at least two eligible positions exist, and applied within each layer/head over the same eligible summary rows. It preserves the exact multiset of treatment-delta rows and therefore row norms, mean, and covariance under permutation. The permutation and invariant checks are persisted.

Arms are constructed and scored sequentially so only one full working cache plus bounded summary slices is live.

## 7. Downstream outcome

The summary-token likelihood is **never an outcome**. It is only a provenance/surprisal diagnostic.

For each frozen plant `j` in conversation `i` and cache state `c`, strictly after the summary and retained tail, score:

```text
M_ijc = mean_logprob(correct_target_ij) - mean_logprob(counterfactual_target_ij)
```

Both answer targets are teacher-forced after the same rendered user probe. The conversation outcome is the equal-weight mean over its two frozen plants:

```text
Y_ic = mean_j(M_ijc)
```

Raw correct-target mean log-probability is a sensitivity outcome. Per-category and per-plant values are descriptive only.

The independent sampling unit is the conversation. Plants within a conversation are never treated as independent replicates.

## 8. Estimands

### Co-primary intersection claim

1. `theta_CF = mean_i(Y_i,C - Y_i,F)`: usable coherent state versus identical visible text freshly encoded.
2. `theta_CW = mean_i(Y_i,C - Y_i,W)`: specificity to the correct evicted history.

A **history-specific coherent summary-state channel** is claimed only if both final two-sided 95% confidence intervals have lower bounds above zero.

### Secondary/exploratory mechanism contrasts

- `theta_VF = mean_i(Y_i,V - Y_i,F)`: value-only benefit.
- `theta_KF = mean_i(Y_i,K - Y_i,F)`: key-only benefit.
- `theta_CV = mean_i(Y_i,C - Y_i,V)`: loss from the old-V/fresh-K split.
- `theta_CK = mean_i(Y_i,C - Y_i,K)`: loss from the old-K/fresh-V split.
- `theta_VD = mean_i(Y_i,V - Y_i,D)`: value-only treatment versus delta-matched perturbation.
- `theta_AF = mean_i(Y_i,A - Y_i,F)`: information/headroom available to recover.

These receive pointwise intervals and are labeled exploratory. No isolated subgroup or secondary interval becomes a confirmatory headline.

## 9. Fail-closed technical gates

All gates run before semantic arm results are trusted:

1. Exact model revision, live bf16 dtype, geometry, GPU residency, and clean/non-null code provenance.
2. Actual incremental generation source recorded as `generated_incremental`; reconstructed batched prefill cannot substitute.
3. Exact summary IDs, request hash, destination span, role boundary, positions, cache shapes, and exact contiguous coverage.
4. No destination outside the summary content span; no tail or wrapper token may be transplanted.
5. Correct actual-generation versus correct stepwise-replay agreement: identical generated IDs and per-token next-logprob maximum absolute difference at most `1e-4`; per-layer summary K/V maximum differences are persisted and must be at most `1e-4` unless the production ladder freezes a stricter observed kernel floor before paid outcomes.
6. Fresh self-replacement and alpha-zero/self-transplant are tokenwise no-ops within `1e-4`.
7. Key rotation zero/round-trip/native-shift identities pass across every layer within the frozen production tolerance.
8. Delta placebo has no fixed points, uses only eligible summary rows, and exactly preserves the treatment-delta row multiset and moment diagnostics.
9. Summary generation ended normally before the 900-token cap.
10. Required per-conversation render, exact IDs, summary text, hashes, traces, arm scores, diagnostics, and manifest are atomically persisted before the next conversation.
11. Interruption after one conversation resumes without rerendering or numerically changing the first checkpoint.

Any failure makes the affected run `VOID_TECHNICAL`. It is preserved and reported, not silently excluded.

## 10. Scientific regime gates and controls

### Competence and headroom at six

- Full history must favor the correct target (`Y_A > 0`) in at least four of six conversations and in the equal-conversation mean.
- `Y_A - Y_F` must be positive in at least four of six conversations and mean headroom must be at least 0.30 nats/token.

Failure means the corpus/regime cannot test recovery; it is not evidence against a state channel.

### Technical positive/negative controls at $0

The Qwen3-0.6B ladder includes:

- an engineered cache-delta positive control proving the transplant/scoring path can move a frozen downstream target in the prescribed direction;
- a fresh self-replacement and irrelevant-history negative control proving the apparatus stays null when no state difference is introduced;
- generated-versus-replay identity, K re-rotation identity, exact-span coverage, and artifact-resume controls.

These validate the machinery. They do not assume an untrained 0.6B model naturally writes an omitted fact into summary K/V.

### Natural channel calibration

Each paid conversation also carries a separate calibration item excluded from primary estimates: the history establishes one of two labels, while an identical forced mini-summary sentence is deliberately ambiguous. The calibration probes the selected label after coherent, fresh, and wrong-history state construction.

It “fires” at six only if mean `C-F` and `C-W` are positive and at least four of six conversations show both directions. Failure does not void an otherwise technically valid run; it enters the futility rule and limits sensitivity claims.

## 11. Serial 6→12 rule

After the first six frozen conversations:

1. Stop immediately for any technical, competence, headroom, artifact, wrong-model, or spend gate failure.
2. Stop for futility only if all three hold:
   - mean primary `C-F <= 0`;
   - mean primary `C-W <= 0`;
   - the natural channel calibration does not fire.
3. Otherwise run the frozen remaining six.
4. Stop at twelve regardless of result. No adaptive model, request, wrapper, scenario, donor, target, tail, position, layer, head, alpha, metric, or analysis change is permitted.

A six-conversation futility stop is an inconclusive bounded canary, not equivalence or proof of absence.

## 12. Uncertainty and multiplicity

- At six, intervals are descriptive; there is no early efficacy declaration.
- At the final N, compute one contrast per conversation and report equal-conversation means with two-sided 95% Student-`t` intervals as the primary uncertainty summary.
- Also report 10,000-resample conversation bootstrap sensitivity intervals with seed `20260711`.
- The channel conclusion is an intersection-union claim: both `theta_CF` and `theta_CW` must independently clear zero. No additional multiplicity correction is required for that joint claim.
- All mechanism, calibration, category, raw-likelihood, and per-plant analyses are exploratory and receive pointwise intervals.
- An interval spanning zero is “inconclusive,” never “zero” or equivalent.
- No smallest effect size or equivalence margin is preregistered; the scale lacks a defensible externally calibrated threshold at this sample size.

## 13. Interpretation matrix

| Final pattern | Licensed interpretation |
|---|---|
| Technical gate fails | Void apparatus run; no substantive inference. |
| Competence/headroom fails | Regime cannot test recovery; no channel inference. |
| Both channel contrasts clear | Correct-history coherent summary state carries a downstream-useful, history-specific channel. |
| `C-F` clears but `C-W` does not | State differs from restart, but correct-history specificity is unestablished; generic coherence/surprisal remains plausible. |
| Channel clears; `V-F` and `V-D` clear | Value-only transplantation exploits part of the channel and beats a geometry-matched perturbation. |
| Channel clears; `V-F` does not; `C-V > 0` | A channel exists, but old-V/fresh-K splitting loses it; K/V coherence is the leading localized limitation. |
| Channel clears; `K-F > 0`, `V-F` does not | Keys/addressing contribute more than the value-only design permits. |
| `V-F > 0` but `V-D` does not | Improvement is not distinguishable from a matched perturbation; no content-specific claim. |
| Both channel means nonpositive while calibration fires | No detected natural-summary channel in this bounded model/regime; assay sensitivity was demonstrated, but intervals govern strength. |
| Both channel means nonpositive and calibration fails | Futility/inadequate sensitivity; no substantive channel conclusion. |
| `V` appears positive without a coherent channel | Suspect geometric/artifactual behavior; replication required before interpretation. |

## 14. Artifacts and reporting

Every expensive render is saved. Per conversation, commit:

- full model-native messages and reply-generation metadata;
- exact request and summary text/token IDs;
- source/destination spans and prefix hashes;
- actual/replay summary-row hashes and bounded numerical diagnostics;
- wrong-history donor and wrong-summary surprisal;
- delta permutation and invariant diagnostics;
- all post-summary per-token target scores;
- arm definitions, model/config/runtime provenance, exclusions/gates, and elapsed time.

Full cache tensors need not be committed if exact replay inputs plus per-layer hashes prove equivalence; unbounded full-cache persistence is prohibited. Summary slices are bounded by 900 tokens and one conversation is live at a time.

All scored results land in `results/` and are committed, including partial, void, or stopped runs. Paid pods are terminated before analysis or writing. The final paper must distinguish the new mechanistic wrapper from the older production-like layout and must report every deviation from this preregistration.

## 15. Pre-outcome implementation clarifications (2026-07-11)

These details were fixed after the initial freeze but before any subject-model arm
result existed:

1. The deterministic native-conversation structural shuffle seed is `20260711`;
   decoding remains greedy, so the seed affects scaffold turn order rather than
   sampled token choice.
2. The independent scientific collaborator froze the `correct` targets as concise,
   faithful answer phrases derived from each scaffold `gold`, not byte-for-byte
   copies of the scaffold's sometimes explanatory prose. The target file records
   that derivation and its immutable hash is part of the run fingerprint. The
   harness additionally carries the exact scaffold gold and its SHA-256 alongside
   each target's provenance. This wording choice was made before any arm result to
   better match the counterfactual target's register and length; neither member of
   a pair may now change.
3. The natural calibration uses labels A/B assigned by SHA-256 of
   `20260711:<conversation-id>:calibration`. Correct and wrong sources state
   opposite labels; fresh states neither. All force the exact rendered sentence
   “The recorded choice remains the approved one.” and score the same downstream
   exact-answer probe. It is excluded from the primary estimates as specified.
