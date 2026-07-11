# Final independent review of the published README paper

**Reviewer:** Sol — OpenAI GPT-5.6, extra-high reasoning effort

**Date:** 2026-07-11

**Paper reviewed:** `README.md` at commit `1d64086cab410365e564e223486428828c7dc893`, blob `87f2a8abfe9ccfb5b6d7dd571caa03039e20ed28`

This review was performed against the published README, stable source code, committed result artifacts, and the repository's explicit methods/provenance requirements. I did not use the paper-writing agent's working notes or draft-review record. Two independent subreviews were used as adversarial checks; I verified their material claims directly against the repository before including them here.

## Bottom line

The paper is exceptionally good as technical prose and unusually candid about negative results. It is also **not scientifically ready for external sharing in its current form**.

The main problem is not a minor statistical disagreement. The paper does not accurately describe the apparatus behind several headline claims:

1. Every held-out synthetic conversation body was foreign-rendered, despite the paper's categorical statement that the Qwen3-30B test model generated its own replies.
2. Three of the four compression-sweep conditions reconstructed the supposed write-time cache under the wrong summary request.
3. The intervention grafted values across both the retained tail and the summary, despite being repeatedly defined as a summary-token intervention.
4. The primary synthetic harness reconstructed the old summary state with a new prefill instead of retaining the actual cache produced during summary generation.

Those are blocking apparatus/provenance errors. They do not make every recorded number meaningless. They change what the numbers answer.

The strongest defensible result is narrower than the paper's current conclusion:

- In the actual **foreign-rendered, combined tail-plus-summary, prefill-reconstructed** synthetic apparatus, four selected graft variants showed no detectable average lift over plain compaction on 18 conversation clusters. The intervals still permit modest positive effects.
- On SWE-Gym-derived trajectories, a coding-selected layer map showed a small positive teacher-forced next-action likelihood effect on the wholly out-of-fitting 102-trajectory set. That is a real proxy result. It has no matched placebo for the selected map, no out-of-sample behavioral corroboration, no task-success evaluation, and only 45 of the planned original 75 confirmation trajectories were completed.
- The four-level compression conclusion is not currently usable.
- The placebo results show that aligned old values are less disruptive than the tested mismatched values. They do not yet isolate semantic or content-specific information.

The paper should be corrected, not discarded. Its honest negative-result framing, the fixed-graft sample heterogeneity, and the process audit are worth preserving.

## What the evidence does support

### 1. The primary held-out point estimates are real measurements of the apparatus that ran

The four synthetic recovery estimates and conversation-clustered intervals in the paper match the committed artifacts. All four intervals span zero. The alpha-zero and identity checks also passed for the headline cells. It is fair to report **no detected average recovery benefit in that measured apparatus**.

It is not fair to call this a definitive null, an equivalence result, or a complete bound. The upper confidence limits are approximately +0.03 to +0.06 nats/token, and no smallest effect size of interest or equivalence margin was specified. “No detectable lift under these conditions” is the appropriate wording.

### 2. The fixed SWE graft is genuinely sample-heterogeneous

The paper handles this part well. The flat alpha graft is positive on the original 75 and absent on the fresh 98, with a pooled interval spanning zero. That is a useful failure of replication, reported instead of hidden.

The two pools are index-defined slices of one local parquet rather than randomized samples from a clearly documented population, so the between-pool difference should be described as dataset/pool heterogeneity, not necessarily a change in the underlying causal effect.

### 3. The selected SWE layer map has a credible continuous proxy signal

The cleanest positive is not the in-sample 98-trajectory or 143-trajectory aggregate. It is the wholly out-of-fitting set:

- fresh evaluation split: 57 trajectories, selected-map minus baseline **+0.0117**, 95% bootstrap CI **[+0.0063, +0.0172]**;
- separate original-pool confirmation: 45 trajectories, reported selected-map minus baseline **+0.0158**, CI above zero;
- pooled out-of-fitting set: 102 trajectories, **+0.0135 [+0.0083, +0.0190]**.

I reproduced the fresh-split result with:

```text
uv run python src/analyze_swegym_tune.py \
  --run results/swegym_tune_20260710T145330Z_brief \
  --champion-eval results/swegym_champeval_20260710T145330Z_brief
```

This supports: **a layer map selected on one coding subset improved teacher-forced likelihood of one demonstrated next action on two out-of-fitting subsets under brief summaries.** It does not support task improvement, uniqueness/correctness of the demonstrated action, or a general coding-agent benefit.

### 4. The process postmortem is valuable

Part II is the most publishable part of the current paper. The precision/provenance incident is concrete and transferable. The paper also deserves credit for exposing adverse cross-architecture results, the fresh-pool failure of the fixed graft, and the proxy nature of SWE scoring.

## Blocking findings

### P0 — Held-out synthetic conversations were not generated by the test model

The paper says:

> “The assistant replies in the conversation body are generated in-context by the test model itself… We do not reuse replies written by another model.”

That statement at [README.md:63](../README.md#L63) is false for every held-out conversation c07–c24.

- The committed headline checkpoints record `native_render: false`.
- c07–c12 identify `mlx-community/Qwen3-4B-Instruct-2507-4bit` as the body renderer; for example [c07](../results/champion_validate/_work_bf16/Qwen__Qwen3-30B-A3B-Instruct-2507/conv_000__c07.json).
- c13–c24 identify external assistant authors such as Sonnet, Opus, and Fable; for example [c13](../results/champion_validate/_work_bf16/Qwen__Qwen3-30B-A3B-Instruct-2507/conv_006__c13.json).
- The source files themselves show the same provenance: [c07](../data/synthetic/c07.json) through [c24](../data/synthetic/c24.json).

The 30B subject did generate the compaction summaries, but it summarized conversation bodies written by other models. This directly contradicts the repository's blocking requirement that foreign replies be disclosed ([METHODS-PROVENANCE-REQUIREMENTS.md:19](../METHODS-PROVENANCE-REQUIREMENTS.md#L19)).

**Consequence:** the synthetic null remains a result about the stored corpus, but the paper cannot call it a per-model-native evaluation or use it to support the claimed “own conversation plus own summary” mechanism. Claims that foreign replies collapse the effect are especially incompatible with presenting a foreign-reply held-out corpus as the definitive test.

**Required correction:** either describe the actual corpus loudly and narrow the inference, or rerender the held-out scaffolds natively with Qwen3-30B and rerun. The existing result should remain in the record as the foreign-body condition.

### P0 — The compression sweep reconstructed three levels under the wrong request

The level-specific request is resolved into local `_REQ` at [cross_arch_probe.py:1886–1891](../src/cross_arch_probe.py#L1886-L1891) and used to generate the summary at [cross_arch_probe.py:2722–2723](../src/cross_arch_probe.py#L2722-L2723).

But `summary_token_layout()` reconstructs `req_ids` with the module-level `SUMMARY_REQUEST`, not the selected `_REQ`, at [cross_arch_probe.py:1323–1334](../src/cross_arch_probe.py#L1323-L1334). The reconstructed old cache is then created from those IDs at [cross_arch_probe.py:2729–2734](../src/cross_arch_probe.py#L2729-L2734).

Therefore:

- `realistic` is internally consistent because it uses the default request;
- `ultra`, `brief`, and `medium` generate text after one request but construct the supposed write-time state after another.

The manifests correctly distinguish the request hashes, but the cache reconstruction does not follow them.

**Consequence:** the compression ratios and full-versus-compacted gaps still describe the generated texts, but the treatment in three cells is not the write-time state for those generations. The paper's centerpiece claim that recovery is flat across a 30-fold compression range, and its statement that this refutes the compression-severity hypothesis, are unsupported.

**Required correction:** fix `summary_token_layout()` to accept the selected request, add an assertion that reconstructed `req_ids` equal the generation prefix, pass the build ladder, and rerun all four cells. Report a direct recovery-by-compression slope or interaction interval rather than inferring flatness from four overlapping intervals.

### P1 — The synthetic “write-time cache” was reconstructed, not retained

`generate_summary_hf()` returns the actual generation-mutated snapshot at [arms_hf.py:53–74](../src/arms_hf.py#L53-L74). The SWE harness uses it directly at [run_swegym_hf.py:403–418](../src/run_swegym_hf.py#L403-L418).

The main synthetic harness discards that snapshot, keeps only summary text, rebuilds `old_ids`, and calls `force_prefill()` at [cross_arch_probe.py:2722–2734](../src/cross_arch_probe.py#L2722-L2734). Thus the synthetic source is a teacher-forced prefill reconstruction of the old layout, not literally the incremental cache saved while the summary was generated.

With an identical token sequence and numerically equivalent kernels this may be a valid surrogate, but the paper claims literal preservation and does not report a generation-snapshot versus reconstructed-prefill equivalence test. The repository itself warns that batched and stepwise kernels must not be assumed identical.

**Consequence:** synthetic and SWE experiments use materially different source-state procedures, and the central phrase “original write-time values” is overstated for the synthetic results. The compression bug makes this more than terminological in three conditions.

**Required correction:** preserve and commit the actual generation snapshot or establish tokenwise/logit equivalence on the production configuration. Describe the two procedures separately until then.

### P1 — The intervention is combined summary plus tail, not summary-only

The abstract, method definition, and glossary repeatedly say values are replaced “at summary-token positions” ([README.md:9](../README.md#L9), [README.md:43](../README.md#L43), [README.md:237](../README.md#L237)). The code constructs two graft regions:

- retained tail;
- summary.

See [cross_arch_probe.py:1366–1383](../src/cross_arch_probe.py#L1366-L1383) and [run_swegym_hf.py:397–404](../src/run_swegym_hf.py#L397-L404). The manifests also record `tail+summary regions`.

README line 55 partially reveals the two regions, but the rest of the paper interprets the result as if it isolated summary-token state.

**Consequence:** neither the synthetic null nor the SWE positive can be attributed specifically to summary values. On coding trajectories, retained-tail values may encode paths, tool outputs, and procedural state. A combined intervention can also conceal opposing summary and tail effects.

**Required correction:** rename the measured intervention everywhere and run summary-only, tail-only, and combined arms before making a mechanistic statement about summary state.

### P1 — The selected SWE positive lacks its own matched placebo

The paper says “every graft is shadowed by a placebo” at [README.md:99](../README.md#L99). The selected-map evaluation did not do this.

- The champion evaluation explicitly disabled SWE placebos at [job_swegym_tune_bf16.sh:103–104](../scripts/job_swegym_tune_bf16.sh#L103-L104).
- Its [manifest](../results/swegym_champeval_20260710T145330Z_brief/manifest.json) records `intervention.placebo: null`.
- The confirmation run contains scalar-alpha placebo arms, not a position shuffle restricted to the selected map's layer set.

**Consequence:** the selected map has a baseline comparison but no matched alignment/content control. The paper cannot say its sole positive “survived its controls” in the same sense as the synthetic variants.

**Required correction:** score a selected-map-matched position placebo and, preferably, a wrong-history same-text and treatment-delta-matched placebo using saved renders.

### P1 — The SWE presentation mixes selection and evaluation rows

The fresh 98 contains 41 fitting and 57 evaluation trajectories. The paper leads with +0.0111 on all 98 and +0.0126 on 143 rows, both of which include the 41 rows used to choose the layer bands. It later gives the clean 102-row out-of-fitting estimate, but that should be the headline.

More importantly, the paper says the tuned graft beats the fixed graft “precisely where the naive graft does not.” On the actual held-out fresh 57:

- selected map minus baseline: +0.0117, CI above zero;
- selected map minus fixed graft: +0.0041, 95% CI [−0.0047, +0.0129].

The selected map beats the baseline out of sample; it does **not** reliably beat the fixed graft out of sample. The favorable +0.0128 head-to-head estimate over all 98 includes fitting rows.

The original-pool confirmation is also incomplete. The job planned 75 trajectories, but the committed directory contains 45 result files and its [manifest](../results/swegym_confirm_20260711T011101Z_brief/manifest.json) remains unfinalized with `instance_ids: null` and no `n_scored`. The table discloses n=45, while prose elsewhere calls it an original-75 replication.

**Required correction:** reorganize the table by role:

| Split | Role | n | What it can support |
|---|---|---:|---|
| Fresh fitting | selection | 41 | exploratory only |
| Fresh evaluation | internal held-out | 57 | primary selected-map test |
| Original confirmation | external to fitting | 45 | partial external confirmation |

Lead with the 102 out-of-fitting rows for selected-map minus baseline. Report selected-map minus fixed separately and as inconclusive. Explain why the planned 75 stopped at 45.

### P1 — The claimed behavioral corroboration disappears out of sample

The paper reports 53% versus 49% and calls this weak same-direction corroboration at [README.md:168](../README.md#L168). On the fresh held-out evaluation split, the selected map is actually **53% versus 54%** for baseline, paired difference −0.0175 with an interval spanning zero. The favorable all-98 pattern is concentrated in fitting rows.

Across the wholly out-of-fitting 102 rows, the independent audit found three baseline failures fixed and three baseline successes broken: net zero. The continuous likelihood result remains; the claimed behavioral corroboration does not.

The metric is also coarser than “exactly-correct action.” The parser keeps only tool, path, and command at [swegym_action_match.py:33–63](../src/swegym_action_match.py#L33-L63). For `str_replace_editor`, two different edits to the same path with the same operation can match because edit payload fields are ignored. A successful historical trajectory supplies one demonstrated action, not necessarily the uniquely correct action.

**Required correction:** call this a demonstrated-action structural match, show tune/eval separately, and state that the out-of-sample result is null.

### P1 — “Content-specificity” is not isolated by the placebo battery

The tested placebos are useful perturbation controls, but they do not justify the semantic claim made at [README.md:13](../README.md#L13), [README.md:83](../README.md#L83), and [README.md:118–127](../README.md#L118-L127).

- Gaussian values are matched to the norm of the source values, not the norm or covariance of the actual treatment delta `V_old − V_fresh` ([cross_arch_probe.py:383–412](../src/cross_arch_probe.py#L383-L412)).
- Position shuffle substitutes other full source vectors and is not guaranteed to be a derangement ([cross_arch_probe.py:349–380](../src/cross_arch_probe.py#L349-L380)).
- Correct same-token old values may simply be much closer to fresh values than random or wrong-position replacements.

Thus “real graft beats placebo” may mean “the aligned source is less destructive than a grossly mismatched source,” not “evicted semantic content was causally isolated.” The slot-count scaling is especially compatible with accumulating generic mismatch damage.

**What is supported:** old values are alignment-sensitive/non-exchangeable, and the tested aligned graft is less harmful than the tested corrupted grafts.

**What is not yet supported:** the difference is semantic, refers to hidden history, or proves a useful latent continuity channel.

**Required control:** compare same summary tokens generated from the correct history with the same tokens teacher-forced under a wrong or ablated history; match placebo perturbations to the treatment delta per layer/head, with derangements and covariance/norm checks.

### P1 — The born-annotated provenance claim is false

The paper says every result records the code commit and “exact text hash” at [README.md:101](../README.md#L101). Inspected headline manifests record:

```json
"code": {"git_commit": null, "git_dirty": null}
```

Examples include the [brief compression manifest](../results/champion_validate/_work_compsweep/brief/manifest__Qwen__Qwen3-30B-A3B-Instruct-2507.json), [SWE tuning manifest](../results/swegym_tune_20260710T145330Z_brief/manifest.json), and [SWE confirmation manifest](../results/swegym_confirm_20260711T011101Z_brief/manifest.json).

The SWE artifacts hash the summary **request**, not each generated summary. Per-trajectory files record summary token count but not summary text, token IDs, or a generated-summary hash. The synthetic checkpoints do preserve summary text, which is good, but the paper's universal claim is still false.

**Required correction:** replace the claim with a precise artifact-by-artifact inventory and add a reproducibility table mapping every paper row to committed result paths, code state, exact prompts, seeds, and analysis commands.

## Statistical and interpretive corrections

### Do not infer a paired comparison from overlapping marginal intervals

[README.md:116](../README.md#L116) says per-head does not beat per-layer because their individual intervals overlap. That question requires a paired interval for `per-head − per-layer`. The current evidence supports only that neither marginal arm has a detected lift over baseline.

### Do not call confidence intervals spanning zero “zero”

“All land on zero,” “definitive,” “refutes,” “never,” and “bound” repeatedly convert failure to reject into equivalence. Replace them with observed estimates and interval limits. If “bound” remains in the title, define the estimand, target population, and excluded effect magnitude.

### Remove the anti-conservative plant-level result from the narrative

The paper itself admits the plant-level interval in the ultra condition uses the wrong inferential unit, then highlights it anyway. That does not strengthen the evidence. Report the conversation-clustered analysis and treat category slices as exploratory with a multiplicity policy.

### Treat the experiment family as exploratory unless a multiplicity policy is supplied

Four selected synthetic variants, multiple placebos, four compression levels, six categories, alpha searches, region searches, two trajectory pools, and action-match slices create many opportunities for selective emphasis. This does not erase the clean held-out selected-map result, but the paper should distinguish confirmatory estimands from descriptive/exploratory ones and avoid isolated significance language.

### Cluster SWE uncertainty at the scientific sampling unit if needed

Bootstrapping trajectories assumes trajectories are independent. The paper does not report whether multiple trajectories share repositories, tasks, or source episodes. If they do, uncertainty should be clustered at repository/task or shown both ways.

## Reproducibility gaps

The paper promises reconstruction from prose at [README.md:33](../README.md#L33) but still omits material information required by [METHODS-PROVENANCE-REQUIREMENTS.md](../METHODS-PROVENANCE-REQUIREMENTS.md):

- exact system/scaffold prompts and exact synthetic source files;
- correct assistant-reply authorship per conversation;
- generation temperature, seed, maximum tokens, truncation, and cache procedure;
- exact summary requests and decoding configuration;
- validation conversation IDs and counts;
- headroom and task-competence gates, thresholds, and exclusion counts;
- gold-token cap and probe aggregation procedure;
- exact selected layer/head maps and table-to-config mapping;
- placebo seeds and permutation diagnostics;
- bootstrap code, seed, and number of replicates;
- exact commands and result paths for every table;
- SWE parquet origin, immutable revision/content hash, successful-trajectory selection, historical trajectory generator identities, and task/repository duplication;
- software environment and hardware in the paper or a linked manifest guide;
- full checkpoint/config details for the additional architectures cited in section 3.4.

Also, “hand-authored” and “not model-generated” are misleading when directed model assistants wrote the scaffolds and golds. The accurate description is **AI-assisted synthetic authorship under human direction, not generated by the experimental subject model**.

## Literature positioning

The paper has no real related-work section or bibliography, despite making novelty claims. The exact untrained old-value/fresh-key transplant across a summary boundary may indeed be unusual. The broad premise that generation-time or prefill KV state can retain information absent after text-only restart is no longer unestablished.

At minimum, the paper should engage directly with:

- [MEMENTO: Teaching LLMs to Manage Their Own Context](https://arxiv.org/abs/2604.09852) — trained memento models retain an additional KV information channel; restarting from the same memento text causes a large task-accuracy drop. This directly establishes the broad dual-stream premise, while differing in training, full-KV retention, and task.
- [Models Take Notes at Prefill: KV Cache Can Be Editable and Composable](https://arxiv.org/abs/2606.17107) — causal evidence that downstream KV positions can encode conclusions derived from earlier fields; adjacent to the “state as notes” interpretation.
- [CacheBlend](https://arxiv.org/abs/2405.16444) — reuses context-dependent cached chunks and selectively recomputes tokens to restore cross-context interactions; a useful contrast to uncompensated transplantation.
- [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467) — trained latent prompt compression, distinct from this training-free intervention.

The novelty claim should be narrowed to the exact **training-free, post-generation, value-only old-state/fresh-key graft evaluated across a text-summary boundary**, plus the negative evaluation and process audit. “Three searches did not find it” is not a substitute for citations.

## Writing and structure

The writing quality is excellent. The title is honest, the intervention is introduced quickly, the tables are readable, and the distinction between likelihood and task success is repeated appropriately. Part II is unusually lucid.

But polish makes the apparatus errors more dangerous, not less: the prose is confident enough that a reader will not suspect the central provenance statement is reversed.

For an external paper, Part II should be shortened after the scientific sections are repaired. Keep the precision/provenance case study and convert the remaining chronology into a compact failure → consequence → guardrail table. Use the recovered space for:

1. Related Work;
2. exact Data Provenance;
3. causal estimands and alternative explanations;
4. a table-to-artifact reproducibility guide;
5. limitations and future controls.

## The conclusion I would publish after correcting the record

Something close to the following is currently defensible:

> We evaluated a training-free value-cache graft across conversation compaction. In an 18-conversation synthetic evaluation using foreign-rendered conversation bodies and a combined summary-plus-tail intervention whose old state was reconstructed by prefill, four selected graft variants showed no detectable average likelihood improvement over plain compaction. Because three compression-sweep conditions reconstructed a different summary-request prefix, that sweep does not presently answer whether benefit varies with compression severity. On SWE-Gym-derived successful trajectories under brief summaries, a layer map selected on a separate coding subset improved teacher-forced likelihood of the demonstrated next action by about 0.013 nats/token over 102 out-of-fitting trajectories. The selected map was not evaluated against a matched placebo; its structural action-match result was null out of sample; no actions were executed and no repair success was measured. The current evidence therefore does not establish a useful semantic continuity mechanism, but it identifies one narrow coding-likelihood lead and several controls needed to test it.

That is less sweeping than the current README, but more credible and, scientifically, more interesting.

## Minimum path to a trustworthy revision

Before external sharing:

1. Correct all corpus provenance statements and table-to-artifact mappings.
2. Remove the compression-sweep conclusion or rerun it with request-prefix identity assertions.
3. Rename the intervention as summary+tail and distinguish reconstructed-prefill from preserved-generation state.
4. Lead the SWE section with the 57/45/102 out-of-fitting results; separate the 41 fitting rows.
5. Remove the claim that selected-map tuning beats the fixed graft out of sample.
6. Report the out-of-sample action-match null and rename that metric.
7. Downgrade “content-specific” to alignment-sensitive/non-exchangeable pending matched controls.
8. State that the selected SWE map lacks its matched placebo and only 45/75 confirmation rows completed.
9. Replace definitive/equivalence language with interval-bounded descriptive claims.
10. Add a real related-work and reproducibility section.

Then, for the science rather than merely the paper:

1. rerun the corrected four-level compression experiment;
2. run summary-only, tail-only, and combined factorial arms;
3. compare preserved incremental generation snapshots with reconstructed prefill snapshots;
4. add same-text wrong-history and treatment-delta-matched placebos;
5. complete or explicitly close the remaining original SWE confirmation rows;
6. only after those gates, consider a small paired live-agent evaluation with task/test outcomes.

## Final verdict

The repository has produced a meaningful answer, but not yet the answer the README says it produced.

The current evidence argues against a large, robust benefit from the naive graft in the tested synthetic apparatus. It leaves a small, credible coding-likelihood lead for a selected layer map. It does **not** yet establish semantic content specificity, a compression-invariant null, a summary-token mechanism, or practical agent improvement.

I would retain the paper's title and negative-result spirit, correct the apparatus and claims, rerun only the invalid/high-value cells, and treat the SWE signal as a lead that has earned better controls—not as the exception that already survived them.
