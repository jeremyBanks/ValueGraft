# Joint research plan after the independent audit

**Authors:** Sol (`gpt-5.6-sol-xhigh`) and Claude (the planning dialogue
self-reported `claude-fable-5`; its underlying runtime was not independently
attested there). The current execution-gate session is independently identified as
`claude-opus-4-8`.

**Status:** Jointly agreed. This note distills the concluded dialogue in
`2026071154-sol-claude-research-plan-dialogue.md`. It is a plan, not a report of
completed work. Subsequent additive preregistration amendments govern the exact
implementation.

## The decision this program should answer

The highest-value unresolved question is whether an ordinary instruction model's
actual generation-time summary K+V state contains useful history-conditioned
information that is absent when the identical summary text is freshly encoded at
the same positions. This channel question must be answered before spending more
money optimizing value-only grafts or claiming practical benefit.

The existing results do not cleanly answer it. In particular, the earlier headline
harness regenerated the summary token sequence and then reconstructed its state by
prefill. It did not retain the exact incremental K+V state that existed while the
summary was generated. Those results remain useful diagnostics, but they are not a
test of the intended generation-state intervention.

## Apparatus gates

Before semantic scoring or paid interpretation, the apparatus must fail closed on
all of the following:

1. Fresh and coherent states use a controlled stepwise path with tokenwise,
   per-layer identity evidence.
2. The primary estimand is confined to the summary region: actual coherent
   correct-history K+V minus fresh same-text K+V. Retained-tail effects are a
   separate position/RoPE question.
3. Correct-history and wrong-history specificity arms use exactly the same summary
   token IDs, positions, continuation, and destination context.
4. The exact incremental generation state is the source of record, with the source
   prefix, summary IDs, positions, component hashes, and branch lineage preserved.
   A reconstructed state may only be used after a separately validated bit-exact
   replay, and must remain explicitly labeled as such.
5. The production checkpoint, revision, dtype, eager attention backend, tokenizer,
   chat-template behavior, causal mask, physical/logical positions, EOS termination,
   and complete technical result are independently reconstructed and validated.
6. Every generated render and summary is persisted before downstream scoring.

The current implementation freezes these requirements additively through
`COHERENT-STATE-PREREGISTRATION-AMENDMENT-10.md`; its exact v10 validators and
artifacts, rather than this prose summary, control execution.

## Budget ladder

### $0: local closure and reuse

Use the 32 GB Apple-Silicon MacBook for apparatus work, not for a citable 4-bit
surrogate of the 30B bf16 experiment.

- Complete the full 0.6B identity/build ladder on the exact v10 code and validate
  production-tokenizer donors.
- Exercise adversarial validator counterexamples, the monitor self-test, and the
  static provenance/input inventory gates.
- Rescore already saved SWE free generations using graded action-equivalence
  criteria where the necessary renders already exist.
- Treat any large-model 4-bit smoke run, if engineering requires one, as a
  non-estimating shape check only.

### $50: recommended mechanistic experiment

Run one model-native bf16 Qwen3-30B technical gate, then six prespecified semantic
conversations and conditionally the full twelve.

The frozen arm family compares:

- coherent correct-history summary K+V;
- fresh same-text summary K+V;
- same-text wrong-history summary K+V;
- V-only and K-only splits;
- a delta-matched derangement/placebo control.

All exact incremental states and generated text must be preserved. The first paid
run is technical-only: no semantic outcome may be produced until its committed
artifact is independently harvested, validated, and authorized.

### $100: follow-up only after a valid channel gate

- Add the native-versus-foreign summary-body/source factorial needed to investigate
  the prior sign reversal without conflating channel existence with graft utility.
- Complete the frozen SWE-map confirmation and its champion-matched wrong-history
  and delta-matched controls in a single render-saving job.
- Keep the SWE champion labeled diagnostic unless it clears out-of-pool
  confirmation, a matched placebo, and a demonstrated coherent-state channel.

This tier does not support a live-agent treatment-effect claim by itself.

### $200: practical agent evaluation, strongly gated

Proceed only if both a coherent-state channel and one usable treatment clear their
earlier gates.

1. First run a no-treatment capability/damage canary: full context versus ordinary
   compaction on tasks where compaction occurs and deterministic tests can score the
   result. This establishes whether there is enough compaction damage to repair.
2. Freeze one treatment from the mechanistic study; do not tune it on the agent
   benchmark.
3. Run paired forks from identical task/repository/container states and seeds:
   standard compaction versus the frozen treatment, with full-context included when
   feasible as a reference rather than as the randomized treatment.
4. Use repository-level coding tasks with deterministic tests, preserve complete
   transcripts, summaries, patches, test logs, token/tool budgets, and every
   compaction event, and cluster inference by task/repository.
5. Analyze paired task success first. Secondary measures may include regression
   count, recovery after compaction, action-equivalence, token use, and latency, but
   they cannot replace the task-level endpoint.

This is a small practical bounding study, not a high-powered product benchmark. A
null should be reported as no detected benefit under the fixed study, not as proof
of equivalence.

## Sequential stopping rules

- Stop immediately for failed identity, prefix, source-state capture, causal-mask,
  provenance, competence/headroom, or independent-harvest gates.
- At six semantic conversations, stop for futility only if both prespecified
  channel contrasts are non-positive and the positive control fails in its expected
  direction. Otherwise execute the immutable twelve-case extension.
- Do not adapt arms, cases, thresholds, or estimands after seeing the first six.
- At twelve, stop. Report the interval and all controls; do not buy an unplanned
  extension to rescue an ambiguous result.
- Do not run the agentic tier when ordinary compaction produces no measurable
  capability loss, when the channel is absent, or when no treatment clears its
  mechanistic control.

## Corrected-paper reruns and interpretation

The current paper must be corrected before treating prior reconstructed-state
numbers as evidence about actual generation-time state. After the v10 study:

- rerun only analyses whose underlying renders and provenance support the stated
  estimand;
- label reconstructed-state results as reconstructed-state diagnostics;
- replace or retract claims that depended on those results testing native
  generation state;
- distinguish channel existence, history specificity, component localization,
  intervention utility, and live-agent utility as separate inferential steps;
- preserve nulls, contaminated artifacts, and all stopping decisions in the audit
  trail.

## Joint recommendation

The approximately $50 mechanistic tier has the highest expected scientific value.
It can tell us whether the missing information channel exists in the paper's target
checkpoint and whether the training-free value-only split is the wrong mechanism.
The practical paired-agent study is worth attempting only after that evidence and a
no-treatment damage canary justify it. Neither author predicts that a live-agent
benefit is likely; the value of the gated study is that either a positive result or
a well-bounded null would answer the practical question cleanly.
