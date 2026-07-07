# Synthetic Referent-Recovery Harness for K/V Grafting Evaluation

Attribution: proposal drafted by Gemini 3.1 Pro, then lightly cleaned and
reframed for this repository.

## 1. Executive Summary

Standard agentic benchmarks fail to isolate the operative regime for context
compaction. They tend to be either too difficult for a 30B model to yield a
clean task signal, or too short to trigger meaningful eviction.

This harness proposes a targeted, offline-generated synthetic pipeline for
evaluating coupled Key/Value (K/V) cache grafting. It forces ingestion of a
dense payload, artificially triggers compaction using a precomputed sparse
summary, and uses a teacher-forced extraction pass to test specific referent
recovery without runtime generation variance.

## 2. Context Architecture And Offline Precomputation

Generate 100 static test cases offline. Each JSONL row should contain the
following frozen blocks:

- **Block A: Anchor**  
  Persona and overarching project goals, roughly 1,000 tokens.

- **Block B: Payload**  
  Fifty bespoke, numbered technical rules, roughly 5,000 tokens. A concrete
  domain could be custom JavaScript/TypeScript AST transformations using Z85
  encoding. The rules must be structurally sound but entirely novel, so the
  model cannot infer the correct behavior from pretraining.

- **Block C: Distraction**  
  Irrelevant but dense engineering debate, roughly 15,000 tokens. For example:
  Git packfile transport bottlenecks, decompressor behavior, and reserved object
  type values. The purpose is to push the payload behind the compaction boundary
  without adding information that helps answer the probe.

- **Block D: Hardcoded Sparse Summary**  
  Generated offline by a frontier model after reading Blocks A, B, and C. The
  summary prompt should explicitly instruct the model to retain rule labels,
  such as "the spec contains 50 rules, including Rule 34," while stripping the
  relations: the actual mechanical instructions of the rules. This guarantees
  the sparse challenge and removes runtime summary variance.

- **Block E: Probe**  
  A zero-shot prompt requesting the exact application of a specific
  summarized-away rule, such as "Apply Rule 34 to this code block."

## 3. Extraction Pass: Teacher-Forced Summary State

Because the summary is hardcoded, write-time K/V activations can be simulated
with teacher forcing rather than live generation.

1. **Load Full Context**  
   Feed the target model Blocks A, B, and C.

2. **Force And Extract Summary State**  
   Teacher-force the model to process Block D token by token immediately after
   the full context.

3. **Freeze Summary Vectors**  
   Extract and freeze the Key and Value vectors computed specifically for the
   Block D tokens. These vectors represent the sparse summary as encoded in the
   original context, not as freshly encoded in isolation.

## 4. Execution And Evaluation Matrix

Clear the context. For the evaluation pass, feed the model only Block D, the
compacted state, followed by Block E, the probe. Graft the extracted vectors
onto Block D's token positions according to this matrix:

1. **Baseline: Plain Summary Compaction**  
   Use the hardcoded summary with freshly computed K/V state. This is the floor.

2. **V-Only Graft**  
   Set `alpha_K = 0` and sweep `alpha_V`. This tests whether value vectors alone
   recover the summarized-away referent.

3. **K-Only Graft**  
   Sweep `alpha_K` with correct RoPE re-rotation and set `alpha_V = 0`. This
   tests addressing-only recovery.

4. **Coupled K/V Graft**  
   Set `alpha_K = alpha_V` and sweep the shared value. This tests whether
   providing both addressing and content resolves the task without catastrophic
   state collapse.

5. **Independent K/V Graft**  
   Optionally sweep independent `(alpha_K, alpha_V)` pairs if the coupled arm
   shows any signal. This is the natural next step if K-only or coupled K/V
   separates from V-only.

## 5. Scoring Metrics

1. **Strict Meaning Judge**  
   Use a frontier model to verify whether the generated code correctly executes
   the target rule's mechanics. If the model confabulates a different rule or
   emits generic code, score it as a failure.

2. **Teacher-Forced Gap Closure**  
   Feed the exact correct continuation token by token and measure how far the
   grafted state moves the probability of that continuation from the compacted
   floor toward the uncompacted ceiling:

   `gap_closure = (logprob_graft - logprob_compacted) / (logprob_full - logprob_compacted)`

This harness is most useful as a falsification gate: before generating a large
100-case suite, run a tiny version and verify that full context beats compacted
context, the sparse summary does not leak the target rule, and at least one K/V
policy separates from the floor.

