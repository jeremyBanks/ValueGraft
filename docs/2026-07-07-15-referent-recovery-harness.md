# File: 2026-07-07-15-referent-recovery-harness.md

# Proposal: Synthetic "Referent-Recovery" Harness for K/V Grafting Evaluation

## 1. Executive Summary
Standard agentic benchmarks (like SWE-bench or τ-bench) fail to exercise the operative regime for context compaction. They are either too cognitively demanding for a 30B model to yield a clean signal, or too short to trigger genuine cache eviction[span_0](start_span)[span_0](end_span). 

This harness provides a targeted, synthetic evaluation pipeline. It forces the model to ingest a massive payload of custom, novel technical rules (the **Referent**), pushes that payload past the summarization boundary using dense, relevant engineering debate (the **Distraction**), and then tests whether the Key/Value graft can successfully retrieve a specific, summarized-away rule to execute a task (the **Probe**).

## 2. The Context Architecture (Per Test Case)
Each test case in the evaluation suite will construct a synthetic context window consisting of four blocks. The target length is ~20,000 to 25,000 tokens to guarantee the compaction wall is hit.

*   **Block A: The Anchor (1,000 tokens)**
    *   Establishes the persona and the overarching project goals. 
*   **Block B: The Payload / The Referents (4,000 - 5,000 tokens)**
    *   A highly detailed technical specification containing 50 distinct, heavily numbered rules. 
    *   *Crucial Design Requirement:* The rules must be structurally sound but entirely bespoke so the model cannot guess them from pre-training.
*   **Block C: The Distraction / Compaction Trigger (15,000+ tokens)**
    *   A massive, highly technical tangent that is semantically adjacent but practically irrelevant to Block B. This forces the model to summarize Block B.
*   **Block D: The Probe (500 tokens)**
    *   A zero-shot prompt requiring the exact application of a rule from Block B.

## 3. Dataset Generation Guide (For the Execution Agent)
*Agent Instructions: Use a capable frontier model to dynamically generate 100 variations of this context structure using the prompts below. Vary the rule numbers and target code blocks for each iteration to prevent evaluation leakage.*

### Prompt for Block B: The Payload
> "You are a principal engineer writing a dense, 5,000-token internal specification for a bespoke JavaScript/TypeScript AST (Abstract Syntax Tree) transformation engine designed for extreme code golf. The engine utilizes a custom implementation of Z85 binary encoding for hashing node identifiers. 
> 
> Define exactly 50 distinct 'Transformation Rules'. Number them clearly (e.g., 'Rule 17:', 'Rule 42:'). Each rule must dictate a highly specific, non-standard AST mutation. For example: 'Rule 23: When encountering a BinaryExpression with an operator of `+` and both sides being StringLiterals, encode the concatenated string using Z85, wrap it in a custom `__z()` decoder function, and replace the entire expression.' 
> 
> The document must be dry, authoritative, and extremely detailed."

### Prompt for Block C: The Distraction
> "Generate 15,000 tokens of highly detailed, back-and-forth engineering debate. The team is aggressively debating the internal protocol specifications of Git. 
>
> The debate must focus deeply on two highly specific edge cases:
> 1. The performance bottlenecks of Git's built-in gzip decompressor on the transport layer during massive packfile transfers.
> 2. The implementation of reserved values for Git object types, and whether the limit on object types is a hard mathematical constraint or a purely architectural one. 
>
> Do not mention ASTs, JavaScript, or Z85. Make the conversation incredibly dense, realistic, and verbose to artificially inflate the token count."

## 4. The Probe & Evaluation Logic

Once the synthetic context is assembled and run through the target model (triggering compaction), we deploy the Probe.

**The Probe Prompt:**
> "We need to finalize the minifier pass. Take the following TypeScript code block and apply **Rule 34** exactly as defined in our AST transformation specification. Output only the transformed code."
> 
> `[Insert standard 15-line TypeScript function here]`

### Scoring Metrics
Because standard Exact Match (EM) string matching is often too brittle for generated code, the agent should evaluate the test suite using the two instruments validated in the base research[span_1](start_span)[span_1](end_span):

1.  **Strict Meaning-Judge (Pass/Fail):** Use a frontier model as a judge to verify if the output code correctly reflects the *mechanics* of the specific target rule (e.g., Rule 34). If the model confabulates a different rule or outputs standard code, it is a failure (0.0).
2.  **Teacher-Forced Gap-Closure:** Feed the target model the exact, correct code continuation token-by-token and read the probability it assigns[span_2](start_span)[span_2](end_span). Calculate how far the K/V grafted state moves the probability from the Compacted (floor) baseline toward the Uncompacted (ceiling) baseline[span_3](start_span)[span_3](end_span). 

## 5. Execution Matrix (The K/V Axis)
To isolate whether Key vectors solve the fragile referent recovery problem, the evaluation harness must be run across these four boundary states:

*   **Baseline (Floor):** Plain text summary compaction.
*   **V-Only Graft:** $\alpha_K = 0$, sweep $\alpha_V$. (Validates the 30B vs 27B discrepancy)[span_4](start_span)[span_4](end_span).
*   **K-Only Graft:** Sweep $\alpha_K$ (with correct RoPE re-rotation), $\alpha_V = 0$. (Tests if addressing alone can pull the required concepts from the summarized text)[span_5](start_span)[span_5](end_span).
*   **Coupled K/V Graft:** $\alpha_K = \alpha_V$. (Tests if providing both addressing and content resolves the task without catastrophic state collapse)[span_6](start_span)[span_6](end_span).










