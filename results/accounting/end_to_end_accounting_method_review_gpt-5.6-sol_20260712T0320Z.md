# End-to-end accounting method review

**Reviewer:** independent GPT-5.6 Sol subagent; orchestration did not expose a
separate reasoning-effort label

**Disposition owner:** Sol — GPT-5.6 Sol, extra-high reasoning

This review audited the ignored provisional accounting implementation without
invoking a paid API or changing tracked code. Its conclusion is that the
prototype is a strong base but is not yet defensible as the final project
accounting.

## Corrections required before final totals

1. The provisional Claude list-price equivalent cannot be computed from
   aggregate model totals. The selected logs contain 5,173 requests above
   200,000 input tokens, including requests near one million tokens. Applicable
   long-context, service-tier, cache-write TTL, cache-read, input, and output
   rates must therefore be applied per request before aggregation.
2. RunPod must be represented as a settled, sanitized per-pod ledger. The
   current prototype retains aggregate totals and response hashes but does not
   preserve safe per-pod rows or prove pagination exhaustion. Repeated rows
   must be reconciled by pod ID; group-by-GPU, balance deltas, and runner cost
   estimates are cross-checks, never additive charges.
3. OpenRouter use is mentioned in the record and a dedicated key exists, but no
   local request ledger was found. Provider activity/transaction exports or
   receipts are required; otherwise both tokens and cost remain `unknown`.
4. No `.gemini` files were modified during the provisional research window
   despite documented Gemini contributions. That usage came through an
   unobserved surface and remains `unknown` without external records.
5. Claude parent `toolUseResult.usage` records can overlap their subagent
   execution. Globally deduplicated assistant message IDs are the appropriate
   base; parent and subagent counters must not simply be added.
6. The Codex reconstruction observed 141 project sessions in five graph
   components and 14,535 meaningful cumulative states, plus six cumulative
   counter resets. Final reconstruction must deduplicate inside session-graph
   components and independently derive fork suffixes; global counter-tuple
   deduplication alone is not a proof.

## Frozen final-audit workflow

- Provisional window start: first repository commit,
  `2026-07-04T20:05:15Z`.
- End: immediately after the last model-assisted paper/audit action. Freeze
  logs first and disclose that the final delivery response necessarily falls
  outside its own snapshot.
- Produce separate ledgers for actual cash transactions, metered consumption,
  subscription workload/invoices, counterfactual list-price equivalents, and
  local experimental inference. Never sum those categories into one purported
  cash total.
- Every row uses one measurement class:
  `exact_source_record`, `reconstructed`, `estimated`, `lower_bound`, or
  `unknown`.
- RunPod requires every page, a safe-field allowlist, unmatched-ID reporting,
  and two stable post-quiescence snapshots.
- Claude requires request-level pricing and message-ID deduplication. Codex
  requires fork-DAG/suffix reconstruction and counter-reset handling.
- Local subject workload deduplicates generation by render hash plus exact
  model revision/config, while separately recording supported repeated scoring
  passes. Judge/result text already represented in model logs is not counted a
  second time.

The final release must include the tested script, synthetic fork/reset/
pagination fixtures, rate tables with hashes, source manifests and cutoffs,
JSON/CSV ledgers, Markdown interpretation, reconciliation tables, explicit
unknowns, and a credential-literal/common-secret-pattern scan.
