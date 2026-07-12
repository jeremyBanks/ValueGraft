# Final GPT-5.5 xhigh paper review

**Invocation-attested runtime:** OpenAI Codex v0.143.0, model `gpt-5.5`,
reasoning effort `xhigh`, read-only sandbox, session
`019f569b-7551-72c2-bef9-ca2635a06fd5`. The CLI reported 126,229 tokens used.
No file modification was authorized inside the review session; this note
preserves its final response.

Requested runtime configuration: OpenAI GPT-5.5, reasoning effort `xhigh` /
extra-high, read-only review, no file modifications.

Verdict: **PASS WITH REQUIRED FIXES**

The scientific framing is appropriately conservative. I did not find a blocker
in the P01/P02 scope: `PAPER.md` correctly treats it as one fixed e01
computation, compares bundled NF4 eligible-linear weights/kernels versus
bfloat16 weights, states K/V cache storage was bfloat16 in both, and avoids
quantization-causality, recovery, population, or agent claims.

## Required fixes

1. **Fix unsupported Claude list-price accounting claim.**

   Location: `PAPER.md:439` and `PAPER.md:627-633`.

   Problem: the paper reports a precise Claude API counterfactual,
   `$3,014.85823075`, with model subtotals. The inspected primary artifact
   `results/end_to_end_accounting/claude-project-usage_claude-cli_20260712T130352Z.json`
   says `counterfactual_list_price_usd: null` and
   `counterfactual_pricing_status: "not computed; no separately frozen,
   time-versioned request-level rate table was supplied"`.

   Required fix: either remove the dollar figure and say the frozen Claude
   token ledger did not compute a price, or add/point to a frozen artifact that
   actually supports the rate table, per-model allocation, subtotals, and
   arithmetic.

2. **Resolve the Claude request-count mismatch.**

   Location: `PAPER.md:626`.

   Problem: the paper says the Claude collector expanded records into `8,944
   request rows`, while the inspected artifact reports
   `unique_message_count: 8943` and `request_count: 8944`. That may be
   explainable, but the paper does not distinguish “request rows” from unique
   messages.

   Required fix: add a short parenthetical clarifying the two counts, or use
   the artifact's exact terms.

## Nonblocking suggestions

- At `PAPER.md:431-449`, keep “not additive” but consider moving the
  resource-accounting table out of the main findings path or trimming
  precision. The long decimals are auditable but visually dominate a
  scientific conclusion section.

- At `PAPER.md:303-314`, the P01/P02 reproduction wording is careful. I would
  keep the sentence “not independent cases” close to the first reproduction
  claim, because readers may otherwise overread exact equality across hosts.

- At `PAPER.md:421-429`, the explicit note that the July 8 checklist's older
  positive claims are retired is important and should stay. It prevents a
  checklist reviewer from treating obsolete nativeness/keys-neutral/
  cross-architecture requirements as current claims.
