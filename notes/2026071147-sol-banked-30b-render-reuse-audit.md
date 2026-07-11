# Banked 30B render-reuse audit

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11

## Question

Can the already-committed Qwen3-30B conversation renders replace a new paid
conversation-rendering pass in the corrected coherent-state experiment?

## Observed inventory and identity

Both of these directories contain all 12 cases, `c01` through `c12`:

- `results/champion_harvest/cs30b/Qwen__Qwen3-30B-A3B-Instruct-2507/`
- `results/redraw_harvest/b0/Qwen__Qwen3-30B-A3B-Instruct-2507/`

For every case, canonical-JSON hashes of the complete `render` object match
between the two directories. The saved `summary_text` also matches in every
case. The second directory introduced the artifacts in commit `f4b6a89`; commit
`dd8d518` copied the same render bank into the champion-harvest directory.
These are duplicate records of one render corpus, not independent renders.

The render fingerprint says:

- model ID `Qwen/Qwen3-30B-A3B-Instruct-2507`;
- `native_render: true`;
- greedy generation (`native_temp: 0.0`);
- 320-token maximum assistant reply;
- 22 assistant replies per conversation, 264 in total.

The aggregate result reports 82,909 reply tokens and mean reply length 314.05
tokens.

## Blocking provenance and quality limitations

The saved artifacts do **not** record the resolved model revision, actual live
parameter dtype, device/backend, tokenizer revision, library versions, or code
commit in their render fingerprint. The repository now identifies the intended
production revision as `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, but the
render artifacts themselves do not prove that this was the revision that wrote
their text.

The old reply-record schema retained only `n_tokens` and `logprob_sum`; it did
not retain the raw generated token IDs or raw pre-trim text. Consequently, it
cannot independently reconstruct or verify the exact generation. The canonical
conversation text survived, but generation-time K/V did not.

Cap incidence is severe:

| Case | replies at 320-token cap / 22 |
|---|---:|
| c01 | 20 |
| c02 | 16 |
| c03 | 21 |
| c04 | 19 |
| c05 | 22 |
| c06 | 20 |
| c07 | 21 |
| c08 | 22 |
| c09 | 22 |
| c10 | 19 |
| c11 | 22 |
| c12 | 22 |

That is 226/264 capped replies. The harness trimmed capped decoded text before
storing it in the canonical conversation. This makes clipped, incomplete, or
locally incoherent turns a high-priority risk, not a hypothetical edge case.

## Decision boundary

These artifacts may be reused as a **frozen model-authored text corpus
candidate**, subject to blind whole-conversation coherence review and the same
counterfactual factual-consistency review used for new inputs. Reuse would avoid
paying to generate the conversation body again.

They may **not** be described as preserved native incremental state, an
independent replication, or a fully provenance-bound exact-revision render.
Replaying their canonical text through the model is a new prefill of imported
text. Any corrected assay must generate the tested summary in the live exact
production configuration and save its complete input IDs, generated summary
IDs/text, write-time state or state hashes required by the assay, resolved
revision/backend/dtype/library/code provenance, and all outputs.

Before choosing this bank for paid work:

1. blind-review every candidate conversation for global coherence and clipped
   or nonresponsive turns;
2. exclude or repair failures before counterfactual authoring, with repairs
   frozen and independently reviewed;
3. call the surviving corpus `banked model-authored canonical text`, not a live
   native render;
4. carry corpus identity and exact text hashes into the preregistration and run
   fingerprint;
5. treat the twelve cases as one purposive scaffold family and keep the
   conversation as the inferential unit.

## Bottom line

The bank can save conversation-generation cost, but it does not remove the need
for input-quality review, live exact-model summary generation, or full run
provenance. Its unusually high truncation rate means a fresh shorter-turn corpus
may still be scientifically cleaner and, after accounting for repair/review,
potentially more economical.
