_This conversation completed a read-only publication-gate review of the current
`PAPER.md`, focusing on provenance, P01/P02 scope, statistical restraint, and
resource-accounting claims. The resulting verdict was PASS WITH REQUIRED FIXES;
the core scientific interpretation was judged conservative and the
NF4-versus-bfloat16 comparison appropriately narrow._

**Participants:** User and gpt-5.5-xhigh.

**Handoff State.** Review configuration was OpenAI GPT-5.5 with `xhigh`
reasoning effort. No files were modified. The paper is framed as an
inconclusive/failure-catalogue report rather than a positive cross-model or
native-design study, and that framing was considered acceptable.

**Required fixes.** At `PAPER.md:439` and `:627–633`, remove or substantiate the
precise Claude list-price analogue of `$3,014.85823075`. The inspected artifact
`results/end_to_end_accounting/claude-project-usage_claude-cli_20260712T130352Z.json`
records `counterfactual_list_price_usd: null` and says pricing was not computed
without a frozen, time-versioned request-level rate table. At `PAPER.md:626`,
reconcile the stated `8,944 request rows` with the artifact’s
`unique_message_count: 8943` and `request_count: 8944`, clarifying the
terminology or adopting the artifact’s exact terms.

The P01/P02 section was validated as a single fixed e01 computation comparing
bundled NF4 eligible-linear weights/kernels against bfloat16 weights, with
bfloat16 K/V cache storage in both arms. It correctly states the N=1 and
bundled-regime limitations and avoids claims about quantization causality,
recovery, population performance, or agent behavior. The exact common-field
reproduction and RunPod accounting totals were supported by the inspected
primary artifacts.

**Additional caveat.** A possible provenance gap remains for the reconstructed
Codex token total: the figures appeared in `PAPER.md`, while no corresponding
frozen Codex accounting artifact was found under
`results/end_to_end_accounting/`; only `snapshot_codex_usage.py` was present.
This was identified for follow-up but was not included as a finalized required
fix in the delivered review.

Nonblocking suggestions were to reduce or relocate visually dominant
long-decimal accounting values, keep “not independent cases” adjacent to the
first P01/P02 reproduction claim, and retain the explicit note that older
positive-design requirements in the July 8 checklist are retired.

## Conversation sources

- `019f569b-7551-72c2-bef9-ca2635a06fd5`
