_This conversation covers final validation of the v12 apparatus, independent
recomputation and scientific interpretation, paper reproducibility corrections,
a bounded NF4 precision probe, and transition to paper drafting and complete
cost reconciliation._

**Participants:** gpt-5.6-sol-ultra.

**Handoff State.** The protocol is formally stopped: no further GPU or treatment
work is authorized. One bounded e01 diagnostic was completed under a matched
Transformers 4.57.6 NF4/bf16 apparatus; the current priority is to finalize the
working paper and end-to-end accounting.

The paper’s defensible conclusion is “no practical benefit was established; the
clean literal natural-compaction question remains unresolved.” It must
distinguish practical black-box efficacy from mechanistic attribution, state
that the mechanism gate was a project policy, and avoid claims of semantic
specificity, deterministic replication, aggregate harm, or general quantization
interaction. The paper should report legacy source-sign differences,
fixed-scalar heterogeneity/null results, selected-map positives, and the missing
behavioral, placebo, and natural-compaction controls with equal care. It is
written as a cautious working paper with an executable reproducibility appendix.

Key verified results include pooled selected-map-minus-baseline SWE movement of
`+0.013548` nats/token, task-clustered CI `[+0.007625,+0.019542]`, and
`+0.014237` after excluding fit-overlap rows; selected-map-minus-fixed-scalar
pooled contrast `+0.001679`, CI spanning zero. Legacy dose reconstruction
recovered 27,893 aligned positions exactly, while SWE summary dose remains
unrecoverable from committed artifacts; at least 93.376% of eval-57 aligned SWE
mass came from the tail. The e01 raw record, token decomposition, legacy tables,
and analysis artifacts were independently reconstructed. Exact score-level
audits are reproducible, but new model-forward reruns require missing
tensors/runtime inputs.

The NF4 route required a separate Transformers 4.57.6 runtime because
Transformers 5 would silently leave 94.95% of packed MoE expert weights
unquantized. The valid fallback covered 18,432 expert linears and 97.961% of
logical weights, with bf16 KV explicitly attested. P01 was preregistered with
full 34-arm repeats, NF4 before bf16, a two-hour/`$4` ceiling, and outcome-blind
extension rules; NF4 proved materially slower than expected, so a narrower P02
was frozen before inspecting outcomes but is not to be launched under the
current handoff.

Operational lessons: shell backgrounding leaked SSH file descriptors;
launchd-based supervision fixed this. A silent provider watchdog failure was
also replaced with independently verified launchd ownership. Future accounting
must separate RunPod consumed credits, actual cash/prepaid funds,
subscription-covered usage, request-level list-price equivalents, local
inference workload, and unknowns; never combine them into one total. Codex and
Claude usage require lineage-aware deduplication because replayed rollouts,
streamed duplicates, and cumulative counters are non-additive.

## Conversation sources

- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f548f-0d8e-76f0-ae6f-06be9df34784`
- `019f548e-e46f-7df2-8fcd-178d2962d024`
- `019f54b9-be07-7413-aa8c-f680c7767190`
- `019f5455-6528-71f3-a64c-a190a65348b4`
- `019f54c9-1978-7211-aab4-76c7dc1f6dbd`
- `019f54c9-01e1-76c1-92d9-9370491df9dd`
- `019f54c8-edc5-7812-800a-44c636e5e4f6`
- `019f54ef-2964-76c1-97b7-3d0b5de7d66d`
- `019f54ef-1121-7a41-b4fa-7c20b5ca5975`
- `019f54ff-30da-7a83-b7cf-d70cb83dfd57`
- `019f54ff-1284-7ce0-9798-27556cbc4b81`
- `019f550e-eb90-72c2-8e11-ad84bbf6b73f`
- `019f5518-7b46-79f2-9b67-1bcbeba1313d`
- `019f5518-3ef1-7603-b2ea-b20d600a7f55`
- `019f5518-64dc-7f43-a868-a5bb1657caa7`
- `019f5526-eccc-7501-b41a-c2d21bd88a6d`
- `019f5527-39c2-7590-a4c6-dfbc21ce0c1f`
- `019f553c-5830-7b31-b110-82d0ed5ad762`
- `019f5527-174c-7890-ac42-3014892072d6`
- `019f5555-20b2-72b0-9568-b5243401909b`
- `019f5566-c0d1-74a1-b63d-2233db2a1b65`
- `019f5586-7022-7041-9db6-26668cedf772`
