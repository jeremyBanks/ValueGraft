# V12 semantic execution — last-valid-head pin after bookkeeping namespace error

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

## Observed error

Commit `17b4f07` added a project-accounting method review at
`results/accounting/end_to_end_accounting_method_review_gpt-5.6-sol_20260712T0320Z.md`.
The file is legitimate research bookkeeping and changed no sealed apparatus
byte, but the frozen v12 release verifier permits every post-authorization
addition only when its path begins `results/coherent_canary_` and its suffix is
JSON or Markdown. A direct verifier run at `17b4f07` therefore returned:

`post-authorization addition is not a result: results/accounting/...`

No pod or model process was launched from that head. The error was detected
during the no-GPU treatment-release audit.

## Additive disposition

The verifier intentionally audits every descendant commit, not merely the net
tree, so deleting or moving the file in a later commit cannot restore this
history. Repository history will not be rewritten. Preserve the accounting
record and pin any remaining sealed v12 semantic execution to
`cbdfa481bded9c8c64a8493416fa900590890cf9`, the last verifier-valid trunk head.
That head contains:

- the complete frozen apparatus and authorization;
- the passing same-host technical report;
- the accepted e01 Phase-A raw/report/receipt;
- the Phase-A acceptance, cost, and pod-stop record.

A fresh clone must check out that literal commit, remain on branch `trunk`, be
clean, prove it is an ancestor of `origin/trunk`, and pass the production
verifier before loading a model. Later trunk notes authorize external
orchestration but are not scientific inputs to the pinned clone.

For e01 treatment, every required release input already exists at the pinned
head. If later cases are authorized while the sealed branch remains active,
their Phase-A/treatment outputs may be written to an explicitly ignored staging
directory outside tracked clone state, so each fresh process can still verify a
clean pinned head. All staged bytes must be receipt-hashed, pulled, independently
validated/harvested, and committed under the required
`results/coherent_canary_*` namespace before a pod is deleted. No scientific
artifact may remain only in staging.

This pin changes no stimulus, input byte, threshold, model, runtime fingerprint,
treatment selector, decision rule, or claim. After sealed semantic execution is
over, the bookkeeping-path mistake and this workaround must be added to the
ordinary incident/reliability documents, and the accounting artifact may remain
in its natural project-wide namespace.
