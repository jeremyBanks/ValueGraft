# End-to-end token and cost audit — required final deliverable

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

The owner requested a reconstructable review of how many tokens and how much
money the project consumed from beginning to end. This is a required closing
deliverable, not an informal estimate. The definitive snapshot must be taken
only after experiments, consultations, paper drafting, and final reviews have
stopped, so that the source logs are quiescent.

## Required separation

The report must not collapse unlike quantities into one deceptively precise
total. It will separately present:

1. **Observed actual cash or credits consumed** by metered services, with
   provider records and settlement cutoffs.
2. **Cash paid or prepaid** (including subscriptions/deposits), which is not
   necessarily equal to project-attributable consumption.
3. **Provider-list-price equivalents** for subscription-included model usage,
   explicitly labeled counterfactual rather than cash paid.
4. **Model-token workload**, separated by provider/model and by input, output,
   cache-read, cache-write, or other available token classes.
5. **Local experimental inference workload**, kept distinct from billed API
   tokens; include only quantities that can be reconstructed defensibly.
6. **Estimates and unknowns**, such as local electricity, missing invoices,
   unsettled pod charges, promotional credits, and sources with no trustworthy
   usage counters.

Every headline number must carry one of `exact from source record`,
`reconstructed`, `estimated`, `lower bound`, or `unknown`. Ranges and lower
bounds are preferable to invented precision.

## Reproducibility and integrity requirements

- Define the research-window start and final cutoff explicitly.
- Preserve source-type/file manifests, byte or record cutoffs, hashes, and the
  deduplication rules used for transcript/session reconstruction.
- Avoid double-counting resumed sessions, nested subagents, copied transcript
  records, cached-token fields, provider billing lag, deposits, or refunds.
- Reconcile pod billing both per pod and in aggregate; preserve unmatched pod
  IDs and late-settling records rather than silently dropping them.
- Distinguish Claude/Codex subscription usage from any direct API/OpenRouter
  calls, and identify exact runtime model names where logs expose them.
- Version and test the audit program; scan generated artifacts to ensure no
  credential literal is present.
- Commit the program, machine-readable detail, summary tables, and a prose
  interpretation. Preserve limitations and unresolved residuals.

A provisional ignored prototype already exists under `.sol-v4/accounting/`.
Its current figures are working snapshots only and must not be reported as the
final total. Before publication it requires an independent Codex session-graph
and suffix-reconstruction check, a final provider-settlement refresh, coverage
of non-RunPod/API/subscription categories that can be evidenced, and rerunning
after all research work becomes quiescent.
