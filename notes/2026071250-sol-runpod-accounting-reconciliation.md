# RunPod Pod-credit reconciliation

**Author:** GPT-5.6 Sol (xhigh)

**Prepared:** 2026-07-12 13:05 UTC

## Result

Two separately captured, complete, inactive annual-bucket provider snapshots
agree exactly: the provider account consumed **$424.871205699397252292 in Pod
credits** across 76 Pod IDs, representing 1,097,688,401 billed milliseconds
(about 304.9134 hours) in the requested 2026-07-01 through 2026-07-12 window.
Both snapshots observed zero active Pods before and after collection.

That exact provider-account number is not automatically an exact project-cost
number. Independent evidence available before the first billing query identifies
73 of the 76 Pod IDs as project activity:

- 45 IDs have local `.pod*_state.json` launch snapshots. Their exact provider
  rows sum to $267.926330000627786196 and 699,132,009 ms.
- 28 additional IDs appear in the project's raw Codex or Claude transcripts
  before `2026-07-12T12:45:14.778043Z`, the first billing request's end-time.
  Their exact provider rows sum to $127.021797557128594006 and 322,114,094 ms.

The independently attributable project lower bound is therefore
**$394.948127557756380202 in consumed Pod credits**, across 73 Pods and
1,021,246,103 billed milliseconds (about 283.6795 hours).

Three provider-window Pod IDs have no independent local-state or pre-capture
transcript attribution:

| Pod ID | Consumed Pod credits | Billed ms |
|---|---:|---:|
| `9jb0bqjrcfuq0t` | $0.10394732188433409 | 729,681 |
| `mgxyzamnu3ece6` | $17.989733720663935 | 45,674,850 |
| `mvn478byps3ifl` | $11.829397099092603 | 30,037,767 |
| **Attribution gap** | **$29.92307814164087209** | **76,442,298** |

These three rows are not silently assigned to the project, but neither are they
declared unrelated. The exact provider total decomposes as:

`$394.948127557756380202 attributable lower bound + $29.92307814164087209 attribution gap = $424.871205699397252292 provider-account total`.

## Aggregation sensitivity

The earlier daily-bucket snapshot sums to $424.871205699397249212, while each
direct annual-bucket snapshot reports $424.871205699397252292. Pod count and
billed milliseconds are identical. The difference is exactly
$0.000000000000003080 in consumed credits and reflects provider bucket-level
decimal spellings. The annual aggregate is the canonical total; the daily sum
is retained only as a nonadditive sensitivity check.

## What remains unknown

Cash deposits and payment history remain unknown. No sanitized transaction
export or payment receipts are present in the evidence bundle. Consumed Pod
credits, balance movement, and cash loaded are different quantities, so this
record does not infer a cash amount from the Pod-billing total.

The 28 transcript-only attributions depend on raw local Codex and Claude
archives that are not committed to the repository. Their appearances were
restricted to events before the first billing capture, preventing the later
billing result from becoming circular attribution evidence. The complete ID
classification, exact arithmetic, source hashes, and limitations are recorded
in
[`runpod-reconciliation_runpod_20260712T130520Z.json`](../results/end_to_end_accounting/runpod-reconciliation_runpod_20260712T130520Z.json).

No new provider or network request was made for this reconciliation.
