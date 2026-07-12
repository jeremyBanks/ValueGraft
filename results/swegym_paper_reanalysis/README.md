# SWE-Gym paper reanalysis artifacts

The canonical artifact is:

`swegym_paper_metrics_Qwen3-30B-A3B-Instruct-2507_20260712T060222Z.json`

SHA-256: `3b577f44af4aa46c6569201b9405c93e3b815c7af128efdde7c783d1193657e4`

It separates the original 75-ID runs by prefill schedule. The July 5 run
predates 4,096-token chunking; the July 10 run and disjoint 98 use chunked
prefill. Only the chunked original 75 and chunked disjoint 98 enter the
schedule-matched pooled estimate. Every source component is sorted numerically
by dataset index before bootstrap.

The `051445Z`, `051600Z`, and `051816Z` v1 files are preserved but superseded.
The first two are preliminary row-order sensitivity records. The `051816Z`
artifact is numerically reproducible but methodologically invalid for citation:
it averaged two different prefill schedules as if they were repeats, which also
made its reported pool-difference interval exclude zero. Do not cite any v1
artifact as the final paper result.
