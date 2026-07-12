# SWE-Gym paper reanalysis artifacts

The canonical artifact is:

`swegym_paper_metrics_Qwen3-30B-A3B-Instruct-2507_20260712T051816Z.json`

SHA-256: `7f1b96509931e462fa6311768873599761fb32cfc6e1cc985a9d77cbf6941a04`

It was generated after making row order fully explicit: every source component
is sorted numerically by saved dataset index before bootstrap, and pooled
components are concatenated in the order recorded in the statistical contract.

The `051445Z` and `051600Z` files are preserved, noncanonical preliminary
row-order sensitivity records. Their point estimates are identical; seeded
finite-bootstrap interval endpoints differ slightly because finite Monte Carlo
resampling maps RNG indices onto row order. They must not be cited as the final
paper artifact.
