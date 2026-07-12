# SWE-Gym task/repository cluster recovery and uncertainty sensitivity

**Date:** 2026-07-12

**Author:** OpenAI GPT-5.6 Sol (extra-high reasoning)
**Scope:** zero-GPU, zero-model-call audit of the 57 fresh selected-map evaluation rows plus the 45-row partial original-pool confirmation. This note does not alter the paper or reinterpret the endpoint as task success.

## Topline

A defensible observable repository key and exact-task clustering surrogate were recoverable from the preserved `messages` column with 491/491 coverage. The selected-map-minus-compacted-baseline likelihood lead remained above zero under task-, versioned-snapshot-, and repository-cluster resampling. The more important new defect was not the interval: because tune/eval assignment hashes **trajectory index**, exact repeated tasks crossed the map-fitting boundary. Seven fitted task clusters recur among eight of the pooled 102 rows. Removing every such row left 94 rows in 69 exact-task clusters, and its task-clustered interval also remained above zero.

Thus this audit did **not** overturn the small likelihood lead. It did identify a blocking manuscript correction: the current paper says stable task/repository clusters were unavailable and does not disclose task-level fit/evaluation overlap. That prose and its statistical artifact should be corrected before release. The absence of a selected-map-matched placebo, action execution, or task-success endpoint is unchanged.

## What was directly observed

- The ignored local `swegym.parquet` was 10,379,409 bytes, SHA-256 `ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1`, with 491 rows and only a `messages` column.
- Every row began with one shared generic system/tool prompt followed by an initial user task. The system prompt was identical in all 491 rows, UTF-8 SHA-256 `1120aa8819abb372428afb82f6a5f49d1d243e4bf58cb27fd481809acd339e84`; it carried no per-task identity.
- Every initial user task began with an explicit uploaded directory of the form `owner__repository__version`. Strict parsing succeeded on 491/491 rows, and the same exact workspace path reappeared in later assistant actions on 491/491 rows.
- Splitting the asserted three-part basename produced 87 versioned snapshots and 11 normalized owner/repository keys across the full parquet.
- Hashing the exact first PR-description block together with its versioned snapshot produced 293 observable task clusters. The exact initial-user-message partition was identical. Of the 293 tasks, 107 repeated across 305 rows with sizes 2–10; all 491 full trajectory transcripts had different hashes. The repeats therefore represented distinct rollouts of the same literal task, not duplicate transcript rows.
- Saved score-row indices joined directly to parquet row indices. This is supported by `src/run_swegym_hf.py`: it loads `df["messages"]` in order, selects `msgs = trajs[idx]`, names output `t{idx:04d}.json`, and writes the same `idx` into the result. The committed JSON filename and embedded index matched for every consumed row.
- As an independent result-only check, the saved demonstrated `gold_action` contained the exact expected versioned workspace key for 101/102 pooled rows and no conflicting key. The one missing row used a generic testbed path; its repository was still recovered unambiguously through the hash-bound parquet index join.

## Deterministic identity contract

For each parquet row:

1. Assert the first roles are `system,user` and exactly one system message exists.
2. Strictly parse the initial user prefix as an uploaded `/workspace/<snapshot>` path.
3. Assert `<snapshot>` has exactly three nonempty `__`-separated fields: owner, repository, version.
4. Set `snapshot_key = owner__repository__version` and `repository_key = owner/repository`.
5. Extract the first closed `<pr_description>...</pr_description>` block, strip scaffold-edge whitespace while preserving all internal bytes, and set:

   `task_sha256 = SHA256-UTF8(snapshot_key + U+0000 + pr_description)`.

This is an artifact-stable, content-addressed **observable-task surrogate**, not recovery of an absent upstream SWE issue/PR ID. The result artifact stores only hashes and normalized owner/repository/version keys, never literal task or system text.

## Pool structure and fit leakage

| Component | Rows | Exact task clusters | Task clusters also in map fit | Rows whose task also appears in fit |
|---|---:|---:|---:|---:|
| Map fitting | 41 | 38 | — | — |
| Fresh evaluation | 57 | 54 | 5 | 5 |
| Partial original confirmation | 45 | 28 | 2 | 3 |
| Descriptive pooled set | 102 | 76 | 7 | 8 |

The fresh and confirmation components themselves shared six task clusters. Pooled task-cluster sizes were 60 singletons, nine pairs, four triples, and three clusters of four. The 45 confirmation rows were exactly the first 45 eligible IDs from the original 75-row pool, confirming that this was a nonrandom stopped prefix.

The task crossing was mechanically explained by `traj_split(idx)` in `src/run_swegym_hf.py`, which hashes `(seed, trajectory_idx)`, not task identity. The prior row-ID disjointness checks were correct but insufficient for task-level held-out separation.

## Statistical contract

The row value was saved `E-champion.tf_mean - B.tf_mean`, in nats per demonstrated-target token. Pooled row order was the 57 fresh IDs sorted numerically, then the 45 confirmation IDs sorted numerically.

All intervals used 10,000 replicates, `random.Random(0)`, ordinary percentile endpoints at sorted indices 250 and 9,750, with no multiplicity adjustment.

- **Existing iid comparison:** sample 102 trajectory rows with replacement; average row deltas.
- **Primary task-cluster sensitivity:** sort the 76 task hashes; sample 76 whole task clusters with replacement; retain all member rows with multiplicity; divide their summed deltas by the total contributed row count. This preserves the paper's trajectory-weighted point estimand.
- **Snapshot/repository sensitivities:** the same whole-cluster algorithm using 39 versioned snapshots or nine normalized repositories.
- **Strict no-fit-task sensitivity:** remove all eight rows whose observable task hash occurred in the map-fitting set, then apply the same task-cluster bootstrap to 94 rows/69 tasks.
- **Alternative-estimand check:** average one within-task mean per sampled task cluster. This changes the estimand and is reported only to show that unequal task multiplicity did not create the sign.

## Results

### Pooled 102 selected-map-minus-B

| Resampling unit / estimand | Point (nats/token) | 95% percentile interval | Clusters |
|---|---:|---:|---:|
| Trajectory iid, row-weighted | +0.013548 | [+0.008335, +0.019087] | 102 rows |
| **Exact task, row-weighted** | **+0.013548** | **[+0.007625, +0.019542]** | **76** |
| Exact task, equal-task sensitivity | +0.014985 | [+0.009603, +0.020819] | 76 |
| Versioned snapshot, row-weighted | +0.013548 | [+0.008002, +0.019690] | 39 |
| Normalized repository, row-weighted | +0.013548 | [+0.010868, +0.018537] | 9 |
| Normalized repository, equal-repository sensitivity | +0.016691 | [+0.010658, +0.023789] | 9 |

The task bootstrap modestly widened the iid interval; its lower endpoint remained positive. Repository resampling happened to narrow the interval because all nine observed repository means were positive, but nine highly unequal repositories are too few for that interval to be treated as a well-powered repository-population result.

Component task-cluster intervals were also above zero:

- Fresh 57 / 54 tasks: +0.011742, task-cluster CI [+0.006115, +0.016939].
- Confirmation 45 / 28 tasks: +0.015836, task-cluster CI [+0.004628, +0.027454].

### Strict task-disjoint-from-fit subset

Removing every row whose exact observable task occurred in map fitting left 94 rows and 69 tasks:

- Row-weighted point: **+0.014237 nats/token**.
- Task-cluster 95% interval: **[+0.007921, +0.020598]**.
- 72/94 row deltas were positive.

The excluded eight rows averaged only +0.005463, so task-fit overlap did not generate the pooled sign. This strict analysis is post hoc and does not retroactively make the original split task-preregistered.

## Limitations and paper consequence

- Exact task text plus snapshot is sufficient to group literal repeated tasks, but it can under-merge semantically identical tasks whose text differs. No canonical upstream task/PR ID was recovered.
- Version strings are explicit directory labels, not immutable source revisions.
- The parquet remains ignored. The committed-sized result serializes the derived hash/key mapping and binds it to the exact parquet hash, so the analysis can be audited without checking in the literal 9.9 MiB task corpus.
- Confirmation remains a nonrandom 45/75 prefix, and pooling 57+45 remains descriptive.
- The repository sensitivity has only nine pooled clusters with sizes 2–31.
- All intervals condition on this observed pool and selected map and are not multiplicity-adjusted.
- Cluster recovery does not address the missing selected-map-matched placebo, lack of executed actions/task success, incomplete upstream/generator provenance, or unsaved summaries.

**Paper blocker verdict:** the statistical lead was not overturned, but correcting the manuscript's claim that cluster IDs were unavailable, replacing or supplementing the iid interval, and disclosing/reanalyzing exact-task fit overlap are blocking before release. Once those zero-GPU corrections are incorporated and checked, this particular reviewer objection is resolved rather than fatal to the reported likelihood lead.

## Reproducible artifacts

- Analysis: `scripts/analyze_swegym_cluster_sensitivity.py`
- Tests: `tests/test_analyze_swegym_cluster_sensitivity.py`
- Machine result: `results/swegym_cluster_sensitivity/swegym_cluster_sensitivity_Qwen3-30B-A3B-Instruct-2507_20260712T063342Z.json`
- Machine-result SHA-256 at generation: `1651803cdb9d45059cefa8474a59c65bce2bf69e4db8c7ff9b4c7344570f58cf`
- Generation command: `PYTHONDONTWRITEBYTECODE=1 uv run python scripts/analyze_swegym_cluster_sensitivity.py --timestamp 20260712T063342Z`
- Focused test command: `PYTHONDONTWRITEBYTECODE=1 uv run pytest -q tests/test_analyze_swegym_cluster_sensitivity.py`
