# Policy Registry Follow-Up

- Model: `Qwen/Qwen3-0.6B`
- Device: `mps`
- Cases: `46`
- Elapsed seconds: `63.5`

This is a noisy local search over candidate benchmark shapes. It is useful
for picking promising regions, not for making claims.

## By Family

| group | n | A>B | positive | mean A-B | mean best E-B | mean best GC | best-policy counts |
|---|---:|---:|---:|---:|---:|---:|---|
| `low_entropy_transform` | 16 | 15 | 12 | 3.453 | 0.454 | 0.206 | {'k010': 2, 'k020': 5, 'v002': 3, 'v020': 1, 'k002': 2, 'kv005': 2, 'kv010': 1} |
| `policy_choice` | 30 | 30 | 28 | 7.238 | 0.265 | 0.036 | {'v030': 2, 'v020': 4, 'v002': 2, 'v010': 12, 'v005': 9, 'kv005': 1} |

## By Family And Summary Style

| group | n | A>B | positive | mean A-B | mean best E-B | mean best GC | best-policy counts |
|---|---:|---:|---:|---:|---:|---:|---|
| `low_entropy_transform / label_only` | 8 | 8 | 6 | 3.353 | 0.377 | 0.169 | {'k010': 2, 'k020': 2, 'v002': 2, 'v020': 1, 'k002': 1} |
| `low_entropy_transform / typed` | 8 | 7 | 6 | 3.552 | 0.542 | 0.249 | {'kv005': 2, 'k020': 3, 'k002': 1, 'kv010': 1, 'v002': 1} |
| `policy_choice / label_only` | 15 | 15 | 13 | 7.440 | 0.274 | 0.035 | {'v030': 1, 'v020': 4, 'v002': 2, 'v010': 3, 'v005': 5} |
| `policy_choice / typed` | 15 | 15 | 15 | 7.035 | 0.257 | 0.036 | {'v030': 1, 'kv005': 1, 'v010': 9, 'v005': 4} |

## By Family And Category

| group | n | A>B | positive | mean A-B | mean best E-B | mean best GC | best-policy counts |
|---|---:|---:|---:|---:|---:|---:|---|
| `low_entropy_transform / dedupe_transform` | 2 | 1 | 1 | 0.496 | 0.101 | 0.074 | {'v020': 1, 'kv005': 1} |
| `low_entropy_transform / identifier_transform` | 4 | 4 | 4 | 4.201 | 0.907 | 0.386 | {'k010': 1, 'k020': 2, 'kv005': 1} |
| `low_entropy_transform / normalization_transform` | 2 | 2 | 2 | 1.462 | 1.022 | 0.679 | {'k010': 1, 'k020': 1} |
| `low_entropy_transform / ordering_transform` | 4 | 4 | 3 | 4.326 | 0.187 | 0.017 | {'v002': 1, 'k020': 2, 'k002': 1} |
| `low_entropy_transform / privacy_transform` | 2 | 2 | 0 | 4.129 | -0.039 | -0.009 | {'v002': 2} |
| `low_entropy_transform / validation_transform` | 2 | 2 | 2 | 4.481 | 0.185 | 0.032 | {'k002': 1, 'kv010': 1} |
| `policy_choice / cache_policy` | 6 | 6 | 6 | 7.574 | 0.242 | 0.033 | {'v005': 1, 'v020': 1, 'v010': 4} |
| `policy_choice / privacy_export` | 6 | 6 | 6 | 8.671 | 0.383 | 0.045 | {'v020': 1, 'v005': 1, 'v010': 4} |
| `policy_choice / release_review` | 6 | 6 | 6 | 6.739 | 0.297 | 0.040 | {'v005': 4, 'v020': 1, 'v010': 1} |
| `policy_choice / routing` | 6 | 6 | 5 | 6.475 | 0.112 | 0.017 | {'v010': 2, 'v005': 3, 'v002': 1} |
| `policy_choice / ui_rendering` | 6 | 6 | 5 | 6.729 | 0.293 | 0.045 | {'v030': 2, 'v020': 1, 'v002': 1, 'kv005': 1, 'v010': 1} |

## Policy Surface

| policy | usable n | positive cases | mean E-B |
|---|---:|---:|---:|
| `v002` | 45 | 36 | 0.059 |
| `v005` | 45 | 36 | 0.119 |
| `v010` | 45 | 32 | 0.074 |
| `v020` | 45 | 17 | -0.418 |
| `v030` | 45 | 9 | -1.068 |
| `k002` | 45 | 9 | -0.051 |
| `k005` | 45 | 6 | -0.176 |
| `k010` | 45 | 6 | -0.514 |
| `k020` | 45 | 6 | -1.494 |
| `kv002` | 45 | 24 | 0.010 |
| `kv005` | 45 | 20 | -0.048 |
| `kv010` | 45 | 8 | -0.422 |
| `kv020` | 45 | 2 | -1.975 |
| `k005_v020` | 45 | 10 | -0.517 |
| `k010_v020` | 45 | 4 | -0.787 |

## Policy Surface For `low_entropy_transform`

| policy | usable n | positive cases | mean E-B |
|---|---:|---:|---:|
| `v002` | 15 | 9 | 0.030 |
| `v005` | 15 | 8 | 0.022 |
| `v010` | 15 | 6 | -0.195 |
| `v020` | 15 | 3 | -1.249 |
| `v030` | 15 | 0 | -2.423 |
| `k002` | 15 | 8 | 0.013 |
| `k005` | 15 | 6 | -0.014 |
| `k010` | 15 | 6 | -0.153 |
| `k020` | 15 | 6 | -0.646 |
| `kv002` | 15 | 7 | 0.037 |
| `kv005` | 15 | 7 | -0.017 |
| `kv010` | 15 | 4 | -0.429 |
| `kv020` | 15 | 2 | -2.159 |
| `k005_v020` | 15 | 1 | -1.150 |
| `k010_v020` | 15 | 0 | -1.261 |

## Policy Surface For `policy_choice`

| policy | usable n | positive cases | mean E-B |
|---|---:|---:|---:|
| `v002` | 30 | 27 | 0.074 |
| `v005` | 30 | 28 | 0.167 |
| `v010` | 30 | 26 | 0.209 |
| `v020` | 30 | 14 | -0.003 |
| `v030` | 30 | 9 | -0.391 |
| `k002` | 30 | 1 | -0.083 |
| `k005` | 30 | 0 | -0.257 |
| `k010` | 30 | 0 | -0.694 |
| `k020` | 30 | 0 | -1.918 |
| `kv002` | 30 | 17 | -0.004 |
| `kv005` | 30 | 13 | -0.064 |
| `kv010` | 30 | 4 | -0.419 |
| `kv020` | 30 | 0 | -1.884 |
| `k005_v020` | 30 | 9 | -0.201 |
| `k010_v020` | 30 | 4 | -0.550 |


## Top Cases

| case | family | category | style | best | E-B | GC | A-B | gold |
|---|---|---|---|---|---:|---:|---:|---|
| `transform-typed-iris` | `low_entropy_transform` | `identifier_transform` | `typed` | `k020` | 1.385 | 0.763 | 1.816 | user_profile_card |
| `transform-typed-azure` | `low_entropy_transform` | `normalization_transform` | `typed` | `k020` | 1.360 | 0.811 | 1.677 | UTC |
| `transform-label_only-iris` | `low_entropy_transform` | `identifier_transform` | `label_only` | `k020` | 1.034 | 0.602 | 1.717 | user_profile_card |
| `policy-label_only-opal` | `policy_choice` | `release_review` | `label_only` | `v020` | 0.819 | 0.095 | 8.591 | block release until rollback notes are attached |
| `transform-label_only-coral` | `low_entropy_transform` | `identifier_transform` | `label_only` | `k010` | 0.792 | 0.109 | 7.298 | userName |
| `policy-typed-citrine` | `policy_choice` | `ui_rendering` | `typed` | `v030` | 0.741 | 0.113 | 6.574 | render the view as a compact comparison table |
| `transform-label_only-azure` | `low_entropy_transform` | `normalization_transform` | `label_only` | `k010` | 0.683 | 0.548 | 1.247 | UTC |
| `policy-label_only-marble` | `policy_choice` | `privacy_export` | `label_only` | `v020` | 0.612 | 0.072 | 8.471 | drop the optional analytics field before export |
| `policy-label_only-citrine` | `policy_choice` | `ui_rendering` | `label_only` | `v030` | 0.540 | 0.088 | 6.105 | render the view as a compact comparison table |
| `policy-typed-nickel` | `policy_choice` | `privacy_export` | `typed` | `v010` | 0.515 | 0.053 | 9.647 | hash account IDs before analytics upload |
| `transform-label_only-cedar` | `low_entropy_transform` | `ordering_transform` | `label_only` | `k020` | 0.504 | 0.072 | 6.946 | tenant id |
| `policy-typed-opal` | `policy_choice` | `release_review` | `typed` | `v010` | 0.447 | 0.060 | 7.492 | block release until rollback notes are attached |
