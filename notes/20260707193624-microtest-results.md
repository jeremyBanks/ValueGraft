# Referent-Recovery Microtest Results

- Model: `Qwen/Qwen3-0.6B`
- Device: `mps`
- Cases: `3`

This is a cheap falsification gate for the referent-recovery harness idea. It is
not a publishable result by itself.

## Verdict

Negative gate at this scale: full context beats compacted context, so the task
creates a real floor/ceiling gap, but every tested graft policy moved below the
compacted baseline. Do not scale this exact harness shape without changing the
sparse-summary design, alpha range, model scale, or target construction.

## Aggregate

| policy        | valid n | mean gap closure | wins over B |
| ------------- | ------: | ---------------: | ----------: |
| `v_only_075`  |       3 |           -3.399 |           0 |
| `k_only_075`  |       3 |           -1.657 |           0 |
| `coupled_075` |       3 |           -1.466 |           0 |
| `coupled_100` |       3 |           -1.475 |           0 |

## Cases

### case01 (Rule 7)

- Tokens: full prefix `2630`, compact prefix `102`, summary graft pairs `73`
- `lp_A`: `-0.3860`
- `lp_B`: `-2.9254`
- `A-B`: `2.5394`

| policy        |    lp_E | gap closure |
| ------------- | ------: | ----------: |
| `v_only_075`  | -9.3246 |      -2.520 |
| `k_only_075`  | -6.8440 |      -1.543 |
| `coupled_075` | -6.6892 |      -1.482 |
| `coupled_100` | -6.6870 |      -1.481 |

### case02 (Rule 4)

- Tokens: full prefix `2615`, compact prefix `102`, summary graft pairs `73`
- `lp_A`: `-0.5086`
- `lp_B`: `-2.8285`
- `A-B`: `2.3199`

| policy        |    lp_E | gap closure |
| ------------- | ------: | ----------: |
| `v_only_075`  | -9.8801 |      -3.040 |
| `k_only_075`  | -5.1382 |      -0.996 |
| `coupled_075` | -4.9540 |      -0.916 |
| `coupled_100` | -4.8975 |      -0.892 |

### case03 (Rule 9)

- Tokens: full prefix `2606`, compact prefix `102`, summary graft pairs `73`
- `lp_A`: `-0.3542`
- `lp_B`: `-2.0481`
- `A-B`: `1.6939`

| policy        |    lp_E | gap closure |
| ------------- | ------: | ----------: |
| `v_only_075`  | -9.9011 |      -4.636 |
| `k_only_075`  | -6.1693 |      -2.433 |
| `coupled_075` | -5.4367 |      -2.000 |
| `coupled_100` | -5.5240 |      -2.052 |
