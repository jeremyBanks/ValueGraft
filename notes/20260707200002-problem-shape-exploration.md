# Problem Shape Exploration

- Model: `Qwen/Qwen3-0.6B`
- Device: `mps`
- Cases: `14`
- Elapsed seconds: `19.5`

This is a local design-search pass. It asks which small problem shapes are worth
turning into a real referent-recovery harness.

## Family Summary

| family                  |  n | A>B | mean A-B | mean best E-B | mean best GC | positive cases | best policies                                                                          |
| ----------------------- | -: | --: | -------: | ------------: | -----------: | -------------: | -------------------------------------------------------------------------------------- |
| `bug_fix`               |  2 |   2 |    6.101 |        -0.048 |       -0.010 |              1 | bugfix-quartz:v005, bugfix-basil:v005                                                  |
| `format_order`          |  2 |   2 |    6.305 |         0.039 |        0.006 |              2 | format-frame-18:v005, format-frame-44:v005                                             |
| `low_entropy_transform` |  4 |   4 |    3.518 |         0.157 |        0.058 |              3 | transform-coral:kv050, transform-slate:k010, transform-amber:v005, transform-pine:k005 |
| `policy_choice`         |  4 |   4 |    6.998 |         0.462 |        0.064 |              4 | policy-citrine:v025, policy-violet:v005, policy-marble:v025, policy-copper:v025        |
| `sense_label`           |  2 |   2 |   11.239 |         0.003 |        0.001 |              1 | sense-nimbus:v005, sense-hydra:v010                                                    |

## Case Details

### sense-nimbus (sense_label)

- Label: `Nimbus`
- Gold: `checkout funnel experiment`
- Rationale: Lower-entropy version of the Pokemon/Nimbus style: recover sense,
  not an exact arbitrary string.
- `lp_A`: `-0.6565`
- `lp_B`: `-12.5028`
- `A-B`: `11.8463`
- Best policy: `v005` (`E-B` -0.1964, GC -0.016581402321967132)

| policy  |      E-B | gap closure |
| ------- | -------: | ----------: |
| `v005`  |  -0.1964 |      -0.017 |
| `v010`  |  -0.6374 |      -0.054 |
| `v025`  |  -1.6333 |      -0.138 |
| `v050`  |  -2.6593 |      -0.224 |
| `k005`  |  -0.2228 |      -0.019 |
| `k010`  |  -0.3328 |      -0.028 |
| `k025`  |  -3.0333 |      -0.256 |
| `kv005` |  -0.4298 |      -0.036 |
| `kv010` |  -1.1370 |      -0.096 |
| `kv025` |  -4.8088 |      -0.406 |
| `kv050` | -10.1754 |      -0.859 |

### sense-hydra (sense_label)

- Label: `Hydra`
- Gold: `multi-account permission collapse`
- Rationale: Lower-entropy version of the Pokemon/Nimbus style: recover sense,
  not an exact arbitrary string.
- `lp_A`: `-0.5529`
- `lp_B`: `-11.1855`
- `A-B`: `10.6326`
- Best policy: `v010` (`E-B` 0.2034, GC 0.019125296199962186)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  |  0.1719 |       0.016 |
| `v010`  |  0.2034 |       0.019 |
| `v025`  | -0.0653 |      -0.006 |
| `v050`  | -1.9941 |      -0.188 |
| `k005`  | -0.2887 |      -0.027 |
| `k010`  | -0.7771 |      -0.073 |
| `k025`  | -3.2594 |      -0.307 |
| `kv005` | -0.0466 |      -0.004 |
| `kv010` | -0.2819 |      -0.027 |
| `kv025` | -3.2472 |      -0.305 |
| `kv050` | -7.3246 |      -0.689 |

### policy-citrine (policy_choice)

- Label: `Citrine`
- Gold: `render the view as a compact comparison table`
- Rationale: A candidate task where the missing relation is familiar and
  action-like.
- `lp_A`: `-0.4095`
- `lp_B`: `-5.9829`
- `A-B`: `5.5734`
- Best policy: `v025` (`E-B` 0.3807, GC 0.06830272808024183)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  |  0.0770 |       0.014 |
| `v010`  |  0.1089 |       0.020 |
| `v025`  |  0.3807 |       0.068 |
| `v050`  | -0.1294 |      -0.023 |
| `k005`  | -0.1013 |      -0.018 |
| `k010`  | -0.3412 |      -0.061 |
| `k025`  | -2.2987 |      -0.412 |
| `kv005` | -0.0223 |      -0.004 |
| `kv010` | -0.1409 |      -0.025 |
| `kv025` | -1.4714 |      -0.264 |
| `kv050` | -1.9596 |      -0.352 |

### policy-violet (policy_choice)

- Label: `Violet`
- Gold: `send the request through the manual escalation queue`
- Rationale: A candidate task where the missing relation is familiar and
  action-like.
- `lp_A`: `-0.4459`
- `lp_B`: `-7.0478`
- `A-B`: `6.6019`
- Best policy: `v005` (`E-B` 0.0927, GC 0.014044585136785434)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  |  0.0927 |       0.014 |
| `v010`  |  0.0828 |       0.013 |
| `v025`  | -0.0665 |      -0.010 |
| `v050`  | -2.0177 |      -0.306 |
| `k005`  | -0.2487 |      -0.038 |
| `k010`  | -0.5486 |      -0.083 |
| `k025`  | -1.2411 |      -0.188 |
| `kv005` | -0.0939 |      -0.014 |
| `kv010` | -0.2032 |      -0.031 |
| `kv025` | -1.7879 |      -0.271 |
| `kv050` | -5.4797 |      -0.830 |

### policy-marble (policy_choice)

- Label: `Marble`
- Gold: `drop the optional analytics field before export`
- Rationale: A candidate task where the missing relation is familiar and
  action-like.
- `lp_A`: `-0.2436`
- `lp_B`: `-8.1938`
- `A-B`: `7.9502`
- Best policy: `v025` (`E-B` 0.7722, GC 0.09712627485726553)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  |  0.4065 |       0.051 |
| `v010`  |  0.6583 |       0.083 |
| `v025`  |  0.7722 |       0.097 |
| `v050`  | -0.8341 |      -0.105 |
| `k005`  | -0.2318 |      -0.029 |
| `k010`  | -0.7650 |      -0.096 |
| `k025`  | -3.0089 |      -0.378 |
| `kv005` |  0.2082 |       0.026 |
| `kv010` |  0.1050 |       0.013 |
| `kv025` | -2.5393 |      -0.319 |
| `kv050` | -8.7343 |      -1.099 |

### policy-copper (policy_choice)

- Label: `Copper`
- Gold: `prefer the cached preview over a fresh screenshot`
- Rationale: A candidate task where the missing relation is familiar and
  action-like.
- `lp_A`: `-0.2230`
- `lp_B`: `-8.0911`
- `A-B`: `7.8680`
- Best policy: `v025` (`E-B` 0.6034, GC 0.07668601380015369)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  |  0.2492 |       0.032 |
| `v010`  |  0.3834 |       0.049 |
| `v025`  |  0.6034 |       0.077 |
| `v050`  | -0.7380 |      -0.094 |
| `k005`  | -0.1310 |      -0.017 |
| `k010`  | -0.2344 |      -0.030 |
| `k025`  | -1.8175 |      -0.231 |
| `kv005` |  0.1237 |       0.016 |
| `kv010` |  0.1440 |       0.018 |
| `kv025` | -1.1303 |      -0.144 |
| `kv050` | -2.4656 |      -0.313 |

### bugfix-quartz (bug_fix)

- Label: `Quartz`
- Gold: `add pointer-events: none to the transparent overlay`
- Rationale: Code-adjacent but not requiring exact full code generation.
- `lp_A`: `-0.0348`
- `lp_B`: `-5.1397`
- `A-B`: `5.1050`
- Best policy: `v005` (`E-B` -0.1253, GC -0.024538591334018638)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  | -0.1253 |      -0.025 |
| `v010`  | -0.4495 |      -0.088 |
| `v025`  | -1.9761 |      -0.387 |
| `v050`  | -4.5151 |      -0.884 |
| `k005`  | -0.4339 |      -0.085 |
| `k010`  | -1.1934 |      -0.234 |
| `k025`  | -4.6651 |      -0.914 |
| `kv005` | -0.4477 |      -0.088 |
| `kv010` | -1.3653 |      -0.267 |
| `kv025` | -5.8292 |      -1.142 |
| `kv050` | -9.8026 |      -1.920 |

### bugfix-basil (bug_fix)

- Label: `Basil`
- Gold: `invalidate the member-count cache after role changes`
- Rationale: Code-adjacent but not requiring exact full code generation.
- `lp_A`: `-0.0687`
- `lp_B`: `-7.1655`
- `A-B`: `7.0968`
- Best policy: `v005` (`E-B` 0.0291, GC 0.0040999137307192205)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  |  0.0291 |       0.004 |
| `v010`  | -0.0454 |      -0.006 |
| `v025`  | -0.6978 |      -0.098 |
| `v050`  | -2.6724 |      -0.377 |
| `k005`  | -0.1238 |      -0.017 |
| `k010`  | -0.4175 |      -0.059 |
| `k025`  | -3.2879 |      -0.463 |
| `kv005` | -0.0782 |      -0.011 |
| `kv010` | -0.4160 |      -0.059 |
| `kv025` | -2.7848 |      -0.392 |
| `kv050` | -6.9779 |      -0.983 |

### transform-coral (low_entropy_transform)

- Label: `Coral`
- Gold: `userName`
- Rationale: Still code-ish, but the answer is a familiar low-entropy
  transformation.
- `lp_A`: `-7.2361`
- `lp_B`: `-10.7727`
- `A-B`: `3.5366`
- Best policy: `kv050` (`E-B` 0.1831, GC 0.05176704908809014)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  | -0.0429 |      -0.012 |
| `v010`  | -0.2254 |      -0.064 |
| `v025`  | -0.7283 |      -0.206 |
| `v050`  | -7.4509 |      -2.107 |
| `k005`  | -0.2549 |      -0.072 |
| `k010`  | -0.7142 |      -0.202 |
| `k025`  | -2.2932 |      -0.648 |
| `kv005` | -0.1468 |      -0.041 |
| `kv010` | -0.6285 |      -0.178 |
| `kv025` | -2.4542 |      -0.694 |
| `kv050` |  0.1831 |       0.052 |

### transform-slate (low_entropy_transform)

- Label: `Slate`
- Gold: `timestamp descending`
- Rationale: Still code-ish, but the answer is a familiar low-entropy
  transformation.
- `lp_A`: `-1.2227`
- `lp_B`: `-7.8119`
- `A-B`: `6.5893`
- Best policy: `k010` (`E-B` 0.4969, GC 0.0754027288115157)

| policy  |      E-B | gap closure |
| ------- | -------: | ----------: |
| `v005`  |  -0.3067 |      -0.047 |
| `v010`  |  -0.7750 |      -0.118 |
| `v025`  |  -3.1540 |      -0.479 |
| `v050`  |  -4.6630 |      -0.708 |
| `k005`  |   0.3229 |       0.049 |
| `k010`  |   0.4969 |       0.075 |
| `k025`  |  -1.0592 |      -0.161 |
| `kv005` |  -0.0307 |      -0.005 |
| `kv010` |  -0.4508 |      -0.068 |
| `kv025` |  -2.4387 |      -0.370 |
| `kv050` | -11.1937 |      -1.699 |

### transform-amber (low_entropy_transform)

- Label: `Amber`
- Gold: `redact email addresses`
- Rationale: Still code-ish, but the answer is a familiar low-entropy
  transformation.
- `lp_A`: `-0.7876`
- `lp_B`: `-4.1263`
- `A-B`: `3.3387`
- Best policy: `v005` (`E-B` -0.1394, GC -0.04173870364239069)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  | -0.1394 |      -0.042 |
| `v010`  | -0.3597 |      -0.108 |
| `v025`  | -1.4970 |      -0.448 |
| `v050`  | -6.4842 |      -1.942 |
| `k005`  | -0.2277 |      -0.068 |
| `k010`  | -0.4239 |      -0.127 |
| `k025`  | -1.3694 |      -0.410 |
| `kv005` | -0.3293 |      -0.099 |
| `kv010` | -0.5845 |      -0.175 |
| `kv025` | -2.1154 |      -0.634 |
| `kv050` | -6.1647 |      -1.846 |

### transform-pine (low_entropy_transform)

- Label: `Pine`
- Gold: `deduplicate by account id`
- Rationale: Still code-ish, but the answer is a familiar low-entropy
  transformation.
- `lp_A`: `-3.4843`
- `lp_B`: `-4.0936`
- `A-B`: `0.6093`
- Best policy: `k005` (`E-B` 0.0885, GC 0.14520374685527)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  | -0.0229 |      -0.038 |
| `v010`  | -0.1610 |      -0.264 |
| `v025`  | -0.8992 |      -1.476 |
| `v050`  | -3.4521 |      -5.665 |
| `k005`  |  0.0885 |       0.145 |
| `k010`  |  0.0383 |       0.063 |
| `k025`  | -2.5780 |      -4.231 |
| `kv005` |  0.0571 |       0.094 |
| `kv010` | -0.0832 |      -0.137 |
| `kv025` | -2.8247 |      -4.636 |
| `kv050` | -4.4045 |      -7.228 |

### format-frame-18 (format_order)

- Label: `Frame 18`
- Gold: `tenant | region | feature | checksum`
- Rationale: A compact structured answer; may be too exact, but less arbitrary
  than full code.
- `lp_A`: `-0.0485`
- `lp_B`: `-6.3543`
- `A-B`: `6.3058`
- Best policy: `v005` (`E-B` 0.0673, GC 0.010669568343477227)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  |  0.0673 |       0.011 |
| `v010`  |  0.0470 |       0.007 |
| `v025`  | -0.6560 |      -0.104 |
| `v050`  | -4.0358 |      -0.640 |
| `k005`  | -0.0213 |      -0.003 |
| `k010`  | -0.0868 |      -0.014 |
| `k025`  | -0.7708 |      -0.122 |
| `kv005` |  0.0599 |       0.010 |
| `kv010` |  0.0234 |       0.004 |
| `kv025` | -1.5258 |      -0.242 |
| `kv050` | -3.0310 |      -0.481 |

### format-frame-44 (format_order)

- Label: `Frame 44`
- Gold: `namespace | key | version | digest`
- Rationale: A compact structured answer; may be too exact, but less arbitrary
  than full code.
- `lp_A`: `-0.0246`
- `lp_B`: `-6.3291`
- `A-B`: `6.3045`
- Best policy: `v005` (`E-B` 0.0103, GC 0.0016409427894867766)

| policy  |     E-B | gap closure |
| ------- | ------: | ----------: |
| `v005`  |  0.0103 |       0.002 |
| `v010`  | -0.1200 |      -0.019 |
| `v025`  | -0.8216 |      -0.130 |
| `v050`  | -3.0531 |      -0.484 |
| `k005`  | -0.0973 |      -0.015 |
| `k010`  | -0.1806 |      -0.029 |
| `k025`  | -1.0076 |      -0.160 |
| `kv005` | -0.1050 |      -0.017 |
| `kv010` | -0.3467 |      -0.055 |
| `kv025` | -2.0289 |      -0.322 |
| `kv050` | -4.1778 |      -0.663 |
