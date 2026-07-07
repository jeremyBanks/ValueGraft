# Policy Registry Follow-Up Notes

This note records the second cheap local screen for synthetic referent-recovery
task shapes. It uses `Qwen/Qwen3-0.6B` on MPS, so the numbers are prospecting
data only. The purpose is to find regions worth testing more carefully, not to
support a paper claim.

## What Changed From The First Screen

The first screen suggested that arbitrary high-entropy payloads were a poor
starting point. The useful shape was lower entropy:

- a private label survives in a sparse summary;
- the label-to-action relation is omitted;
- the model is asked to recover the omitted relation;
- grafted summary-token K/V is tested by teacher-forced gold logprob.

This follow-up widened that shape into two families:

- **Private Policy Registry:** labels such as `Citrine` or `Opal` map to
  familiar implementation policy phrases.
- **Named Transform Registry:** labels such as `Coral` or `Azure` map to short
  code/data transforms such as `userName` or `UTC`.

For each case, the runner measures:

- `A`: full payload context, used as the reachable ceiling;
- `B`: sparse-summary compacted context, used as the floor;
- `E`: the compacted context after grafting write-time summary-token K/V onto
  the same summary-token positions.

The sweep included V-only, K-only, coupled K/V, and small mixed policies. The
runner writes the complete output to `outputs/policy_registry_followup.json` and
a compact table report to `outputs/policy_registry_followup.md`.

## Main Results

Full run: 46 cases in 68 seconds.

| family                | cases | usable A>B | positive best graft | mean best E-B | mean best gap closure |
| --------------------- | ----: | ---------: | ------------------: | ------------: | --------------------: |
| policy choice         |    30 |         30 |                  28 |        +0.265 |                +0.036 |
| low-entropy transform |    16 |         15 |                  12 |        +0.454 |                +0.206 |

The policy family is the stable lane. Every policy case had `A > B`, and 28 of
30 had at least one graft policy improve over `B`. The signal is small as a
fraction of the gap, but broad.

The transform family is more selective. It has fewer cases and more failures,
but the strongest cases have much larger gap closure. This is where the
K-sensitive structure appears.

## Policy Registry Read

The policy family mostly prefers low-dose V-only grafting:

- `v005` and `v010` are the best broad policies.
- High-dose V-only (`v020`, `v030`) can win individual cases but is worse on
  average.
- K-only is generally not useful for the policy family.

This makes the policy registry a good candidate for a robust behavioral gate: it
is cheap, creates a clear full-vs-compacted gap, and usually shows a small
positive graft effect.

The best policy categories in this tiny screen were privacy/export,
release/review, cache policy, and UI rendering. Routing was weaker but still
mostly positive.

## Transform Registry Read

The transform family is the better K-focused search area. The global K-only
average is negative, but several specific transform categories strongly prefer
K-only:

| case                         | gold                | best policy |    E-B | gap closure |
| ---------------------------- | ------------------- | ----------- | -----: | ----------: |
| `transform-typed-iris`       | `user_profile_card` | `k020`      | +1.385 |       0.763 |
| `transform-typed-azure`      | `UTC`               | `k020`      | +1.360 |       0.811 |
| `transform-label_only-iris`  | `user_profile_card` | `k020`      | +1.034 |       0.602 |
| `transform-label_only-coral` | `userName`          | `k010`      | +0.792 |       0.109 |
| `transform-label_only-azure` | `UTC`               | `k010`      | +0.683 |       0.548 |

That pattern is narrow, but interesting. Identifier transforms and timezone
normalization are the clearest local hits. Privacy redaction did not work in
this screen, despite having a normal `A > B` gap.

The next transform screen should expand around identifier normalization,
filename/case conversion, and timezone/unit normalization. It should not assume
that "transform registry" as a whole is K-sensitive.

## Summary Style

Typed summaries helped slightly. A typed summary keeps labels plus coarse
categories while omitting the exact relation. It made all policy cases positive
and improved the transform family mean.

That suggests the sparse summary should preserve a semantic anchor. A bare label
list may be too under-conditioned for some cases; a full relation would make the
task trivial. The useful middle is "label plus type, relation omitted."

## Recommended Next Step

Use a two-lane synthetic harness:

1. **Policy lane:** 30-50 private labels mapped to familiar policy actions. Use
   this as the stable V-sensitive gate. Sweep low-dose V-only and low-dose
   coupled policies first.
2. **Transform lane:** 30-50 labels concentrated around identifier/case
   conversion, filename normalization, timezone normalization, and simple unit
   normalization. Use this as the K-sensitive diagnostic lane. Sweep low-dose
   K-only and coupled K/V.

Both lanes should keep summaries sparse but typed. Each generated case should
freeze the payload, summary, probe, and gold answer in JSONL before evaluation
so runtime generation variance is not mixed into the graft measurement.

## Caveats

- This is a 0.6B local model screen, not a result about the larger subject
  models.
- Context lengths here are small compared with the intended long-context
  compaction regime.
- Teacher-forced logprob is a clean cheap signal, but generated answers and a
  strict meaning judge are still needed after a candidate shape passes the
  screen.
- The current table records best-of-policy outcomes, so the next serious run
  should pre-register the policy sweep and report the full surface, not only
  winners.
