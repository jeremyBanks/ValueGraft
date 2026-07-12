# METHODS-PROVENANCE-REQUIREMENTS.md — successor checklist

This checklist is blocking for any scientific report. It is outcome-neutral:
no finding is assumed in advance.

## Question, estimand, and scope

- Exact scientific question and causal/associational claim boundary.
- Sampling recipe/frame, independent unit, target population (if any), and why
  the uncertainty procedure applies. Fixed benchmark inference is named as
  such.
- Primary/secondary estimands with equations, sign convention, raw components,
  units, damage normalization, inclusion rules, multiplicity, SESOI/bound (if
  used), sequential looks, stopping, and heterogeneity handling.
- Which schedules, regions, precisions, repeats, plants, and hosts are nested
  measurements rather than N.

## Origin of every token and label

For every system/user/assistant/tool message, carrier, retained tail, summary,
probe, target, countertarget, and judged label, record:

- literal text/token IDs and canonical rendering procedure;
- author/generator runtime model and exact version/effort/session where known;
- whether subject-generated, subject-forced replay, independently authored,
  imported, post-edited, or deterministic template text;
- timestamp, source path/hash, selection/rejection procedure, review/adjudication
  record, and whether any outcome was visible to its creator;
- compaction boundary, message/token spans, logical/physical positions, event
  call widths, and leakage class.

Never call imported replay live/native, authored text organic, a fixed carrier a
natural summary, or an execution repeat an independent sample.

## Subject and runtime

- Exact model repository ID, resolved immutable revision, configuration and
  tokenizer/template hashes, license, parameter/cache dtype, quantization,
  attention backend, geometry, context limit, decoding settings, and output
  caps/stop rules.
- Exact code commit/inventory and dirty state; Python, Torch/Transformers/MLX
  versions and lockfiles; GPU UUID/type/memory/driver/CUDA, host/provider tier,
  datacenter when available, and residency/offload checks.
- Every load conversion or packed-MoE topology mapping with independent count/
  tensor evidence.

## Intervention and controls

- Literal source/destination layouts; K/V source for every arm; selected rows,
  layers, heads, alpha, dtype conversions, RoPE/position policy, schedule, and
  causal recomputation boundary.
- Exact identities, confinement/nonselected preservation, positive path control,
  can-it-fail fixture, wrong-history control, nonfocal outcome control, identity
  placebo/noise floor, and matched placebo construction.
- Applied-dtype/state-geometry availability evidence. Missing controls and
  failed constructions remain visible.
- Numerical schedule/kernel floor and how it limits the smallest interpretable
  bound.

## Corpus and exclusions

- Full authored pool, frozen ordering/randomization, domain/length/subtype
  balance, author clustering, reserves, attrition, decoded/mechanical reviews,
  oracle/headroom/damage gates, and every exclusion with treatment-blindness
  evidence.
- No silent fixture replacement or post-outcome repair. Failed candidates and
  contaminated cases remain archived.

## Execution, persistence, and cost

- Exact run command/config, unique output path printed at start, start/end UTC,
  wall times by stage, provider rate/balance window, session/case spend, retries,
  and termination verification.
- Per-case atomic renders/checkpoints contain full text, token IDs, summary,
  traces, row hashes, raw arm scores, environment, config, and resume state.
- State bounds, keys, eviction, ownership, resume equivalence, independent
  harvest, byte hashes, result commit, and push evidence.

## Analysis and reporting

- Analysis code committed before outcomes; exact command, seeds, raw-to-summary
  derivation, missing-data handling, collapse-to-independent-unit step, interval
  formula, alpha spending, sensitivity analyses, and adversarial tests.
- Report every raw component needed to distinguish correct-target gain from
  countertarget suppression and utility from history specificity.
- Point estimates, simultaneous intervals/bounds, per-unit distribution,
  behavioral flips, placebo/noise results, schedule/host/precision sensitivity,
  and null/negative results.
- Explicitly name what the evidence does not establish. A confidence bound is
  scoped to its sampling recipe and tested model/config/locus.

Before publication, independent reviewers check the paper line by line against
this file and the literal artifacts. Any unresolved provenance gap blocks ship.
