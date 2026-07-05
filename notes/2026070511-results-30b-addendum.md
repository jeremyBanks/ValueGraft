# RESULTS addendum — 30B scale replication (Qwen3-30B-A3B-Instruct-2507, 4-bit)

Targeted arm set (A, B, B-min, C, D, H-gap, E-post α∈{0.25, 1.0}), both summary
conditions, same corpus, same protocol; ladder L0–L4 re-passed on this model
before any result. Judged by Sonnet (798 verdicts, logged). Negative controls at
30B pending (queued after Phase 2). n = 12 synthetic + 8 natural — pilot scale.

## Continuation NLL (synthetic, n=12; paired vs B, bootstrap CIs)

| contrast         | mean Δ                   | wins  | 95% CI          | 4B comparison                        |
| ---------------- | ------------------------ | ----- | --------------- | ------------------------------------ |
| H-gap − B-min    | +0.128                   | 12/12 | [0.097, 0.158]  | +0.093, 10/12 — **grows with scale** |
| E-post α=1 − B   | +0.052 (~29% of A−B gap) | 10/12 | [0.021, 0.086]  | −0.135 (harmful) — **sign flips**    |
| E-post α=.25 − B | +0.018                   | 11/12 | [0.010, 0.026]  | +0.020 — stable                      |
| C − B            | −0.019                   | 5/12  | [−0.057, 0.017] | −0.165 — harm vanishes               |

The 4B's K/V-decoupling fragility (high-α collapse, E-inter disaster, C's NLL
cost) largely disappears at 30B; full-strength contiguous value grafting becomes
net-positive. **Caveat — natural corpus does not replicate:** A−B gap there is
only 0.07 nats (little compaction damage to recover) and all contrasts are ≈0 or
slightly negative (H−B-min −0.020, 2/8). The gains are conditional on the
continuation actually depending on evicted content, which the synthetic corpus
guarantees and free-form chat often doesn't.

## Probe accuracy, clean cut (brief/shadow condition, judged)

| category     | B     | B-min | C         | H-gap | E α=.25 | E α=1 |
| ------------ | ----- | ----- | --------- | ----- | ------- | ----- |
| referent     | 2/17  | 1/17  | 3/17      | 1/17  | 2/17    | 2/17  |
| sense        | 11/22 | 4/22  | **15/22** | 5/22  | 11/22   | 10/22 |
| evicted-fact | 0/24  | 0/24  | 0/24      | 0/24  | 0/24    | 0/24  |

C beats B by +18pp on clean sense probes (was +8pp at 4B); referent remains
floored for every compacted arm at both scales; the laundering control stays
perfect (oracle 9/9 in std condition, all compacted arms ~0).

## The honesty effect at 30B (evicted facts, fabricated : admitted)

| arm          | std     | brief    |
| ------------ | ------- | -------- |
| B            | 9:1     | 15:9     |
| B-min        | 8:1     | 16:8     |
| C            | 4:6     | 5:19     |
| H-gap        | **3:7** | **1:23** |
| E-post α=.25 | 9:1     | 14:10    |
| E-post α=1   | 8:3     | 11:13    |

Replicates the 4B finding and strengthens it: with only the summary's write-time
cache entries retained, the 30B fabricates once in 24 unknowable probes; the
identical summary as fresh text fabricates 15–16 times. The effect ranks by how
much original cache state an arm retains (H-gap > C > E > B), consistent with
retained state carrying a usable signal about the _extent_ of what was actually
discussed.

## Interim synthesis (pre-Phase-2)

At 30B, the deployable interventions produce: (1) a large, consistent reduction
in post-compaction fabrication (H-gap), (2) a modest clean-sense accuracy gain
(C), (3) a real continuation gain on compaction-sensitive content (E-post α=1,
~29% gap closure) with none of the 4B's fragility. Referent recovery remains
unsolved by every method tested. Phase 2 (H-pack vs B-min-pack) tests whether
(1) survives the packed, deployable layout; negative controls at 30B are
required before (2)/(3) are certified.
