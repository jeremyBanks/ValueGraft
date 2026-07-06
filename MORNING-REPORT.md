# Morning report — night of 07-05 → 07-06

*Everything below is committed with full data trails; DECISIONS.md has the
per-item entries. Spend: started the night ~$101 after your top-up;
$62.35 at 04:50 with 7 pods running (~$9.5/hr at full width).*

## Headline: the end-to-end agent experiment works, and a tuned config leads

The serving shim + OpenHands pipeline ran ~80 objective coding-task
episodes overnight (real agent, server-side compaction, pytest verdicts).
Table as of 04:50 (runs still accumulating):

| condition | pass rate | n |
|---|---|---|
| A (no compaction) | 100% | 5 |
| B (plain compaction) | 91% | 11 |
| B @ tighter thresholds (c6000/c4500) | 10/11 | 11 |
| E global α=0.75 (all thresholds) | **29%** | 17 |
| E global α=1.0 | 78% | 9 |
| E other α (0.25/0.5/1.5) | 6/7 | 7 |
| **E per-layer tuned (cfg=layers)** | **100%** | **9** |
| E anti-graft (α=−0.5) | 40% | 5 |
| E shuffled control | 2/2 | 2 |

Three big reads, in decreasing confidence:
1. **The per-layer tuned config is the night's champion — 9/9 including
   the harder c4500 tier** — and it passed its wrong-conversation guard
   (content-dependence confirmed, see Tuning below). This is the "how far
   can we push it" exhibit you asked for, pending confirm-phase n.
2. **The first instrument was too easy**: B passes ~90% because single
   seeded constraints survive summaries/tail echoes (round-1's dramatic
   B-fails/E-passes didn't generalize — it was run on unseeded tasks
   pre-cache-semantics). The c4500 confirm tier is running now; harder
   multi-constraint tasks are the next lever if B stays ceilinged.
3. **Anomaly, honestly flagged**: scalar α=0.75 specifically underperforms
   (5/17) while neighboring α values and the map-based config do well.
   Round-1 (pre-summary-cache shim) had α=0.75 winning 4/4. Suspects:
   noise clustering (17 runs is small), or an interaction between scalar-α
   grafting and the frozen-boundary summary cache. NOT resolved; treat all
   α=0.75 agent rows as provisional until re-run under a no-cache A/B of
   the shim semantics.

## Tuning story (the paper's methodological spine)

- Global α=0.75 transfers 4-bit→bf16 (validated). 30B profiles are diffuse
  (no layer band, no star head; heads explain 1.4% of variance — your
  factored-form prediction confirmed).
- The 57-slot mask beat global on TF holdout (+0.038, 10/10) — **and then
  FAILED its pre-registered wrong-conversation guard** (foreign values
  through those slots help ≈ as much: content-INDEPENDENT, a
  regularizer-like artifact). Dead as a finding; reported honestly.
- The per-layer coarse config ({0, 0.75, 1.0} from the layer profile)
  **PASSED the same guard** (foreign values hurt, −0.09..−0.21) and leads
  the agent table. The guard methodology discriminates real structure from
  fished structure — that contrast is itself a result.
- 4B side-story: factored clamped > signed (your geometry argument lost
  its holdout test; both below the simple mid-band rule).

## What else landed tonight

- **Stage 2 (SWE-Gym replay) FINAL**: E-tuned recovers +0.0156 nats on
  real agent traces, 45/75 wins, CI [0.005, 0.027], 10% of the A−B gap.
- **Stage 1 (LongMemEval, now closed per your call)**: damage number
  52.5% → 4.1% (n=350, bf16); arms equivalent on fact-QA; graft does not
  increase fabrication. 1b/H200 killed before spend.
- **Honesty suite at bf16**: complete (12 convs × 72 probes × 4 arms);
  Sonnet judging launched this hour — fabrication:admission table lands
  this morning.
- **Gemma**: 6 failures deep, parked at a real boundary (hybrid
  sliding-window attention needs its own scoring path); 5 template fixes
  committed; the porting-cost ledger is write-up material. Resume point
  documented.
- **Wildcard pod**: α=2.0 agents pass 1/2, α=1.5 passes 2/2 — extrapolation
  tolerance in-agent extends further than TF metrics suggested; recall
  probes captured on every run (dissociation analysis pending judging).
- **Mistral pre-tuning**: template adapter ready (system-less rendering);
  pod run not yet scheduled — today's queue.

## Failure ledger (all in DECISIONS with fixes)

Silent-death class fixed (5-min dead-job + new-error alarms, no
exemptions; count-validated completion markers); ssh-detach race
documented + patterned; orphan pod from launcher race found and adopted
(e3); LME full-protocol needs >80GB for A-arm (documented, then LME
killed anyway); ceiling effect caught by A/B controls.

## Recommended next steps (your morning decisions)

1. **E2 scale-up**: run the confirm design (A/B/E-map/E-1.0) to n≥20/arm
   on harder multi-constraint tasks — the champion's 9/9 needs adversarial
   n. ~$25-35. My recommendation: yes.
2. Resolve the α=0.75 anomaly (cheap: 8 re-runs with cache disabled).
3. Honesty + dissociation tables land this morning automatically.
4. Mistral variety run (~$5) and/or Gemma hybrid scoring path (~2h code)
   — optional breadth.
5. The sealed final eval (s50-s99 + sight-unseen templates) stays locked
   until the champion is stable.
