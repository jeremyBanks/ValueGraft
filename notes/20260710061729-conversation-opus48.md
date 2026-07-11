_Validation spend-down is nearing completion: the SWE-Gym 2×2 is banked, while
the per-head four-way scan and compression curve remain in progress. Results
support a small reproducible effect under brief summaries, null effect under
production-faithful summaries, and a narrower champion-over-scalar advantage in
the realistic regime._

**Participants:** claude-opus-4-8.

**Handoff State.** The completed SWE-Gym table reports brief scalar and champion
effects of +0.0128 versus baseline, both statistically positive, with champion
tied to scalar. Under production summaries, scalar is −0.0015 and champion
+0.0097 versus baseline, neither clearing zero; however, champion beats scalar
head-to-head by +0.0112 [0.005, 0.018]. The intended interpretation is that
tuning adds value specifically in the harder realistic regime, but does not
rescue the graft above baseline.

The production-champion run completed all 75 examples with both arms preserved,
and its pod was terminated to concentrate resources on the remaining headline
analyses. Two pods remain at roughly $8.00 balance and an estimated ~$2.9/hour
burn. The per-head scan is in late passes; the compression sweep is on its final
`realistic` level, with completion previously estimated at roughly 45 minutes.
Earlier estimates indicated both full 12-pass analyses should fit the remaining
budget, leaving a small reserve for per-head follow-up if a winner emerges.

Operational checks repeatedly confirmed that flat validation counts reflected
long overlapping passes rather than stalls: logs were fresh, GPU utilization was
active, and conversation counts continued advancing. Results were harvested
incrementally, annotated with provenance, and scalar-only backups were retained.

Next priority is to wait for both headline runs to finish, read the complete
confidence intervals, and spend any residual budget only on a targeted per-head
follow-up. The final writeup should preserve the nuanced result and avoid
claiming that champion tuning improves production performance relative to
baseline; the stronger defensible claim is a small brief-summary effect plus a
realistic-regime champion-over-scalar difference.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
