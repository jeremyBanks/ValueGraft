_This conversation covers the completed 4B/30B KV-cache semantic-continuity
experiments and the pivot to mitigation: packed write-time summaries and value
grafts improve post-compaction behavior when continuations depend on evicted
context._

**Participants:** User, claude-sonnet-5, and claude-fable-5.

**Handoff State.** Phase 1 is closed and committed with its addendum; Phase 2
was run, judged, merged, and committed at both scales. Claude Sonnet 5 subagents
judged the queues, with exact key/count verification. The durable interpretation
is conditional: write-time cache state carries usable context, but natural
conversations showed little recoverable compaction damage (30B A−B gap ~0.07
nats; H-gap−B-min −0.020), unlike synthetic dependency-designed conversations.

At 30B, H-gap exceeded B-min by +0.128 nats in all 12 conversations, CI [0.097,
0.158]. E-post α=1 exceeded B by +0.052 in 10/12, closing roughly 29% of the
compaction gap; α=0.25 gave +0.018 in 11/12. At 4B, the tuned mid-band α=0.25
graft produced a holdout gain of +0.017 nats in 10/10 conversations, CI [0.012,
0.024], roughly 10% gap closure. The completed synthesis reports packed
write-time summaries as nearly eliminating post-compaction fabrication and value
grafting as closing approximately 24% of the continuation gap at 30B versus ~10%
at 4B.

Judging consistently found that full-context A answers recovered referents and
evicted facts, while degraded B-family conditions often confidently fabricated
plausible specifics. H-pack/H-gap/wrong-source variants more often admitted
uncertainty, avoiding fabrication but failing legitimate referent recovery.
Negative controls and the 30B replication certified the mechanism; caveats are
that recall of evicted referents remains unsolved, some honesty improvement is
layout-induced caution, results are pilot-scale (n=12 synthetic plus n=8
natural), use one model family in 4-bit, and do not generalize to content
independent of evicted context.

Deployment applicability is limited to compaction chosen while the original
cache remains available. H-pack only retains the summary cache; ValueGraft
requires the old cache transiently at graft time, after which it can be freed,
so neither addresses cache loss after restart or migration.

The α sweep found different optima by scale: 4B favored mid-band α=0.25, while
30B initially favored late-band gating and α=1.25 extrapolation, requiring the
completed holdout analysis for final selection. The experimental program is
complete; remaining work is documentation and scouting: produce the HF-style
blog-post draft with an honest literature pass, then assess benchmark fit,
especially LongMemEval, SCBench/RULER, and possibly LongBench v2, including
token-length feasibility and whether 4B/30B baselines show measurable effects.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `aace54520a911d304`
- `ad67f93ca685091b3`
- `ad744fa57a134fd78`
- `a4ed6475705d2452e`
- `a171f187aaeaacc50`
- `a07bd4453ba350867`
- `a152ede29ed57e700`
- `a9864cc95a498ce1d`
- `ad64b2d88f0b91869`
- `a93c2ae1790d34366`
- `aa526c35c0b47b524`
- `abd1de70a11ec4542`
- `a6cd255f880cdbaa8`
- `ab350d26d8b84a47c`
- `a4e19c420dda5fae5`
- `a29d5ad486a383e34`
- `a8e07a54a32eeea0b`
