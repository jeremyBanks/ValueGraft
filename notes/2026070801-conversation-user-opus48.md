Correct model verified (+0.1246 referent, matching the +0.136 known-positive,
correct sense/stance dissociation) — the gate is genuinely green on the original
12-conversation corpus. The new c13–c54 conversations show real evicted content
(gap 1.12) but the graft recovers almost none of it (+0.009), so doubling the
corpus currently weakens rather than strengthens the signal.

**Participants:** User and claude-opus-4-8.

I'd run the wide cross-architecture sweep now on the trusted original 12 convs —
that's a clean, validated positive control and gets the actual paper result (the
sign-map across 16 models) without further delay. The new-conversation quality
issue is a separate, non-blocking problem: investigate afterward why
model-mix-rendered scenarios don't evict/recover as cleanly (likely something
about how the plants are embedded or how far they sit from the compaction
boundary), then decide whether to fix and re-add them or keep the corpus at 12.

Do you want me to proceed that way — fire the wide sweep on c01–c12 now, park
the corpus-augmentation fix for after?
