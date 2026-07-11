_The project continued with chain-tier validation as the gated exploration
phase, reserving tau2-bench for a funded, pilot-qualified confirmatory phase.
The completed 20-cell grid confirmed that per-layer tuning prevents
full-strength graft collapse, but coarse pass/fail results showed no compaction
damage, shifting priority to finer-grained diagnostics._

**Participants:** User, claude-opus-4-8, and claude-fable-5.

Tau remains the designated future confirm-phase backbone: tau2
`banking_knowledge` with core tau domains as controls. It is conditional on (1)
a genuine chain-tier graft effect, (2) funding, and (3) a roughly $3 pilot
verifying compaction depth, deterministic rewards, and tolerable simulator cost.
If the pilot fails, the fallback is scaling the chain tier. No tau execution was
authorized during this phase.

The remaining budget was initially reported as approximately $36.50 after
$163.50 spent from $200 loaded; a later update reported balance at $64.92 after
a top-up. Remaining spend was directed to completing the chain-arm table. The
expected table completion window was 3–5 hours, with immediate escalation for
implausible results; champion verdicts were expected within roughly 30 minutes
once prioritized.

Operationally, the grid stalled because follower scripts had been created with
spaces in their filenames by an unquoted shell loop, making them unlaunchable. A
later `tail -f` design also failed because it replayed completed queue lines and
followers shared a spec file. The workaround was to replace followers with three
sequential runners over the available tunnels, prioritizing champion episodes. A
summary-script parsing bug then falsely displayed champion cells as blank
because a `-c` splitter matched the “c” in `cfg`; raw data inspection corrected
the display.

The final chain table was:

- s1: Original 4/4, Compacted 4/4, graft@0.75 4/4, graft@1.0 0/4, champion 4/4
- s3, s4, s6: every measured arm 4/4, including champion

Thus the per-layer-tuned champion replicated successfully on all four seeds,
including recovery of s1’s full-strength graft failure. This supports the
mechanical hypothesis that α=1.0 pushes some layers off-manifold and that tuning
or zeroing harmful layers removes the instability. However, Compacted also
passed 4/4 on every seed, so the chains currently provide no coarse evidence of
compaction damage for grafting to repair. The result is presently “tuned
grafting does not harm,” not evidence of recovery.

Next priority is analysis of per-exercise scores and recall probes, especially
late-chain behavior under peak context pressure, since binary chain pass/fail
may conceal degradation. If those finer signals also show no compaction effect,
the chain instrument must be hardened through longer chains, tighter compaction
thresholds, or stronger cross-exercise dependencies before confirm-phase
spending.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
