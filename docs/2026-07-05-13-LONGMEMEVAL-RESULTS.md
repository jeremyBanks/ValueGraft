# LongMemEval results (standard-benchmark validation)

Setup: LongMemEval-S (xiaowu0162/longmemeval-cleaned, MIT), question types
single-session-user / single-session-assistant / knowledge-update. Per
question we assemble a ≤16K-token multi-session conversation with the
evidence session inside the evicted region and distractor sessions as the
retained tail; terse summary; question appended as the final user turn.
n=48 at 4B, n=36 at 30B (memory incident cost a batch — see DECISIONS).
Sonnet-judged CORRECT/FABRICATED/ADMITTED; summary-leak flagged per item
(1/84 leaked; excluded from clean cuts).

## Tables (counts)

**4B (n=48):**
| arm | correct | fabricated | admitted |
|---|---|---|---|
| A (full) | 34 | 6 | 8 |
| B (production) | 1 | **17** | 30 |
| E-tuned | 3 | 15 | 30 |
| B-min-pack | 3 | 12 | 33 |
| H-pack | 4 | **11** | 33 |
| H-gap | 4 | 11 | 33 |

**30B (n=36, clean):**
| arm | correct | fabricated | admitted |
|---|---|---|---|
| A | 29 | 3 | 3 |
| B | 2 | 8 | 25 |
| E-tuned | 2 | 8 | 25 |
| B-min-pack | 3 | 9 | 23 |
| H-pack | 4 | 9 | 22 |
| H-gap | 4 | 9 | 22 |

## Reading

1. **The compaction problem itself replicates perfectly on standard data:**
   full context answers 71–81% of questions; every compacted variant
   collapses to ≤11% correct. Nothing we built recovers evicted *recall* —
   consistent with all prior phases.
2. **The honesty effect is frame-dependent.** At 4B, H-pack cuts fabrication
   ~35% vs production compaction (11 vs 17) — the direction from our
   synthetic corpus holds on real data. At 30B it vanishes (9 vs 8): the
   larger model's "I don't have access to your personal history" calibration
   already fires on this benchmark's personal-fact questions, so the
   production baseline fabricates rarely and there is no headroom. Contrast
   the synthetic corpus at 30B (B fabricated 16:8; H-pack 1:23): there, the
   compacted frame — an in-progress working conversation with a retained
   tail — invites the model to keep confabulating as a participant.
3. Judges independently noted the fabrication channel that remains on real
   data: plausible *world knowledge* substituted for lost personal facts
   (wrong-but-real song titles, states, policies) — a channel our synthetic
   decoys couldn't offer. And narrative/descriptive questions invite far
   more confabulation than plain factual ones, in every arm.
4. Implication for deployment claims: cache-state honesty interventions
   matter where compaction happens mid-task in agentic/working contexts
   (which is where production compaction actually operates), not in
   retrieval-style personal QA where refusal training already covers the
   gap. That is a scope statement, not a retraction — but it belongs in any
   write-up.
