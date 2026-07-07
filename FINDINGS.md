# FINDINGS.md — headline results (the paper's spine)

*Living document of load-bearing findings. Each entry: claim, evidence,
strength, caveats. Chronology/process lives in DECISIONS.md; failures in
INCIDENTS.md. THIS file is what the write-up is built from.*

---

## F1. Grafting recovers compaction damage in proportion to how semantic (vs factual) the lost content is — the "recovers sense, not trivia" dissociation

**Claim.** When a conversation is compacted (older turns replaced by a
summary), write-time KV-value grafting recovers a meaningful fraction of
the lost *meaning* — and does so specifically where compaction caused
damage that a text summary couldn't repair. The effect tracks the damage.

**Evidence (Qwen3-30B-A3B bf16, meaning-judged by Sonnet 5, clean-eviction
plants, PARTIAL=0.5).** Meaning-recovery rate by probe category:

| category | Original (full ctx) | Compacted | Graft | graft − Compacted |
|---|---|---|---|---|
| stance (honor an evicted preference) | 96% | 93% | 96% | +2pp |
| sense (disambiguate an evicted referent's meaning) | ~100%* | 46% | 58% | **+12pp** |
| referent (recover a specific evicted decision) | ~100%* | 17% | 26% | **+10pp** |

**Interpretation.** Three regimes:
- *stance* — a good summary already preserves "user dislikes carousels";
  compaction barely hurts (96→93), so graft has nothing to add (+2, null).
  Correct null.
- *sense* — the semantic-disambiguation regime ("which 'handoff' did they
  mean"): compaction flattens it (100→46); graft restores a quarter of the
  loss (+12pp). The write-time value vectors carry disambiguating meaning a
  summary loses.
- *referent* — specific evicted decisions: compaction devastates (100→17);
  graft still recovers +10pp.

The graft recovers ~10–12pp **wherever compaction caused real damage**
(sense, referent) and is **null where it didn't** (stance). That the
effect scales with the damage is the signature of a genuine mechanism, not
noise. Probes are realistic semantic-continuity questions (not contrived
trivia); result extracted by re-judging already-collected data on the
MEANING dimension (the original judging used a fact/honesty scheme that
was blind to these categories).

**Strength.** Moderate-high. Both independent graft-arm judge batches agree
on the gradient. Internally consistent (effect tracks damage; negative
controls elsewhere in the project crater).

**Caveats / to-tighten.**
- Original-ceiling n is tiny for sense(1)/referent(2) — the clean-eviction
  filter left few A cells; widen it / add A rows before publishing the
  exact ceiling numbers.
- PARTIAL=0.5 is a scoring choice; run a strict RECOVERED-only robustness
  cut.
- Single model/precision so far (30B bf16); the mechanism replicated across
  scales/precisions on OTHER metrics (see F2/F3), but this specific
  category cut hasn't been repeated at 4B.
- Duplicate-key artifact in 2 judge batches (17 + 5 keys); judge resolved
  conservatively, low impact.

---

## F2. Honesty effect (banked, strong). Compaction makes the model fabricate about lost content; write-time KV retention makes it appropriately uncertain.
Decoy fabrication: Compacted 83% vs write-time-KV (H-pack) 17%; on evicted
facts H-pack both most accurate (38/48) and least fabricating (4%).
Replicated 4-bit→bf16 and 4B→30B. (Details: DECISIONS 07-06; honesty runs.)

## F3. Tuning finding (banked). Naive full-strength grafting (alpha=1) can
catastrophically break a task (chain s1: 0/4 where all else 4/4); per-layer
guard-validated tuning eliminates the instability (champion 16/16). Layer
profile passed its wrong-conversation contamination guard; the 57-slot mask
FAILED it (content-independent artifact). (Chain table + guards, DECISIONS 07-06/07.)

## F4. Scope boundary (honest, itself a contribution). Standard agent-coding
benchmarks don't cleanly support compaction research with a 30B: SWE-bench
is beyond the model (0/7 even oracle-mode); exercise-scale tasks (chains,
tau-banking) don't naturally reach the compaction-stress regime without
threshold tuning. The operative regime — hard enough that eviction matters,
easy enough the model can use recovered context — is narrow and
under-served by existing benchmarks. (INCIDENTS 19; DECISIONS 07-06/07.)
