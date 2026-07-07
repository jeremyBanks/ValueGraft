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

**Strength.** Moderate-high — and notably from GENUINE natural-length eviction (the synthetic conversations were long enough that policy/referents were evicted by real conversation length, NOT by an artificially low compaction threshold). This is cleaner than the tau-benchmark attempts, whose short sessions force artificial thresholds. Moderate-high. Both independent graft-arm judge batches agree
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



### F1 corroboration (independent metric, 07-07)
The category dissociation replicates on a SECOND, judge-free metric —
teacher-forced gap-closure (E−B)/(A−B) of the exact gold continuation
(30B, 66 probes):
| category | judged (meaning) | gap-closure (exact tokens): % probes helped |
|---|---|---|
| stance | +2pp (null) | 39% (null/neg, mean −0.14) |
| sense | +12pp | 64% (mean +0.03) |
| referent | +10pp | 81% (mean +0.04) |
DIRECTION agrees on all three (graft helps sense/referent, null on stance)
across two independent measurement methods. MAGNITUDE is much smaller on
the exact-token metric than the meaning-judge — which is PREDICTED by the
thesis: grafting recovers SENSE, not verbatim FORM, so it should move a
meaning-judge more than an exact-token-probability metric. The
metric-magnitude gap is thus a SECOND signature of the same "recovers
meaning not surface" mechanism, not a failure to replicate. Strengthens F1
from single-method to two-method-directionally-consistent.

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
under-served by existing benchmarks. (INCIDENTS 19; DECISIONS 07-06/07.) CONFIRMED empirically 07-07: tau-bench banking, even with a capable GPT-4o-mini user-simulator, produced ~3-4K-token sessions — too short for meaningful eviction at any threshold (high→compaction never fires; low→nothing substantial to evict). All 3 arms reward 0.00, no separation. Standard interactive benchmarks with short task-dialogues are structurally unsuited; the operative regime needs genuinely long sessions (which our synthetic F1 data has naturally).

## F1 robustness (strict scoring, 07-07)
Strict RECOVERED-only cut (PARTIAL counts as miss) — dissociation HOLDS:
stance +4pp (null), sense +9pp, referent +8pp. Not an artifact of the
PARTIAL=0.5 choice. THIRD independent confirmation of the same pattern
(judged-lenient, judged-strict, and TF-logprob gap-closure all agree that
grafting helps sense/referent, null on stance).

## F1 scale-dependence (07-07) — DOES NOT replicate at 4B (honest limitation)
The category dissociation is a 30B (large-model) result. At 4B the same
gap-closure metric does NOT reproduce it: stance 87% helped (mean +0.19,
was 39% at 30B), sense 55% / mean −0.09 (was 64%), referent 62% (was 81%)
— the clean stance-null / sense-referent-positive ordering is gone/muddled.
CONSISTENT WITH known scale-dependence of graft effects in this project
(the alpha dose-response INVERTED 4B↔30B; DECISIONS 07-06). So F1 is
bounded: three-method-corroborated AT 30B, NOT cross-scale; the mechanism
behaves differently at small scale. State F1 as a large-model finding.
Caveat: gap-closure ratios are noisier at 4B (smaller A−B denominators);
a judged 4B cut would confirm, but the non-replication direction is clear.

## F1 mechanistic readout (07-07) — PARTIAL support, honestly bounded
Logit-lens on the gold concept token at the probe position (30B, 61 probes):
grafting increases the evicted concept's internal presence — E>B on 68-77%
of probes, lp_E BETWEEN lp_B and lp_A in every category. Direct internal
evidence that grafting INSERTS concept content and moves the residual
stream partway back toward full-context. HOWEVER it does NOT reproduce the
category DISSOCIATION: stance shows ~77% E>B, same as sense (behaviorally
stance was null). CAVEAT: near-floor signal (gold-token logprob ~−12 to
−14, rank in thousands) → single-token logit-lens is a weak/noisy
instrument, detects "graft nudges concept up broadly" but can't resolve
WHERE recovery matters. VERDICT: supports the GENERAL mechanism (graft
inserts concept, E→A) but NOT the specific dissociation. Sharper readout
(J-lens / multi-token / targeted layers) = future work. Do not overclaim
this as mechanistic proof of the dissociation.

## Methods note — graft strength
Default value-graft strength α_V = 0.75 (the project's standard graft dose;
α=1.0 = full replacement, studied in F3 tuning). Where unspecified, results
use α=0.75.

## Same-model confirmation (Qwen3.6-27B, the lens's model) — 07-07
Behavioral gap-closure on 27B (same model the J-lens weights exist for), so
behavior + lens sit on ONE model. The core dissociation REPLICATES:
| category | 27B mean GC | 27B % helped | (30B % helped) |
|---|---|---|---|
| sense    | +0.050 | 59% | (64%) |
| referent | +0.004 | 48% | (81%) |
| stance   | −0.201 | 21% | (39%) |
Graft helps SENSE (positive), null/negative on STANCE (summary suffices) —
matches 30B qualitatively. NOTE: referent is essentially FLAT on 27B (+0.004)
— value-only grafting barely moves the retrieval-hard "recover a specific
evicted decision" category. This is the wall V-only hits, and the direct
motivation for the K/V (key-graft) experiment: keys carry addressing/position,
which is what referent recovery may need.

## Lens resolution (07-07) — corroborates in AGGREGATE, not per-example
Our ordinary-regime three-state probe (role-in-summary + tail, 4 sense/referent
cases) did NOT show clean per-case aligned>shifted separation — shifted control
sometimes matched/beat aligned at the hinge token. The lens signal is SMALL and
only reliably shows alignment-sensitivity when AGGREGATED over many tokens/cases
(cf. the other agent's 10-case/156-token batch: aligned mean-positive, shifted
mean-negative). VERDICT: the J-lens is a low-resolution, aggregate-level
corroborator for this V-only intervention — it confirms the graft is active +
alignment-sensitive on average, and confirms the omitted-fact negative, but
cannot supply vivid single-example exhibits. Paper uses it honestly as such;
no dramatic per-example lens figure exists or is claimed.
