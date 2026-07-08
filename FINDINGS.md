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

## Phase 2 F-entry — Key-grafting is technically sound (07-07)
Built K-grafting with RoPE RE-ROTATION (re-rotate a stored write-time key by
the position delta p_new−p_old, since RoPE composes by angle: R(p_new)=
R(delta)·R(p_old)). VALIDATED on 0.6B to fp32 precision: re-rotated key vs
freshly-encoded key at new position = cosine 0.99999982, max-diff 3e-05 (vs
0.9964 un-rotated control). α_K=0 bit-identical to fresh; α_K=1 changes output.
So the K-graft is not an approximation — keys can be moved across positions
exactly. Enables the V/K/coupled/independent sweep. src/kv_graft.py.

## Phase 2 result — Key-grafting does NOT help (coarse untuned, 30B) — 07-07
Behavioral gap-closure, 6 global-uniform policies, 30B, sanity-passed (V-only
referent +0.057 ≈ paper's positive 30B value → K-graft path trustworthy):
| cat | v_only | k_only | coupled | ind_k50 | ind_v50 | ind_k100 |
|---|---|---|---|---|---|---|
| sense | +0.027 | −0.283 | −0.024 | +0.005 | −0.103 | −0.014 |
| referent | +0.057 | −0.096 | +0.029 | +0.023 | −0.016 | +0.076 |
| stance | −0.017 | −0.211 | −0.128 | +0.035 | −0.165 | −0.151 |
VERDICT: VALUE is the operative axis. K-only actively HURTS all categories
(re-rotated keys perturb attention). Coupled/independent don't beat V-only on
referent or sense. ind_k100 referent +0.076 vs +0.057 = within noise (n=21).
The RoPE-addressing hypothesis (keys recover referent where values floored) is
NOT supported by UNIFORM grafting. The 0.6B hint (keys help referent) was noise.
BOUNDS/OPEN: (1) this is 30B where V-only ALREADY works on referent (+0.057) —
less room for keys to add; the exact "where-V-floored" test is 27B (referent
flat +0.004), not yet run (needs kv_graft 27B-path port). (2) COARSE UNTUNED
(uniform αK all layers) — uniform K-graft could average out layer-specific
effects; PER-LAYER/PER-HEAD key tuning (kv_graft supports it) might find a
key-profile that helps, but the strongly-negative K-only lowers that prior.

## Phase 2 CONCLUDED — Key-grafting doesn't help, even per-layer (30B) — 07-07
Per-layer key probe (add α_K=0.75 at ONE layer L on top of the working value
graft, referent, n=21 plants, 12 layers sampled). v_only baseline +0.0565.
Best layer (24) lift = +0.0111 (only 10/21 plants positive = coin flip);
k_only@L NEGATIVE at ALL 12 layers. NO layer meaningfully lifts referent.
COMBINED WITH the coarse uniform sweep (keys don't help, K-only hurts all
cats): the K/V exploration is CONCLUSIVE — VALUE is THE operative axis;
key-grafting does not recover referent uniformly OR per-layer. RoPE-addressing
hypothesis (keys carry "where to look" for retrieval) thoroughly UNSUPPORTED.
Per-head not run (per-layer clean-negative + k_only-negative-everywhere makes
it a dead direction; stopped per direction-against discipline). This STRENGTHENS
the paper: not "value is the axis we tested" but "value is THE axis — keys
checked uniformly + per-layer, don't help."

## Phase 2 SCOPING CORRECTION (Fable cross-check vs referent_recovery_microtest) — 07-07
My "VALUE is THE operative axis" was OVERCLAIMED — universal-sounding, but our
corpus has only ONE target morphology: semantic referent PHRASES (Nimbus=signup
funnel). A separate microtest (referent_recovery_microtest, 0.6B prospecting)
found that for a DIFFERENT shape — label→short TOKEN-like identifier (Coral→
userName, Azure→UTC) — KEY-grafting often WINS (k020/k010 best, GC up to ~0.8).
CRUCIAL: where the two studies OVERLAP (semantic policy_choice phrases) they
AGREE — the microtest reproduces our result (V-only dominates, K-only 0/30
positive). So the divergence is in UNTESTED territory (short identifiers), not a
contradiction. The microtest signal is statistically WEAK (0.6B, winner's-curse
best-of-15-policy selection, tiny cells n=2-4, K-only global mean still negative)
— it CANNOT move our claim toward "keys help", only stop it being UNIVERSAL.
Mechanism makes the flip plausible: keys=positional addressing; short-identifier
recovery = retrieval-by-address, not semantic reconstruction.
HONEST SCOPED CLAIM (replaces the earlier universal one):
> For SEMANTIC-referent recovery (decision/label → semantic phrase), VALUE-
> grafting is the operative axis — key-grafting doesn't help (uniform or
> per-layer, 30B rigorous), K-only hurts; RoPE-addressing unsupported FOR
> SEMANTIC-PHRASE TARGETS. Whether keys matter for SHORT TOKEN-LIKE IDENTIFIER
> targets (addressing/retrieval) is OPEN — only noisy 0.6B evidence hints yes.
TESTABLE PREDICTION: target morphology MODERATES K-graft utility (keys help for
short identifiers, inert for semantic phrases). WARRANTED: a focused 30B run —
add a short-identifier target lane, pre-register K/V/coupled sweep, report FULL
policy surface (not best-of) — before ANY general claim about keys.

## Lens free-divergence result — clean pre-registered NEGATIVE (27B, N=43) — 07-07
The free-generation fix (remove teacher-forcing pin → let A/B/E freely generate,
find fork, lens at fork) did NOT surface a vivid internal exhibit.
THREE-BUCKET: FORK_TOWARD_A = 0, SUBTLE_LEAN = 17, DISCONFIRMING = 26.
ZERO cases show a clean fork toward the correct concept with decisive margin +
decoding stability. (referent: 0/9/12; sense: 0/8/14.) Because the DISCONFIRMING
bucket was PRE-REGISTERED (no fork, OR fork-away, OR right-lean doesn't survive
3 decodings), the 26 can't be relabeled "subtle" — this is a genuine null, not a
heads-I-win. CONCLUSION: even the sharper free-gen method finds no dramatic
internal fork; the effect is genuinely subtle at the token level. The paper's
existing honest framing (lens = low-resolution aggregate corroborator, no vivid
per-example figure, not manufactured) STANDS and is STRENGTHENED — we tried the
method built to reveal it and it didn't appear. (Fable's pre-registered
disconfirming bucket is what makes this honest not spun.)

## Ops note (incident 28 addendum): process-gone + GPU-freed = COMPLETION *or* crash
Distinguish before diagnosing: check (a) output file exists, (b) log reached the
LAST expected item — if both, it FINISHED (terminate idle pod); only if neither,
it crashed. I mis-called a completed run as "crashed, no output" by reading
proc-dead as failure. RULE 26 addendum: on proc-gone, check finished-vs-crashed
explicitly; a done job means terminate the now-idle pod promptly.

## Effect-bound (placebo-controlled) — 27B, N=43, K=12 — 07-07
Teacher-forced shared gold[:12] pre-divergence window; A/B/E/placebo bit-identical
keys; placebo = norm-matched random-value graft (seeded derangement).
- E−placebo = +0.445, CI[+0.286,+0.612] EXCLUDES 0 → graft DECISIVELY beats random
  values; placebo−B = −0.428 CI[−0.590,−0.276] (random-value graft HURTS badly).
  = rigorous, spin-proof "alignment/content-specific, not generic perturbation."
- E−B = +0.017, CI[−0.033,+0.065] INCLUDES 0 → PRE-REGISTERED VERDICT: NULL. Over
  the pre-divergence window on 27B the graft doesn't clearly beat plain compaction.
  Mean weakly positive, consistent with 27B's modest effect (sense +0.05, referent
  +0.004 §7); underpowered at n=43 + narrow window (first 12 gold tokens ~ answer
  preamble). NOT a "graft does nothing" — content-specificity is strong (vs placebo).
NEXT (primary-strengthen): run SAME bound on 30B (strong model) — E−B should
resolve positive there; that's the placebo-controlled CI that strengthens the primary.

## Effect-bound 30B (K=12 pre-window) — SURPRISE, needs full-window re-run — 07-07
30B, N=43, K=12: E−B = −0.066 CI[−0.130,−0.007] EXCLUDES 0 NEGATIVE (graft HURTS
next-token pred over the first 12 gold tokens!); placebo−B = −0.306; E−placebo =
+0.241 CI[+0.033,+0.457] EXCLUDES 0 (graft still content-specific, beats random).
27B was E−B null; 30B is E−B negative — on BOTH the pre-window E−B ≤ 0.
INTERPRETATION (honest, not spin): K=12 measures the answer PREAMBLE, not the
content tokens where the +10-12pp gap-closure benefit lands. Graft helps the model
commit to right CONTENT (later tokens), slightly perturbs generic opening tokens →
K=12 catches perturbation, misses payoff. In teacher-forcing there's NO divergence
to avoid (all arms score identical gold tokens) so the K=12 cap (from the free-gen
design) was over-conservative. RIGHT measurement = FULL gold continuation (matches
gap-closure). RE-RUNNING at full window to validate: if E−B resolves POSITIVE over
full continuation → confirms primary + adds placebo control; if still ≤0 → REAL
tension with the gap-closure headline, must confront. Content-specificity (E−placebo
>0 both models) is solid regardless.

## ⚠️ CRITICAL — effect-bound 30B does NOT reproduce F1 positive gap-closure (07-07)
Full-window (K=48) 30B effect-bound, 43 sense+referent cases, UNIFORM α=0.75,
per-model 30B-generated summary. Computed the PAPER's gap-closure metric from the
same data:
- Overall 37% cases helped (E>B); ratio mean −0.034 median −0.016.
- sense: 27% helped, ratio −0.054 (NEGATIVE). referent: 48% helped, ratio −0.013.
- Compaction DID damage (A−B +1.96, 100% cases A>B) — setup valid.
- E beats PLACEBO (content-specific) but does NOT beat plain compaction B.
PAPER HEADLINE (F1, 66 probes, 30B): sense 64% helped, referent 81% helped. This
run is OPPOSITE. This is a POTENTIAL NON-REPRODUCTION of the core result and must
be resolved before trusting the primary or expanding (cross-arch HELD).
LIKELY EXPLANATIONS (to verify, NOT assume in our favor):
1. CONFIG: this uses UNIFORM α=0.75; paper §8 says uniform disrupts, per-layer
   CHAMPION needed. Original F1 config (champion? uniform?) must be checked;
   re-run effect-bound with champion to see if effect returns.
2. CASES/SUMMARY differ (43 free_div plants vs 66 F1 probes; per-model summary).
3. Original F1 less robust than presented.
ACTION: hold cross-arch, diagnose config vs original F1, re-run w/ champion, Fable
conceptual read, report to user. If primary not robust → paper (on README) OVERCLAIMS,
correct before external repro. HONEST — do not spin.

## ⚠️⚠️ APPARATUS INSTABILITY — F1 does not reproduce run-to-run (07-07, CRITICAL)
Re-ran the ORIGINAL F1 code (gap_closure_cat.py, α=0.75, same 43 cases) LIVE on 30B.
Does NOT cleanly reproduce the saved F1:
- sense: 64% helped BUT mean_gc −0.14 (saved +0.03) — SIGN FLIP on mean.
- referent: 71% helped, +0.10 (saved 81%/+0.04) — directionally ok, noisy.
- stance: 58% helped, +0.09 (saved 38%/−0.31) — DID NOT reproduce as null! flipped.
Same code, same inputs, DIFFERENT results → NOISY apparatus. Root causes:
1. GEN_TEMP=0.0 → summary is GREEDY (NOT sampled). But MoE argmax flips on near-tie
   tokens across hardware/runs → different summary cascade. (Ruled out sampling.)
2. Qwen3-30B-A3B is MoE — routing on bf16/hardware is NONDETERMINISTIC → summary
   AND teacher-forcing logprobs vary run-to-run/hardware. Ratio metric (small A−B
   denominators) AMPLIFIES the variance.
IMPLICATION: the saved F1 numbers are ONE sample from a noisy distribution,
presented in the paper as point estimates WITHOUT error bars. The dissociation
(esp. stance-null) is NOT stable run-to-run. Effect-bound wasn't necessarily buggy
— it may be within the noise band. THE APPARATUS MUST BE STABILIZED before ANY
conclusion. FIX: (a) FIXED summary (remove summary variance — use fixed_summaries),
(b) run N seeds → gap-closure mean±CI per category, (c) re-establish dissociation
WITH error bars or honestly report it's noisier than presented. Paper (on README)
currently OVERSTATES robustness — must fix before external repro. HOLD everything
downstream. Do NOT spin — this is a real problem with the measurement.

## ✅ RESOLVED (Fable + user push) — effect is REAL; instability was the MEAN-RATIO estimator, not the effect (07-07)
Mined the TWO saved runs (results/gap_closure_cat vs _live, 67 common probes) — NO
GPU needed (I was thrashing on env/GPU re-runs; the answer was on disk).
FINDING: the effect REPRODUCES on every ROBUST metric; only mean-of-ratio swings.
  raw E−B:      referent +0.156/+0.125, sense +0.062/+0.047, stance −0.026/+0.002
  median ratio: referent +0.118/+0.101, sense +0.033/+0.090, stance −0.068/+0.019
  % helped:     referent 81/71, sense 64/59, stance 38/54
  mean ratio (BROKEN): sense sign-flips, stance −0.308 vs +0.091.
Both runs, all robust metrics: SAME dissociation — graft raises gold logprob for
referent+sense, ≈0 for stance. The "−0.31 stance-null" = TWO probes with near-zero
|A−B| denominators (Cauchy blow-up). Condition |A−B|>0.5 → even mean ratio stable
(stance −0.029/+0.039 ≈ 0 both).
METRIC FIX (Fable): DEMOTE mean-ratio. Primary = raw lp_E−lp_B (bounded) + %-helped;
median ratio secondary; if ratio kept, winsorize/condition on |A−B| + report n.
NEVER report bare mean-ratio again.
HONEST POSITION: effect REAL, reported with UNSTABLE estimator + NO error bars.
README overstates PRECISION (point estimates, dramatic −0.31, no CI), NOT existence/
direction. = numbers correction + error bars, NOT retraction.
NOTE: this supersedes the "APPARATUS INSTABILITY / does NOT reproduce" alarm above —
that alarm conflated "mean-ratio swung" with "effect unstable"; they're different.
The MoE/summary-cascade nondeterminism is real but 2nd-order (lp_A varies ~0.03).
NEXT: (1) recompute paper table on robust metrics (both runs, side by side, no GPU);
(2) T1 fixed-summary 2-run determinism floor; (3) T2 N=5 → mean±CI; (4) cross-arch
uses raw E−B not mean-ratio. Effect-bound probe uses a DIFFERENT teacher-forcing
(raw E−B negative there) — retire it, trust gap_closure_cat.

## PRIMARY EFFECT with proper CONFIDENCE (bootstrap CIs over probes, raw E-B, live/5.13) — 07-07
Apparatus is DETERMINISTIC within-env (live1==live2 exact) → confidence = bootstrap
over the PROBE SAMPLE (n=21-24/cat), N=10000:
- referent: +0.125 CI[+0.030,+0.218] EXCLUDES 0 → REAL significant effect. 71% helped.
- sense:    +0.047 CI[−0.038,+0.131] SPANS 0 → suggestive but UNDERPOWERED (n=22).
- stance:   +0.002 CI[−0.036,+0.049] SPANS 0 → genuinely NULL (as claimed).
HONEST: effect REAL for referent (significant), suggestive-underpowered for sense
(logprob metric; judged +12pp carries it), null for stance. The clean "sense+referent
recovered" dissociation is MORE NUANCED on the logprob metric than the paper implies —
only referent is significant there. n=21-24/cat is TOO SMALL for tight CIs = the real
limitation, FIXABLE by more probes.
STANDARD METRIC GOING FORWARD: raw E-B (bounded) + %-helped + bootstrap 95% CI over
probes. Never bare mean-ratio. The JUDGED +12pp is now load-bearing for sense →
needs its own bootstrap CIs (next audit).
EXTENDED PLAN now has statistical PURPOSE: more probes/scenarios to POWER the effect
(esp. sense), CIs throughout. Cross-arch, champion-tune, judged-audit all on this footing.

## ⚠️ K/V conclusion ALSO used the broken ratio metric (user caught it, 07-07)
kv_layer_probe.py line 19: gap_closure=(E-B)/(A-B) — SAME unstable mean-ratio.
So "K-only hurts all categories (sense −0.283, referent −0.096, stance −0.211)"
and "per-layer negative" are RATIO-INFLATED artifacts. Recomputed on RAW E-B:
- v_only referent: +0.120 CI[−0.001,+0.228] — REAL value effect (matches F1 +0.125).
- k_only per-layer: ALL near-zero (−0.020..+0.016), tiny — keys are ~NEUTRAL,
  NOT dramatically harmful. The dramatic "keys hurt −0.28" was the estimator.
CORRECTED K/V read: "value is THE operative axis" HOLDS (value positive, keys
don't HELP), BUT "keys actively hurt" is FALSE on robust metric — keys ~neutral.
Per-layer negatives are real but tiny perturbations, not impossible.
ACTION: the paper §10 K/V section needs the SAME robust-metric correction as F1 —
replace ratio numbers (−0.283 etc.) with raw E-B; reframe "K-only hurts" →
"keys ~neutral, don't help". Add to task 29 (paper correction). GENERAL LESSON:
EVERY result using (E-B)/(A-B) mean-ratio is suspect — audit all of them on raw E-B.

## Cross-arch Qwen2.5-32B: NEGATIVE raw E-B, diagnosing (07-08)
Qwen2.5-32B (dense GQA, validation model) with fixed Sonnet summary: raw E-B AGGREGATE
-0.28 CI[-0.37,-0.20] EXCLUDES 0 (graft HURTS), pre_gap A-B +0.61 (compaction DID
damage, setup valid). Significant HURT (not null). CANDIDATES: (1) architecture doesn't
transfer (user's hypothesis, live); (2) cross-arch harness bug specific to Qwen2.5 template
(region detection). RULED OUT: alignment (difflib aligns 100%, 0 dropped on Qwen2.5) and
fixed-summary (alignment perfect). DECISIVE TEST running: trusted gap_closure_cat.py on
Qwen2.5 (bypasses cross-arch harness). Don't conclude from 1 model — exploring all 7.
ALIGNMENT NOTE (user flagged difflib): build_alignment difflib is fragile overkill (drops
<8-tok runs silently) but VERIFIED NOT compromising results (100% aligned everywhere
checked). Being simplified to direct span map (task 31) for robustness, equivalence-gated.

## ✅ Cross-arch datapoint 1: Qwen2.5-32B grafts NEGATIVE — REAL ARCHITECTURE (07-08)
Value graft REVERSES on Qwen2.5-32B (dense GQA). Confirmed by 3 independent runs:
cross-arch fixed-summary raw E-B -0.28, cross-arch self-gen -0.32, and TRUSTED
gap_closure_cat.py (exact F1 code, model-gen summary) referent -0.26/sense -0.38/
stance -0.25 (all significantly negative, %pos 4-19%). Harness VALIDATED (trusted
matches cross-arch). NOT a bug — Qwen2.5 genuinely grafts negative where Qwen3-30B-
MoE grafts +0.12. INTERPRETATION: the effect is ARCHITECTURE-SPECIFIC and can
REVERSE — strong evidence it's a real mechanism, NOT a generic artifact (an
artifact wouldn't flip sign by architecture). Map entry: Qwen3-MoE +, Qwen2.5-dense −.
Keep exploring (Gemma next, parallel pod2). Note: Qwen2.5 vs Qwen3 differ in
dense-vs-MoE AND generation — cause of the reversal is open (n=2).

## 🚨 NOTABLE: cross-arch POSITIVE CONTROL FAILED — fixed-summary suspected of breaking the graft (07-08)
Ran Qwen3-30B-A3B (our F1 model, known +0.156 referent / +0.062 sense via
gap_closure_cat) through the CROSS-ARCH HARNESS with the FIXED SONNET summary.
RESULT: referent +0.004 (CI spans 0), sense −0.147 (CI EXCLUDES 0, NEGATIVE),
agg −0.073. pre_gap +0.51 (compaction did damage, valid). The harness does NOT
reproduce the known positive — it's null-to-negative.
THE ONLY DIFFERENCE from F1: summary source. F1 = model's OWN GENERATED summary;
cross-arch = FIXED SONNET (foreign) summary. STRONG SUSPICION: the fixed-summary
design SUPPRESSES the graft. MECHANISTIC FIT: the graft re-injects the model's
write-time state from GENERATING its own summary (its own compression act) — a
foreign summary the model merely READ may not carry that continuity. Would mean
the fixed-summary sweep is BIASED toward null/negative → understates the effect →
Qwen2.5's "architecture" negative is partly suspect (though Qwen2.5 was ALSO
negative on the trusted model-gen path, so that one may be real).
ISOLATION TEST RUNNING: cross-arch harness SELF-GEN summary on Qwen3-30B. If ~+0.156
→ harness OK, FIXED SUMMARY is the culprit → switch whole sweep to self-gen (accept
Fable's summary-quality confound, handle via pre-gap normalization). If still null →
deeper harness bug. IMPLICATIONS: (a) Mistral currently running on FIXED summary =
biased, needs self-gen re-run; (b) all fixed-summary sweep numbers suspect until
resolved; (c) POSSIBLE MECHANISTIC FINDING: graft needs the model's OWN summary
(would be a real insight about how ValueGraft works, pending confirmation).
LESSON: always run a POSITIVE control (reproduce a known result) before trusting a
new harness — the negative-agreement (Qwen2.5) was NOT sufficient validation.

## ✅✅ RESOLVED + MECHANISTIC FINDING: graft needs the model's OWN summary (07-08)
Isolation CONFIRMED. Qwen3-30B via cross-arch harness:
- FIXED Sonnet summary: referent +0.004 (null), sense −0.147 — POSITIVE CONTROL FAILED.
- SELF-GEN (own) summary: referent +0.136 CI[+0.034,+0.23] 81% helped, sense +0.045
  64% helped, agg +0.090 CI[+0.022,+0.154] SIGNIFICANT_POSITIVE — MATCHES F1
  (+0.156/81%, +0.062/64%).
Same model+harness, only summary source differs → the FIXED (foreign) summary
SUPPRESSES the graft; the model's OWN generated summary reproduces the effect.
CONCLUSIONS: (1) HARNESS VALIDATED (reproduces known positive on self-gen).
(2) FIXED-SUMMARY DESIGN BROKEN → sweep switches to SELF-GEN summaries.
(3) MECHANISTIC FINDING (real, not speculation now): ValueGraft re-injects the
write-time state of the model's OWN summarization ACT — a summary the model merely
READ doesn't carry the recoverable continuity. Enriches the paper's mechanism.
MAP status: Qwen2.5-dense NEGATIVE is REAL (self-gen −0.32 AND trusted model-gen
−0.30, both). Qwen3-MoE POSITIVE. Mistral (fixed-summary ~null) = BIASED, re-run
self-gen. CONFOUND (Fable): self-gen summary quality varies across models → report
pre_graft_gap + summary token-count per model, use conditioned ratio.

## POSITIVE CONTROL PASSES on correct model (07-08) — effect confirmed real
After the wrong-model saga (incident 34), the trusted apparatus (gap_closure_cat.py,
difflib, self-gen) on the CORRECT model Qwen3-30B-A3B-Instruct-2507 reproduces the effect:
  orig c01-c12: referent raw_EB=+0.1246 (71% helped), sense +0.047, stance +0.002 (~null).
The dissociation (referent>sense>stance~0) reproduces exactly. The known +0.136 is confirmed.
CAVEAT: new convs c13+ show WEAKER referent (+0.009 ~null), sense +0.083, stance -0.096 -->
the doubled corpus DILUTES rather than strengthens on referent. Checking pre_graft_gap (A-B)
on new convs to decide: small gap = effect-tracks-damage (fine); large unrecovered gap =
new convs are lower-quality rendering (fix before wide).

## ⚠️ NATIVENESS CONFOUND (Fable, 07-08) — potentially paper-fatal, test BEFORE wide spend
Fable's general read caught a confound one level up from the wrong-model: the +0.1246 positive
control on c01-c12 may be reliable BECAUSE the original corpus was generated IN-CONTEXT by a
Qwen-family model (old MLX pipeline = Qwen3-4B). So c01-c12 is NATIVE to Qwen, FOREIGN to
everyone else — the SAME condition that killed the new convs (foreign replies -> under-recover).
IF SO: +0.1246 is a Qwen-native artifact, and on a fixed shared corpus the cross-arch "SIGN"
would track PER-MODEL NATIVENESS, not attention geometry -> a gorgeous sign-map that's really a
NATIVENESS map. Same ghost class as the wrong-model, one level up.
DECISIVE PRE-SPEND TEST: run the c01-c12 referent positive control on ONE non-Qwen arch
(Mistral/Llama). If raw_EB COLLAPSES there like the new convs did -> nativeness dominates ->
redesign before spending on 16. One model, existing corpus, hours not days.
MECHANISM REFINEMENT: my "foreign write-side" hypothesis is the WEAKER half; the DOMINANT term
is likely the TARGET side — E's continuation is ALSO foreign, and the graft has no reason to
raise the probability of text the model wouldn't produce. Also can't yet exclude: the 1.12 new-conv
A-B gap is generic distributional surprise (foreign text lower-prob), NOT graft-shaped continuity.
CHEAP DISTINGUISHERS (data in hand): (1) nativeness regression — score each conv by mean per-token
logprob of its assistant replies under the test model, regress raw_EB on it; (2) dissociation
decomposition — does the new-conv A-B gap have the referent>sense>stance signature? if flat, it's a
different non-graftable gap.
OTHER RED FLAGS (Fable): per-arch value-graft INDEX ALIGNMENT must be re-verified PER MODEL (differs
by tokenizer/arch; silent misalignment = plausible garbage — what crashed before). NOISE FLOOR:
stance +0.002, new +0.009 -> sign resolution near zero is marginal at n=12; a sign inside the noise
band isn't a sign (need per-conv bootstrap CIs). Attention-geometry predictor must be computed
INDEPENDENTLY of the outcome (pre-register) or it's post-hoc fitting.
PATH (Fable): run wide on reliable c01-c12 BUT gate on (i) the one non-Qwen nativeness control +
(ii) the ~free nativeness/dissociation analysis FIRST. REJECT re-rendering the corpus per-model
(16 native corpora = new confound). If nativeness dominates, the honest strong paper is the
mechanism + dissociation + the SCOPE CONDITION itself (recovery needs self-native context).
STRONGEST PAPER (Fable): "A model's own write-time value vectors can be re-injected to recover
evicted semantic continuity — specifically referent binding — but only for on-distribution
context, and the sign of recovery is predicted by attention geometry across architectures."

## DESIGN v2.1 VALIDATED on native Qwen (07-08) — the redesign works
Per-model native render (Qwen3-30B-A3B-Instruct-2507 generates its OWN in-context replies + own
self-gen summary; SHARED gold continuation) on c01-c12 REPRODUCES the effect:
  referent CI [+0.012, +0.195] (mid +0.10, headroom 1.71) — EXCLUDES ZERO, matches the pre-rendered
  +0.12. sense +0.03 (weak-positive). stance -0.05 (~null, headroom 0.51). ruled_out FLOORED
  (headroom 0.18<0.3, correctly EXCLUDED — summary preserved it = nothing to recover, NOT "harm").
  evicted_fact null. identity_ok+alpha0_ok pass. reply_covariates captured (mean 316 tok/reply).
=> The matched-scaffold model-filled design is SOUND: native replies reproduce the effect, the
dissociation holds, and the headroom gate correctly floors preserved-content categories. Gate GREEN
on the design. Remaining before wide: (1) render scaling (batched decode + snapshot-replay — render
was ~2.5hr/12conv), (2) finalize pre-registered geometry hypothesis (fix QK-norm detect first).

## Cross-arch (in progress, 2026-07-08) — first full-run result + two methodology corrections

**Mistral-Small-24B-Instruct-2501 (no-QK-norm, dense), n=12 convs, conversation-clustered CIs:**
- referent [+0.010, +0.063] — POSITIVE, excludes 0
- sense    [-0.067, -0.005] — NEGATIVE, excludes 0
- stance   [-0.059, -0.014] — NEGATIVE, excludes 0
- ruled_out / evicted_fact — include 0 (null)
Smoke identity_ok/alpha0_ok pass (valid run). Signature DIFFERS from Qwen (referent+/sense+/stance-null):
Mistral recovers referent but the graft HURTS sense+stance. Directly relevant to H1: a no-QK-norm model
with a POSITIVE referent challenges H1's "no-QK-norm → null/negative referent" prediction — but the
sense/stance flip shows the architectures act differently. PRELIMINARY: n=12 is a borderline cluster
count; the Qwen 24-conv gate + the QK-norm ablation are the anchors. Not to be over-read as one model.

**Methodology correction 1 — CI clustering unit.** The headline raw_EB_ci is now the CONVERSATION-
clustered bootstrap (was plant-clustered = anti-conservative; plants within a conv are correlated).
Plant-level kept as raw_EB_ci_plant; ci_method records the unit. On Mistral the conv CI was ~the same
width as plant (low between-conv correlation), so the result held — but this is not guaranteed per model.

**Methodology correction 2 — OLMo-2 "empty alignment" was a MISDIAGNOSIS.** OLMo-2 alignment works
(108/0 smoke-align). The real cause of its ERROR: the absolute competence floor (task_lpa_floor=-8.0)
excluded EVERY OLMo plant (its gold logprobs sit lower); the old code guessed "empty alignment." New
counters (empty_alignment_convs / short_gold_drops / task_excluded_plants) now name the true cause.
OPEN: to get an OLMo result the floor likely needs to be per-model/relative — a methodology call.
