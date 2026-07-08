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
