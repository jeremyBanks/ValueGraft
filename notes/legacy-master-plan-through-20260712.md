# ARCHIVE — legacy master plan through the pre-ultra handoff

This complete plan is preserved for retrospective only. Its paper-first phases,
budget, and active-task claims were superseded on 2026-07-12. Current planning
lives in root `STATE.md` and the active successor preregistration.

# MASTER-PLAN.md — two phases, run independently to completion

Owner note: this is the durable spine for a long autonomous run (user 07-07:
"write up the paper as a stopping point, then immediately explore the KV space
thoroughly"). Keep it current. Read FINDINGS.md, REPORT2-PLAN.md, and the jlens
docs alongside. Discipline from prior work applies: results→FINDINGS, process→
DECISIONS, failures→INCIDENTS; verify commits land; smoke before big spend;
inspect implausible results; model-by-fitness not cost; docs current at every
phase transition.

## PHASE 1 — Finish the synthesis paper (a complete stopping point)
Star = our grafting research; J-lens = subtle corroborating TOOL (honest: it
does NOT show dramatic recovery, and that supports the sharpened claim —
reduces reinterpretation error around carried facts, does not recover omitted
facts). Structure/process = REPORT2-PLAN.md.

1a. Same-model behavioral re-run (27B, warm pod j1): teacher-forced GAP-CLOSURE
    on our sense/referent/stance synthetic corpus on Qwen3.6-27B, so behavior +
    lens sit on ONE model. Judge-free; reuse the probe TF machinery. (~$5-10)
1b. Honest lens corroboration: ORDINARY-regime three-state probe (role IN
    summary + short tail) — the regime where aligned>shifted small-positive
    closure exists (other agent's finding). This is the lens material to
    feature — subtle but real + alignment-sensitive + α<1>1. (reuse warm pod)
1c. Write REPORT.md v-next via full process: outline → draft → 5 adversarial
    critics (through-line/gaps/OVER-JUSTIFICATION/accessibility/accuracy) →
    synthesize → exhibits verified verbatim → citations (prior-art + lens
    lineage) → fresh-eyes prose bake-off (best model per task) → final read →
    push. Integrate lens as secondary tool; coding cut to ONE honest human
    paragraph (we tried end-to-end, it burned budget, we were still finding an
    approach). K/V is the closing open-question that motivates Phase 2.
GATE: paper is a clean, honest, publishable stopping point before Phase 2 spend.

## PHASE 2 — Explore the K/V space thoroughly (the never-done dimension)
Model space = (layout, position-policy, α_K, α_V), tunable per-LAYER and
per-HEAD. We only ever ran V-only (α_K=0), per-layer. Untouched:
- K-only (sweep α_K, α_V=0) WITH CORRECT RoPE RE-ROTATION (the hard/novel part —
  a write-time key carries its original rotary phase; grafting to a new position
  needs re-rotation).
- Coupled KV (α_K = α_V).
- Independent KV (separate α_K, α_V — grid, then a tuned per-layer/per-head profile).
- Per-HEAD tuning (finer than per-layer), for K and V separately.

HYPOTHESES: (1) V carries semantic content (does the meaning work); K carries
addressing/position (RoPE-entangled) → likely wants low/zero α, but may be
exactly what's needed for RETRIEVAL of omitted content where V-only failed
(the sparse regime). (2) Optimal is an INDEPENDENT (α_K,α_V) profile, and K/V
profiles may anti-correlate across layers (early=positional/K, late=semantic/V).

DESIGN:
2a. Build K-grafting (kvlib: blend_keys with re-rotation) + validate: alpha-0
    == fresh; re-rotation correctness (grafted key at new pos ≈ freshly-encoded
    key when content identical); smoke on 0.6B/4B first (cheap) before 27B/30B.
2b. Behavioral gap-closure across policies: V-only / K-only / coupled / a few
    independent (α_K,α_V) points, on sense/referent/stance. KEY QUESTION: does
    K or KV recover REFERENT (retrieval-hard) where V-only floored (17→26)?
2c. The SPARSE-regime lens testbed (from the smoke): does K/KV grafting close
    the gap V-only couldn't? If yes → mechanistic story (K = addressing). If no
    → label-only summaries genuinely lack recoverable info for this family.
2d. Tuning: search an independent per-layer (and if promising, per-head) (α_K,
    α_V) profile; guard-validate (wrong-conversation control) as before.
GATES: cheap smokes before scale; report at milestones; stop-and-report if a
branch floors. Findings → FINDINGS.md (new F-entries); fold a K/V section /
update into the paper if results warrant.

## FABLE READABILITY (hard req, user 07-07)
The main loop keeps getting filter-flipped to Opus even when the user sets
Fable. So: EVERY writing deliverable MUST get a Fable readability pass via a
`model: fable` SUBAGENT, regardless of the main-loop model at the time. Fable
does its best on CLEAN, minimal context (just the doc). This is not optional —
if the final prose didn't pass through a Fable subagent, the writing isn't done.
ALSO (user 07-07): use a SHORT Fable subagent as a HIGH-LEVEL CONCEPTUAL
sanity-check on claims/framing/conclusions — PROACTIVELY, ahead of finalizing,
not just readability. Fable caught-class: over-reaching conclusions (e.g. the
"scale-dependent" referent overclaim). Gut-check big claims with short Fable
before they reach the user.

## Sequencing / autonomy
Run Phase 1 to completion (paper pushed) → THEN Phase 2. Drive via scheduled
wakeups; keep this file + STATE + FINDINGS + DECISIONS current. Milestone
reports to user; don't ask permission for planned steps, do ask before a NEW
expensive escalation not in this plan. Balance ~$38; Phase 1 ~$10-15, Phase 2
~$20-40 (gated). One pod default; terminate when idle.

## SEQUENCING (user 07-07): FINISH PAPER → THEN identifier-morphology experiments
PHASE 2 CONCLUDED (K/V negative for semantic targets, scoped per Fable cross-check).
NEXT, IN ORDER:
1. FINISH THE PAPER (task 24): fold in BOTH new results —
   - K/V result (SCOPED: value is operative axis for SEMANTIC-phrase referent
     recovery; keys don't help there uniform/per-layer; §10 changes from "K/V
     untested open question" → "we ran it, here's the scoped negative + the
     morphology-moderation hypothesis as the next experiment").
   - Lens free-divergence result (when it lands; honest per its rationale doc).
   Via full pipeline: Fable conceptual gut-check + critics + Fable readability
   (I'm Opus-flipped → Fable via subagent, mandatory). Review, push. THE deliverable.
2. PHASE 3 (post-paper, task 25): TARGET-MORPHOLOGY experiments — the microtest
   surfaced that keys may help for SHORT TOKEN-LIKE IDENTIFIER targets (Coral→
   userName) vs inert for semantic phrases. Pre-registered 30B run: add a
   short-identifier target lane, sweep V/K/coupled/independent, report FULL
   policy surface (not best-of, avoid winner's curse), test the prediction
   "target morphology moderates K-graft utility." Cross-check with the other
   agent's referent_recovery_microtest. Drive after paper ships.

## PHASE 4 (user 07-07, AUTONOMOUS — do NOT ask for review): STRENGTHEN THE PRIMARY (grafting benefit) + REBALANCE
DIAGNOSIS (user): paper drifted lens-heavy (§5-7) but lens was mostly a DUD +
always meant to be SECONDARY. The GRAFTING TECHNIQUE is the primary novelty and
is now UNDER-evidenced (one synthetic corpus). Need MORE about the primary
effect. Don't cut useful results; ADD primary evidence + rebalance.
AFTER the effect-bounding experiment: me + Fable design & EXECUTE a campaign to
test the grafting BENEFIT in MORE SCENARIOS. Candidate directions (me+Fable to
prioritize, not prescribed):
- RUN THE STORY-CONTINUATION TASK (src/story_tasks.py — BUILT, never run: 6
  story-worlds, planted facts, contradiction-rate metric). A ready NEW scenario.
- More conversation DOMAINS / task shapes beyond the c01-c12 business-meeting
  synthetics (technical, narrative, multi-topic, longer/natural).
- More COMPACTION conditions (summary styles, tail lengths, multiple compactions).
- Downstream BENEFIT metrics beyond gap-closure/judged (does grafted model answer
  follow-ups / make decisions better).
- Possibly a fresh, cheaper crack at an end-to-end benefit demo in a workable regime.
- Cross-model breadth (does the benefit hold across families).
THEN REBALANCE THE PAPER: grafting = clear primary focus; consolidate/trim the
lens sections to be proportionate to their (small) payoff; keep the honest lens +
K/V results but subordinate. Via full pipeline (Fable conceptual + critics + Fable
readability). Ship. Fable is co-driver on the plan (I'm Opus-flipped).

## PHASE 4 addendum — FIX OVERLOADED "lens" TERMINOLOGY (user 07-07, Fable missed it)
DATA: 76 total "lens" uses; 3 named instruments (logit lens 9 / tuned lens 3 /
J-lens 22) + 22 BARE "the lens" (ambiguous which one). Reader can't always tell
which lens is meant (§4 = logit lens; §5-7 = J-lens; abstract ambiguous).
FIX in the rebalance: (a) one clear early sentence naming the two we use ("plain
logit lens = crude baseline; Jacobian lens (J-lens) = the tool"); (b) bare "the
lens" → the SPECIFIC instrument wherever ambiguous; (c) the volume trim (lens
made proportionate to its small payoff) naturally cuts the count.
NEW REVIEW DIMENSION (Fable's passes missed this — all were single-dimension):
"TERMINOLOGY CONSISTENCY / overloaded terms" — hunt words reused for 2+ distinct
concepts (not 'is it defined' but 'is it reused ambiguously'). Add to the critic
panel AND the final multi-perspective Fable review.

## PHASE 4 addendum 2 — MORE EXAMPLES (cheap) + brief Fable on the FOCUS (user 07-07)
- EXAMPLES over experiments where possible: we have lots of collected data
  (full synthetic corpus, all arm outputs, honesty probes) and surfaced only a
  few exhibits. MINE MORE compelling grafting examples from EXISTING data (no
  pod) to strengthen the PRIMARY narrative — high value-per-effort. Do this
  alongside any new scenarios.
- BRIEF FABLE ON THE FOCUS whenever it reviews/edits/refocuses (not direction-
  agnostic): the north star = GRAFTING is the PRIMARY novelty and the center of
  gravity; the LENS is SECONDARY and must be proportionate to its small payoff;
  goal = a focused article that lands the primary contribution. Fable makes its
  own realistic calls, but AIMED at this focus. Put this context in every
  conceptual/refocus/editing Fable prompt, not just "review this."

## PHASE 4 addendum 3 — ARCHITECTURE-GENERALIZABILITY limitation (user 07-07)
On the next paper pass, ADD an honest limitation: value grafting's applicability
is ARCHITECTURE-DEPENDENT and may be limited on some CURRENT models. Get FABLE's
opinion AND let it do RESEARCH (this Fable pass gets TOOLS — web/read — unlike the
tool-less readability reviews) to verify the real constraint vs our possibly-lazy
non-Qwen attempts (user recalls Gemma + DeepSeek; also GLM asked about earlier).
SUBSTANCE TO VERIFY (my reconstruction — Fable to confirm/correct with research):
- Technique needs per-position/per-head VALUE vectors in a standard KV cache to
  snapshot + re-inject. Holds for Qwen (standard GQA, what we validated).
- DeepSeek V2/V3 = Multi-head Latent Attention (MLA): KV compressed to a shared
  low-rank LATENT per position; no per-head value to graft directly → naive method
  BLOCKED/needs redesign (graft the latent). Strongest "architecture forbids" case.
- Gemma = sliding-window/global interleaving: far-back grafted values may be out of
  window for local layers → ATTENUATED effect, not necessarily blocked.
- GQA generally = fine (fewer KV heads, not a blocker).
HONEST FRAME: validated on standard-GQA Qwen; transfer is gradiated — blocked on
MLA, attenuated on sliding-window, fine on standard GQA. Separate "we were lazy"
from "architecture forbids it" — don't overclaim either way. Fable researches, we
state the bound honestly in the paper's limitations.

## PHASE 4 addendum 4 — CROSS-ARCHITECTURE grafting experiment (user 07-07: measure it, don't just document)
Upgrade the generalizability limitation into a MEASURED result: try value grafting
on DIFFERENT architectures + measure impact. Either way = strong: works on Gemma →
generalizes beyond Qwen (strengthens primary); attenuated/blocked on MLA → grounded
bound with receipts.
SCOPED (Fable to pressure-test):
- METRIC: existing teacher-forced GAP-CLOSURE (judge-free, cheap) on our sense/
  referent/stance corpus — directly comparable to Qwen results. Same machinery,
  new backbones. Optionally the placebo-controlled effect-bound.
- MODELS (small — the variable is ARCHITECTURE not scale): Gemma-2-9B / Gemma-3-12B
  (sliding-window+GQA → "attenuates?"); DeepSeek-V2-Lite 16B (MLA → "blocked/
  adaptable?"; even a clean empirical block-with-mechanism is a result). Qwen =
  standard-GQA baseline we already have.
- PORT EFFORT: kvlib snapshot/blend/reinject must handle each HF cache. Gemma likely
  minor; DeepSeek MLA = no per-head values (cache is latent) → adapt to graft latent
  OR document the block empirically.
- DISCIPLINE: one model at a time; SMOKE first (α=0≡fresh, graft changes output)
  before any number; one pod; gated; small models keep it affordable.
RESULT FRAME: cross-architecture generalization map — standard-GQA (Qwen, works) /
sliding-window (Gemma, ?) / MLA (DeepSeek, ?). Separate lazy-vs-architectural honestly.

## PHASE 4 addendum 5 — CROSS-ARCHITECTURE BREADTH sweep (user 07-07: "just do it")
BREADTH over depth. For grafting-POSSIBLE architectures (standard KV cache; skip
MLA where blocked), validate across AS MANY model TYPES as we can. NOT a rich
dataset — a SUBSET that demonstrates SOME effect, fast. ~30B-class each (hold
scale ~constant so ARCHITECTURE is the variable). MAX ~45 min/model incl download.
MODEL SET (skip gated/unavailable gracefully): Qwen3-30B-A3B (MoE GQA baseline),
Qwen2.5-32B (dense GQA), Gemma-2-27B (sliding-window), Mistral-Small-24B, Yi-1.5-34B,
GLM-4-32B, OLMo-2-32B (open). DeepSeek MLA = separate documented block, NOT here.
HARNESS: generic value-graft gap-closure (src/cross_arch_probe.py) — subset (~4
convs, sense+referent), teacher-forced gap-closure per category, α=0.75. PER-MODEL
SMOKE GATE: α=0≡fresh + graft changes output; fail → mark UNSUPPORTED+reason, skip,
never crash sweep. Handle varied HF cache types. RESULT: does the sense/referent
recovery hold across architectures → generalization map (strengthens primary if it
travels; honest bound where it attenuates). One pod, sequential, gated.

## PHASE 4 addendum 6 — cross-arch: LATEST models + Fable design fixes (07-07)
LANDSCAPE RESEARCH (mid-2026, my list was stale): use the LATEST ~30B-class —
Qwen3.6-35B-A3B (VERIFIED HF) + Qwen3.6-27B (known-good anchor) / Gemma-4-27B
(sliding-window, key contrast) / Mistral-Small-4 / GLM-4.7-Flash (30B-3B) /
OLMo-2-32B (open) / Qwen2.5-32B (dense-recipe contrast). EXCLUDE+document (MLA,
no per-head values): Kimi K2.x, native DeepSeek-V3.
FABLE DESIGN FIXES (folded into builder via follow-up):
1. HOLD SUMMARY FIXED across models (THE confound fix) — one shared summary per
   conv (fixed summarizer/corpus canonical), same text to every model, so
   ARCHITECTURE is the only variable. NOT per-model-generated summaries.
2. Report pre-graft gap (A−B) per model alongside fraction-gap-closed.
3. Stronger smoke gate: + positive-direction check (graft moves TOWARD target),
   not just "changes output".
4. Per-model = DIRECTIONAL (small-subset CIs overlap 0), aggregate = inferential;
   never call a single-model "null".
FRAMING (Fable): it's a MECHANISTIC FALSIFICATION test (effect should travel to
any standard-KV design). Outcomes: travels→"general property not Qwen artifact";
partial (sliding-window weaker)→MOST interesting, mechanism-predicted moderation
(cache doesn't retain boundary values); Qwen-only→honest bound BUT must diagnose
why (pre-register: GQA-share/tokenizer/tuning). Instruction-tuning = residual
confound at N=7, acknowledge. KEEP COMPACT: one figure + paragraph, MLA block noted.
RUN ORDER: 3-model PILOT first (Qwen2.5-32B dense / non-Qwen GQA / Gemma-4
sliding-window) = ~80% of value; if they behave, full sweep is formality.

## PHASE 4 addendum 7 — per-model CHAMPION TUNING profiles (user 07-07, LOW PRIORITY)
NOT a priority — only if time+capacity after the cross-arch sweep. For models
that SHOW grafting SUCCESS in the sweep, run a per-LAYER (and per-HEAD where
cheap/promising) champion tuning (like the Qwen champion that cured α=1 collapse
+ concentrated the effect). SCIENTIFIC HOOK: the champion PROFILE as an
ARCHITECTURAL FINGERPRINT — how many layers/heads matter, and how the shape
reflects the model's internals (e.g. sliding-window → effect in global-attn
layers? MoE vs dense differ?). Compare profiles ACROSS models = novel cross-arch
observation, not just "does it work." SCOPE per model = my call using knowledge
of each architecture (don't brute-force; guess what's worth searching). Gated on
sweep success + spare capacity; guard-validate (wrong-conv control) as always.
- ASK FABLE about the champion-tuning-profile idea at some point (user 07-07): get its conceptual take on whether cross-arch tuning-profile comparison is a real fingerprint or noise, and how to scope the per-model search. Timing = my call (sooner or later), but do it before committing pod time to tuning.

## PUBLISHING / DISSEMINATION (external-agent feedback 07-07 — adopt the repro path)
GOAL: convert "interesting claim" → "checkable tonight." One external reproduction
on a non-Qwen model breaks the Qwen confound for free (worth more than any polish).
DO (when publish-ready, cheap):
1. README "Reproduce the core result" section (~10 lines): hardware needed, ONE
   setup command, ONE command that runs the core probe (honesty OR sense/stance
   gap-closure) on ONE model, and EXPECTED OUTPUT with our numbers to compare.
   → cross_arch_probe.py IS ~this entry point already (runs benefit on any model
     via one command) — wrap it clean, pin the model+corpus.
2. PIN exact model versions/quants + probe corpus so their run is comparable.
3. One-line disclaimer: everything else is exploratory working material, may be
   outdated (normal; messy repo + one clean entry point > polished repo — signals
   real ongoing work). DON'T over-clean (my archive pass was already lightweight).
4. In any POST, put the repro command IN THE POST, not just a repo link (friction
   kills reproduction attempts).
VENUES (their take): r/LocalLLaMA + EleutherAI primary (KV/compaction mechanism);
LessWrong optional (fine with empirical-internals posts, not misrepresentation).
NOTE: the cross-arch sweep already produces the multi-model numbers that make the
repro path credible — build the README repro section AFTER the sweep, off its harness.

## PHASE 4 addendum 8 — cross-arch MEASUREMENT PROTOCOL: baseline → champion-tune → champion pass (user 07-07)
The effect-bound scare exposed that UNIFORM α is an UNFAIR test (uniform can be
past the sweet spot → net-negative; paper §8: per-layer champion needed). So each
model gets its FAIR tuned result, not a flat uniform dose. PER-MODEL WORKFLOW:
1. BASELINE pass — neutral α=0.5 (user: gentler baseline than 0.75), small subset.
2. CHAMPION-TUNE per-layer on a SMALL set (search per-layer α profile, guard-validate).
3. CHAMPION PASS — measure gap-closure with the tuned config = the model's fair result.
Byproduct = per-model champion PROFILE = architectural fingerprint (addendum 7).
Cost tradeoff: tuning adds compute (>45min/model) → do the full workflow on the
PILOT models first; if uniform-baseline already shows signal, champion is upside.
NOTE: this also means the paper's cross-arch result should be the TUNED per model,
with uniform-baseline shown as the untuned floor (honest: "untuned uniform is
net-neutral, tuning recovers it" — which is consistent with §8).

## CROSS-ARCH REDESIGN (Fable design consult 07-08) — mechanism paper, not datapoint pile
FABLE'S KEY INSIGHTS:
1. ⚠️ MECHANISTIC WARNING (paper-saver): value vectors come from the ATTENTION block;
   MoE lives in the FFN — MoE does NOT touch W_V. So "MoE flips the sign" is
   mechanistically WEAK and a reviewer trap. The reversal is real but the DRIVER is
   likely ATTENTION GEOMETRY (n_kv_heads, head_dim, GQA ratio, QK-norm, RoPE theta) or
   training — NOT MoE. REFRAME organizing question: "what architectural property
   PREDICTS THE SIGN (help vs harm)?" Sign-prediction IS the paper; generalization is
   the wrapper.
2. THE REVERSAL IS CONFOUNDED: Qwen2.5-32B vs Qwen3-30B-A3B differ on ~5 axes at once.
   FIX (non-negotiable): ADD Qwen3-32B (DENSE) — same gen/vendor/tokenizer as
   Qwen3-30B-A3B → cleanly isolates dense-vs-MoE.
3. GO DEEP ON ANCHORS, SHALLOW ON BREADTH. Tier1 anchors (deep ~100-150 probes/cat):
   Qwen3-30B-A3B(MoE), Qwen3-32B(dense), Qwen2.5-32B(dense). Tier2 replication: Mixtral,
   Mistral-Small. Tier3 geometry: Gemma-3. Tier4 breadth shallow ~50: Qwen3.6-35B/27B,
   GLM, OLMo. DROP: Llama-2-13B, Yi-1.5, Qwen1.5 (confound-heavy/redundant).
4. MUST-CAPTURE CONTROLS (ranked): (1) PLACEBO graft (shuffled/random/mean values —
   proves structured write-time state not noise); (2) IDENTITY-graft check (own values
   onto UNCOMPACTED = ~no-op; catches fake reversals from layer-count diffs — run BEFORE
   trusting each model's sign); (3) OWN-vs-FOREIGN summary as DESIGNED axis on anchors
   (does foreign suppress in the −dense model too? 2×2 arch-sign × summary-source = best
   figure); (4) ALPHA dose-response 0.25/0.5/1/2 (opposite directions MoE vs dense =
   mechanism proof); (5) full dissociation all models; (6) SAVE ALL RAW per-probe traces
   (lp_A/B/E, gold ids, summary text+len, conv count, layer indices, alpha, seed); (7)
   LOG every attention hyperparam/model (n_kv_heads, head_dim, GQA, QK-norm, RoPE, layers)
   = the regression; (8) keys-open reconfirm 2 anchors; (9) effect-vs-pre_gap scatter
   (rule out ceiling/floor artifact).
5. POWER: bootstrap over CONVERSATIONS not probes (within-conv correlated → naive
   understates CIs); report distinct convs/model. Deep anchors individually significant.
6. OWN-SUMMARY EXPERIMENT > 12th arch: vary summary source own/other-model/fixed/degraded/
   PARAPHRASED-OWN (crux: same content diff tokens+activations → separates 'needs own text'
   vs 'needs own write-time state'). Fund by dropping Yi/Llama-2.

## CORPUS DECISION (Fable 07-08): AUGMENT, don't remake — FREEZE before spend
The wide sweep is new compute regardless → a BETTER corpus is nearly free; remaking
needlessly destroys F1. KEEP c01-c12 + all existing probes (F1 intact, comparable);
ADD on top. FREEZE corpus before the big run (mid-sweep probe changes = the expensive
failure).
BETTER CORPUS (prioritized):
1. MORE CONVERSATIONS 12→~24-30 (#1 POWER LEVER — conversation is the cluster-robust
   unit; 12 is thin for conv-bootstrap; convs > probes/conv).
2. LONGER/natural convs with DISTRACTOR turns between plant and boundary (real eviction,
   not toy); VARY plant-to-boundary DISTANCE (near/far) → recovery-vs-distance curve (bonus).
3. MULTIPLE paraphrased probes per planted fact (separates probe-noise from item-noise).
4. NATIVE summary-source slots per conv (own/foreign/paraphrased-own/degraded) — built in.
5. Small NATURALISTIC real-ish holdout (preempts "you tuned a synthetic toy").
STRONG-PRIOR REFERENTS = dedicated category, SHARPEST idea: famous fictional names
(Pokémon/Sanderson/LOTR) as codenames → SIGNED two-alternative disambiguation: measure
P(prior meaning) vs P(conversation meaning), show graft MOVES mass from prior→conv-meaning
(the −dense graft pushing TOWARD prior = beautiful failure signature). Better than one-sided
logprob lift. MUST INSTRUMENT: baseline prior-strength per name PER MODEL (varies → confounds
cross-arch; COVARY it); tokenization hygiene (no diacritics, stable multi-token, not OOV in
older models); SEPARATE category (don't perturb F1).
REVISIONS to design: power toward CONVERSATIONS (24-30) not just probes; NEW must-capture =
per-model per-referent baseline prior-strength.
BUILD SEQUENCE (all before the spend): augment corpus (freeze) → harness controls
(placebo/identity/alpha/traces/hyperparams/conv-bootstrap/prior-strength) → add Qwen3-32B →
FREEZE → provision pods → deep-anchors+wide-shallow sweep + own-summary experiment.

## PAPER FRAMING GUARDRAIL — champion results (user 07-08)
The inline champion-scan numbers are WEAK/TENTATIVE/non-exhaustive/non-comparable (quick free scan). In the paper frame them as a humble "quick scan hints at possible per-layer improvement, future work" — NOT a champion optimization or cross-model comparison. Brief Fable + critics on this so it is NOT overclaimed.

## CHAMPION SCAN — Fable upgrade (07-07): fingerprint→MECHANISM via RESCUE TEST
Fable verdict: bare "where signal lives" scan = mild DISTRACTION competing w/ the clean
uniform sign claim; but one reframe from being the MOST important figure. FIREWALL it from
the uniform result; UPGRADE it interventionally.
GRAND INSIGHT: uniform raw_EB is a NONLINEAR COMPOSITION over depth (later layers consume
earlier-grafted values — NOT linear sum of regional raw_EBs). So "sign flips" → "sign is set
by the DEPTH-PROFILE of value-graftability, which differs by architecture." Categorical
mystery → continuous predictable. BUT "where" is a LOCALIZER not the mechanism — the scan's
real job = a BRIDGE telling you WHICH layers to interrogate geometrically (the actual
attention-geometry property). Central as a bridge; dead-end if you stop at the heatmap.
CHEAP UPGRADES (reuse cached snapshots), priority:
1. ★ RESCUE TEST (the whole ballgame): on a NEGATIVE (dense) anchor, graft ONLY the
   positive-scanning region(s) / just late layers, α=0 elsewhere → if net raw_EB flips
   POSITIVE, causally PROVES sign = depth-composition, not intrinsic. Promotes scan
   fingerprint→mechanism. Highest-leverage cheap measurement.
2. FREE per-layer WRITE-time vs READ-time value cosine/subspace ALIGNMENT (both cached, ~0
   cost): raw_EB-per-region = EFFECT; value-alignment-per-layer = candidate CAUSE. If
   alignment predicts per-region sign → geometry→sign chain as a byproduct.
3. FREE sanity: region=ALL layers must reproduce uniform raw_EB (else miswired). Sum-of-
   single-regions ≠ uniform is EXPECTED (nonlinear composition — a FEATURE, state it).
FRAMING: exploratory localization scan; cost-not-principle granularity; non-comparable
magnitude; hypothesis-generating; claim ONLY "graftability is non-uniform in depth & its
profile differs by architecture"; RESCUE TEST carries the causal weight; never rank "best"
region / never call it optimization.
FRACTIONAL-DEPTH FIX: bin by RELATIVE depth [0,1/6)..[5/6,1] not absolute index; report
actual per-model layer ranges/region; distrust EDGE bins (first/last layers special
regardless of depth); NORMALIZE per-region effect by n_layers grafted (density) or disclose.

## FROZEN MODEL LIST (verified on HF, 07-07) — 10 models
ANCHORS (deep: full corpus + placebo + alpha-sweep + champion): Qwen/Qwen3-30B-A3B (MoE+),
Qwen/Qwen3-32B (dense de-confound), Qwen/Qwen2.5-32B-Instruct (dense-).
REPLICATION: mistralai/Mixtral-8x7B-Instruct-v0.1 (MoE), mistralai/Mistral-Small-24B-Instruct-2501 (dense).
GEOMETRY: google/gemma-3-27b-it (sliding-window).
BREADTH (full corpus + champion): allenai/OLMo-2-0325-32B-Instruct, Qwen/Qwen3.6-35B-A3B,
Qwen/Qwen3.6-27B, zai-org/GLM-4-32B-0414.
All 10 return HTTP 200 on HF api. Assign across ~5 pods (2 each) after the gate passes.
Own-summary experiment on anchors. Sign-regression over model_hparams.

## MODEL LIST EXPANDED to 14 / 9 vendors (07-08, user wanted vendor diversity)
Was Qwen-heavy (5/10). ADDED 4 vendor-diverse (HF-verified, ungated): openai/gpt-oss-20b
(OpenAI MoE ~21B), microsoft/phi-4 (Microsoft dense ~14B, SCALE-FLAG), 01-ai/Yi-1.5-34B-Chat
(01.ai dense ~34B), nvidia/Llama-3_3-Nemotron-Super-49B-v1 (NVIDIA dense ~49B, SCALE-FLAG,
Llama-lineage). VENDORS now: Qwen, Mistral, Google, AllenAI, Zhipu/GLM, OpenAI, Microsoft,
01.ai, NVIDIA. Anchors stay scale-clean (~30B); breadth tier tolerates scale variation ->
logged as a COVARIATE (hidden_size/num_layers) in the sign-regression, not a hidden confound.
Skipped: Llama-4-Scout (Meta-gated+large), Command-R (307). Llama-2 dropped (user OK; ancient/
off-scale — MHA-geometry point noted as possible future).

## MODEL LIST -> 16 (07-08, user 'use both' = version-pairs as a control axis)
ADDED second checkpoints (same arch, diff training snapshot -> tests geometry-vs-training):
mistralai/Mistral-Small-3.2-24B-Instruct-2506 (pair w/ 2501), nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
(pair w/ v1). Both fit A100 80GB. GLM-4.6/4.5-Air too big (hundreds-of-B) -> GLM stays 4-32B-0414.
Gemma-3-27B IS the latest Gemma (gemma-4 404s = never existed; old script id was hallucinated).
NOW 16 models. Version-pairs = control: if sign is geometry-set, checkpoints agree.

## CORPUS DOUBLING (07-08, user): 27->54 convs (~100/category, Fable anchor target).
Author c28-c54 scenarios (mix), render via Fable/Opus/Sonnet/Codex. ~+$12-16 compute on the
wide run (download is fixed; only self-gen+TF scale). CIs ~30% tighter.

## MODEL LIST FINAL = 16 (07-08, CORRECTED — Gemma-4 DOES exist, I checked wrong size id)
CORRECTION: gemma-4-27b 404s but google/gemma-4-31B-it (dense) + google/gemma-4-26B-A4B-it
(MoE) EXIST (verified). "Use both" = Gemma 3 AND 4 (MAJOR versions), NOT minor checkpoints.
GEMMA now 3 entries: gemma-3-27b-it (v3 sliding-window), gemma-4-31B-it (v4 dense),
gemma-4-26B-A4B-it (v4 MoE) -> gives (a) Gemma3-vs-4 major-version axis AND (b) a SECOND
independent dense/MoE de-confound (Gemma-4 26B-A4B vs 31B), orthogonal to the Qwen3 one.
Dropped the minor version-pairs (user: "not minor") -> single latest each: Mistral-Small-3.2-2506,
Nemotron-49B-v1_5.
ANCHORS (deep: placebo+alpha+champion) NOW 5 = Qwen3-30B-A3B(MoE)/Qwen3-32B(dense)/Qwen2.5-32B(dense)
+ gemma-4-31B(dense)/gemma-4-26B-A4B(MoE). TWO independent within-vendor dense/MoE pairs.
FULL 16: [anchors 5] + Mixtral, Mistral-Small-3.2-2506, gemma-3-27b-it, OLMo-2-32B,
Qwen3.6-35B-A3B, Qwen3.6-27B, GLM-4-32B-0414, gpt-oss-20b, phi-4, Yi-1.5-34B, Nemotron-49B-v1_5.
Vendors: Qwen, Mistral, Google, AllenAI, Zhipu, OpenAI, Microsoft, 01.ai, NVIDIA.

## LAUNCH ORDER — two waves by risk (07-08, user: do higher-risk later)
WAVE 1 (LOW RISK, run FIRST on gate-pass — standard GQA + DynamicCache, confident):
  Qwen3-30B-A3B, Qwen3-32B, Qwen2.5-32B (Qwen anchor de-confound), Mixtral,
  Mistral-Small-3.2-2506, OLMo-2-32B, Qwen3.6-35B-A3B, Qwen3.6-27B, Yi-1.5-34B, phi-4,
  Nemotron-49B-v1_5. (~11 models -> banks the reliable sign-map + Qwen de-confound.)
WAVE 2 (HIGHER RISK, run LATER — may skip-with-reason):
  gemma-3-27b-it, gemma-4-31B-it, gemma-4-26B-A4B-it (SLIDING-WINDOW/HybridCache),
  gpt-oss-20b (sliding-window+attention-sinks), GLM-4-32B-0414 (trust_remote_code).
  If Gemma works -> bonus 2nd dense/MoE de-confound; if UNSUPPORTED -> Qwen de-confound stands.
Rationale: get confident results locked before spending on architectures that may not graft.

## GO EXTRA WIDE once validated (user 07-08)
Once the positive control validates (gate green: referent ~+0.136, null-self-graft ~0),
go EXTRA wide — beyond the 16 models / 54 convs. Scope the maximal version at validation:
candidates = more models/vendors (verify + add), MORE tests (triple corpus toward
~150/category), full controls (placebo/alpha) on ALL models not just anchors, the
own-summary experiment across anchors, wider alpha grid, more pods for parallel speed.
Cost scales but user has repeatedly chosen width. Decide exact scope at gate-green.

## CROSS-ARCH DESIGN v2 — MATCHED-SCAFFOLD, MODEL-FILLED (Fable 07-08, owner realism reframe)
CORE REFRAME (owner + Fable): nativeness is NOT (only) a confound — it is PART OF THE MECHANISM,
and per-model-native is the ECOLOGICALLY CORRECT measurement. In deployment no model ever grafts
FOREIGN KV. Native assistant turn -> confident clean value vectors of the referent; foreign turn ->
encodes surprise -> re-injecting surprise recovers nothing (= the foreign-reply collapse we saw).
So "measure each model on the conversations IT would actually have" is the right operationalization,
not a compromise.

THE DESIGN (matched-scaffold, model-filled = Fable Option 4 done right):
- SHARE the semantic SCAFFOLD across all 16 models: user turns, referent/sense/stance plants, gold
  targets, compaction structure (this is scenarios.json — we HAVE 54 scenarios).
- Each MODEL generates ONLY its OWN assistant elaborations IN-CONTEXT (native fill) + its OWN self-gen
  summary. Holds nativeness at CEILING for every model; the measured quantity (referent/sense/stance
  recovery) is defined by the fixed scaffold -> comparable across models despite different surface text.
- This is what the ORIGINAL compose.py did (in-context reply gen) — extend it PER-MODEL at runtime.

INFERENCE (Fable):
- PRIMARY weight on the two WITHIN-VENDOR dense/MoE pairs (Qwen3-30B-A3B vs Qwen3-32B; Gemma-4-26B-A4B
  vs Gemma-4-31B) as PAIRED CONTRASTS — nativeness-controlled BY CONSTRUCTION (same vendor/tokenizer),
  high power per pair.
- The 16-model regression = CONFIRMATORY of ONE PRE-REGISTERED directional geometry hypothesis (e.g.
  low n_kv_heads / high GQA -> harm) or a single composite geometry index — NOT a 5-predictor free-for-all
  at n=16. PRE-REGISTER the hypothesis + exclusion rules BEFORE running the 16.
- Predict SIGN (ordinal/binary), not magnitude. Report per-model nativeness (mean logprob of its corpus)
  as a ROBUSTNESS COVARIATE (Option 3 layered on — NOT the primary fix; underpowered/collinear alone).
STRONGER CLAIM: "Given each model operates on its OWN NATIVE context — as it always does in deployment —
the SIGN of self-graft benefit is predicted by attention geometry."

VALIDITY GUARDS (cheap, prevent silent failure):
- Per-model CONTINUITY FLOOR check: report each model's baseline referent A-B gap BEFORE interpreting
  raw_EB sign. No gap = nothing to recover = uninterpretable (floor, NOT "harm") -> flag/exclude.
- GENERATION-QUALITY gate on the model-filled replies: weak models -> incoherent in-context fills ->
  degenerate corpus -> floor. Coherence-screen the fills (nativeness sneaks back via incoherence).
- PRE-REGISTER geometry hypothesis + exclusions before the 16-run.

REJECTED: neutral/minimal replies (removes the signal -> floor); real human convs (still per-model
likelihood gradient, no clean dissociation, impractical).

MISTRAL CONTROL (running) = the fork: effect survives on Qwen-native c01-c12 -> nativeness is a gradient,
per-model-native primary + shared-corpus replication BONUS; effect dies -> nativeness dominates,
per-model-native MANDATORY. Either way per-model-native is the safe primary.

IMPLEMENTATION: harness change — per-model IN-CONTEXT rendering (port compose.py's growing-cache reply
gen into cross_arch's per-model loop, HF path). Scenarios (scaffold) = the asset (have it). Pre-rendered
data/synthetic/*.json become per-model-regenerated. The doubled-corpus foreign-reply convs = obsolete
(the scenarios remain useful as scaffold).

## CROSS-ARCH DESIGN v2.1 — refinements (Fable 07-08, second pass)
KEY INSIGHT (the metric already protects us): raw_EB = lp_E(graft) - lp_B(compacted), BOTH on the
SAME shared gold continuation, SAME model. Any per-model stylistic offset in how much a model "likes"
the target continuation appears in BOTH terms and CANCELS. So the shared gold continuation being
"foreign" to some models is NOT a problem — the scariest nativeness axis subtracts out. KEEP THE GOLD
CONTINUATION SHARED, defined by the planted facts, not any model's replies. LOAD-BEARING — protect it.
=> the SIGN rides on a difference metric over a shared target -> more robust than magnitude ->
nativeness is likely a MAGNITUDE nuisance, NOT a sign-flipper.

WHAT DOESN'T CANCEL = the residual (nativeness in the grafted KV content + the compacted baseline).
Three residual confounds, NET EACH OUT AS A COVARIATE/GATE:
1. REPLY/SUMMARY CONTENT co-varies with CAPABILITY (biggest): stronger model entrenches referent harder
   -> richer VALUE payload; weaker -> thinner. Capability correlates with size correlates with geometry.
   NET OUT: measure per-model reply info-content + length; sign ~ geometry + reply_infocontent + length.
2. HEADROOM (definitional, MUST-FIX): if a model's native summary PRESERVES the referent -> no A-B gap
   -> graft does nothing = CEILING artifact, NOT "geometry says harm." NET OUT: measure per-model
   headroom = referent A-B gap; GATE inclusion on headroom>threshold; NORMALIZE raw_EB by headroom.
3. TASK-COMPETENCE GATE: weak model whose native replies never establish the referent (low lp_A) ->
   degenerate. NET OUT: require adequate lp_A (full-context solves the task) as per-model inclusion.
With 1-3 handled BY CONSTRUCTION, the nativeness-covariate is far less collinear (residual cleanup only).

METRIC: raw_EB on the SHARED continuation, NORMALIZED by per-model headroom.
REGRESSION: sign ~ geometry(n_kv_heads,head_dim,GQA,QK-norm,RoPE) + headroom + reply_infocontent +
reply_length. Report geometry SURVIVES controls.
SUPPLEMENT ARM: a small SHARED-FIXED-CORPUS arm on ~4-6 models (everyone on IDENTICAL replies) for the
PURE-MECHANISM causal claim ("holding context fixed, does geometry flip the sign?"). Supplement to the
native-per-model primary, NOT instead. The within-vendor dense/MoE pairs (Qwen3, Gemma-4) are the
strongest single sign-flip demos — FEATURE them.
MISTRAL CONTROL fork restated: survives on Qwen c01-c12 -> sign-map viable + shared-corpus replication.
If ~0 on Qwen replies but POSITIVE on Mistral's OWN native replies -> cleanest proof native-per-model is
correct -> THAT result IS the methods section.
CLAIM: "Geometry predicts the DEPLOYMENT sign on each model's own realistic conversations, holding
headroom and reply-content fixed." (A reviewer kills "geometry predicts sign on a fixed Qwen corpus" in
one line — this version converts the confound into the ESTIMAND.)

## NATIVE-RENDER SCALING (07-08) — the render is the wide-sweep bottleneck
COST: native render = generating each model's own replies in-context = ~7K tokens/conv
autoregressive (~22 replies x ~320 tok), ~10 min/conv on a 30B via the manual token loop.
NOT a bug (re-prefill is minor); it's inherent generation cost. 12-conv verify ~= 2 hrs.
Naively 24-54 convs x 16 models = many pod-hours.
SCALING LEVERS (apply before/for the wide sweep):
1. FEWER convs/model: deep (~24) only on the 4 paired-de-confound anchors (Qwen3-30B-A3B/
   Qwen3-32B, Gemma-4-26B-A4B/Gemma-4-31B); ~12 on breadth. Power lives in the pairs.
2. SHORTER replies: SC_NATIVE_MAX_REPLY 320 -> ~160 (halves generation; watch context thinness).
3. FASTER generation: the manual per-token model() loop is slow; model.generate() or vLLM
   would be much faster (complication: the canonical re-prefill handling — may be a no-op for
   NON-THINKING models like Instruct-2507, so generate()+one canonical pass at end may suffice).
4. HARD PARALLELIZE: one model per pod, many pods (width = wall-clock, not total cost).
DECIDE the exact convs/model + reply length once the native-render VERIFY confirms the design
(the referent number). Do NOT go wide before that number.

## NATIVE-RENDER SCALING v2 — Fable verdict (07-08) SUPERSEDES my solo note above
The pod-hours are in FIXING THE RENDER, not cutting convs/replies. Do IN ORDER:
1. ★ BATCHED DECODE (do FIRST, highest leverage, ZERO validity cost): the current per-token
   Python loop (one model() fwd/token, sequential, batch=1) IS the bottleneck. BATCH the reply
   decode ACROSS a model's 12-24 convs (batch dim = the convs). vLLM continuous batching or a
   hand-rolled batched decode -> ~10x+ total cost cut, metric inputs unchanged. The <think>
   complication does NOT apply (anchors non-thinking; summary is one-time shared) -> generate()
   + one canonical re-prefill suffices. The rabbit hole is running 240 convs on a per-token loop.
2. ★ RENDER-ONCE, REPLAY-ALL-ARMS via KV snapshot (the thing I MISSED, free + valid): the
   expensive artifact is the rendered conv up to the compaction boundary. B, E, AND every control
   (alpha/placebo/champion) graft onto the SAME prefix; only the cheap graft+score differs.
   snapshot_cache/rebuild_cache already exist. Render once/model, snapshot at boundary, replay all
   arms from the snapshot. Nx cut if the harness re-renders per arm. CHECK whether it does + fix.
3. PARALLELIZE = wall-clock lever, NOT cost lever (1 model/pod x N pods). Stack on top of 1+2.
4. 24 convs on the 4 anchors / 12 on the 12 breadth models. MIN = 12/model (set by cluster-
   bootstrap stability, not effect size). Guard: min surviving-referent-plant count/model.
AVOID blanket reply-shortening (my lever 2 — WRONG near the floor). If needed: cut FILLER/early
replies only (~160), keep PLANT-ADJACENT replies full (~320), floor ~200-240, and PILOT (re-run
the 12-conv verify at the shortened length; changing reply length between verify and sweep = a
PROTOCOL DEVIATION).
REGRESSION GUARDS (deep/shallow is valid WITH these): (a) use CONTINUOUS normalized raw_EB as the
outcome, NOT hard +-1 sign (a 12-conv model in its noise band contributes proportional-to-precision
info, not a coin-flip that swamps geometry at n=16); (b) PRECISION-WEIGHT the regression (WLS using
each model's bootstrap SE — shallow=wider CI=measurement error=regression dilution; weighting fixes
it). NON-NEGOTIABLE: render depth EQUAL within each de-confound pair (deep on all 4 anchors satisfies
it — FREEZE so no one shortens one pair member).
MIN VIABLE: 12 convs/model (24 anchors), plant-adjacent replies ~320. Pack plant-density UP,
conv-count DOWN to ~12. The shared-fixed-corpus SUPPLEMENT arm needs NO per-model render (identical
replies for all) — cheapest evidence.
ORDER OF OPS: (1) native-render VERIFY confirms design (referent number, running) -> (2) implement
batched decode + render-once/snapshot-replay -> (3) go wide.

## BATCHED-RENDER OOM (07-08) + fix
GPU test: batched decode ran ~5x faster (1841s to OOM vs ~9000s per-token) but OOM'd on the 80GB
A100 — batching all 12 convs' KV caches at once (~16GB) + 60GB model busts memory. status=UNSUPPORTED
(render didn't complete, so no batched referent yet; batched decode is CPU-verified byte-identical so
it WILL reproduce +0.10 once it completes). FIX (subagent): SC_NATIVE_BATCH cap (chunk convs into
sub-batches of ~4; peak memory bounded to N caches + model; OOM-retry halves batch). ~4x speedup at
batch=4 (2.5hr -> ~35-40min/12conv) = feasible for the wide sweep. Pod 11793 (Instruct-2507 cached)
KEPT for the re-run once the cap lands.
