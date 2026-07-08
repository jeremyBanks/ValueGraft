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
