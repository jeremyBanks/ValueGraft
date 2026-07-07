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
