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

## Sequencing / autonomy
Run Phase 1 to completion (paper pushed) → THEN Phase 2. Drive via scheduled
wakeups; keep this file + STATE + FINDINGS + DECISIONS current. Milestone
reports to user; don't ask permission for planned steps, do ask before a NEW
expensive escalation not in this plan. Balance ~$38; Phase 1 ~$10-15, Phase 2
~$20-40 (gated). One pod default; terminate when idle.
