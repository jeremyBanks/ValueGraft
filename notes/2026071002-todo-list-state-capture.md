# To-do list state capture (2026-07-10)

Full snapshot of the session task list at the point of the in-domain SWE-Gym confirm run.
Captured verbatim on owner request. Completed tasks kept for history; pending tasks carry
their full descriptions (the outstanding state).

## Full list (id · status · subject)

- #16 · completed · Cloud phase execution (approved): pod stages 0-3 per cloud-plan.md
- #20 · completed · P1a: Same-model 27B behavioral gap-closure
- #21 · completed · P1b: Ordinary-regime lens corroboration
- #22 · completed · P1c: Write & ship the synthesis paper
- #23 · completed · P2: Explore the K/V space (K-graft, coupled, independent, per-head)
- #24 · completed · Lens free-divergence probe + batch paper update
- #26 · completed · Effect-bounding experiment (Fable-reframed from trajectory)
- #27 · **pending** · Phase 4: strengthen primary grafting evidence + rebalance paper
- #28 · completed · Cross-arch REDESIGNED sweep (mechanism paper) — build then run
- #29 · completed · Correct paper: robust-metric audit of ALL ratio results + CIs
- #30 · completed · Audit judged +12pp metric with bootstrap CIs
- #31 · completed · Harden alignment: replace difflib with direct span-offset map
- #32 · **pending** · Fix cross-arch harness Gemma chat-template (alternation) in summary-snapshot path
- #33 · completed · Unified reproduction entry-point script (filterable, reproduces all results)
- #34 · completed · AUGMENT corpus (freeze before spend): 24-30 convs, distractors, strong-prior category, summary-source slots
- #35 · completed · Harness controls: placebo, identity-check, alpha-sweep, trace capture, hyperparam log, conv-bootstrap, prior-strength
- #36 · **pending** · Own-summary mechanism experiment (> a 12th model)
- #37 · **pending** · END-OF-PROJECT (very low priority): archive code+data remnants not needed for repro or active use
- #38 · **pending** · PAPER (BLOCKING): fully document data provenance + methodology per METHODS-PROVENANCE-REQUIREMENTS.md
- #39 · completed · Harvest cs32b champion + PODS OFF (verify by PID)
- #40 · completed · Fable re-spine paper/DRAFT.md (honesty-led, honesty-directive bound)
- #41 · completed · Assemble full paper from CLAIMS numbers + re-captioned exhibits
- #42 · completed · Run review stack (Fable angle passes + critics + terminology + GPT-5.5)
- #43 · completed · Promote to README + commit + push; STOP before external publish
- #44 · completed · Champion-config validation: placebo-controlled, 4-bit + bf16
- #45 · completed · Per-head champion scan on primary bf16 30B (Fable designing)
- #46 · completed · Strengthen SWE-Gym +0.0156 anchor on primary bf16 30B
- #47 · completed · After paper is promoted: move draft into notes/
- #48 · **pending** · FINAL (after paper + everything else): re-run note-naming then summarization scripts, commit + push
- #49 · **pending** · Design in-domain graft optimization on brief-SWE-Gym (the unexplored live margin)
- #50 · **pending** · After full close-out: fresh Fable writes a speculative note on WHY the graft fails

## Pending tasks — full descriptions

**#27 — Phase 4: strengthen primary grafting evidence + rebalance paper.** AUTONOMOUS (no review-ask, me+Fable). After bounding experiment. Test grafting BENEFIT in MORE scenarios (story-continuation task built-never-run first; more domains/task-shapes; compaction conditions; downstream benefit metrics; maybe cheaper end-to-end). Then rebalance paper: grafting=primary focus, lens/K-V subordinate+proportionate to small payoff, keep honest results. Full pipeline (Fable conceptual + critics + Fable readability). Ship.

**#32 — Fix cross-arch harness Gemma chat-template (alternation) in summary-snapshot path.** Gemma-3 errored: TemplateError 'roles must alternate' — build_summary_snapshot uses generic render_hf, but Gemma needs the alternation-safe path (harness already has build_b_messages_gemma for the B-context; the summary-snapshot construction wasn't adapted). Fixable harness bug, NOT an architecture verdict. Fix so Gemma (sliding-window — the key architectural test) can run. Also: verify the sliding-window HybridCache snapshot detector behaves (Gemma may still hit padded-cache Unsupported after the template fix — that WOULD be a real architectural boundary to document). Do after seeing Mistral/other results; Gemma is high-value (sliding-window) so worth fixing.

**#36 — Own-summary mechanism experiment (> a 12th model).** Fable: worth more than a 12th architecture. On 2-3 anchors (one +MoE, one -dense): vary summary source own / different-model's-own / neutral-fixed / degraded(truncated/shuffled) / PARAPHRASED-OWN (crux: same content, diff tokens+activations → separates 'needs own text/content' from 'needs own write-time state'). Combined w/ placebo + alpha sweep = nails the mechanism. This confirms today's finding (graft needs model's OWN summary) as a designed result.

**#37 — END-OF-PROJECT (very low priority): archive code+data remnants not needed for repro or active use.** User (07-08, do at the very end, NOT now): move the repo toward containing ONLY code/data necessary for (a) REPRODUCTION of the final paper's results or (b) active use — no remnants of past experiments whose data isn't in the final paper (superseded/replaced-by-better/dead branches). ARCHIVE (move to an out-of-the-way folder, like notes→docs/), don't necessarily delete — keep findable. Candidates once paper is final: jlens_boundary_probe/ (lens work done, effect_bound retired), the many one-off src/ scripts (kv_sweep/kv_layer_probe/effect_bound/free_divergence/run_phase2/audit_corpus/compose-mlx/etc. not in the repro path), stale results/. Gate: is its data in the final paper OR in the reproduction path OR active? If no to all → archive. Do AFTER the paper + repro entry-point are settled.

**#38 — PAPER (BLOCKING): fully document data provenance + methodology per METHODS-PROVENANCE-REQUIREMENTS.md.** User (07-08, emphatic): every prior writeup FAILED to document the structural/experimental methodology — above all WHO/WHAT generated each piece of data (prompts authored by us; assistant replies model-native-in-context; summary self-gen; gold from planted facts; exact model checkpoint ids — the thinking-vs-instruct issue cost hours). This omission makes the result un-reproducible/un-trustable and is UNACCEPTABLE to repeat. The paper MUST satisfy every item in METHODS-PROVENANCE-REQUIREMENTS.md. Wired as a blocking requirement into writeup-guidelines.md + AGENTS.md. Provenance gaps = blocking review failure. (NOTE 2026-07-10: the current README paper's §2 already addresses much of this — the born-annotated provenance discipline is now a paper strength; verify it fully satisfies the requirements doc.)

**#48 — FINAL (after paper + everything else): re-run note-naming then summarization scripts, commit + push.** RE-RUN as the VERY LAST step of THIS round, after: in-domain SWE-Gym experiment finishes → result valued → paper (README) updated (§3.3 free-step + confirm/α-sweep results, §10 future-work) → committed + pushed → any new draft/notes archived. THEN regenerate ALL note paths + summaries: scripts/normalize_notes_archive_names.py (naming), then update_notes_meta_summaries.py (orchestrates daily+month+overall meta-summaries). Commit + push. (Ran once already this session at commit d4cbf16; must re-run now that new notes + a paper revision exist.)

**#49 — Design in-domain graft optimization on brief-SWE-Gym (the unexplored live margin).** Owner's key insight (2026-07-10): the graft's ONLY net-positive cell is brief-SWE-Gym (scalar α=0.75 → +0.013), but every champion we built was tuned on the NULL synthetic-conversation corpus and transferred out-of-domain. We NEVER tuned/optimized the graft ON the coding task. Design+build: α-sweep + per-layer/head profiling ON held-out brief-SWE-Gym trajectories, placebo-controlled, to test if an in-domain-tuned graft beats the naive +0.013. Held-out discipline; unique per-run output dir (the filename collision lost the prod-champion data). (STATUS 2026-07-10: BUILT + LAUNCHED — funded $25; run in progress at STAGE2. EARLY RESULT: positive collapses to null on independent disjoint N (~70): scalar α=0.75 = −0.0035 CI[−0.018,+0.010]; no α beats baseline. → paper becomes clean complete null. Awaiting full N + champion held-out eval before finalizing.)

**#50 — After full close-out: fresh Fable writes a speculative note on WHY the graft fails.** Owner (2026-07-10): once EVERYTHING queued is done and pushed — experiment valued, paper (README) updated to clean-null + pushed, draft archived, notes regenerated + pushed (#48) — dispatch a FRESH, unled Fable to write a standalone NOTE in notes/ speculating about potential explanations for WHY value-grafting write-time V vectors across a compaction boundary does not recover meaning / beat the compacted baseline. Pedagogical/speculative, NOT empirical research — reasoning about mechanism (redundancy of values with the re-read summary text, reconstructability of write-state, why content-specificity is real yet non-additive, positional/attention considerations), framed as hypotheses. Keep it OUT of the main paper. Fresh Fable, minimal leading. Output = a committed + pushed notes/ file.

## Completed this session — final-state notes (for the record)

- **#42** Review stack ran: Fable adversarial reviewer (caught §5 terseness-vs-domain error), proofreader, independent cold-reader readability review (6 fixes applied). Paper commit d94df0b.
- **#43** Final paper promoted to README.md, committed + pushed to origin/trunk (4b05806). External publish NOT done (held for owner).
- **#44** Per-layer champion validation: raw_EB null (CI spans 0) under declared self-gen realistic summary, bf16 primary. Commit 37714c6.
- **#45** Four-way (heads/layers/intersection/union) held-out validation, shuffle_pos: raw_EB NULL for all four — per-head does NOT beat per-layer; no champion clears baseline. Commit 37714c6.
- **#46** Prod-SWE-Gym N=75 born-annotated: E−B = +0.0006 (NULL) under production-faithful summary vs BRIEF anchor +0.0156 → compression-dependent. (Later superseded: the brief positive itself collapses on independent N — see #49 early result.)
- **#47** paper/DRAFT.md git-mv'd to notes/2026070920-paper-draft-archived.md; paper/ no longer holds the draft.

## Headline result state (2026-07-10)

The paper is a **clean, complete null** on the value graft: no champion family (per-layer / per-head / ∩ / ∪) beats the compacted baseline on held-out recovery; the compression sweep is flat/slightly-negative (refutes the concentration hypothesis); and the one apparent positive (brief-SWE-Gym +0.013) **does not survive independent replication** (−0.0035 on ~70 disjoint trajectories; no α beats baseline). Content-specificity is real but non-additive. Effect measured on a teacher-forced-logprob proxy, one model (bf16 Qwen3-30B-A3B). Paper = README.md on origin/trunk; working draft archived in notes/.
