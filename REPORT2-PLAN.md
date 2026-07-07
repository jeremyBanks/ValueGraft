# REPORT2-PLAN.md — the synthesis piece (grafting research, seen through the J-lens)

## Deliverable
A ~70% academic / 30% blog technical piece for people in the area. OUR
grafting research is the star; the J-lens is a SECONDARY theme and a TOOL
used to look *into* the effect and explain what's happening. Rich enough to
contribute to the literature, with a through-line that carries a skimmer via
compelling examples (lens-based and behavioral). Long is fine if it flows.

## Roles (do not invert)
- STAR: ValueGraft — write-time value grafting across a summarization boundary;
  the behavioral dissociation (sense/referent recovered, stance null; the
  Lena-Cho / Ruben / countdown exhibits); tuning (champion cures α=1 collapse);
  honest bounds (30B-only, no clean agent benefit).
- TOOL/SECONDARY: the J-lens (Gurnee et al. 2026 Global-Workspace / Neuronpedia
  Jacobian-lens). Explained briefly, contrasted with next-token and logit lens,
  used to watch the graft act on internal state. We borrowed it; we didn't make it.

## Hard reality that shapes everything
- J-lens weights exist ONLY for Qwen3.6-27B. Our BEHAVIORAL results are on
  Qwen3-30B-A3B-2507. Different models. → lens illustrations are on 27B; the
  behavioral dissociation is on 30B-A3B. MUST be framed honestly ("we illustrate
  the mechanism with the lens on a comparable model for which lens weights exist").
- The other agent's honest finding (intervention_probe_findings.md,
  valuegraft_four_sequence_intervention_probe.md): the graft is ACTIVE and
  ALIGNMENT-SENSITIVE (shifted control much worse), α=0.75 > α=1.0 internally,
  and it does NOT rescue facts a summary aggressively omitted (sparse challenge
  ~zero closure). Their examples were WEAK because the answer was already visible.
  Their explicit recommendation: build lens demos from "label preserved, role
  lost" cases. That is exactly our sense/referent corpus.

## Cross-method convergences to feature (two instruments, same conclusions)
1. α=1.0 harmful, dose down — lens (worse closure at α=1) AND behavior (0/4 break).
2. Alignment matters — lens shifted-control craters AND behavioral wrong-conv guard craters.
3. Mitigation not magic — lens sparse-challenge null AND our referent only 17→26%.
4. Small internal shifts — lens closure ~0.01–0.04 AND my logit-lens near-floor.
→ The lens SHARPENS our claim: reduces REINTERPRETATION error around carried
  facts (sense), does NOT recover omitted facts (referent barely moves). Adopt
  this reframed, more-honest claim.

## NEW RESEARCH NEEDED (the expensive part — makes the piece sing)
Run the three-state intervention design (full / fresh-compacted / aligned-graft
+ shifted control + α sweep 0/.25/.5/.75/1) on STRONG "label preserved, role
lost" examples mirroring our synthetic sense/referent cases (e.g. Nimbus=signup-
funnel-not-cloud-migration; Hydra=multi-task-model-not-GPU-cluster; the sandbox),
on Qwen3.6-27B, sampling the J-lens WIDE across layers at the four boundary
points. Reuse intervention_challenge_probe.py / the three-state design.
Goal: 4–8 compelling exhibits where the lens VISIBLY shows the concept
neighborhood snap from wrong→right when grafted, at a real semantic hinge token
(not a path/punctuation token). Target ~$10–20 pod time.

## Structure (through-line)
1. Hook: compaction loses the built-up *sense*; where does it live?
2. The graft + behavioral dissociation (STAR) — keep Lena-Cho / Ruben / countdown exhibits.
3. "But what's happening inside?" — introduce the J-lens (what it is; vs next-token
   & logit lens; the Nanda hypothesis-gen caveat) and walk the FOUR boundary points.
4. The lens exhibits: concept-neighborhood snap on strong sense/referent cases;
   the α sweep and shifted control read from the inside.
5. Convergence + sharpening: two instruments agree (α, alignment, mitigation);
   the honest reframe (reinterpretation-reduction, not omitted-fact-recovery).
6. Tuning/deployment (champion). 
7. Bounds: scale (4B), lens limits (hypothesis-gen; model-mismatch caveat), the
   reframed claim; end-to-end agent benefit undemonstrated — ONE honest paragraph,
   coding-benchmark detail CUT/de-emphasized per user.
8. So-what + open questions (K/V axis — motivated by both our plan and the lens work).

## Process (repeat of what worked; write down + recheck as I go)
P0 source-of-truth: FINDINGS.md + the jlens docs + NEW lens artifacts. Every
   number/quote traces. P1 outline to this through-line. P2 draft. P3 the 5
   adversarial critics (through-line / gaps / OVER-JUSTIFICATION / accessibility /
   accuracy). P4 synthesize. P5 exhibits verified verbatim vs data. P6 citations
   (both prior-art review + the lens lineage: logit lens, tuned lens, J-lens,
   Global Workspace, KV-edit papers). P7 fresh-eyes holistic prose (bake-off,
   best model per task — Opus/Fable, not cost-driven). Final read + diff-check + push.

## Decisions pending user (see message)
1. Green-light new 27B lens sampling on strong examples (~$10–20)? [rec: YES]
2. Model-mismatch: lens on 27B w/ our-style examples + honest caveat, DON'T
   re-run behavior on 27B? [rec: YES]
3. Cut coding-benchmark detail to one honest bounding paragraph? [rec: YES]

## DECISIONS (user, 07-07) — LOCKED
1. Lens sampling: BIGGER sweep (~$30-50) on strong examples.
2. Model mismatch: RE-RUN BEHAVIOR ON Qwen3.6-27B too — one model, two
   instruments, no mismatch caveat. (Was: lens-only on 27B.)
3. Coding-benchmark: CUT the detail, but keep an HONEST HUMAN note — we tried
   end-to-end agent validation, it burned too much budget, we were still
   searching for a workable approach. Not buried, not over-defended: one candid
   paragraph.

## TECHNICAL RISK introduced by decision 2 (must smoke first)
Qwen3.6-27B is a newer HYBRID architecture; our behavioral grafting machinery
(kvlib_hf value-graft, serve_shim arms) was built for Qwen3-30B-A3B (standard
MoE). The other agent's intervention probe DID V-graft on 27B (so V-graft is
possible), but our behavioral pipeline needs verification there.
PLAN:
- Phase A (GATING): smoke that value-grafting runs on 27B and reproduces the
  arms correctly (alpha-0 == fresh; graft changes output). Reuse the intervention
  probe's grafting path.
- Phase B (behavior on 27B): teacher-forced GAP-CLOSURE on our sense/referent/
  stance synthetic cases on 27B (judge-free; no serve_shim port needed) →
  establishes the dissociation on the SAME model as the lens. Add judged-answer
  meaning-recovery IF the shim ports cheaply; else gap-closure carries it.
- Phase C (lens): bigger three-state intervention on strong label-preserved-
  role-lost examples, wide layers, four boundary points, on 27B.
- Phase D: synthesis writing (both instruments, one model).

## CONFIDENCE + GATED SPEND (07-07, user asked "are you confident")
- HIGH confidence the PIECE is valuable — the two-instrument convergence +
  sharpened honest claim already exist in on-hand data; ships strong even if
  new sampling underwhelms.
- MEDIUM confidence on DRAMATIC lens exhibits — other agent got only subtle
  rank shifts (closure ~0.01-0.04); good examples should be sharper but "snap"
  not guaranteed.
- GATE: cheap ~$5 SMOKE first — (a) verify grafting runs on hybrid 27B (gates
  the same-model behavioral re-run), (b) 2-3 strong lens examples to see if
  exhibits pop. If yes → full bigger sweep + 27B behavioral. If lens stays
  subtle → report to user before the rest; ship the honest synthesis anyway.
  Don't commit $30-50 on faith.
