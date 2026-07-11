# Corrections to weigh after Sol's final review — DO NOT apply yet

**Status (2026-07-11):** Holding pattern. Sol (independent GPT-5-family auditor) reviewed the *old* README + the real data and produced four notes (2026071003–06). It has **not yet** reviewed the *final* paper (commit 1d64086); a proper final review is pending and may take a while. Per owner: make NO paper changes now — capture what may need correcting, then do it **all at once, holistically**, once Sol's final review lands. This file is that capture. It is my (Opus) assessment, with what I independently verified vs. what still needs checking.

## Verified by my own recompute (load-bearing — these are real)

1. **BLOCKING — synthetic held-out body provenance is misstated in §2.3.** README §2.3 states the conversation-body assistant replies are "generated in-context by the test model itself" and "we do not reuse replies written by another model," and makes it load-bearing. The files say otherwise: `data/synthetic/c07–c12.json` `meta.model = mlx-community/Qwen3-4B-Instruct-2507-4bit`; `c13–c24.json` `meta.author = sonnet/opus/fable`. Validation ran `SC_NATIVE_RENDER=0` (pre-rendered bodies, not 30B-rendered). **Only the summary is 30B-self-gen; the bodies are 4-bit-4B or Claude/Fable-authored.** Verified directly against files + the final README. This is the project's core provenance sin, recurring inside the held-out set the whole paper rests on.

2. **BLOCKING for the synthetic framing — the "clean held-out null" is a source-dependent sign reversal.** Independent recompute of the per-layer champion `raw_EB` (conversation-clustered bootstrap over the trace data), matching Sol to 3 decimals:
   - c07–12 (Qwen3-4B-4bit bodies): **+0.1197 [+0.056, +0.180]** — strong positive, CI excludes 0.
   - c13–24 (Claude-authored bodies): **−0.0503 [−0.075, −0.028]** — significant negative, CI excludes 0.
   - Pooled (18): −0.0026 [−0.041, +0.042] — the reported "null."
   The −0.003 null is an arithmetic average of two opposite, individually-significant subgroups. §3.1's "clean null / no champion beats baseline" framing is misleading; the honest statement is "source-conditional sign reversal." The subgroups are exploratory (6 vs 12 clusters) and source is confounded with length/authoring, but the interaction is large and real.

## Verified by my own recompute — round 2 (from Sol notes #004/#005, checked 2026-07-11)

8. **Graft is summary + retained tail, NOT summary alone.** `src/run_swegym_hf.py:397-404` builds TWO alignment regions — the retained tail `(b_starts[2]→end)` and the summary `(b_starts[1]→b_starts[2])` — and blends values across both. Same pattern in `cross_arch_probe.py`/`run_tune_hf.py`. The abstract/method "replacement at the summary-token positions" is a **misdescription**: no result can be attributed to summary-state retention specifically; tail-state (identifiers, paths, tool state) could carry or cancel the effect. Fix the description everywhere to "summary + retained tail," and the mechanism is un-attributed until a summary-only / tail-only factorial runs. CONFIRMED by code.

9. **SWE-Gym generated summary text is NOT saved.** Result JSONs carry `summary_tokens` (an integer count) + arm scores + `gold_action`; no summary text, no summary/target token IDs, no cut/tail context, no aligned-pair map. README's "exact text hash" is `summary_request_sha256` — the hash of the *request prompt*, not the generated summary. The expensive coding renders are therefore **not exactly reproducible/auditable** from committed artifacts — violates the repo's own "SAVE EVERY RENDER" rule. CONFIRMED by inspecting `results/swegym_tune_.../t0215.json`.

10. **`code.git_commit: null` in the SWE-Gym/champion manifests.** README says the exact code commit is recorded; the pod workspace lacked git metadata, so `code.git_commit` and `code.git_dirty` are both null. Model repo/revision/dtype/transformers-version ARE recorded well; the code commit is not. CONFIRMED by manifest.

## Credible, still to verify in the holistic pass

3. **Compression sweep (§3.2) request-consistency.** Sol: ultra/brief/medium generate the summary under a level-specific request but reconstruct the write-time V source under the constant realistic request, so only `realistic` is self-consistent; the flat-severity result is then invalid. I confirmed each level has its own summary request (`compression_levels()`), but did NOT trace whether the *write-time V source* uses the level request or a fixed one. **VERIFY by code-trace before trusting or correcting §3.2.**
4. **Discrete action-match coarseness (§3.3).** Sol: 84/98 targets are `str_replace_editor`; the scorer matches tool + path + editor-command only and ignores `old_str`/`new_str`, so "match" ≠ correct edit. Plausible — verify in `src/swegym_action_match.py`; if true, the discrete metric caveat in §3.3 needs to say "action-type/path match, not edit correctness."
5. **Placebo not delta-matched.** Sol: gauss/shuffle placebos aren't matched to the treatment-delta row-norm/covariance, so "content-specific" is largely "avoids the harm a wrong graft causes," not proof of hidden-history semantic specificity. §3.1/abstract already lean this way ("non-additive… a matter of not inflicting harm"), but could state the placebo-matching limitation explicitly.
6. **SWE-Gym target provenance.** The gold next action is from the original SWE-Gym trajectory (GPT-4o/Claude-authored), not the Qwen subject — so §3.3 is teacher-imitation of a foreign trajectory, not the subject's own behavior. Worth a scope sentence.
7. **"Bounded raw_EB" terminology.** Log-prob differences are not mathematically bounded; the word "bounded" (if it survived into README) is inaccurate. Minor.
11. **Discrete action-match significance is softer than the bootstrap says.** §3.3 / Sol #005: the 98-traj discrete match "+7.14pp, bootstrap LB>0" rests on 11 discordant pairs (9 graft-only vs 2 baseline-only); an exact paired McNemar gives two-sided **p=0.0654** — suggestive, not conventionally significant. Also 84/98 targets are `str_replace_editor` and the scorer ignores `old_str`/`new_str`, so it's "demonstration-action-type/path match," not edit correctness. Rename "correct action" → "demonstration-action match" and soften the significance. Verify the scorer in `src/swegym_action_match.py`.
12. **"bit-for-bit / exact production identity" language.** Smoke gates compare *averaged* logprobs at ~5e-3 tolerance; tokenwise cancellation can hide errors. The identity discipline is real and good, but "bit-for-bit reproducible" overstates it. Soften if that wording is in the final README.
13. **Provenance-claim overstatements to reconcile with #9/#10:** README's "every result records the summary condition and its exact text hash" and "exact commit recorded" are false for the SWE-Gym artifacts. Correct the provenance prose to what's actually stored.

## Prior art / novelty — Sol #005 (UNVERIFIED citations — check before relying)

Sol argues the broad premise (generation-time KV state carries task info beyond the summary *text*) is **already established prior art**, so the paper must not frame it as unknown. Cited (I have NOT verified these arxiv IDs exist — do so before citing; treat as leads):
- **MEMENTO** (MS Research, ~Apr 2026) — the near-exact conceptual experiment: keep memento KV vs re-prefill identical memento text; restart drops Qwen3-8B AIME24 66.1%→50.8%. Uses full KV + compaction-aware *training*, not our untrained V-only graft.
- **"Models Take Notes at Prefill"** (~Jun 2026) — conclusions get memoized onto downstream token states; naive local KV edits fail because the info lives elsewhere. Directly explains why our token-local post-hoc edit can fail.
- CacheBlend, Gist Tokens, Cache-to-Cache, H2O/StreamingLLM/SnapKV — adjacent KV-reuse/compression work.
- **Defensible novelty framing** (Sol's, worth adopting): don't claim "useful info beyond text summaries was unknown"; claim we test whether a *simpler training-free* post-hoc V-graft (old V, fresh K) recovers continuity in an ordinary instruction model — and find it generally doesn't, with a possible small in-domain coding exception. **Action:** verify each citation is real (not hallucinated) before it enters the paper; then add a Related Work paragraph and temper the novelty claim.

## Other items that depend on what survived into the FINAL README (check during holistic pass)

Sol reviewed the OLD README; these may already be fixed by Fable's finalization — check each against the final text, correct only if still present:
- H-pack honesty result presented as ValueGraft evidence (it's OOD packing/calibration; wrong-summary control equal/stronger; and `phase2_30b_scored.json` is 4-bit MLX, not bf16). Keep separate from the value-only story.
- LongMemEval "52.5%→4.1%, n=320" triple — no committed scored roll-up found; use the reproducible n=36 (80.6%→5.6%) instead.
- Cross-arch final gate not uniform (Qwen2.5/32B/phi used absolute floor; final rule is relative; Qwen2.5 excluded 9 plants; dtype inferred not born-annotated) — only matters if architecture claims remain.
- QK-norm architecture hypothesis was falsified (Mistral positive without QK norm); no 16-model regression. Temper any architecture-causal language.
- "own-summary is required" — from a render-fragile comparison; state as "observed in one controlled comparison; not independently replicated."

## Already OK in the FINAL paper (Sol's old-README critiques that are moot)

- "§3.3 says the SWE positive replicated" / "latest champion omitted" — these were about the OLD README. Fable's finalization already added the pooled SWE-Gym result, the in-domain champion, and retired the "replicated" framing. Sol's pending review of the *final* paper should drop these.
- The SWE test-retest-isn't-independent point and the pooled +0.0053 borderline null **match Fable's own pooled analysis** already in final §3.3 — consistent, no correction needed there.

## Net effect on the paper's conclusion (for the holistic rewrite)

The final paper currently reads: synthetic recovery = clean null; compression = flat/negative; SWE = naive graft heterogeneous, in-domain-tuned champion a small out-of-sample positive. After corrections it likely becomes:
- Synthetic held-out: **not a homogeneous null — a source-conditional sign reversal** (positive on Qwen-family-rendered bodies, negative on Claude-authored), on a corpus whose body-provenance §2.3 currently misstates. This is arguably a *more interesting* finding (nativeness/source interaction) but it dismantles the "clean bounding null" spine.
- Compression: possibly only the `realistic` cell is valid (pending #3).
- SWE: largely stands (Fable's pooled analysis is sound), with the discrete-metric + target-provenance caveats tightened.
- Postmortem: should explicitly own that the provenance error **recurred in the finalization itself** (I pushed a "final" paper with a false native-body claim, caught by an independent auditor) — a live instance of the paper's own thesis.

## Process notes

- Do the corrections in ONE holistic pass after Sol's final-paper review — don't patch piecemeal.
- The real meta-lesson: an independent adversarial provenance audit needed to run **before** "final," not after. The internal review stack (including me) missed a false load-bearing provenance claim that a fresh external auditor caught in one pass.
- External publishing stays held regardless until this is reconciled.
