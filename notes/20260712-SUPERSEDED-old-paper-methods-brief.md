# METHODS BRIEF (extracted from src by Explore subagent, 07-09) — keep for the real take

Two apparatuses: src/cross_arch_probe.py (current main, design v2/v2.1) and
src/gap_closure_cat.py (trusted original F1). Shared: src/arms_common.py, arms_hf.py,
kvlib_hf.py.

## Graft mechanic
- VALUES ONLY blended: V[new] ← (1−α)·V_fresh + α·V_old, float then cast; keys untouched
  (kvlib_hf.py:130-155). alpha_v default 0.75 (SC_GC_ALPHA). All layers/heads by default.
- TWO graft regions (both harnesses; cross_arch_probe.py:1090-1094, gap_closure_cat.py:41-42):
  (1) TAIL: B's tail tokens ← write-time snapshot values from the ORIGINAL conversation tail
  (computed when the evicted middle was still in context);
  (2) SUMMARY: B's context-note tokens ← write-time summary-token values (from the model's
  own summarization act).
- Snapshot capture: F1 = actual KV grown during greedy summary generation (arms_hf.py:59-74);
  cross_arch = force_prefill of [full conv + summary request + full summary incl think] —
  deterministic values, equivalent. Only think-free clean sub-block grafted.
- Alignment: difflib SequenceMatcher(autojunk=False) per region, blocks ≥ MIN_BLOCK=8,
  skip sinks (N_SINK=4) + special tokens (arms_common.py:138-157). Strict direct aligner
  exists but reverted from live path (incident 33: breaks on thinking-model self-gen).

## B-context
system + assistant "[Context note] Earlier parts of this conversation were compacted.
Summary of what came before:\n\n<summary>" + tail msgs from sections.middle_end_msg.
(Gemma family: user-note + assistant ack variant.)
SUMMARY_REQUEST (verbatim, arms_common.py:13-22): "Please write a thorough context note
summarizing our conversation so far, for someone who will continue this conversation
without seeing it. Cover: decisions made and what was chosen over what; open threads and
next steps; definitions, names, and terms we introduced and what they mean; constraints
and preferences either of us stated; approaches or options we tried and ruled out, and
why. Be redundant and specific; use retrieval-friendly wording. Write it as flowing prose
or bullet points, roughly 300-500 words. Do not add commentary before or after the note
itself."
Summary gen: greedy (GEN_TEMP=0.0), top_p 0.8, seed 17, max 900 tok, last token dropped.

## Native render (default ON)
Shared from scaffold: system, ALL user turns, plants, probes, GOLD continuations, keywords.
Per-model: every assistant reply (greedy, max 320 tok, per-conv seed 1000+idx; capped
replies trimmed to sentence) + own self-gen summary (fixed-summary files IGNORED in native
mode). Canonical re-prefill after each reply (think-block handling). Batched decode
byte-identical, SC_NATIVE_BATCH=4.

## Metric
raw_EB = lp_E − lp_B, PER-TOKEN MEAN teacher-forced logprob of shared gold (tokenized
add_special_tokens=False, cap SC_MAX_GOLD_TOK=80, <2 tok dropped). Arms A/B/E + E0 (α=0
check). Multi-probe averaging: gold scored under every probe paraphrase, averaged.
Ratio (E−B)/(A−B) DEPRECATED (median-conditioned |A−B|>0.5 secondary only).
Bootstrap: percentile, ROBUST_N_BOOT=10000 seed=42; headline CI = CONVERSATION-clustered
(plant-level kept as ci_plant). Headroom = mean(lp_A−lp_B)/category;
raw_EB_normalized = raw_EB/max(headroom,1e-3).

## Gates
#1 machinery (status OK vs UNSUPPORTED, never sign): alpha0_ok max|lp_E0−lp_B|≤5e-3;
graft_changes ≥1e-3; identity_ok (self-graft A onto A no-op ≤5e-3); snapshot length must
equal input length (padded/Hybrid/sliding-window rejected); head-axis == num_key_value_heads.
#2 headroom floor 0.3: floored category = uninterpretable, excluded from sign (not harm).
#3 task-competence on per-token lp_A: absolute −8.0 default; pre-registered relative mode
median−3·MADN (MADN=1.4826·median|x−med|), min 8 finite values else no drop.
Coherence screen: ABSENT as gate (empty/truncated replies recorded as covariates only).
Verdict from aggregate cluster CI: SIGNIFICANT_POSITIVE/NEGATIVE/null-underpowered.

## Judged pipeline (scripts/judged_bootstrap.py)
Sonnet-5 verdicts RECOVERED/PARTIAL/MISSED = 1/0.5/0. Graft = POOLED doses a0.25+a1.0
(results/judge_semantic), baseline = B (judge_semantic_base). theta_c = mean(graft) −
mean(B) per category; conversation-clustered bootstrap over the 12 convs, nboot=20000
seed=0. 64 plants: sense 23, referent 18, stance 23.

## Block analysis (scripts/block_analysis.py)
BASELINE c01-c12 (positive control) vs FRESH c13-c24 (held-out; extend to c36 per
stopping rule if fresh referent CI spans 0). ONE pooled competence floor (median−3·MADN,
K=3, min 8) over ALL plants both cells. Per cell×category raw_EB + conv-clustered CI
(N_BOOT=10000 seed=0). HEADROOM_GATE=0.3. Referent morphology split: short_id (gold ≤3
words) vs sem_phrase.

## Constants table
alpha_v 0.75 | bf16 | N_SINK 4 | MIN_BLOCK 8 | summary: greedy/top_p .8/seed 17/max 900 |
native reply: greedy/max 320/seed_base 1000 | gold cap 80 tok min 2 | alpha0_tol 5e-3 |
change_tol 1e-3 | headroom floor 0.3 | lpA floor −8.0 (abs) / median−3·MADN (rel) |
ROBUST_N_BOOT 10000 seed 42 | cluster CI = conversations | ALPHA_SWEEP (0.25,.5,.75,1,2) |
judged nboot 20000 | block N_BOOT 10000 | categories cross-arch: sense referent stance
ruled_out evicted_fact strong_prior | gap_closure_cat: sense referent stance |
default summarizer model (legacy fixed path): Qwen3.6-27B | gap_closure_cat default model:
Qwen/Qwen3-30B-A3B-Instruct-2507.

## Caveats
- lp values are PER-TOKEN MEANS not sums.
- Harnesses differ only: snapshot capture method (equivalent), metric emphasis, cross_arch
  extras (native render/gates/controls/multi-probe).
- Judged doses (0.25/1.0) differ from logprob-metric dose (0.75) — state in paper.
