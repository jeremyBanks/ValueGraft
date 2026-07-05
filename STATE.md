# STATE.md — session handoff notes

*Last updated: 2026-07-04 ~21:40 (update this file at every phase transition).*

## What this project is

Testing whether KV-cache state written at generation time carries semantic
continuity that text-recomputation loses across a chat-compaction boundary.
Read in order: `semantic-continuity-experiment-brief.md` (main design),
`followup-explorations-arms-GH.md`, `phase2-scaleup-and-coding-extension.md`,
`amendments-from-external-review.md` (adds B-causal, negative controls,
leakage classes, metric hierarchy). `DECISIONS.md` = every deviation + verified
model/runtime facts (READ IT before touching cache code — it documents the
traps: think-block template instability, batched-vs-stepwise kernel mismatch,
alignment region crossing).

## Environment

- uv project; `uv run python src/...`. mlx-lm 0.31.3 + transformers pinned 5.0.0.
- Dev model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (all results so far).
- Final model downloaded, unused yet: `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`.
- Long jobs: launch detached (`nohup ... & disown`, PID to scratchpad
  `pipeline.pid`) because harness-tracked background tasks got killed twice.
  `python -u` + `tee` to scratchpad log; monitor greps the log.

## Pipeline state (as of 2026-07-04 ~23:45)

**4B pilot COMPLETE — see RESULTS.md.** Headline: probe-accuracy mitigation
null; mechanism supported (H-gap>B-min +0.093 nats CI[.04,.14]; E-post α=.25
closure +0.08 both corpora; micro-sense positive; negative controls clean);
novel honesty effect (H-gap flips fabricate:admit from B's 19:5 to 4:20 in the
brief condition, robust across 10/12 convs). Judging: 2089 Sonnet verdicts,
all logged.

**NOW RUNNING:** L-ladder on the 30B (SC_MODEL env; scratchpad
ladder30b.log). If green → launch targeted 30B run overnight:
`SC_MODEL=...30B... SC_OUTDIR=results/raw_30b SC_E_POST=0.25,1.0 SC_E_INTER=
uv run python -u src/run_arms.py` then run_brief with
SC_OUTDIR_BRIEF=results/raw_30b_brief. Purpose: do the three live effects
(honesty, α=.25 CONT gain, H>B-min) survive scale? Negative controls at 30B
via supplement_arms.py only if 30B shows effects. Then: score both new dirs
(extend CONDITIONS in score.py or point at new dirs), judge new answers,
compare 4B vs 30B in RESULTS addendum.

## Pipeline state (older, for context)

- DONE: L0–L4 ladder (all pass; identities exact), micro sense experiment
  (results/micro_sense.json — V-swap carries sense, KV-swap ~half of oracle),
  corpus (12 synthetic in data/synthetic + 8 natural in data/natural,
  tail-contamination repaired, 115/120 plants clean), main 4B batch
  (results/raw/*.json: CONT + probes for 12 arms per conversation).
- RUNNING now (detached, ~21:30 start): supplement pass (B-causal +
  E-wrongconv + E-shuffled) merging into results/raw; then run_brief.py
  (terse-summary shadow condition) → results/raw_brief/.
- QUEUED after that (in order):
  1. `uv run python src/fix_bcausal_cont.py` — B-causal CONT was skipped by a
     template quirk during supplements (see DECISIONS); this repairs it.
  2. `uv run python src/score.py export` — builds results/judge_queue.json
     (judgments + paraphrase leakage checks).
  3. `uv run python src/judge_batches.py split` — batch files; judge each
     batch with a Sonnet subagent (user: no Haiku — use Sonnet; prompt:
     answer each item's prompt
     with the single word demanded; write verdicts_NN.json as {key: verdict});
     max 2 agents at a time. Then `judge_batches.py merge`.
  4. `uv run python src/score.py apply` — final scores (results/scores.json).
  5. `uv run python src/analyze.py` + `src/plots.py` — tables + figures.
  6. Decide 30B run scope (trim α sweep; include H/B-min/B-causal/negative
     controls); switch MODEL constant in src/run_arms.py etc., rerun L0/L3
     identities on 30B first (never report from an un-laddered config).
  7. RESULTS.md write-up structured per refocusing-and-reframing.md
     (mitigation-first claim hierarchy; E/H/negative-controls center of
     gravity; C kept for mechanism/factorization, not the practical claim;
     funny observations noted per user). Paper-style draft, Phase-2 memo,
     and 30B run are DEFERRED (user decision 2026-07-04 ~22:20) until after
     RESULTS + a careful re-review of refocusing-and-reframing.md.
     If H-gap shows judged clean-cut signal, H-pack (re-rotation + identity
     test) is the top follow-up candidate.

## Key interim findings (4B, pre-judge — do not over-claim)

- Calibration perfect: evicted-only facts A 9/9, ALL compacted arms 0/9.
- Summary leakage: 81/120 plants explicit-in-summary → clean referent/sense
  cut underpowered; hence the brief-summary shadow condition now running.
- CONT (both corpora consistently): E-post α=0.25 small reliable gain
  (closure ~+0.08, CI excludes 0 on both); heavier α hurts monotonically;
  C ~ −0.2..−0.7; D ≈ 0. Negative controls (c01): shuffled/wrong-conv graft
  crater to −2.6 — effect is content+alignment-specific.
- Probes (leaky cut, keyword-only): C/H-gap ≥ B on referent/sense; E ≤ B.
- Stance keyword numbers are junk pre-judge (anti-keyword penalizes quoting).

## File map (src/)

kvlib.py (cache primitives, GappedKVCache, teacher-forcing), arms.py (arm
builders, summary gen, alignment), run_arms.py (main driver, ArmSet,
12 variants), supplement_arms.py (B-causal + negative controls),
run_brief.py (shadow condition), fix_bcausal_cont.py (repair),
compose.py / compose_natural.py / fix_tails.py (corpus), audit_corpus.py
(contamination flags), score.py (leak classes, judge queue/apply),
judge_batches.py, analyze.py (gap closure), plots.py, micro_sense.py,
l0..l4 scripts (ladder — rerun on any new model/config).
