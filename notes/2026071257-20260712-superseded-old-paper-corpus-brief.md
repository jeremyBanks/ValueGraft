# CORPUS PROVENANCE BRIEF (extracted by Explore subagent, 07-09) — keep for the real take

## Counts
- 54 synthetic scenarios c01–c54 (data/scenarios.json = master scaffold; rendered convs
  data/synthetic/cNN.json). c01-c12: ~8.3-9.4K tokens, 45 msgs (22 user turns), 10 plants
  each (5 categories × 2: referent/sense/stance/ruled_out/evicted_fact). c13-c54: ~2.2-3.2K
  tokens, 30-36 user turns, 12 plants each (adds strong_prior; c20/c21 have 13).
  Total plants: 626 (120 in c01-c12; 506 in c13-c54).
- 8 natural conversations n01-n08 (data/natural/): 16 turns, ~7.4-11.4K tokens, ZERO
  plants; holdout_msgs=4 as scored continuation; personas hand-written in
  src/compose_natural.py, all turns sampled by Qwen3-4B-Instruct-2507-4bit (seeds 2000-2007).

## Plant structure
id, category, distance (near/far), middle_user (planted-fact user msg in evicted middle),
tail_user_fragment (tail turn referring back without restating), probe (+ probes list of
~4 paraphrases in augmented), gold (correct answer derived from planted fact, median
~16-18 words, capped 80 tok at scoring), keywords/anti_keywords (contamination audit +
keyword scoring). Originals carry audit annotations from src/audit_corpus.py.

## Authorship
- c01-c12 scaffold: hand-authored at project start (Claude subagents wrote scenario/probe
  text per 07-04 daily summary; exact per-file author not recorded in files).
- c13-c54 scaffold: authored 07-08 by subagent mix, meta.author per file:
  wave 1 (c13-c27): c13-15 sonnet, c16-18 opus, c19-21 fable, c22-24 sonnet,
  c25-27 codex-gpt5.5. Wave 2 (c28-54, 6 parallel shards, batch files
  data/scenarios_batch_c{28,33,38,43,48,52}.json): c28-37 sonnet, c38-42 opus,
  c43-47 fable, c48-51 sonnet, c52-54 opus (with -render suffix tags).
- c01-c12 stored assistant replies: mlx-community/Qwen3-4B-Instruct-2507-4bit via
  src/compose.py (growing KV cache, temp 0.7, seed 1000+i, MAX_REPLY_TOKENS=420, capped
  replies trimmed to sentence, canonical think-block re-prefill with per-turn identity
  assertion; meta.tail_repaired 3-6/conv). compose.py = dev tool only now.
- c13-c54 stored replies: written directly by authoring subagent — SUPERSEDED by native
  render for all experiments (foreign-reply confound).

## Native render (v2.1 matched-scaffold model-filled)
FIXED shared: system, all user turns (early/middle/tail incl. plant turns), plants,
probes, GOLD continuations. REGENERATED per model: every assistant reply + own self-gen
summary. Gold shared deliberately: raw_EB difference cancels target-nativeness
(LOAD-BEARING — protect).

## Screens
- Contamination audit (audit_corpus.py): plant keywords must appear in own middle_user,
  not in early/tail; annotations in-place. (Original corpus: 115/120 clean after repair.)
- Runtime gates: headroom 0.3, task-competence floor, short-gold drop (<2 tok).
- Coherence gate on native replies: mandated by MASTER-PLAN, NOT implemented as exclusion
  (empty/truncated replies recorded as covariates only) — state honestly.
- Augmented convs schema-verified per file at authoring.

## Open flag (07-09)
c13-c54 did NOT carry +0.10 on the OLD (foreign-reply) render (~+0.009); native-render fix
validated on c01-c12 only; block-design held-out reproduction in flight.
