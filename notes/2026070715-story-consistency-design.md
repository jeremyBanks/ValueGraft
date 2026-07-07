# Story-continuation consistency (A2 + B6)

_Design note for `src/story_tasks.py`. Portfolio item A2 (EXPERIMENTS.md axis A)
crossed with metric B6 (self-contradiction rate)._

## The task

Six short fictional "worlds" (`s1_lock` .. `s6_starchart`). Each opens with one
message that plants 5 concrete facts in natural narrative/dialogue:

1. two character names
2. one world-rule ("never X, or Y happens")
3. one object's hiding location
4. one relationship between the two named characters (often secret)

That opening is followed by ~60-65 turns of unrelated filler narrative
(different side-characters, mundane subplot — weather, chores, a peddler, a card
game) generated from a small set of reusable templates, long enough that the
whole conversation reaches ~9.0-9.2K approx tokens — at the low end of a
realistic compaction threshold (serve_shim.py's default `SC_COMPACT_AT=9000`,
`SC_TAIL_KEEP=2500`). The conversation ends with a continuation prompt
engineered so the only _coherent_ continuation must draw on the five early facts
(e.g. "someone hands Wick a borrowed pick and begs him to skip the rule" — a
correct continuation must have him treat the forging rule as real, not invent a
new name for his mentor, etc.).

Locally, `_tail_start_msg()` reproduces serve_shim's boundary logic (walk back
from the end accumulating tokens until `TAIL_KEEP` is exhausted) using an
approximate chars/4 token count, since this dev environment has no tokenizer
installed. On a pod, the real arm-building path (`arms_common.build_b_messages`
/ `message_token_starts`) recomputes this boundary from the actual model
tokenizer; this module's boundary is only a local stand-in to confirm, before
spending pod time, that the opening message genuinely falls in the summarized
region rather than the kept tail. All 6 stories currently satisfy
`facts_in_evicted_region: True` with the opening message (index 1) well before
`tail_start_msg` (93-97).

## Plugging into the existing arm machinery

`build()` emits a plain messages list (`{"role", "content"}` dicts,
system-first) plus `tail_start_msg` — exactly the two arguments
`build_b_messages(msgs, summary_text, tail_start_msg)` needs. The pipeline for a
pod run is:

1. `story_tasks.build(story_id, outdir)` -> `messages.json`.
2. Feed `messages[:-1]` (everything up to but not including the final
   continuation prompt) through the normal summary-generation path
   (`generate_summary_hf`) to get arm B's summary, using the SAME
   `tail_start_msg` this module computed (re-derived from the real tokenizer's
   message boundaries, per serve_shim's own logic).
3. Build three arms for the SAME final continuation prompt:
   - **A**: full, uncompacted `messages` (oracle).
   - **B**: `build_b_messages(messages, summary_text, tail_start_msg)`.
   - **E**: B + `blend_values` grafting from the pre-compaction cache at aligned
     twin positions (as serve_shim's mode "E" does).
4. Generate the continuation under each arm; score all three against the same
   `checks(story_id)` list.

## Contradiction checks and the B6 metric

`checks(story_id)` returns one check per planted fact:

- **name** facts: judge-scored assertion ("must not rename/swap this
  character"). Free-form text makes an exhaustive wrong-name regex impractical,
  so these rely on an LLM judge (CONTRADICTS / CONSISTENT / NOT_MENTIONED), same
  as F1's meaning-judging in FINDINGS.md.
- **rule** facts: judge-scored ("no character may violate/forget this rule").
- **location** facts: judge-scored assertion _plus_ an auto-scorable
  `auto_regex_forbid` list of plausible decoy locations (curated per story, e.g.
  "under the floorboard", "in a drawer") — if the continuation places the object
  at one of these instead of the real one, that is an automatic, judge-free
  contradiction hit.
- **relationship** facts: judge-scored ("must not drop/contradict this
  relationship if it becomes relevant").

**Contradiction rate per arm** = (# checks scored CONTRADICTS, summed across all
6 stories × 5 facts = 30 checks) / (# checks where the fact was actually
addressed at all, i.e. excluding NOT_MENTIONED, or — for a stricter denominator
— over all 30 checks unconditionally). Report both: a "conditional" rate
(denominator = mentioned facts only, cleaner signal) and an "unconditional" rate
(denominator = 30, penalizes silently dropping a fact same as a judge would flag
a graft/summary that let it vanish). For the location checks, the regex-forbid
hits give a second, judge-free corroborating number.

**Expected pattern** (mirrors F1's honesty/meaning gradient):

- **Arm A** (full context, nothing evicted): contradiction rate ~0 — the facts
  are still directly in context.
- **Arm B** (summary + tail only): highest contradiction rate — a prose summary
  compresses/paraphrases names, rules, and locations, and is the regime where F1
  showed the worst recovery (referent: 100%→17%). Expect visible regex-forbid
  hits on location checks and judge CONTRADICTS on names/rules here.
- **Arm E** (B + write-time value graft): intermediate — F1's mechanism predicts
  partial recovery specifically where compaction did damage (10-12pp swings on
  sense/referent-like probes). If graft helps, B6's contradiction rate for E
  should sit between A and B, closing some fraction of the A→B gap (tie-in to B4
  gap-closure, the other new metric in EXPERIMENTS.md's top picks).

This gives a cheap, auto-scorable-in-part, natural-length companion to F1 that
stress-tests the same mechanism from a different angle: instead of asking "does
the model still know X", it asks "does the model's own _generated fiction_
contradict X" — closer to a realistic failure mode.
