_This chunk covers a codex (gpt-5.5) session on the ValueGraft compaction
project, spanning post-mortem reassessment of the collapsed live-agent matrix,
an extended and repeatedly corrected effort to establish clear conceptual
terminology for key/value grafting, revision of paper-facing draft documents to
reflect that terminology, and a series of read-only status refreshes on live
infrastructure as the project pivoted from synthetic to standard coding tasks._

**Participants:** User and gpt-5.5-xhigh.

**Incident assessment.** Following a large data loss from the overnight matrix
run, the session opened with a joint assessment that the failures were
implementation/measurement-protocol failures (per-mode session isolation bug,
instrument-broken recall probe, duplicate concurrent runs into shared scratch
directories, task-sensitivity/ceiling problems) rather than evidence against the
underlying ValueGraft hypothesis. The bf16 honesty/H-pack result, the SWE-Gym
replay logprob result, and the guard-contamination hygiene finding (posslots
failed, layers passed) were judged to remain salvageable. The repository
subsequently added `INCIDENTS.md` as required reading, quarantined the `e1`
lane, demoted the prior night's matrix to exploratory/tainted, and
pre-registered a stricter clean rerun (unique run dirs, preflight-A-must-pass,
no cross-arm sessions, a harder self-tested `t3` task, and a stop rule
invalidating the batch if the control arm fails).

**Terminology repair.** A large portion of the session was spent correcting
confused terminology around the `H-pack`/`B-min-pack` (packed-summary)
experimental arms versus the live `E`-arm ValueGraft. Multiple rounds of user
pushback (the assistant's explanations were repeatedly flagged as incoherent,
self-contradictory, or conflating unrelated axes) forced successive corrections.
The terminology eventually converged on a precise two-axis model: attention
**keys** encode position (via RoPE) plus content-derived addressing features
from the hidden state at generation time; **values** encode payload content read
out when attended to. This yields a general two-parameter family, `alpha_K` and
`alpha_V`, blending fresh vs. write-time keys and values respectively:

- Plain compaction: `alpha_K=0, alpha_V=0`
- V-only Graft (the current live `E` arms `E:a0.75`, `E:a1.0`, `E:cfg=layers`):
  `alpha_K=0`, `alpha_V` constant or tuned
- K-only Graft: `alpha_K` tuned, `alpha_V=0` (not yet implemented/tested)
- Full KV-Graft: both tuned (not yet implemented/tested)

A key clarification: the historical `H-pack`/`B-min-pack` packed-summary-only
experiment is a _different, confounded_ comparison — it uses a minimal
summary-only context (no retained tail) and varies K and V together rather than
isolating one axis — so it cannot be treated as a clean instance of the current
taxonomy; it is auxiliary historical evidence only. The user also directed that
"Replace" not be used as a named method, since it is just the `alpha=1` endpoint
of the same continuous parameterization, not a distinct behavioral category. Two
documents captured this: a revised
`paper-working/valuegraft-synthesis/nomenclature-reframing-note.md`
(paper-facing aliases only, no in-flight code/identifier renames) and a new
root-level `controlled-key-graft-reframing.md`, which explicitly states the
scientific-control requirement (only the axis under test may vary; summary text,
layout, tail, positions, prompt, and decoding must be held fixed) and was
iterated several times to reach the final two-alpha framing, committed as of
`3ad5015`.

**Draft document work.** A new tentative draft,
`paper-working/valuegraft-synthesis/valuegraft-focused-draft-2026-07-06.md`, was
created reusing the existing focused draft's structure but updated to the
`alpha_K`/`alpha_V` vocabulary, folding in the newest clean/standard-task
status. This draft initially reintroduced the rejected "Packed"/"Summary-State"
terminology and lost citation detail (Parallel Context Compaction, LLMLingua,
RECOMP, MemGPT, provider-doc citations) relative to the broader draft; both
issues were corrected in follow-up commits, with the historical summary-only
comparison ultimately reframed in the prose (not as a named method) as auxiliary
historical evidence, its exact controlled variables spelled out, and explicitly
separated from the current main-method taxonomy so a reader cannot mistake it
for a live design axis. A subsequent readability pass removed AI-typical hedging
patterns (excessive "not X, but Y" constructions, defensive caveats,
self-referential guardrail prose) consistent with an established editorial
standard from earlier passes on the main draft. All draft edits were committed
to single-file, explicitly-pathed commits and pushed to `origin/trunk` per
explicit standing instruction to commit and push frequently, even for non-final
work, to preserve a clear history.

**Infrastructure status (read-only refreshes).** Several scheduled/requested
read-only refreshes tracked: Incident 11 (unbounded incremental KV cache VRAM
growth, fixed by one-entry eviction with an added assert) and Incident 12 (dead
shims burning false score rows; fixed by purging burned rows, bounding `ccache`
like `icache`, and changing the driver so invalid episodes no longer write
`score.json`). The trust rule for scored data became: only rows with both
`score.json` and an `E1_AGENT_DONE` log marker are valid. The project pivoted
toward incorporating real SWE-bench-Lite tasks (`src/swebench_tasks.py`,
`scripts/swb_filter.py`) alongside the synthetic `t1`/`t2`/`t3` tasks, with a
pre-registered rule that already-scored synthetic rows remain valid, only unrun
synthetic rows may be replaced by standard-task rows (conditional on a scouting
pass), and analysis must stratify by task source rather than pooling. As of the
final refresh, trusted synthetic-only clean-run results were: `A` 4/4, `B` 2/3,
`E:a0.75` 4/4, `E:a1.0` 3/3, with no trusted `cfg=layers` or standard-task rows
yet — a small but directionally favorable signal for V-only grafting over plain
compaction, explicitly caveated as too small (`n`) to support any real
conclusion. Live shim/lane health was intermittently poor (multiple ports
refused/reset/timed out during health checks), and standard-task scoring had not
yet produced a single trusted result by session's end.
