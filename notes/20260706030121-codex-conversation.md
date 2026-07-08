_This conversation covers naming the two ValueGraft variants, adding repo
availability to the working draft, a long sequence of periodic read-only
refreshes tracking the live E-track coding-agent experiment through a major
data-invalidation incident, and an extended effort to fix the project's core K/V
taxonomy._

**Participants:** User and gpt-5.5-xhigh.

**Naming and draft availability.** The two ValueGraft variants were named
`ValueGraft-Pack` (renamed from `H-pack`) and `ValueGraft-Blend`, with a matched
fresh control renamed `FreshPack`; this was applied only to
`valuegraft-focused-draft.md` and committed/pushed (`70081ee`). A **Code
Availability** section pointing to `https://github.com/jeremyBanks/ValueGraft`
was added (`cf3d67d`) because the current repo is not being anonymized for
review. Other off-project distribution logistics from the conversation are
intentionally omitted here.

**Supervisor-mode readiness check.** The user asked whether the assistant would
be comfortable taking over autonomous supervision of the whole experiment board
(spawning/monitoring pods, running experiments, checkpointing every ~30 minutes
via handoff files). The assistant affirmed readiness but stated it would require
explicit approval before mutating files, spending money, or killing/restarting
pods, and outlined a supervisor-loop shape (stabilize board → fix blockers →
advance phases → checkpoint → stop cleanly). No such handoff was actually
granted in this conversation; work continued via periodic read-only refreshes.

**E-track coding-experiment trajectory (via repeated refreshes).** The project
pivoted fully from LongMemEval (declared dead as a live path, kept only as a
damage-quantification table) to live agent coding tasks run through OpenHands
against a custom OpenAI-compatible serving shim (`src/serve_shim.py`), using
`Qwen/Qwen3-30B-A3B-Instruct-2507` in bf16 on RunPod A100 pods. Early signal
(`t1`/`t2` tasks, tiny n) showed B failing and E (ValueGraft) passing, but this
collapsed after a **major incident**: a per-mode session-isolation bug in the
shim let state leak across A/B/E arms, a recall/dissociation probe was
instrument-broken, a "c4500" confirm tier collapsed even for A/B, and at least
one duplicate concurrent run wrote into a shared scratchpad directory. The
user's assessment — confirmed by the assistant — was that this invalidated the
_live-agent measurement apparatus_, not the underlying ValueGraft hypothesis;
the failures were implementation/protocol bugs (shared session state, wrong-mode
endpoint, broken probe, duplicate run dirs, task ceiling issues), not evidence
against the mechanism. Cleaner non-agent evidence (bf16 honesty/H-pack result,
SWE-Gym replay logprob result, TF/cache-surgery validation) remained intact.

The recovery response, tracked via repeated refreshes, included: `INCIDENTS.md`
added as required reading in `AGENTS.md`; the tainted pod (`e1`) quarantined;
the old matrix demoted to exploratory; a pre-registered clean rerun defined with
a sensitivity gate (abort/harden again if B doesn't fail enough, including
requiring ≥2 t3-B rows); contaminated scores (155 files) versioned into
`results/_QUARANTINE_agent_runs/` with a do-not-use README; a rule added that
all scored results must enter the repo audit trail; clean scores versioned under
`results/agent_clean_run/`. Further incidents surfaced and were fixed during
subsequent refreshes: a `cfg=layers` shim bug referencing `sess` before
creation; Incident 11 (unbounded incremental KV-cache VRAM growth, fixed with a
one-entry eviction bound and assert); Incident 12 (dead shims silently burning
rows into false score files, fixed by purging burned rows, bounding `ccache`
like `icache`, and changing the driver so invalid episodes no longer write
`score.json` — the new trust rule became "only `score.json` + `E1_AGENT_DONE`
counts"). A parallel track added a standard-task pivot: a pre-registered
conditional amendment allowing already-scored/in-flight synthetic (`t1/t2/t3`,
harness-authored, not a standard benchmark) rows to stay, while unrun capacity
shifts to real SWE-bench-Lite instances (`src/swebench_tasks.py`,
`scripts/swb_filter.py`) contingent on a scouting pass, with a rule that
analysis must stratify by task source rather than pooling synthetic and standard
rows. At conversation's end, endpoint health was poor (most matrix ports
unresponsive/timing out) and a further round of provenance-related documentation
and A-arm solvability pre-screening for real tasks was underway. Throughout, no
result set reached enough trusted rows to support any arm-comparison conclusion;
each refresh's honest answer was "operationally promising, scientifically too
early."

**Core taxonomy overhaul.** A long, repeatedly corrected discussion clarified
the actual mechanistic distinction between the project's compaction-mitigation
variants, which had been muddled under names like
`H-pack`/`B-min-pack`/`ValueGraft-Pack`. The assistant initially gave several
incorrect or confusing explanations (conflating "packing," `alpha`, and separate
summarization/layout methods), which the user rejected as incoherent; through
iteration the following became the agreed, documented model:

- Attention keys and values are both learned projections (`W_k`, `W_v`) of a
  token's hidden state; keys determine attention routing (an "address"), values
  determine payload content. RoPE positional rotation is applied to keys (and
  queries) after projection, and is reversible/re-appliable to move a token to a
  new position without recomputing its content-derived component — but
  re-rotating an old key is not equivalent to freshly generating a key from a
  different context, since the underlying hidden state differs.
- The general parameterized family is:
  `K(alpha_K) = (1-alpha_K)*K_fresh + alpha_K*K_write_time_rerotated`,
  `V(alpha_V) = (1-alpha_V)*V_fresh + alpha_V*V_write_time`. Plain compaction is
  `alpha_K=alpha_V=0`; the current live `E` arms are **V-only Graft**
  (`alpha_K=0`, `alpha_V` constant or tuned per layer); a full write-time KV arm
  is `alpha_K=alpha_V=1`.
- "Replace" was rejected as a named method since it is just `alpha=1`, a point
  on the same continuum, not a distinct behavior.
- The historical `H-pack`/`B-min-pack` result is a _summary-only_ context
  (sinks + summary tokens, no tail), varying fresh vs. write-time KV together —
  a different, more confounded axis than the current production-shaped V-only
  Graft experiments. After user pushback that an invented term ("Summary-State
  Graft") was meaningless, unclear, and risked misleading readers into treating
  it as part of the current design, the assistant revised the draft so this old
  result is presented only as plainly-described historical/auxiliary evidence
  (no method name), with its exact differences from the current
  `alpha_K`/`alpha_V` design spelled out (context shape, tail presence, coupled
  K+V change).

This taxonomy was written up in a new root-level document,
`controlled-key-graft-reframing.md`, through several committed revisions
(`74bd6b2`, `3ad5015`), establishing the rule that old code/result identifiers
must not be renamed while experiments are in flight (paper-facing names are
aliases only), but that future controlled key-policy experiments must isolate
`alpha_K`/`alpha_V` as independent variables with everything else (summary text,
layout, tail, positions, decoding) held fixed.

**Draft hygiene.** A new dated draft,
`paper-working/valuegraft-synthesis/valuegraft-focused-draft-2026-07-06.md`, was
created separately from the original focused draft to incorporate the new
terminology and latest clean-run status, then went through several corrective
passes after commit: removing reintroduced "Packed"/"Summary-State" terminology
(restoring `H-pack`/`B-min-pack` as historical-only labels), restoring
citation/related-work detail that had been dropped (Parallel Context Compaction,
LLMLingua, RECOMP, MemGPT), and a general readability pass removing "AI-writing"
patterns (excessive "not X, but Y" constructions, defensive hedging, invented
pseudo-methods) per standing style guidance from the original draft's earlier
cleanup pass. The operative rule reaffirmed repeatedly by the user: commit early
and often, even on non-final work, to preserve a clear history — each committed
document change should be scoped to only the intended file and pushed to
`origin/trunk` promptly. This document form is intended to read as a paper-style
draft: precise, non-defensive, no invented terminology to mask confounds.

---
