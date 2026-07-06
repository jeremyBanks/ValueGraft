_This conversation, conducted with the codex/gpt-5.5 agent, tracked the
aftermath of a data-quality crisis in the live coding-agent experiment
(E-track), then shifted into a lengthy conceptual and terminological
clarification of the ValueGraft method family, applied to a separate tentative
paper draft, followed by several more scheduled read-only refresh check-ins on
the evolving live-agent run._

**Participants:** User and gpt-5.5-xhigh.

## Incident aftermath and data status

Following the earlier discovery of multiple infrastructure failures in the live
coding-agent tranche (per-mode session isolation bug, broken recall probe,
duplicate concurrent runs into shared scratch directories, c4500-tier
task/harness failures), the repo response was: add `INCIDENTS.md` as required
reading in `AGENTS.md`, quarantine the `e1` lane, formally demote the earlier
night matrix to exploratory/tainted, and pre-register a stricter "clean run"
protocol (fixed shim, unique run directories, mandatory A/control pass as a
preflight gate, no cross-arm sessions, a harder self-tested `t3` task, and a
stop rule invalidating the batch if the control arm fails). The assessment
reached with the user: these were implementation and evaluation-protocol
failures, not evidence against the underlying ValueGraft hypothesis. Non-agent
evidence (TF/cache-surgery validation, wrong-conversation guards, bf16
honesty/H-pack result, SWE-Gym replay logprob result) remained intact and is the
more trustworthy body of evidence pending the clean rerun.

The clean rerun subsequently went through several further hardening cycles,
documented as additional incidents: an unbounded incremental-KV-cache VRAM
growth bug (Incident 11, fixed with one-entry eviction and an assert), and dead
shims silently producing false score files (Incident 12, fixed by purging burned
rows, bounding the cache, and changing the driver so invalid episodes no longer
write `score.json`). The trust rule for scored data became: only rows with both
`score.json` and an `E1_AGENT_DONE` marker in the log count as valid.

## Task-source evolution

The experiment's task source moved through several stages over the course of the
tracked period: synthetic harness-authored `t1`/`t2`/`t3` templates (seeded
variants like `s30`–`s39`) → a conditional pre-registered pivot toward standard
SWE-bench-Lite tasks (contingent on a local scouting pass) → plain SWE-bench
(`swb:`) rows, which came back essentially all-failing (0/17) and were assessed
as too hard for the model at the current tier → an oracle-retrieval variant
(`swbo:`) that also went 0/7 → eventual retirement of SWE-bench as a track, with
the live program pivoting to `chain:*` tasks as the current usable real-task
tier, and tau2 scouted as a possible future standard benchmark (not yet
evidence). Analysis rules require stratifying by task source rather than pooling
synthetic and standard rows together.

Throughout, synthetic-task results (using an intentionally detail-starved "brief
summary" prompt) consistently showed a positive pattern for value-grafting arms
over plain compaction (e.g., `A` near-ceiling, `B` around 40-50% pass,
`E:a0.75`/`E:a1.0` markedly higher), but this was explicitly downgraded to
mechanism-level evidence ("if the summary drops specifics, grafting can help")
rather than production-faithful headline evidence, since real compaction should
use a better summary. The `chain:*` tier is the closest thing to current
production-relevant evidence but remains too small (single digits per arm) to
interpret.

## ValueGraft naming and taxonomy clarification

A substantial portion of the conversation was spent correcting confused
terminology around what had been called "H-pack" and "B-min-pack." Successive
naming attempts (ValueGraft Pack, Summary-State Graft, KV-Graft, etc.) were each
found inadequate on inspection, with repeated user pushback that answers were
hedged, self-contradictory, or reintroduced the very ambiguity being removed.

The clarification that was eventually reached and adopted as the durable
framing:

- A transformer layer computes separate key, value, and query projections from a
  token's hidden state. Keys function as attention addresses (with RoPE
  positional rotation folded in); values are the payload read out when attention
  is paid.
- "Rotating" a cached key adjusts only its positional component to match a new
  position; it is not equivalent to freshly recomputing the key from a different
  context, because the underlying content component (derived from the hidden
  state) differs depending on what context was available when the key was
  generated.
- The correct general parameterization of the method family is two independent
  blend parameters:
  - `K(alpha_K) = (1 - alpha_K) * K_fresh + alpha_K * K_write_time_rerotated`
  - `V(alpha_V) = (1 - alpha_V) * V_fresh + alpha_V * V_write_time`
- Named "methods" (e.g., a separate "Replace" label for alpha=1) were rejected
  as unnecessary; alpha values are settings within one parameterized family, not
  distinct techniques. Plain compaction is `alpha_K=0, alpha_V=0`; the current
  live `E` arms (`E:a0.75`, `E:a1.0`, `E:cfg=layers`) are V-only grafts
  (`alpha_K=0`, `alpha_V` constant or tuned per layer); a full K/V or K-only
  graft is not yet implemented or tested in the live coding run.
- The old `H-pack`/`B-min-pack` comparison is a categorically different,
  historical experiment: it used a summary-only context (no retained tail) and
  varied both K and V together (write-time vs. fresh) in a fixed minimal layout,
  so it cannot be read as an `alpha_K`-isolated result. It must be described in
  writing as auxiliary/historical evidence, explicitly separated from the
  current `alpha_K`/`alpha_V` taxonomy, with its actual varying dimensions
  spelled out rather than given a method-sounding name.
- Scientific control principle established: whichever axis (K policy, V policy,
  or both) is under test, all other factors — summary text, layout, tail, token
  positions, prompt, decoding — must be held fixed across compared arms.

This taxonomy was written up in a root-level document,
`controlled-key-graft-reframing.md`, through several committed revisions, and
the guidance that in-flight experiment identifiers/code should not be renamed
mid-run — only reporting/paper-facing vocabulary changes — was made explicit and
repeatedly reconfirmed.

## Paper draft work

A separate tentative draft,
`paper-working/valuegraft-synthesis/valuegraft-focused-draft-2026-07-06.md`, was
created (distinct from the existing focused draft, which was left untouched) to
incorporate the new terminology and the latest clean/live-agent status. This
went through multiple correction passes: an initial pass reintroduced the
rejected "Packed"/"Summary-State Graft" terminology and lost citation detail
from the broader draft, both of which were corrected — paper-facing names became
`V-only Graft` and the historical pair was demoted to plainly-labeled auxiliary
evidence rather than a named method, and the fuller related-work citation list
(Parallel Context Compaction, LLMLingua, RECOMP, MemGPT, provider docs) was
restored. A subsequent readability pass removed "AI-ish" hedging patterns
(excessive "not X, but Y" constructions, defensive caveats, self-referential
clarifications) in favor of direct paper prose, consistent with an established
style standard from earlier draft passes: state the claim directly, qualify only
where it changes interpretation, and keep old/auxiliary results visibly
separated from the current design rather than blended into main-method prose.

Operational note reinforced during this work: the user requires frequent commits
(including of non-final drafts) to preserve a clear history, and requires an
explicit push when requested — a standing instruction that was under-followed at
one point and had to be corrected mid-session.

## Refresh-protocol correction

Repeated scheduled read-only refreshes tracked the clean-run's evolution (lane
health, endpoint health, active vs. stalled matrix processes, trusted vs.
untrusted score counts) across many iterations, with the agent explicitly
avoiding any edits, commits, restarts, or kills during these checks. Toward the
end of the tracked period, the user identified a systematic failure mode in
these refreshes: the agent was treating "refresh" as an operational/process
check rather than a full state reconstruction, relying on tailing large or
non-chronologically-structured status files (which caused stale top-of-file
content to be read as current and newer superseding sections to be missed), and
under-weighting narrative/decision changes relative to easily-measured live
metrics like process status and score counts. The corrected refresh protocol
going forward: track the last-seen commit explicitly, diff
`git log <last>..HEAD --name-status` and read the actual changed hunks in all
status/decision/incident/handoff files, and only then separately summarize live
processes and results — explicitly calling out which notes changed and what was
learned from them, rather than re-sampling the whole repo state each time.
