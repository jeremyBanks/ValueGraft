_This conversation established and completed a clean, provider-configurable
transcript archive rebuild using Luna (`gpt-5.6-luna`) or Claude, with
nonpersistent summarization, selective subagent-final inclusion, contributor
provenance, referent-level filtering, model-based filenames, and recursive
rollups. It also produced a scientific audit and staged research plan concluding
that naive post-prefill value-only grafting has not shown a robust benefit,
while identifying stronger causal and agentic experiments._

**Participants:** User and gpt-5.6-sol-xhigh.

**Handoff State.** The archive workflow is consolidated behind one command with
shared stage flags for extraction, summarization, filename normalization, and
hierarchical rollups. Provider/model selection propagates through conversation,
daily, monthly, yearly, and archive summaries; raw `--command` remains an escape
hatch. Codex workers use ephemeral sessions, Claude workers use
`--no-session-persistence`, and explicit summary-worker detection prevents
historical summarizer sessions from entering their own input. A real
user-message requirement alone was insufficient because print-mode workers
contain synthetic user prompts.

Only final assistant assessments from subagents are included, excluding prompts,
reasoning, tool calls, and progress updates. Claude extraction takes the final
nonempty response from each nested subagent session; Codex extraction accepts
child `FINAL_ANSWER` response items. Duplicate finals are removed, included
material is labeled as subagent evidence, and finals are inserted at their
chronological positions during clean rebuilds. The old manifest is ignored
during regeneration, and all replacement notes are generated and validated in a
temporary directory before replacement.

Participant provenance is required at every summary layer. Conversation notes
retain deterministic users and source-supported model/version/effort
identifiers; rollups carry exact contributor identities represented in their
immediate sources. Models are never invented when metadata is unavailable. Each
conversation note ends with an unlinked `Conversation sources` list containing
opaque parent and subagent conversation IDs.

Filename conventions use only compact user and contributing model/version
identifiers. Effort levels, task labels, subagent counts, opaque IDs, and
byte-limit machinery are excluded. Opaque source IDs remain only in the footer.
Examples include `...-conversation-user-gpt56.md` and
`...-conversation-user-fable5-sonnet5.md`.

Referent filtering is defined as subject-level obscurity rather than literal
word avoidance. When a source matches an exclusion filter, summaries must remove
the underlying subject, synonyms, distinctive facts, entities, locations,
events, and contextual clues, or replace the entire topic with a generic
non-identifying description. This is still primarily prompt-guided and requires
semantic review beyond literal-match checks.

The implementation corrected custom-command argument swallowing, summary-worker
self-ingestion risks, contributor/filename conflation, and provenance placement.
The unified custom-command dry run was verified not to execute its summarizer.
The implementation passed the final focused and repository test checks, with 56
tests passing at the final audit.

The clean Luna rebuild completed successfully:

- 55/55 conversation candidates passed generation, formatting, and validation.
- Atomic replacement committed as `8e69584`.
- Filename normalization committed as `a577ad6`.
- Daily, monthly, yearly, and archive rollups were regenerated with Luna.
- 55/55 conversation notes carry Luna provenance, model-based participant lines,
  and source footers.
- 269 opaque parent/subagent source IDs were preserved.
- 10/10 daily/month/archive rollups contain participant/model paragraphs.
- No filenames contain task labels, effort labels, or opaque IDs.
- No literal exclusion-filter matches remained.

A post-generation audit found that rollup prompts initially allowed generic
roles such as “labeled subagents.” Those prompts were tightened to require only
users and exact source-supported model identifiers, versions, and effort levels.
A deterministic required-participant ledger was added so rollups cannot silently
omit identities present in lower-level notes. July 4 was regenerated after it
incorrectly omitted the available `claude-fable-5` identity. A forced
continuation update also refreshed the current task’s own note so it records the
completed rebuild and final audit rather than the earlier in-progress state.

The scientific consultation concluded that the repository supports a qualified
negative result about training-free, post-prefill, value-only exact-token
grafting, not a general negative result about latent continuity across
summarization. The strongest caveats are:

- The held-out synthetic pool reverses sign by conversation source:
  Qwen3-4B-rendered and Claude-authored blocks behave differently, so the pooled
  null is not a homogeneous definitive test.
- Three of four compression-sweep conditions reconstructed the wrong summary
  request and cannot support claims about compression severity.
- Existing placebos are not matched to the treatment perturbation and do not
  isolate semantic content.
- SWE-Gym trajectories are imported teacher demonstrations rather than
  model-native agent episodes; the result measures action imitation, not
  coding-agent success.
- The coding champion remains provisional and has not established superiority
  over a simpler scalar treatment.

The recommended research direction is to test coherent original K+V state
against identical-text fresh restart, wrong-history controls, K-only/V-only
variants, treatment-delta-matched placebos, and summary-versus-tail arms. A live
agent study should use one model-native investigation prefix, snapshot the
workspace at compaction, fork full-history, ordinary-compaction, and treatment
branches, and score executed tests or resolved patches. The proposed $0 Mac
program prioritizes strict rescoring of existing SWE results and a small
controlled state-channel experiment; staged $50, $100, and $200 plans use
sequential capability and futility gates before expanding to live agent tasks.

## Conversation sources

- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f4f16-8cc4-7d10-b574-9f41c2e4d817`
- `019f4f16-73cb-7be3-acff-071c4f9df64b`
- `019f4f2d-27ee-7f10-a846-f24facc12a19`
- `019f4f2d-1b08-7b61-aa7c-8d55f26f2f5b`
