_This conversation established and implemented a clean, provider-configurable
transcript archive rebuild using Luna (`gpt-5.6-luna`) or Claude, with
nonpersistent summarization, subagent-final inclusion, contributor provenance,
and referent-level filtering. The implementation is validated, but the full Luna
regeneration remains an in-progress, pre-swap operation pending final candidate
validation._

**Participants:** User and gpt-5.6-sol-xhigh.

**Handoff State.** The archive workflow is being consolidated behind one command
with shared stage flags for extraction, summarization, filename normalization,
and hierarchical rollups. Provider/model selection propagates through
conversation, daily, monthly, yearly, and archive summaries; raw `--command`
remains an escape hatch. Codex workers use ephemeral sessions, Claude workers
use `--no-session-persistence`, and explicit summary-worker detection prevents
historical summarizer sessions from entering their own input. A prior audit
found that requiring a user message alone was insufficient because print-mode
workers contain synthetic user prompts.

Subagent inclusion is intentionally narrow: only final assistant assessments are
included, excluding prompts, reasoning, tool calls, and progress updates. Claude
extraction takes the final nonempty response from each nested subagent session;
Codex extraction accepts child `FINAL_ANSWER` response items. Duplicate finals
are removed, and included material is labeled as subagent evidence. For the
clean rebuild, these finals are inserted at their actual chronological positions
rather than appended as compatibility segments; the old manifest is ignored and
all replacement notes are generated into a temporary directory before any swap.

Participant provenance is required at every summary layer. Conversation notes
retain deterministic users and source-supported model identifiers, with
reasoning effort available in the participant block; rollups identify
contributors represented in their immediate sources. Models must not be invented
when metadata is unavailable. Each conversation note also ends with an unlinked
`Conversation sources` list containing opaque parent and subagent conversation
IDs. These IDs and subagent task labels are not filenames.

Filename conventions were corrected after several implementation mistakes.
Filenames contain only compact user and contributing model/version identifiers,
with effort levels omitted; they do not contain task labels, opaque IDs,
subagent counts, or arbitrary byte-limit machinery. Real examples now resolve to
forms such as `...-conversation-user-gpt56.md` and
`...-conversation-user-fable5-sonnet5.md`, while the participant line can
include `gpt-5.6-sol-xhigh` and the opaque IDs remain in the bottom provenance
list.

The summarization prompts now require referent-level obscurity for excluded
topics. A forbidden match identifies the underlying subject, not merely a
literal word: summaries must remove synonyms, distinctive facts, entities,
locations, events, and contextual clues, or replace the entire subject with a
generic non-identifying description. This requirement is applied at all summary
levels and is now preemptive when the source itself matches an exclusion filter.
The implementation remains a prompt-guided safeguard rather than a complete
semantic guarantee, so semantic review is still appropriate.

The clean rebuild initially exposed and corrected several blockers:
custom-command arguments were swallowing orchestration flags such as `--dry-run`
and `--resummarize-all`; contributor labels had been incorrectly reused as
filename content; and opaque subagent IDs had been placed directly in filenames,
producing the false filename-length problem. The final design separates
participant identity, filename identity, and source provenance. The corrected
implementation passed 57 repository tests, and a unified custom-command dry run
was verified not to execute its summarizer command.

At the last recorded state, the implementation commit had been pushed, the
corrected Luna run had generated and validated the first three of 55 replacement
conversation notes, and existing notes remained untouched because the atomic
replacement gate had not yet been reached. The intended sequence is to finish
all 55 candidates, validate contributor/model coverage and formatting, replace
the old conversation notes and manifest in one clean operation, normalize names,
and recursively regenerate daily and higher-level rollups with the same Luna
configuration.

## Conversation sources

- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f4f16-8cc4-7d10-b574-9f41c2e4d817`
- `019f4f16-73cb-7be3-acff-071c4f9df64b`
- `019f4f2d-27ee-7f10-a846-f24facc12a19`
- `019f4f2d-1b08-7b61-aa7c-8d55f26f2f5b`
