_This conversation continues the transcript-summarization pipeline work: making
the incremental updater less prone to churn on small live-tail continuations,
adding a mechanically generated and validated participant paragraph to every
conversation note, and catching and fixing a regex bug that had truncated
several older notes before it reached a commit._

**Participants:** User and gpt-5.5-xhigh.

**Live-tail deferral.** Repeated re-running of the updater during testing was
rewriting the most recent notes for only a handful of new messages, forcing
manual cleanup after nearly every run. This was identified as a sign that the
tool's default behavior was wrong, not that manual touch-ups were an acceptable
workflow: a well-behaved one-command updater should not need hand-editing after
ordinary runs. The updater was changed to defer small continuations by default —
a conversation only triggers a note revision once enough new material has
accumulated, with an explicit force flag to override for deliberate runs — and
the final status message was corrected so it accurately reports when work was
deferred rather than implying prompts were written. The transcript README was
updated to describe this deferral behavior. This work, plus the earlier
manifest-coverage rollback fix and the resulting `notes/AGENTS.md` reference
update, was committed as `2126663` ("Harden conversation summary updater") and
pushed to `trunk`. One unrelated modified file,
`results/cross_arch_wide/Qwen__Qwen3-30B-A3B-Instruct-2507.json`, was left
unstaged throughout.

**Model/participant provenance requirement.** A gap was identified: many
existing notes did not name which models had taken part in the conversation,
which is treated as required information going forward — every model that
participated in a conversation must appear at least once in that conversation's
note, ideally including provider and version. Because relying on summarizer
prose compliance is unreliable, this was implemented as deterministic,
tool-generated metadata rather than freeform summary content: a "Participants"
heading is computed directly from the raw transcript's per-message model
metadata (the same metadata already captured in transcript headings) and
inserted into each note, independent of the summarizer. The design went through
one revision during the conversation: an initial "Models" list was replaced with
a "Participants" block per the following rules — `User` is listed only when the
conversation actually contains user messages (some agent-only conversations may
have none), followed by assistant models sorted in descending order by the
volume of text each contributed, rendered as a single compact heading/sentence
rather than a bulleted list. Separately, the summarizer prompt was updated to
instruct the model to narrate switches between assistant models within the flow
of the summary when more than one model participated, since the mechanical block
only captures the roster and overall order, not when and why a switch mattered.
This logic was implemented in both the incremental updater and the
full-rebuild/splitter path, and a validation gate was added requiring every
model ID recorded in the manifest for a note to actually appear in that note's
text, so a summarizer omission is now caught mechanically instead of relying on
manual review. Top-level and transcript-specific documentation (`AGENTS.md`,
`notes/AGENTS.md`, `scripts/transcripts/README.md`) were updated with the new
heading name and example shape.

**Clarification on formatting.** A brief miscommunication was resolved during
this work: an aside about `deno fmt` rewrapping an example paragraph was misread
as describing wrapping itself as undesirable. It was clarified that paragraph
wrapping is intended and correct; the actual issue was that the patch tool
matches on exact surrounding text, so a patch built against pre-formatting
content failed to apply after `deno fmt` had already rewrapped the paragraph — a
tooling context-matching problem, not a style problem. No formatting behavior
was changed as a result.

**Regex truncation bug caught before commit.** While inserting the new
Participants block, a diff review ahead of staging revealed that several older
notes had been drastically shortened. The cause was a block-removal regex that
only recognized bold (`**`) headings as a stopping point when replacing the old
inline block, so on notes using `##`-style Markdown headings it deleted body
content after the insertion point along with the block it was meant to replace.
This was caught before any commit was made. The fix restored the affected
conversation notes and the manifest to their last known-good committed state,
corrected the regex to stop at any Markdown heading rather than only bold text,
and reran the mechanical sync so the Participants block could be reinserted
without further data loss. The rerun was verified before commit: older notes
retained their bodies, each source-specific note contains a Participants block,
and every model ID recorded in the manifest appears in the corresponding note.
