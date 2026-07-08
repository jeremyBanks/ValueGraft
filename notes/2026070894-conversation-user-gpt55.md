_This entry covers continued work on the conversation-note archive: building an
incremental updater with continuation handling, separating Claude and Codex
transcripts into distinct notes, establishing a style convention with a
generated participant line, moving from timestamped filenames to compact per-day
counters backed by git commit dates, and adding a retry-and-filter mechanism for
generated summary text._

**Participants:** User, gpt-5.5-xhigh, and gpt-5.5-medium.

## Incremental updater and manifest

An incremental updater was built on top of the existing extract/shard tooling,
backed by a manifest recording exactly which raw message ranges each note
covers. Running it appends notes for new, uncovered ranges and only revises an
existing note when a source conversation it already covers has continued past
its recorded boundary. The updater groups multiple continued source streams that
land in the same note into a single revision pass rather than issuing
conflicting prompts. A later correction made the updater defer very small
continuations (e.g., a handful of fresh messages appearing while the updater
itself is running) instead of triggering a full revision each time; only
continuations past a size threshold, or an explicit force option, trigger a real
rewrite. This was needed because repeated testing runs were causing needless
prose churn in recently generated notes.

Running the updater with no arguments now performs the standard update using
repo-specific defaults (known transcript locations, `notes/`, the manifest, the
default summarizer, and automatic formatting), so a future agent does not need
to know the underlying pipeline to keep the archive current.

## Source separation and content policy

Claude Code and Codex conversations are now always split into separate notes
rather than being packed together, even when their timestamps interleave,
because mixing application sources in one note was judged confusing. The full
note archive was regenerated from raw transcripts under this rule rather than
patched by hand. A default lint layer scans for token/secret-shaped strings
(AWS-style keys, OpenRouter keys, RunPod keys, Hugging Face tokens); an optional
`--forbid-regex` argument allows ad hoc additional checks without hardcoding
project-specific content policy into the tooling itself.

Generated notes are expected to open with an italicized summary paragraph (not a
title), use prose sections, and reserve bullet lists for compact enumerations
rather than converting the whole note into a list. Corrections and requirements
are described as such rather than attributed to participant mood or temperament;
project priority or blocking status is preserved when it should change what a
future agent does, expressed as priority/blocker language rather than as a
description of how someone felt. Off-topic logistics about external publication,
venues, or coauthor process are omitted from notes unless they changed a tracked
repository artifact. Several rounds of validation found generated notes drifting
into disallowed patterns — narrating the cleanup process itself, quoting removed
phrasing, or otherwise being self-referential about the archive-cleanup work —
and these were corrected by tightening the generation prompt and by direct edits
to the affected notes.

## Participant metadata

A generated metadata line was added identifying who took part in each
conversation, derived mechanically from the raw transcript rather than left to
the summarizer. After several rounds of format correction, the final agreed form
is a single sentence: `**Participants:** User, model-a, and model-b.`, with
`User` included only when user messages are present, and assistant models listed
in descending order of contributed text volume. Full model identifiers are used,
with reasoning effort appended directly as part of the identifier via a hyphen
(e.g., `gpt-5.5-xhigh`), not in parentheses and not as a separate
switch-sequence block — an earlier version that added a decorated multi-field
heading with runtime/provider detail and a semicolon-separated list, plus a
separate "assistant model sequence" line, was rejected and removed along with
the code paths that generated it. Placeholder/status labels appearing in raw
transcripts (e.g., synthetic error or credit-exhaustion markers) are filtered
out of participant detection since they are not real model identifiers. The
summarizer prompt separately instructs the model to note in the prose when the
acting model changes mid-conversation, but that is left to narrative text, not
enforced metadata.

An earlier fix also caught a regex bug where inserting the participant line
stripped the rest of an existing note's body in files using `##`-style headings
instead of `**`-style ones; affected notes were restored from the last good
commit and regenerated correctly.

The term "shard," used internally for transcript chunking, was removed from all
reader-facing prose (notes, prompts, docs) since it is meaningless to an outside
reader; it remains acceptable in internal script/flag names describing
file-chunking mechanics.

## Filename and history changes

The archive originally used full-timestamp filenames
(`YYYYMMDDHHMMSS-{source}-conversation.md`). This is being replaced with a
compact scheme: `YYYYMMDDNN-conversation-user-<slug1>-<slug2>...md`, where `NN`
is a two-digit per-day counter starting at `01`, rolling over past `99` into
base-36 (`A0`, `A1`, ...), and the suffix lists compact participant slugs (brand
prefixes and version punctuation stripped, e.g. `claude-opus-4-8` → `opus48`,
`gpt-5.5-xhigh` → `gpt55`) in the same order as the generated participant line.
Reasoning effort and full identifiers are deliberately omitted from the filename
and kept only in the note body, since filenames were already long.

Because the full date is leaving the filename, the git commit date for each note
file becomes the canonical record of when the underlying conversation occurred.
The agreed rule: editing an existing note in place needs no special handling,
since its creation date is already set; but creating a new note, or changing an
existing note's path, requires that file to be introduced in its own commit
dated to the conversation's first-message time (via author/committer date),
always, not as an optional mode. A single combined rename commit was avoided
because git's rename-detection could trace back through history to an earlier
commit where multiple notes were introduced together, defeating the per-file
dating goal — instead, path changes are done as an explicit delete-then-add
pair.

This was implemented as a shared naming/planning helper used by both the
normalizer script and the transcript generation scripts, with unit tests for
counter rollover, prefix parsing, and slug collapsing. Migration ran in stages:
existing non-conversation notes were renamed to the compact scheme first (78
files, one dated commit each); the 16 conversation notes were then deleted; a
colliding demo file (`...pokemon-demo-conversation.md`) was renamed off the
conversation-note glob; and a subsequent pass recomputed final compact slots and
reinserted the 16 conversation notes with per-file dated commits, shifting some
existing counters as needed. Along the way, two related bugs were found and
fixed: git's ISO-8601 dates with a trailing `Z` weren't parsed correctly on the
local Python version, and helper code was comparing relative and absolute
repository paths inconsistently, both of which were causing incorrect
date/rename decisions. After migration, the normalizer's dry run reports zero
further renames needed, confirming naming stability.

## Retry-and-filter mechanism for generated text

A general retry mechanism was added to the summarizer path: an optional
case-insensitive forbidden-pattern argument can be supplied; if a generated
summary matches it, the summarizer is re-run with a fresh context that lists
everything matched so far (accumulated across all attempts, not just the latest
one) and explicit instructions to avoid those terms or similar language,
substituting vaguer, less detailed phrasing instead. This repeats up to a
configurable limit (default five attempts), after which any remaining matching
lines are deleted outright as a blunt fallback. The default committed pattern
set covers only token/secret-shaped strings; a broader pattern set — covering a
specific historically sensitive topic plus wording associated with informal
reactions, discontent, or coarse language, including the literal word describing
that category of wording — is used only for interactively run rebuilds and is
not checked into the repository defaults. Supporting fixes included treating a
missing manifest as empty coverage (so a full from-scratch rebuild is possible)
and allowing new dated notes to fill numbered gaps left by earlier deletions
rather than only appending after the highest existing counter.

## State at end of chunk

Tooling and test changes for the retry mechanism, the gap-filling counter fix,
and two path-handling bugs were committed in small scoped commits, keeping the
unrelated modified result file
(`results/cross_arch_wide/Qwen__Qwen3-30B-A3B-Instruct-2507.json`) unstaged
throughout. A full conversation-note archive rebuild — deleting all 16 generated
notes and the manifest, then regenerating from raw transcripts through the
retry/filter pipeline with the broader run-only pattern set — was in progress at
the end of this chunk, restarted after fixing the path bugs, with no generated
notes yet committed in this final pass.
