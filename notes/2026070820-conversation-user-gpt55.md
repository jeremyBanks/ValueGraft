_This chunk covers continued work on the transcript-summary archiving pipeline:
adding incremental updates, splitting Claude and Codex conversations into
separate notes with a compact naming scheme tied to git commit dates, adding
mechanical participant/model metadata, building a retry-and-scrub mechanism for
filtering unwanted output language, and finally revisiting the
conversation-grouping rule and rebuilding the full note archive._

**Participants:** User, gpt-5.5-xhigh, and gpt-5.5-medium.

Work proceeded in stages across several full rebuilds of the conversation-note
archive in `notes/`. Early in this chunk, the incremental updater was built to
append new notes and revise existing ones when a source conversation continued
past its last recorded boundary, tracked via a manifest that records exact
message ranges and content hashes per note. A design decision followed: Claude
Code and Codex conversations should never be combined into one note, since they
occur in separate applications; this required regenerating the archive from raw
transcripts rather than editing already-combined summaries. Filenames were
changed to be source-specific and later compacted further per user request: a
`YYYYMMDDNN.md`-style prefix with a two-digit (rolling into base-36) per-day
counter, followed by a kebab-case suffix built from ordered participant slugs
(e.g., `user-gpt55`, `user-fable5-opus48`), with full model identifiers and
reasoning effort levels kept only inside the file body's `**Participants:**`
line, not in the filename.

A significant operational requirement emerged: once the date component would
eventually be dropped from filenames, git commit history had to become the
authoritative timestamp source. This meant every newly created or renamed note
file needed its own commit with author/committer dates set to that
conversation's first-message time, and rename operations had to avoid
`git mv`/`--follow` semantics that could trace back through unrelated prior
history. Several bugs surfaced and were fixed in this area: a `git log --follow`
date-lookup bug where deleted-and-recreated files at the same path picked up the
wrong (older) creation date, a Python 3.9 incompatibility parsing UTC
`Z`-suffixed ISO timestamps, relative-vs-absolute path handling in the commit
helper, and a normalizer bug that failed to insert a new file into a gap left by
previously deleted notes. Each was caught via dry-run and unit tests before
being applied.

Required content rules established for generated notes: (1) a deterministic,
script-inserted `**Participants:** User, model-a, and model-b` line (not a
separate heading, no semicolons, no CLI/runtime version noise, no generated
"model switch sequence" block) built from raw transcript metadata, sorted by
contributed text volume, omitting "User" when no user messages exist, and
filtering out non-model status/error labels like `<synthetic>`; (2) prose-first
summaries with an italicized opening-summary paragraph at the top (not a title)
and bullets reserved for compact enumerated lists, not full ledgers; (3) no
internal "shard" terminology in reader-facing text, though internal script/flag
names may keep it; (4) content should describe corrections and priorities as
project facts (requirements, blockers, priority) rather than describing
participant affect, and colorful phrasing should be paraphrased neutrally; (5)
publication/venue/upload/coauthor logistics should be omitted from notes unless
they changed a tracked repo artifact; (6) headings within notes should use bold
paragraph labels rather than inconsistent Markdown heading levels.

To enforce the language constraints mechanically rather than relying on prompt
compliance, a retry mechanism was added to the summarizer: a case-insensitive
`--forbid-regex` argument (default: only secret-shaped patterns like
AWS/OpenRouter/RunPod/HuggingFace key formats) causes the tool to detect matches
and re-run the summarizer with a fresh context that lists the accumulated
(deduplicated across all prior attempts) forbidden matches and instructs it to
use vaguer, more generic phrasing instead. After five attempts, any remaining
matching lines are deleted as a crude fallback. For rebuilds run directly by the
user, a much broader, uncommitted-by-default regex was added covering
sensitive-topic terms and terms describing personal reactions, colorful
language, or similar wording — including the literal word describing that
category — to strengthen the negative examples fed back into the retry loop;
this broader filter is intentionally not part of the committed defaults.

Partway through the final rebuild, the user identified that the note-grouping
logic re-coalesced raw transcript segments into notes without respecting the
one-hour gap or day-boundary splits used during raw extraction, meaning
genuinely separate conversations (including several hours-long assistant-only
status-update stretches with no user participation) were being merged into
larger notes. After comparing several grouping thresholds, the user chose a rule
of coalescing raw segments into one note only when the gap between them is under
2 hours, with crossing UTC day boundaries allowed. Applying this to the current
transcript set yields 22 notes (8 Claude-side, 14 Codex-side) versus 16
previously, with zero notes ending up assistant-only, though it was noted that a
stricter one-note-per-raw-segment policy would yield 40 notes including 7 with
no user participation. This 2-hour coalescing rule was implemented as the new
default (`--max-coalesce-gap-hours 2.0`, with a negative value preserving old
size-only behavior), covered by new tests, and committed.

At the point this chunk ends, the plan in progress (not yet executed) is: delete
all existing generated conversation notes and the manifest in a single batch
commit, then regenerate the full 22-note archive from raw transcripts using the
updated 2-hour coalescing rule, the heading-style guidance update, and the broad
run-only forbidden-term regex, committing each new note individually with its
correct historical date and pushing when complete. The user confirmed they are
fine with the process being slow due to apparent upstream API latency, as long
as it continues to make progress, and does not want unrelated in-flight
repository changes (e.g., an unrelated modified results JSON file, other agents'
unrelated commits) touched during this work.
