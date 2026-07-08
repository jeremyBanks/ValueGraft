_This conversation covers the design, implementation, and iterative refinement
of an incremental conversation-note summarization pipeline for the notes/
archive, including its file-naming scheme, generated metadata, and
content-filtering safeguards._

**Participants:** User, gpt-5.5-xhigh, and gpt-5.5-medium.

The starting design point was an incremental updater layered on the existing
extract/split tooling: a manifest records exactly which raw message ranges each
note already covers, new transcript ranges are appended as new notes, and only a
note whose source conversation continued past its recorded end is revised (the
summarizer is given the prior summary plus old and new messages and asked to
extend it rather than restart). Prompt-only/dry-run passes are used before any
paid generation call, so continuation cases and planned file operations can be
checked first. The committed default content filter is scoped to secret-shaped
strings only (AWS-style access key IDs, OpenRouter keys, RunPod API keys,
Hugging Face tokens), reframed explicitly as a repository-safety default rather
than a content-style preference; a separate `--forbid-regex` argument exists for
ad hoc checks.

**Style contract.** Existing notes were inconsistent (some had an opening
summary, some used headings, bullet density varied). The agreed fix, applied as
both a prompt change and a light edit pass rather than a full rewrite: every
note opens with a short italicized summary paragraph (not a title), body content
is prose-first with bullets reserved for compact enumerated lists, and generated
text should describe corrections and requirements as project facts/priorities
rather than participant mood — importance should still be preserved, but
expressed as priority, blocking status, or required follow-up. An early literal
misreading of "keep the opening to one line" (which produced overly short,
forced single-physical-line summaries) was corrected back to "one paragraph,
wrap length doesn't matter," with tooling/docs wording updated accordingly. A
separate small terminology objection (an internal placeholder term used in early
guidance drafts) was fixed in the docs and prompt, but scrubbing every trace of
it from already-generated note text was flagged as unnecessary effort and
dropped.

**Source separation.** A significant correction mid-session: Claude Code and
Codex conversations should never be combined into one note, since they occur in
separate applications; existing combined notes needed to be regenerated from raw
transcripts rather than patched apart. This became the default in both the
initial splitter and the incremental updater (with an explicit legacy flag
preserving the old combined behavior only if ever needed). Summarization was
made sequential per source stream, carrying a small prior-context snippet
derived from the previous note's opening summary, purely for orientation, not to
encourage merging or rewriting history.

**Participants metadata.** After several iterations (an over-decorated first
attempt using a heading with semicolon-separated entries and CLI/runtime detail
was rejected as poorly formatted, and a follow-on generated "assistant model
sequence" line was rejected outright as unrequested), the settled design is a
single deterministic paragraph, generated mechanically from raw transcript
metadata rather than left to the summarizer: **Participants:** in bold, followed
by a plain-English list (comma-separated, Oxford comma, "and") ordered by each
participant's contributed text volume, descending. `User` appears only when user
messages exist in that conversation, and is otherwise omitted entirely. Each
assistant entry is the full model identifier with reasoning effort appended via
a hyphen (e.g., `gpt-5.5-xhigh`), with no provider name, runtime/CLI version, or
other decoration. Placeholder status/error entries emitted by some tooling (not
real model identifiers) are filtered out of this computation. The prompt
separately instructs the summarizer to note a change of assistant model within
the prose when one occurs mid-conversation, but the roster line itself is not
subject to summarization.

**Filenames and dating.** Filenames evolved from full-timestamp, source-suffixed
names to a compact `YYYYMMDD` plus two-character per-day counter (`01`-`99`,
rolling into base-36 `A0`, `A1`, … beyond that), with a descriptive suffix built
from compact, lossy participant slugs in participant order (no brand prefix, no
effort/version — e.g. `conversation-user-gpt55`,
`conversation-user-fable5-opus48`); full model identifiers and effort levels
live only in the note body. A later cosmetic refinement makes the two-character
counter continue across days rather than resetting to `01` daily, except that a
day whose continued range would cross `99` instead resets its leading digit
while keeping the trailing digit's alignment (mod-10) — explicitly requested as
inexpensive to implement (a naming-rule and test change) and not to be achieved
by re-running any summarization. Because the date is leaving the filename, the
canonical per-note date (first message time, in UTC) now must live in git
history: it was made a hard rule, honored by dry-run, that any newly created or
renamed path must land in its own commit whose author/committer date equals that
canonical date; ordinary edits to an already-existing path do not require this.
Implementing this required deleting and regenerating the full 16-note archive
with per-file dated commits (to avoid git rename-detection walking back into old
bulk-created history), and separately fixing a `git log --follow` trap where a
reused filename's history traces through a prior, deleted lifetime — date lookup
now uses the current lifetime's first "add" commit rather than the oldest commit
reachable by follow, since synthetic commit dates are not monotonic with git log
order.

**Content-filtering retry loop.** A self-correction mechanism was added: if a
case-insensitive forbidden-content regex matches generated output, the
summarizer is rerun from a fresh context listing the deduplicated set of matches
accumulated across all attempts so far, instructing it to avoid those terms or
similar phrasing in favor of vaguer, more generic language. The retry cap
defaults to five attempts, after which any remaining matching lines are deleted
outright as a last-resort fallback. The committed default pattern set covers
only secret-shaped strings; a considerably broader pattern set — covering
certain sensitive historical/political subject matter and strongly charged or
coarse phrasing, including the literal category name for that phrasing — is
deliberately kept out of the repository defaults and supplied only as a manual
argument whenever the archive owner personally runs a full rebuild.

**Execution and fixes.** The full archive was regenerated from an emptied
manifest using the retry-capable summarizer with the broader owner-supplied
filter; the run took a non-trivial amount of wall-clock time attributable to
upstream API latency rather than any local issue, progressing through 16 notes
with periodic progress checks. Partway through, an approximate estimate of five
to ten more minutes was given based on observed per-note pacing, later
superseded by the run's actual completion. The retry mechanism triggered
multiple times during this run, and at least one note exhausted its retry budget
and had matching lines removed per the fallback rule; final validation reported
no remaining forbidden matches. Several implementation defects were found and
fixed along the way: an overly greedy regex for removing an old metadata block
had been truncating note bodies that used `##`-style headings (fixed, with
affected notes restored from the last good commit); relative-vs-absolute path
handling bugs in new date-lookup and commit helpers; treating an absent manifest
as empty coverage so regeneration from scratch is possible; allowing a new note
to fill a freed numeric slot instead of being blocked by a later same-day file;
and a Python 3.9 compatibility gap parsing `Z`-suffixed ISO-8601 timestamps from
git log output, fixed in all affected script locations with a regression test.
One two-file swap (adjacent same-day notes) was initially done as a content swap
at stable paths, recognized as not preserving git-history identity, reverted,
and redone as a genuine rename chain through a temporary filename.

**Handoff state.** A one-command default now exists
(`python3 scripts/transcripts/update_conversation_notes.py` with no subcommand)
that performs the normal incremental update against known transcript locations,
the manifest, the default summarizer, and automatic formatting; documentation
was updated (top-level pointer, transcripts README, notes-directory guidance)
with concrete style examples rather than only abstract rules. A deferral
threshold was added so tiny live-tail continuations (e.g., messages generated by
the very session doing the update) do not trigger a full note rewrite, since
that behavior had been producing unwanted churn during testing; only a
meaningful new chunk or an explicit force triggers revision. A non-transcript
demo note (unrelated Pokémon content) was kept out of scope throughout and
renamed once to avoid an accidental filename collision with the
conversation-note pattern. Unrelated in-flight files elsewhere in the repository
(a results JSON and a couple of other scripts) were consistently left untouched
and unstaged across every commit in this session. At the end of the session the
archive was in a validated state: dry-run renaming reported nothing left to do,
the relevant test suite passed, manifest integrity checks passed, and the
forbidden-content scan reported zero matches.
