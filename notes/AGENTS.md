# AGENTS.md — notes archive

This directory is an archive for research notes, drafts, reviews, and small
reports. Archive notes should be Markdown files.

Agents may add, edit, or move note content here, but they should not hand-manage
archive filename prefixes. Put the content in the directory, then run the notes
scripts; the scripts own prefix normalization, manifest updates, and
meta-summary generation.

Markdown archive files are normalized to UTC, sortable, kebab-case names:

```text
YYYYMMDDNN-short-kebab-case-title.md
```

`YYYYMMDD` is the UTC date. `NN` is assigned by each file's git creation
timestamp and normally continues across day boundaries. If carrying the prior
day's suffix forward would push a day past `99`, that day starts at the same
trailing digit in the `01` through `10` range instead. The full timestamp
belongs in git history, not the filename. Do not invent or manually edit these
prefixes; run the normalizer.

Daily meta-summaries are the one intentional naming exception:

```text
YYYYMMDD.md
README.md
```

`YYYYMMDD.md` files synthesize the ordinary notes for that UTC day and are
excluded from the archive counter. `README.md` synthesizes the daily summaries
into one project-history overview. Generate or refresh all meta-summaries with:

```bash
python3 scripts/update_notes_meta_summaries.py
```

Generate or refresh one daily summary with:

```bash
python3 scripts/update_daily_meta_summary.py YYYYMMDD
```

The daily-summary manifest keys staleness by the sorted set of source git blob
IDs for that day, not by source filenames. Conversation summaries are included
in full up to `16 KiB`; above that they use a `12 KiB` leading excerpt plus a
`4 KiB` tail excerpt. Other notes are included in full up to `8 KiB`; above that
they use a `6 KiB` leading excerpt plus a `2 KiB` tail excerpt, with an explicit
omitted-content marker in the gap.

If any daily summaries change during an all-days run, the overall `README.md` is
regenerated from the full daily summaries. The overall prompt includes date
headers before each source day so the model can understand sequence, but it is
instructed not to copy those date headers into its output.

Summary generators accept `--forbid-regex` and also read
`VALUEGRAFT_NOTES_FORBID_REGEX` or `NOTES_FORBID_REGEX` from the environment or
from git-ignored dotenv files at `.env` and `notes/.env`. Use that for local
topic/sensitive-term filters that should be respected by the daily and overall
summary layers without committing the filter text.

```bash
python3 scripts/normalize_notes_archive_names.py
```

Usual workflow: drop or edit notes in `notes/`, then run the normalizer and the
meta-summary updater. Use `--dry-run` first when reviewing planned changes. The
normalizer accepts `.md` and markdown-like `.txt` notes, outputs `.md`, and
updates conversation-summary manifest paths when conversation notes are renamed.
The summary generators run `deno fmt` on generated Markdown/JSON before
recording manifest hashes. `AGENTS.md` is the intentional unprefixed instruction
file in this directory; `README.md` is the generated overview.

Conversation-summary notes use participant-based names:

```text
YYYYMMDDNN-conversation-user-gpt55.md
YYYYMMDDNN-conversation-user-fable5-opus48.md
```

Keep Claude Code and Codex conversations separate even when their dates
interleave. Start each conversation summary with one italicized opening summary
paragraph, containing one sentence or at most two short sentences, then use the
generated `**Participants:** ...` paragraph, then use prose paragraphs. If
section labels help, use optional bold paragraph-opening labels such as
`**Handoff State.**` rather than Markdown heading syntax. The participant
paragraph includes `User` only when the source range has user messages, followed
by full assistant model identifiers sorted by contributed text volume. If
reasoning effort is present, append it to the model identifier with a hyphen,
such as `gpt-5.5-xhigh`; every source model ID for the note must appear there.
Filenames intentionally collapse model identifiers to compact participant slugs
(`fable5`, `opus48`, `sonnet5`, `gpt55`) and omit effort levels. Bullets are
fine for compact lists of named results, rules, arms, or open questions, but
avoid turning a whole conversation into a bullet ledger. Preserve priority when
it affects future work, while describing it as project priority, blocking
status, or required follow-up rather than participant mood. Do not flatten
importance: if emphasis changes what a future agent should do first, keep that
as a project fact or required next action. Preserve concrete ETAs, deadlines,
durations, recurring check-back cadences, and expected completion windows as
handoff facts, including later revisions or cancellations. Do not infer
timestamps or convert relative timing into false precision.

Preferred shape:

```markdown
_This conversation covers the concrete transition or investigation, plus the
durable decision/result a future agent should know._

**Participants:** User, claude-sonnet-4-20250514, and gpt-5.5-xhigh.

**Short Topic.** Use prose paragraphs for the main summary. Capture the
important ideas and state changes, not every exchange.

**Handoff State.** Use bullets only for compact lists of named rules, results,
arms, blockers, or required follow-up.
```
