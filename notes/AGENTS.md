# AGENTS.md — notes archive

This directory is an archive for research notes, drafts, reviews, and small
reports. Archive notes should be Markdown files.

Markdown archive files must use UTC, sortable, kebab-case names:

```text
YYYYMMDDNN-short-kebab-case-title.md
```

`YYYYMMDD` is the UTC date. `NN` is assigned by each file's git creation
timestamp and normally continues across day boundaries. If carrying the prior
day's suffix forward would push a day past `99`, that day starts at the same
trailing digit in the `01` through `10` range instead. The full timestamp
belongs in git history, not the filename. If a note is created or renamed, that
file must be introduced in its own commit with author and committer dates set to
the note's canonical timestamp. Existing-path edits do not need special
timestamp handling.

```bash
python3 scripts/normalize_notes_archive_names.py
```

Usual workflow: drop a note into `notes/`, run the normalizer, and let it commit
any required per-file rename. Use `--dry-run` first when reviewing planned
changes. The normalizer accepts `.md` and markdown-like `.txt` notes, and
outputs `.md`. `AGENTS.md` is the intentional unprefixed instruction file in
this directory.

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
as a project fact or required next action.

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
