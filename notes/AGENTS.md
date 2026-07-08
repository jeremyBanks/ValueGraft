# AGENTS.md — notes archive

This directory is an archive for research notes, drafts, reviews, and small
reports. Archive notes should be Markdown files.

Markdown archive files must use UTC, sortable, kebab-case names:

```text
YYYYMMDDHHMMSS-short-kebab-case-title.md
```

The timestamp should be the document's original creation time, traced through
git renames when possible, not the time it was moved into `notes/`.

```bash
python3 scripts/normalize_notes_archive_names.py
```

Usual workflow: drop a note into `notes/`, run the normalizer, then commit the
resulting rename. The normalizer accepts `.md` and markdown-like `.txt` notes,
and outputs `.md`. `AGENTS.md` is the intentional unprefixed instruction file in
this directory.

Conversation-summary notes should use source-specific names:

```text
YYYYMMDDHHMMSS-claude-conversation.md
YYYYMMDDHHMMSS-codex-conversation.md
```

Keep Claude Code and Codex conversations separate even when their dates
interleave. Start each conversation summary with one italicized opening summary
paragraph, containing one sentence or at most two short sentences, then use the
generated `Participants in this Conversation` block, then use short titled
sections and prose paragraphs. The participant block includes `User` only when
the source range has user messages, followed by assistant models sorted by
contributed text volume; every source model ID for the note must appear there.
Bullets are fine for compact lists of named results, rules, arms, or open
questions, but avoid turning a whole conversation into a bullet ledger. Preserve
priority when it affects future work, while describing it as project priority,
blocking status, or required follow-up rather than participant mood. Do not
flatten importance: if emphasis changes what a future agent should do first,
keep that as a project fact or required next action.

Preferred shape:

```markdown
_This shard covers the concrete transition or investigation, plus the durable
decision/result a future agent should know._

**Participants in this Conversation.**

User; `claude-sonnet-4-20250514` (Claude Code `1.0.61`).

**Short Topic.** Use prose paragraphs for the main summary. Capture the
important ideas and state changes, not every exchange.

**Handoff State.** Use bullets only for compact lists of named rules, results,
arms, blockers, or required follow-up.
```
