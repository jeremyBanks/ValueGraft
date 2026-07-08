# Transcript Summary Pipeline

These scripts reproduce the mainline conversation-summary workflow used for the
`notes/*-conversation-*.md` files.

The pipeline is intentionally plain:

1. Extract mainline Claude Code and Codex messages from local JSONL stores.
2. Build contiguous transcript segments sized for a summarizer, keeping Claude
   Code and Codex in separate source streams by default.
3. Summarize each segment with a CLI model.
4. Combine segment summaries if desired.
5. Split a combined summary back into compact dated `notes/` files. Each created
   note is committed individually with author/committer dates matching the first
   source message it covers.
6. Keep a manifest of covered source-message ranges so future updates are
   incremental.

## Usual Command

For the normal "bring conversation notes up to date" workflow, run this from the
repo root:

```bash
python3 scripts/transcripts/update_conversation_notes.py
```

With no arguments, the script defaults to the known local Claude Code and Codex
transcript files for this repo, writes prompts/candidates under
`/tmp/valuegraft_transcript_incremental`, runs `claude --print --model sonnet`,
updates `notes/*-conversation-*.md`, runs `deno fmt` on generated Markdown files
when Deno is available, and refreshes
`scripts/transcripts/conversation-summary-manifest.json`. It also inserts a
deterministic `**Participants:** ...` paragraph from raw transcript metadata:
`User` when present, then assistant models sorted by contributed text volume. If
reasoning effort is present, it is appended to the model identifier with a
hyphen, such as `gpt-5.5-xhigh`; provider names, app runtimes, and CLI versions
are not included.

Raw transcript extraction splits at UTC day boundaries and at gaps over one
hour. The update workflow may coalesce adjacent raw segments into one note, but
only within a source stream and only when the inter-segment gap is at most
`--max-coalesce-gap-hours`, default `2.0`. Crossing a UTC day boundary is
allowed when that gap condition is still satisfied.

Use `--no-command` to write prompts only, or pass `update --command ...` to use
a different summarizer command. Small continuations of an existing note are
deferred by default so the script does not keep rewriting the latest note for
the live tail created while an agent is working; use
`--force-small-continuations` only when that is intentional.

Generated summaries are checked case-insensitively against `--forbid-regex`
patterns. Defaults only cover common access-token shapes. If a candidate
matches, the script retries with a fresh prompt that lists the accumulated
forbidden matches and asks for vaguer language around those topics. After
`--max-forbid-attempts` attempts, it deletes matching lines as a last-resort
scrub.

The summaries should focus on ideas, decisions, methodology, results, caveats,
and handoff state. If a discussion established an intended writing form, such as
paper-style, blog-style, article-style, or report-style, capture that. Preserve
priority when it affects future work, but describe it as project priority,
blocking status, or required follow-up rather than participant mood. Do not
flatten importance: if emphasis changes what a future agent should do first,
keep that as a project fact or required next action. Avoid side logistics unless
they directly affect repository workflow.

Conversation-note style:

- name files as `YYYYMMDDNN-conversation-<participants>.md`, where `NN` is the
  per-day git-creation-time order and participant slugs are compact (`user`,
  `fable5`, `opus48`, `sonnet5`, `gpt55`)
- start with one italicized opening summary paragraph containing one sentence,
  or at most two short sentences, describing the conversation
- include the generated `**Participants:** ...` paragraph immediately after the
  opening summary; every source model ID for that note must appear there, with
  reasoning effort appended by hyphen when present
- use prose paragraphs; if section labels help, use optional bold
  paragraph-opening labels like `**Handoff State.**` rather than Markdown
  heading syntax
- use bullets only for compact lists of named results, rules, arms, or open
  questions
- keep Claude Code and Codex conversations separate even when their dates
  interleave

Style sketch:

```markdown
_This conversation covers the move from mixed transcript notes to
source-specific Claude/Codex conversation summaries, plus the tooling needed to
update them incrementally._

**Participants:** User, claude-sonnet-4-20250514, and gpt-5.5-xhigh.

**Pipeline Decisions.** The archive now treats Claude Code and Codex as separate
conversation streams. Each note records the ideas, decisions, results, caveats,
and handoff state that matter for future work, without trying to preserve every
exchange.

**Operational Rules.** Priority should remain visible when it affects what a
future agent should do next:

- keep source streams separate
- preserve blockers and required follow-up as project facts
- omit side logistics unless they changed repository workflow
```

## Full Rebuild Example

```bash
WORK=/tmp/valuegraft_transcript_work
mkdir -p "$WORK/filtered/claude-main" "$WORK/filtered/codex-main"

python3 scripts/transcripts/extract_claude.py \
  ~/.claude/projects/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370.jsonl \
  "$WORK/filtered/claude-main"

python3 scripts/transcripts/extract_codex.py \
  ~/.codex/sessions/2026/07/04/rollout-2026-07-04T22-07-20-019f3007-bab0-7e50-b019-2625d1538f63.jsonl \
  "$WORK/filtered/codex-main"

python3 scripts/transcripts/build_summary_shards.py \
  --filtered-root "$WORK/filtered" \
  --out-dir "$WORK/summary_shards"

python3 scripts/transcripts/summarize_shards.py \
  --shards-dir "$WORK/summary_shards" \
  --out-dir "$WORK/shard_summaries" \
  --prompt-dir "$WORK/prompts" \
  --command claude --print --model sonnet

python3 scripts/transcripts/combine_shard_summaries.py \
  "$WORK/shard_summaries" \
  "$WORK/summaries/mainline-full-conversation-summary-draft.md"
```

To split a combined summary into separate notes, creating one dated git commit
per generated segment:

```bash
python3 scripts/transcripts/split_summary_into_notes.py \
  --combined "$WORK/summaries/mainline-full-conversation-summary-draft.md" \
  --summary-shards-dir "$WORK/summary_shards" \
  --claude-jsonl ~/.claude/projects/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370.jsonl \
  --codex-jsonl ~/.codex/sessions/2026/07/04/rollout-2026-07-04T22-07-20-019f3007-bab0-7e50-b019-2625d1538f63.jsonl \
  --notes-dir notes
```

The note-generation scripts run `deno fmt` automatically on generated Markdown
files when Deno is available.

## Incremental Updates

After the initial notes exist, initialize a manifest once:

```bash
python3 scripts/transcripts/update_conversation_notes.py init-manifest \
  --summary-shards-dir "$WORK/summary_shards" \
  --notes-dir notes \
  --claude-jsonl ~/.claude/projects/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370.jsonl \
  --codex-jsonl ~/.codex/sessions/2026/07/04/rollout-2026-07-04T22-07-20-019f3007-bab0-7e50-b019-2625d1538f63.jsonl
```

Then future runs can update only uncovered transcript ranges:

```bash
python3 scripts/transcripts/update_conversation_notes.py update \
  --notes-dir notes \
  --claude-jsonl ~/.claude/projects/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370.jsonl \
  --codex-jsonl ~/.codex/sessions/2026/07/04/rollout-2026-07-04T22-07-20-019f3007-bab0-7e50-b019-2625d1538f63.jsonl \
  --work-dir "$WORK/incremental" \
  --command claude --print --model sonnet
```

If an already-covered conversation segment has continued, the updater writes a
revision prompt that includes the existing summary, the previously summarized
messages, and the new messages after the cutoff. If there are entirely new
segments, it writes new source-specific dated note files. Without `--command`,
it writes prompts only. Generated notes are checked for common secret-shaped
strings before being written.
