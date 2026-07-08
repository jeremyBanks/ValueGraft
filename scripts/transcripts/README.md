# Transcript Summary Pipeline

These scripts reproduce the mainline conversation-summary workflow used for the
`notes/*-conversation.md` files.

The pipeline is intentionally plain:

1. Extract mainline Claude Code and Codex messages from local JSONL stores.
2. Build contiguous transcript shards sized for a summarizer.
3. Summarize each shard with a CLI model.
4. Combine shard summaries if desired.
5. Split a combined summary back into dated `notes/` files, with optional git
   commits whose author/committer dates match each file prefix.

The summaries should focus on ideas, decisions, methodology, results, caveats,
and handoff state. If a discussion established an intended writing form, such as
paper-style, blog-style, article-style, or report-style, capture that. Avoid
side logistics unless they directly affect repository workflow.

## Example

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

To split a combined summary into separate notes and create one dated git commit
per shard:

```bash
python3 scripts/transcripts/split_summary_into_notes.py \
  --combined "$WORK/summaries/mainline-full-conversation-summary-draft.md" \
  --summary-shards-dir "$WORK/summary_shards" \
  --claude-jsonl ~/.claude/projects/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370.jsonl \
  --codex-jsonl ~/.codex/sessions/2026/07/04/rollout-2026-07-04T22-07-20-019f3007-bab0-7e50-b019-2625d1538f63.jsonl \
  --notes-dir notes \
  --commit
```

Run `deno fmt notes/*-conversation.md` after generating notes.
