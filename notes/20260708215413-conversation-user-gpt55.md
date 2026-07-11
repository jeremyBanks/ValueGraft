_This conversation finalized the archive’s two-hour coalescing regeneration and
added durable timeline-preservation guidance, then began designing a six-hour
maximum conversation policy with an in-script dry-run mode._

**Participants:** User and gpt-5.5-xhigh.

**Handoff State.** The archive was regenerated using a maximum two-hour
inter-segment coalescing gap, producing 22 notes (8 Claude, 14 Codex), and the
result was normalized, validated, committed, and pushed to `trunk`. Validation
found 22 matching manifest records, no forbidden-term matches, no Markdown
heading lines, and passing naming tests. Root `REPORT.md` was moved into
`notes/` with required archive renames in one commit. The transcript normalizer
was subsequently made manifest-aware so note renames automatically update
manifest paths; 17 focused tests passed, and that fix was pushed. An unrelated
dirty Qwen result JSON remains intentionally untouched.

Summarization guidance now requires preserving explicit ETAs, deadlines,
durations, recurring check-back intervals, expected completion windows, and
later revisions or cancellations as neutral handoff facts, without inventing
precision. Generated summaries should use optional bold paragraph-opening labels
rather than inconsistent Markdown heading levels.

The current requested policy is: retain ordinary gap-based splitting, but ensure
a conversation note spans no more than six hours; if a split is required, select
the largest message-to-message gap whose split point falls between four and five
hours after the note start. The user requested a real script dry run before any
regeneration. Implementation is underway in
`scripts/transcripts/update_conversation_notes.py`: planning is now
message-level, so long raw transcript segments can be split internally; explicit
chunk/split structures explain decisions; `--dry-run` loads manifests and
transcripts, reports affected existing notes and uncovered ranges, and exits
before prompts, summaries, writes, or commits. Six-hour maximum and preferred
four-to-five-hour split-window options are threaded through the real update
path, with tests covering preferred-window and fallback-before-maximum behavior.

The first focused test fixture accidentally placed a large gap exactly at the
configured inclusive boundary, making the preferred-gap selection ambiguous; the
fixture was adjusted. The next required action is to rerun the focused tests and
the script’s dry run, inspect which notes would actually split and the resulting
count, then report findings before deciding whether to modify or regenerate the
archive. No six-hour regeneration has yet been authorized or completed.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
