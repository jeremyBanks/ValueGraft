_This chunk covers a workflow discussion (no repository changes made) about how
to make the transcript-summary/archiving pipeline incremental so that future
updates only process new conversation content rather than re-summarizing
material already captured._

**Participants:** User and gpt-5.5-xhigh.

The user raised a concern about the existing conversation-archiving system:
because past summarization work appears to scan back through the full documents
directory, there was a risk that running an update in the future could redo work
on conversations that had already been captured and archived, rather than only
picking up new material. The user asked explicitly for options and reasoning
only, with no changes to be made in this session.

The assistant's recommendation was to make the summarization pipeline
incremental and range-aware rather than re-running over the entire history each
time. The proposed mechanism is a small tracked manifest file (suggested path:
`notes/conversation-summary-manifest.json`) recording, per generated
conversation note: the source stream identifier (mainline session id or Codex
thread id), the start and end message keys (date plus source line or message
index) covered by that note, a normalized input hash for the covered shard, the
resulting output note filename, and the commit hash that introduced it.

The proposed future-update procedure: re-extract all raw mainline messages, sort
them into the same canonical merged chronological order used previously, load
the manifest, determine the latest already-covered message per source stream (or
latest covered merged position), and generate new summary shards only from
messages after that boundary. A small overlap window could be included for
context when generating a new shard, but explicitly marked as context-only and
not re-summarized as new content. New shard files should use synthetic dates
derived from their first new message, consistent with the existing archive
naming convention.

A related rule proposed: filenames alone are not a reliable resume checkpoint
for this pipeline (they're useful for archive ordering but insufficient for
determining what's already been summarized) — the durable checkpoint should be
based on source-message ranges and content hashes tracked in the manifest, not
on file naming. The assistant also recommended that ordinary "update
conversations" runs should never edit or resummarize existing shard notes; if an
old summary needs a style fix, that should be a direct, explicit patch to that
specific note rather than something triggered by a routine update pass.

No manifest file or pipeline code was created in this session — this was purely
a design discussion. A follow-up task, if the user proceeds, would be to
actually implement the manifest file and wire the incremental-range logic into
whatever script currently generates the conversation summary notes.
