_This message is the user's request for a workflow to make future
conversation-note generation incremental rather than reprocessing the full
transcript archive each time, and this is the assistant's design response
outlining that approach._

**Participants:** User and gpt-5.5-xhigh.

The user asked for thoughts only, with no action taken in this exchange, on how
to prevent the transcript-summary pipeline (built earlier for extracting,
sharding, and summarizing mainline conversations from the documents directory)
from re-summarizing conversation ranges that have already been captured in
existing notes.

The proposed approach is to make the pipeline incremental and range-aware rather
than reprocessing all history each run. The core mechanism is a small manifest
file (suggested path: `notes/conversation-summary-manifest.json`), tracked in
the repo, recording per generated note: source stream identifier (Claude
mainline session id or Codex thread id), start and end message keys (date plus
source line or message index), a normalized input hash for that shard, the
output note filename, and the commit hash that introduced it.

The proposed update workflow: re-extract all raw mainline messages, sort into
the same canonical merged order used previously, load the manifest, determine
the latest covered message per source stream (or latest covered merged
position), and build new summary shards only from messages after that boundary.
An optional small overlap window may be included for context but should be
marked context-only and not resummarized as new content. Only new
`*-conversation.md` files should be written, dated from their first new message.

A stated principle: filenames alone are an insufficient resume checkpoint for
archive ordering; the durable checkpoint should be source-message ranges plus
content hashes, not just dates. Old shard summaries should not be edited as part
of routine "update conversations" runs — if an existing summary has a style
problem, it should be patched directly and separately; ordinary updates should
only append new summary files for new transcript ranges.

No repository changes were made in this exchange; this is a design proposal
pending implementation.
