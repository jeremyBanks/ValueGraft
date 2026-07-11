_This conversation established an incremental, range-aware workflow for updating
conversation summaries without reprocessing already archived material._

**Participants:** User and gpt-5.5-xhigh.

**Handoff State.** The proposed durable checkpoint is a tracked manifest such as
`notes/conversation-summary-manifest.json`. Each generated note should record
its source stream and identifier, covered start/end message keys, normalized
input hash, output filename, and introducing commit hash. Future runs should
re-extract raw mainline messages, reproduce the canonical merged ordering,
compare against manifest coverage, and generate shards only for messages beyond
the latest covered boundary.

A small overlap window may be included for context, but overlapping messages
must be marked context-only and excluded from newly summarized content.
Filenames alone are insufficient as resume checkpoints; source-message ranges
plus hashes are required for reliable incremental processing. Normal updates
should append new `*-conversation.md` files using synthetic dates based on the
first new message, while existing shards remain unchanged unless an explicit
request targets a correction or style issue.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
