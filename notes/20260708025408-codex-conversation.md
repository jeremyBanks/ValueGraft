_This shard covers completing and hardening the transcript-summarization
pipeline: separating Claude Code and Codex conversation notes by source,
building an incremental manifest-based updater with automatic revision of
continued conversations, iterating on generation prompts and validation until
output required minimal manual cleanup, executing a full source-split
regeneration of the notes archive, and making the updater a single
default-argument command._

**Incremental update design.** To avoid resummarizing already-covered
conversation history on every run, the pipeline uses a tracked manifest
(`scripts/transcripts/conversation-summary-manifest.json`) recording, per note,
the source stream, covered message range, input hash, output filename, and
commit hash. Updates re-extract all mainline messages, sort them into canonical
order, and compare against the manifest to find uncovered ranges. New ranges
become new notes. When a source conversation continues past a note's previously
recorded end, the updater passes the existing summary, the old covered messages,
and the new messages to the summarizer with an explicit boundary marker, and
asks for an updated version that extends rather than replaces the note. An edge
case where a single note covers multiple source streams and more than one of
those streams advances was found and fixed so continuations are grouped per note
into one revision prompt rather than overwriting the same prompt path
repeatedly.

**Source separation and filenames.** Claude Code and Codex conversations were
judged confusing when interleaved in a single note, so the pipeline's default
was changed to always split by source application, even though the manifest
format itself can support multi-source notes. Filenames became
`notes/<UTC-prefix>-claude-conversation.md` and
`notes/<UTC-prefix>-codex-conversation.md` (never a bare `-conversation.md`),
applied both to the initial batch sharding and to incremental updates, with a
legacy flag retained for the old packed behavior. Sequential per-source-stream
summarization carries forward a short prior-context block derived from the
previous note's opening summary paragraph, to aid interpretation without merging
threads.

**Style consistency pass.** Existing notes were inconsistent — some opened with
a title, some used dense bullet ledgers, others were prose. The required target
style was clarified: no top-level header, an opening italicized summary
paragraph (a paragraph, not a single physical line — wrapping is fine;
over-shortening these paragraphs to force one visual line was tried and
explicitly rejected as incorrect), then short titled prose sections, with
bullets reserved for compact lists of named items rather than one bullet per
exchange. Priority/urgency from the original conversation should still be
captured, but expressed as project priority, blocking status, or required
follow-up rather than as participant mood — the tone rule is not meant to
flatten emphasis into blandness. A default secret-shaped-string lint (AWS-style
keys, OpenRouter keys, RunPod keys, Hugging Face tokens) was kept as a generic
repository-safety default rather than a content-specific forbidden-terms list. A
brief, concrete style example was added to both the transcript README and
`notes/AGENTS.md` so future generation has a style target without becoming a
rigid template. One internal naming choice used briefly in the tooling
("capsule" for the opening paragraph) was judged an unnecessary term and
replaced with plain wording ("opening summary paragraph"); this was a minor
terminology fix, not a substantive change to the style rule, and did not warrant
the exhaustive cleanup pass it initially received.

**Generation iteration and validation.** Reaching output that needed minimal
manual cleanup took several rounds of prompt tightening, discovered by
generating samples and reading them rather than trusting the summarizer's
compliance. The durable prompt guidance is to return only the note body, keep
the opening-summary-plus-sections structure, preserve priority as project
priority or required follow-up, omit side logistics unless they changed
repository workflow, and avoid retelling the tooling's own cleanup process
inside the archive notes. After each tightening, only the shards that had failed
validation were regenerated, using the existing manifest and rolling context to
avoid redundant work.

**Automation and defaults.** The updater script was set up so that running it
with no subcommand performs the standard incremental update using known local
transcript paths, `notes/`, the manifest, the default summarizer, and automatic
`deno fmt` formatting of generated/updated notes before hashing and commit — so
a future agent only needs to run one script with no required arguments for the
common case, with explicit flags available for non-default runs.
`notes/AGENTS.md` and the transcripts README were updated to lead with this
single-command path rather than the full manual pipeline, and to include a brief
concrete style example so a summarizing agent has something to steer toward
beyond abstract rules. A minor CLI ordering bug in this convenience path was
found and fixed before the default path was relied on.

**Full regeneration.** With the split-source, incremental, and style tooling in
place, the entire notes archive was rebuilt from raw transcripts into a
temporary directory: 17 Claude segments and 23 Codex segments packed into 16
source-separated summary shards, run sequentially (single-threaded, to keep
rolling per-source context meaningful) through the summarizer CLI. The run took
several minutes per large shard; progress was tracked by polling output file
counts rather than assuming silence meant failure. Two further
validation-and-regeneration rounds were needed after the initial full run, each
time regenerating only the shards that failed the style/content checks rather
than the whole set. The resulting notes (10 Claude, 6 Codex) were validated
against the style/secret lint before being copied over the old mixed-source
conversation notes in the repository, with the manifest rebuilt against final
repo paths and hashes. One incidental issue was caught and reverted: running
`deno fmt` broadly against `notes/*-conversation.md` also reformatted an
unrelated non-transcript note (the archived Pokemon demo), which was restored to
its prior formatting since it is not a transcript-summary shard.

**Post-regeneration currency check.** After the full regeneration, the archive
was current only as of the rebuild's transcript snapshot. A prompt-only check
subsequently showed two conversations (the latest Claude and Codex notes) had
continued past that snapshot; the updater was run for real, producing revised
notes for both and refreshing the manifest from the updater's current state. One
intermediate manifest refresh mistakenly used the initial shard ranges, which
would have rolled recorded coverage backward; this was caught before commit and
corrected by rerunning the updater rather than reinitializing from scratch.

Unrelated local source-file edits (e.g., `src/arms_common.py`,
`src/cross_arch_probe.py`, `scripts/launch_pod.sh`, `scripts/preflight*`,
`CLAUDE.md`, `results/cross_arch_wide/`) were repeatedly noted as out of scope
and left untouched throughout this work.
