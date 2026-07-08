_This shard covers finalizing the notes archive naming convention (docs/ renamed
to notes/, stray files fixed, jlens_boundary_probe cleanup), building a full
transcript-extraction and summarization pipeline for Claude Code and Codex
mainline conversations, and iterating on an incremental updater with style/tone
guidance to keep future conversation-summary notes consistent._

**Notes archive cleanup and naming.** The UTC-based archival renamer
(`scripts/normalize_docs_archive_names.py`) was fixed for two bugs:
`git log --follow --reverse` unreliably hides the earliest commit for renamed
files, so the oldest-date lookup was rewritten to not depend on `--reverse`; and
untracked files with an already-valid timestamp prefix need to keep that prefix
rather than falling back to filesystem mtime (relevant right after a rename,
before commit). The archive directory was then renamed from `docs/` to `notes/`,
the script renamed to `scripts/normalize_notes_archive_names.py`, and
`docs/README.md` was replaced with a short `notes/AGENTS.md` aimed at agents
rather than humans. The normalizer was extended to catch `.txt` archive notes (a
stray `robust-f1-numbers.txt` had escaped `.md`-only matching) and given clearer
per-file logging (mode, scanned/planned counts, date source, reasons).
`jlens_boundary_probe/` was checked for note files (none found) and deleted
entirely, including untracked logs/pod-state/cache. A separately orphaned
`demos/pokemon-demo-conversation.md` (unreferenced anywhere in the repo) was
archived into `notes/` with its real git creation date. Root-level JSON files
were audited for references: `tune_configs.json` and `tune_rules.json` are
actively used by `src/run_guard_hf.py`, `src/serve_shim.py`,
`scripts/launch_pod.sh`, and `src/run_tune_hf.py`; most `.pod*_state.json` files
are gitignored runtime state with no direct references except the generic
`.pod_state.json` default and two files mentioned in `STATE.md`.

**Codex commit-attribution config.** Enabled Codex's experimental
`codex_git_commit` feature and set
`commit_attribution = "OpenAI GPT-5.5 <noreply@openai.com>"` as a top-level key
in `~/.codex/config.toml` (root keys must precede table headers in TOML),
verified via `codex --strict-config --version` and `codex features list`.

**Transcript summarization pipeline.** Built a full extraction and summarization
workflow, first prototyped in `/tmp/valuegraft_transcript_work` then promoted
into the repo under `scripts/transcripts/`. Key design decisions, several
corrected mid-stream by direct user instruction:

- Only mainline (non-subagent) conversation streams are extracted: the Claude
  Code root project JSONL under `~/.claude/projects/-Users-jeb-experimentation/`
  (excluding `subagents/`), and the canonical Codex main-thread rollout JSONL
  (excluding side/review/subagent threads that copied the same workspace
  metadata).
- Filtered output keeps only user and assistant messages — no tool
  calls/results, no system/developer records, no heartbeats or task
  notifications (an early filter pass leaked automation notifications as "user"
  text and had to be tightened).
- Assistant messages carry model provenance (e.g.
  `model=claude-fable-5`/`claude-opus-4-8` plus `claude_code_version`;
  `model=gpt-5.5`, `provider=openai`, `codex_cli=...`, `effort=...`) so a
  downstream summarizer knows which model produced each reply.
- No clock times or ISO timestamps are exposed to the summarizer or in final
  notes; dates are used only internally to group messages by UTC day and split
  on >1 hour gaps, and files are named with a per-day incrementing sequence, not
  embedded time-of-day.
- Large transcripts are sharded (~180k characters per shard, split only at
  message boundaries) and each shard is summarized independently, then combined.
- Summaries must use neutral, professional language: corrections, disagreements,
  and requirements are described as project facts, while still preserving
  genuine project priority/urgency framing (blocking status, required follow-up)
  rather than flattening everything to bland minutes.
- Side logistics are excluded from conversation notes unless they directly
  affect repository workflow; only the intended document style (e.g.
  "paper-style," "blog-style") should be retained if established.
- A default secret-shaped-string lint (AWS-style keys, OpenRouter keys, RunPod
  API keys, Hugging Face tokens) was added to the tooling as a repo-safety
  default, replacing an earlier, more content-specific forbidden-terms list; ad
  hoc `--forbid-regex` remains available.
- Final per-conversation notes should be
  `notes/<UTC-prefix>-<source>-conversation.md` (source = `claude` or `codex`,
  not mixed), containing no top-level title/header, starting directly with an
  italicized one-to-two-sentence capsule sentence, then short titled prose
  sections; bullets reserved for compact lists of named results/rules, not one
  bullet per exchange.
- Claude and Codex streams are treated as separate conversations by design
  (previously some shards mixed both apps in one file, packed by date/sequence)
  — this was identified as a correction to redo, since interleaving separate
  applications is confusing. The pipeline's default was changed to isolate
  sources per shard/note, with a legacy flag retained for the old packed
  behavior.
- Sequential per-source-stream summarization should carry forward a short
  prior-context block (derived from the previous note's opening capsule) to help
  interpretation without merging threads or rewriting prior notes.
- A manifest (`scripts/transcripts/conversation-summary-manifest.json`) records,
  per generated note, the source stream, covered message range, input hash,
  output filename, and commit hash, enabling incremental updates: re-extract
  everything, diff against the manifest's last-covered position per stream, and
  only summarize new ranges — except when a source conversation already covered
  by an existing note continues, in which case the updater regenerates that note
  from old-summary + old-messages + new-messages together (uncommon case),
  grouping multiple advanced streams per note into one revision prompt rather
  than clobbering the same prompt path.

**Iteration and known-good state at shard boundary.** An initial
incremental-update dry run correctly detected two notes needing revision (their
source conversations continued) plus one new note to append. Generated summaries
exposed style and source-mixing problems, so the prompt and pipeline were
iterated before committing generated notes. The source split, filename
convention, rolling context, and style guidance were patched into the scripts. A
sample run (large Claude shard, small Codex shard) landed closer to target:
italic capsule present, prose-first sections, and priority preserved as
rules/results. At the point this shard ends, the plan in progress is to
regenerate the entire source-split conversation archive from raw transcripts
into a temp directory using the updated prompt, validate it thoroughly, and only
then replace the existing tracked `notes/*-conversation.md` files — this full
regeneration and replacement had not yet been executed or committed.
