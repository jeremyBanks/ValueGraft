_This conversation covers building and refining a documentation/notes-archiving
system for the repository, plus an interpretability side-investigation into
whether a model's internal representations of a politically sensitive historical
topic diverge from its generated output._

**Participants:** User and gpt-5.5-xhigh.

The interpretability thread began from an observation using a newer probing tool
(J-lens): for a Chinese-language prompt referencing a public square, the model's
internal readout at the final prompt token surfaces event/protest-related
concepts in late layers, even though the generated answer redirects toward
reform and development themes. Case-specific continuation scoring supported
this: the reform/development redirect scored highest, direct factual
continuation lower, and refusal much lower — indicating the behavior is
redirection rather than refusal or non-recognition. To test whether this
required the newer tool, a comparison was built against raw logit lens
(unembedding applied directly to residual stream activations at the same token
positions, with and without final normalization) plus concept-category scoring
and English/Chinese internal-similarity analysis (residual cosine and
lens-transported cosine). Two rounds of read-only subagent critique were used to
sharpen the evaluation criteria before running the job, converging on the
standard that the new tool should only be credited with added insight if it
captures something behavior/logprob/raw-lens baselines miss or blur. Methodology
decisions: same token positions across methods, matched controls,
layer-robustness checks, category scores rather than anecdotal top-k lists, and
cautious language avoiding claims of mechanism, intent, or identical
representation. After running on a cloud GPU pod (queued behind another job,
then executed once free), the result was more nuanced than the initial framing
suggested: raw logit lens also recovers much of the same signal (event-related
tokens surfacing in later layers at the same position), so the newer tool is not
uniquely necessary — it mainly produces a cleaner, more semantically expanded,
more easily interpreted readout, and adds a useful transported-space similarity
view. English/Chinese same-topic representations were found to be more similar
to each other than to most controls, especially in transported space, but this
was explicitly written up without claiming identical representation. The
corrected framing (raw lens sees it too; the new tool improves clarity, not
access) was written into the probe's observations/notes and the generated
comparison report, then committed and pushed. The pod was later confirmed idle
and terminated after verification.

Separately, and taking up most of the work in this stretch, a durable
documentation/archiving convention was established and iterated repeatedly. The
end state: a repository-root `notes/` directory (renamed from an initial
`docs/`) holds standalone Markdown files named with a UTC
`YYYYMMDDHHMMSS-kebab-case-title.md` prefix, derived from each file's earliest
git history (tracing renames) or filesystem creation time for new/untracked
files, with numeric suffixes for same-second collisions. A Python normalizer
script (`scripts/normalize_notes_archive_names.py`) enforces and can apply this
convention, treating an already-correctly-prefixed but untracked filename as
authoritative when git history isn't yet available, and is idempotent; it was
extended to also catch and convert markdown-like `.txt` notes. A brief
`notes/AGENTS.md` (replacing an initial `README.md`, since this is agent-facing
process guidance) documents the rule and reserves itself from renaming. Several
standalone content folders were swept into this convention and then deleted once
their Markdown was extracted: an interpretability probe folder, a
synthesis/paper-working folder, and an unreferenced demo transcript; folders
with only code/data and no note files were deleted outright after confirming no
note files existed. Root-level JSON files were also audited for references (two
config files are actively used by code; most `.pod*_state.json` files are
ignorant lifecycle state referenced by nothing beyond one default path).

A large effort went into building a mainline conversation-transcript
summarization pipeline, now checked into `scripts/transcripts/`. It extracts
only top-level user/assistant messages (explicitly excluding subagent, tool,
system, and developer records) from the Claude Code project JSONL store and the
canonical Codex session JSONL, tags each assistant message with its
model/provider/runtime version, groups messages by date with per-day sequence
numbers, and splits into new segments on day boundaries or gaps exceeding one
hour — but strips all visible timestamps and the word describing that concept
from anything a summarizing model would read, using internal dates only for
ordering. Large filtered transcripts are chunked into shards sized for practical
processing, each shard is summarized independently (originally intended for one
lightweight model per shard, executed instead via available subagent workers run
in parallel waves due to a concurrency cap), and shard summaries are combined
into a draft. Instructions to summarizers emphasize capturing ideas, decisions,
corrections, and open questions at an appropriate level of detail, without
verbatim transcription, without unnecessary elaboration, and with only necessary
key terms/structures preserved — that is the general form now encoded in the
tooling.

Other operational items: a Codex CLI setting (`commit_attribution` under
`[features].codex_git_commit` in `~/.codex/config.toml`) was enabled to
auto-append a co-author trailer to agent-generated commits. Two unrelated
source-code files (`src/arms_common.py`, `src/cross_arch_probe.py`) remained
locally modified throughout this stretch as part of an active separate
experiment stream and were consistently left untouched by all the
archiving/documentation commits described above.

**Handoff state.** All notes/tooling work is committed and pushed to `trunk`.
The cross-architecture generalization experiment is the active parallel frontier
(not detailed in this chunk beyond noting its untouched dirty files) — a prior
chunk recorded that Qwen2.5-32B looked negative, Mistral was null/underpowered,
Gemma errored, and a Qwen3-30B fixed-summary positive control had failed, making
a Qwen3-30B self-generated-summary run the pending decisive test; a live pod was
reported running that job as of the last refresh in this chunk, with a second
pod idle after writing the Mistral result. A refresh-and-push was requested and
initiated at the end of this chunk but not yet confirmed complete.
