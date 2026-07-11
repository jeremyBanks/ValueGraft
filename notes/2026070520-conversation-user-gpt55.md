_The conversation reviewed the active LongMemEval/Phase 2 work and added a
practical, non-prescriptive deployment framing for opaque compaction state. It
also clarified that current experiments show reduced fabrication but do not yet
distinguish calibrated honesty from generalized refusal._

**Participants:** User and gpt-5.5-xhigh.

**Handoff State.** The repository experiment remains a single-process, in-memory
cache-surgery study: full history and KV cache are available during
summarization, experimental arms are built from in-memory snapshots, and only
textual results are saved. It does not yet serialize KV state, persist it across
requests, or implement an API protocol. H-pack’s re-rotation ladder and HP-0
identity check were considered methodologically appropriate for the mitigation
pivot.

At review time, `src/run_longmemeval.py` was actively running under `uv`,
producing roughly one JSON output per minute; seven outputs existed, five were
new and untracked. The run was left untouched. Identified follow-up issues were
atomic temporary-file writes with validation before skipping existing outputs,
an A-success filter or semantic judge before comparing arms, a detached long-job
workflow, configurable Hugging Face cache paths, and a STATE cleanup to reflect
active work. One A answer demonstrated that exact-string matching is
insufficient: it gave the correct entity without the full gold wording.

The deployment discussion concluded that raw KV sidecars are too expensive for
ordinary stateless requests. For Qwen3-30B-A3B, fp16 KV was estimated at
approximately 96 KiB/token, making a 500-token summary about 47 MiB versus
roughly 1.1 GiB for a 12K-token cache. The practical framing is provider- or
session-side state behind an opaque handle, analogous to prompt-cache or
continuation artifacts, rather than client transport of raw tensors. Relevant
caveats include model/tokenizer/template binding, privacy and revocation,
expiration, non-portability, auditability, and possible future quantization or
selective retention.

A narrow document, `opaque-compaction-handle-notes.md`, was created and
committed as `f9473c9`, then expanded and recommitted as `79a0423`. Its intended
form is a lightweight deployment note, not a new research branch or prescriptive
product specification. The proposed conceptual interface is
`compact_from(boundary, instructions) -> {summary_text, compaction_token}`; the
harness replaces old history with visible summary text, an opaque token at the
compaction boundary, and the recent tail, with summary-only fallback when
unsupported. The active LongMemEval outputs remained untracked and uncommitted.

**Methodology Conclusion.** Existing probes establish fewer confident invented
specifics on unknowable or evicted facts, but that may reflect either improved
epistemic calibration or generic caution. B-min-pack and H-pack-wrongS becoming
highly abstention-prone, together with referent/sense probes showing weak
recall, suggest H-pack currently demonstrates abstention more clearly than
preserved memory. LongMemEval is the needed complementary test because it
contains answerable questions with gold answers: compare full-context A,
text-compacted B, and H-pack on answerability as well as abstention.

The required evaluation target is a calibration matrix: retain accuracy on
answerable LongMemEval and tail-visible items while abstaining on evicted facts
and never-discussed decoys. “More I don’t know” alone is not sufficient evidence
of better honesty; the next interpretation should depend on whether H-pack
answers answerable items better than B without restoring fabrication.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
