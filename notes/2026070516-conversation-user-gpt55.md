_This conversation covers a follow-up read-only review of an actively running
LongMemEval benchmark job, plus a discussion (culminating in a committed notes
document) about the practical deployment implications of retaining summary-KV
state, and a methodological discussion of whether observed "reduced fabrication"
reflects genuine calibrated honesty versus generic refusal._

**Participants:** User and gpt-5.5-xhigh.

**Review findings (read-only pass, GPT-5.5/xhigh, Codex CLI).** A LongMemEval
smoke-test run (`src/run_longmemeval.py`) was actively in progress in another
process throughout the review; the reviewer made no changes and treated the
growing set of untracked `results/longmemeval_4b/*.json` files as in-flight
output. Several issues were identified but not fixed: (1) the script writes
final JSON directly without an atomic temp-file/rename pattern, so an
interrupted run can leave a corrupt file that later runs will skip without
validation — a real risk given this job's tendency to run long and be
harness-fragile; (2) the LongMemEval path records raw answers but has no
scoring/judging layer or "did A actually answer" gate — one sample's answer
("University of Melbourne") didn't contain the full gold string ("University of
Melbourne in Australia"), so exact-string matching will misclassify semantically
correct answers, and arm comparisons should not be interpreted until an
A-success filter or semantic judge is added; (3) the run was launched as a plain
foreground `uv run python -u ...` process rather than detached with
nohup/PID/log per the repo's long-job convention, increasing risk if the shell
is interrupted; (4) the local Hugging Face cache path is hard-coded, which will
break portability; (5) STATE.md was stale, claiming all work complete and GPU
idle while the run was active — flagged as needing a cleanup pass before any
handoff. On the positive side, the H-pack machinery (re-rotation ladder, HP-0
identity check) and reframed Phase 2 results were assessed as well-aligned with
the project's mitigation-focused pivot, and the two previously committed
LongMemEval smoke samples showed the expected qualitative pattern (full-context
arm retrieves correctly, compacted/packed arms mostly abstain).

**Deployment-overhead discussion and committed notes.** The user raised, as a
topic to potentially fold into a future writeup, how much overhead retaining
summary-KV state would add if this approach were deployed against
normally-stateless LLM APIs, proposing that only the KV for the summary portion
(not full history) would need to be retained/recomputed. The assistant
quantified this for Qwen3-30B-A3B: roughly 96 KiB of fp16 KV per token, so a
300–700 token summary sidecar would run ~28–66 MiB, versus ~1.1 GiB for a full
12K-token cache — much smaller than full-cache retention but still orders of
magnitude larger than the summary text itself (a few KB), so it is not a cheap
payload for a literal stateless client-to-server transport. The user noted this
still seems expensive relative to plain text and suggested a smarter mechanism,
drawing an analogy to existing opaque state patterns in current LLM APIs (e.g.,
opaque "thinking" tokens). The assistant developed this into a concrete framing:
rather than clients shipping raw KV, providers could expose an explicit
compaction primitive —
`compact_from(boundary, instructions) -> {summary_text, compaction_token}` —
where the opaque token is stored/resolved server-side and inserted into future
requests alongside the visible summary text and recent tail, analogous to
prompt-cache handles or session continuation IDs. The user asked for this
captured in a document without being prescriptive ("here's a way we might think
about it" framing, not a design mandate), and the assistant wrote and committed
`opaque-compaction-handle-notes.md` (commit `f9473c9`), then expanded it with
the explicit `compact_from(...)` harness/API shape per the user's follow-up
(commit `79a0423`). Both commits touched only that single notes file; the
in-flight LongMemEval result files were explicitly left untracked/untouched
throughout. It was clarified and agreed that the current experiment
implementation is purely single-process, in-memory KV-array manipulation (load
model, build full cache, snapshot arrays, construct arms, evaluate, write only
JSON scalars/text results) — it does not implement any real
serialization/handle/transport layer, and building that transport layer is
considered an out-of-scope implementation detail following established patterns
(prompt caching, session handles), not a research question the project needs to
study further.

**Methodological question: honesty vs. refusal.** A separate thread examined
whether the project's existing results distinguish genuine calibrated honesty
(abstaining because the model correctly senses information is absent) from
generic refusal/caution (abstaining more broadly regardless of whether the
answer was actually available). The assessment is that current experiments only
partially distinguish these: evicted-fact and decoy probes show H-pack/H-gap
correctly abstaining when they should, which is good evidence of honesty in the
negative direction, but referent/sense probes show H-pack failing to preserve
recall even where it should be able to answer, and other packed-summary-only
configurations (B-min-pack, H-pack-wrongS) also become uniformly cautious —
suggesting at least part of the abstention effect may be generic caution induced
by the compacted-context shape rather than calibrated knowledge-sensitivity. To
properly separate the two, the discussion proposes scoring a calibration matrix
across item types (LongMemEval-answerable-from-summary, answerable-from-tail,
evicted-but-absent-from-summary, never-discussed decoys), with the target being
accuracy maintained/improved on answerable items combined with reduced
fabrication on unanswerable items — not just an overall increase in "I don't
know" responses. The LongMemEval-style benchmark currently running is identified
as directly useful for this: it supplies the missing "should answer and does the
model still answer correctly" half that the decoy/evicted-fact probes alone
don't test, and early live samples already show the qualitatively expected split
(full-context arm answers correctly; B and H-pack arms report inability to
answer), which the project should analyze specifically for whether H-pack ever
outperforms B on genuinely answerable items — that would be the discriminating
signal between calibrated honesty and generic refusal.

**Handoff state.** As of this chunk: the LongMemEval run was still actively
producing output (process still live, growing file count) and was left
untouched; the notes document `opaque-compaction-handle-notes.md` is committed
in two stages; STATE.md is known to be stale and needs updating once the run
completes; the A-success/judging gate for LongMemEval scoring has not yet been
implemented and remains a prerequisite before trusting arm-comparison
conclusions from that benchmark.
