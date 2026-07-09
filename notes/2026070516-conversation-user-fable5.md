_This chunk covers a compute-hardware recovery, an autonomous continuation of
the compaction-mitigation research using LongMemEval as a standard held-out
dataset, a quantified overhead analysis for turning the SelfGist/ValueGraft
methods into a stateless-API-compatible design, and the start of a cloud-rental
plan requested for the user's return._

**Participants:** User and claude-fable-5.

**Continuation gap, defined.** Per user request, the metric was clarified: hold
out a genuine next reply in a conversation, teacher-force it, and measure mean
per-token log-probability under (A) the full uncompacted conversation vs. (B)
the standard compacted version. The gap is A−B; an intervention "closing 24% of
the gap" moves 24% of the way from B back toward A. This is a
fluency/coherence-flavored measure, intentionally weighted below behavioral
(fabrication) probes in the writeup's hierarchy of evidence.

**New work directive.** The user, being away for the day, asked the agent to (a)
find a better/standard dataset and continue evaluation using it, given slow
local compute means only a modest sample is feasible until server rental
happens, and (b) produce, within roughly three to four hours, a
beginner-oriented plan for renting cloud GPU hardware (assuming AWS-level
unfamiliarity, possibly via SSH or an API key handed to the agent), with
explicit spend guardrails given a prior overspend incident. The agent was
authorized to use its own judgment on research direction for the rest of the
day.

**Dataset and execution.** LongMemEval was selected as the standard dataset
(real multi-session chat histories with answers in earlier, now-evicted
sessions), building ≤16K-token instances. Arms tested: A/B baselines,
SelfGist/H-pack, B-min-pack, tuned ValueGraft. Plan: 4B batch (~48 questions) →
Sonnet-judged (correct/fabricated/admitted) → 30B batch → judge → results doc
and blog-draft update → SWE-Gym trajectories as a stretch goal. The 4B batch
completed cleanly (48 + 2 smoke test) after a mid-run crash from integer-typed
gold answers was fixed and the batch resumed without redoing completed
questions. Both 4B judge passes completed and a benchmark table was committed.

The 30B batch ran into escalating per-question latency (191→380s), diagnosed as
memory pressure; it was relaunched with a memory fix and reduced event-batching
noise (progressing 2min→3min→5min intervals over the session per the operator's
quiet-monitor convention). Contingency set: if per-question time stabilizes near
~2.5 min, the full 48 questions finish by early afternoon; if it stays above ~4
min, the batch will be capped at ~24–30 questions, judged sufficient for the
fabrication contrast. At the end of this chunk, the 30B batch was alive
post-restart with memory still recovering (34% free), and the agent noted it
would run no further local model work (including CPU-side experiments) while the
30B batch was in progress.

**Stateless-API overhead quantification.** Responding to a user question about
practical deployment overhead (since the method breaks API statelessness by
requiring a live KV cache), the agent computed: per-token fp16 KV cache size ≈96
KB/token at 30B-A3B (48 layers × 2 × 4 kv-heads × 128 dims × 2 bytes) and ≈144
KB/token at 4B; a SelfGist summary blob ≈8 MB at 30B for a terse ~80-token
summary plus 4 sink tokens (56 MB for a 600-token summary), with 8-bit/4-bit KV
quantization projected at ~4 MB/~2 MB but untested since runs deliberately used
fp16. For reference, the summary as plain text is ~0.5 KB (blob ~15,000× larger
than the text it replaces), while the full 12K-token context's KV is ~1.2 GB, of
which only ~0.7% would be shipped. ValueGraft, by contrast, requires the entire
original cache (~1 GB) resident at compaction time and is inherently a
server-side/local-only technique, never a request-attachable blob. Constraints
noted: the blob is valid only for an exact model/revision/tokenizer, and
accepting client-supplied KV introduces a new trust surface (injected KV as
unauditable soft prompts), suggesting providers would more likely implement this
as signed, expiring portable prompt-cache entries rather than raw blob upload.

**Framing note adopted.** The user supplied framing notes
(opaque-compaction-handle-notes.md) proposing a "handle" abstraction and split
cost accounting (client bandwidth ≈ nil; provider storage in tens of MiB with
cache-style expiry; one-time packing compute). This was adopted into the draft's
overhead section, preserving the user's caveats on auditability and
version-specificity.

**Outstanding handoff item.** The cloud-rental plan (for AWS-unfamiliar GPU
rental with spend guardrails) was committed to be delivered before the user's
return, expected in roughly three to four hours from message 050; as of this
chunk's end, the plan was reported as being written but not yet confirmed
complete, and the 30B LongMemEval batch was still in progress with its final
scope (n=48 vs. capped ~24–30) undetermined pending latency stabilization.
