_This conversation is a strategy and code-review session (Codex CLI, GPT-5.5,
effort xhigh) conducted read-only in parallel with another agent actively
working in the same repository, culminating in a committed notes file capturing
an adaptive-blending design idea._

**Participants:** User and gpt-5.5-xhigh.

The session opened with a read-only review of commits landed since the prior
reframing memo, covering the 4B pilot completion and a partially-landed 30B
targeted run. Code review focused on `src/score.py`, which drives both standard
and brief-condition scoring and keys judge verdicts by condition. Sanity checks
confirmed 2,520 score rows and 2,089 judge verdicts with no missing entries, and
the headline numbers in RESULTS.md were verified to reconcile with
`results/scores.json`, including the mitigation-null table, the clean
evicted-fact control, negative-control cratering, and the H-gap honesty shift.
No mechanical inconsistencies were found in the reported 4B numbers.

Four review findings were recorded, none requiring immediate action but flagged
for the active working agent: (1) `score.py local`'s fallback path
(src/score.py:157) still combines judge verdicts with `kw_pass`, while the
external-verdict path (src/score.py:268) intentionally trusts the judge alone to
avoid over-penalizing quoted-but-compliant answers — the two paths now diverge,
so the local fallback would not reproduce committed `results/scores.json`
values; (2) STATE.md (line 21) is stale, still describing the final 30B model as
downloaded-but-unused despite `results/raw_30b/c01–c04.json` existing and the
30B ladder having reportedly gone green with a targeted run launched, and its
top-of-file date does not match body state; (3) DECISIONS.md has not been
updated for the 4B final scoring interpretation, the Sonnet-judge change,
B-causal repair completion, or 30B ladder facts, leaving methodology/runtime
facts split across commits, STATE.md, RESULTS.md, and writeup-guidelines.md; (4)
`src/analyze.py` (line 87) added B-causal, B-min, and H-gap to aggregate tables
but the per-conversation appendix still only lists C, D, E-post-a1.0, and
E-inter-a1.0, omitting the now-central H-gap/B-min contrast and E-post-a0.25.

A substantive design discussion followed on the experiment's framing. The
assessment: the pilot is scientifically careful and already informative, with
strong controls (L0–L4 ladder, B-causal, wrong-conversation/shuffled grafts,
leakage stratification, brief-summary condition, logged external judging), but
arm C is too confounded (old keys, gapped positions, different causal order, OOD
cache geometry) to carry the central claim. The key correction established in
this conversation: the original design over-centered the mechanism question
"does write-time KV state contain context-conditioned information?" when the
more valuable question is "can a small, practical cache-state intervention
reduce compaction damage in a deployable way?" This reframing was validated by
the user as correcting a genuine misfocus in the original design, not just a
rephrasing.

Given this reframing, a two-phase handoff instruction was drafted for the
working agent (after two earlier drafts were rejected for still centering the
old design as default): Phase 1 — finish only the smallest coherent chunk of
current work to a clean stopping point (e.g., an already-underway partial 30B
run), update STATE.md/DECISIONS.md/RESULTS.md to reflect actual state, and
commit that chunk; Phase 2 — stop treating the old design as default and execute
a narrower mitigation-first experiment, with success defined as: B underperforms
A on a compaction-sensitive behavior, a small intervention improves over B, the
improvement is not explained by leakage/wrong-conversation/shuffled controls or
generic fluency, and the method is plausibly deployable. Candidate directions
offered (not mandated): H-pack vs. B-min (same summary text and packed
positions, differing only in write-time encoding, requiring key re-rotation and
identity validation); low-alpha E-post focused near the useful dose range with
negative controls retained; a sharper v2 probe set engineered so A succeeds, B
fails, and the answer is absent from both summary and tail, avoiding current
ceiling/floor effects; a focused honesty/uncertainty experiment testing whether
the H-gap fabrication-reduction effect is real and deployment-relevant;
deferring G/SoftGraft until simpler tests justify the added machinery. This
instruction was passed to the other working agent. A single thread heartbeat was
scheduled (not a repo change) for follow-up check-ins at approximately +60 and
+240 minutes.

Further discussion refined the plain-language framing of the experiment for a
reader with basic transformer knowledge (KV-cache entries as context-conditioned
interpretation, not just text; A/B/C/E/H arm comparison; success defined as
narrowing the B-to-A gap via a deployable intervention beyond what negative
controls achieve). This led to an idea for adaptive/confidence-weighted blending
of old and fresh value vectors — gating the blend per token/layer/head based on
attention-retrieval confidence (sharp vs. diffuse attention over old cache,
top-match margin, multi-head agreement) rather than a fixed global alpha, with a
caution against using raw value-vector cosine similarity as a closeness metric
since value space lacks guaranteed geometric meaning; attention QK scores were
suggested as the more principled retrieval-confidence signal. This
adaptive-gating idea is not yet implemented: the current pilot's `E-post` arm
only supports fixed-alpha blending (0.25/0.5/0.75/1.0) at exact token-match
positions. The adaptive approach was identified as conceptually aligned with the
already-planned but unimplemented `G`/SoftGraft direction.

**Repository state change.** A new uncommitted-then-committed file,
`adaptive-blending-notes.md`, was created at the repo root capturing this
adaptive-blending discussion, isolated from in-progress experiment outputs. It
was committed alone (commit `347602c Add adaptive blending notes`) at explicit
user request, without touching untracked 30B result files
(`results/raw_30b/c07.json`, `c08.json`) that belonged to the other agent's
concurrent work. No other repository or experiment files were modified during
this session.
