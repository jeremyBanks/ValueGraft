This chunk documents an extended J-lens/ValueGraft investigation cycle: a first
pass corrected a compressed, evidence-poor writeup into one with concrete
examples, then a series of user-driven methodological corrections progressively
exposed that the J-lens work had never actually measured the ValueGraft
intervention itself, only diagnosed a pre-existing state gap — leading to new
experiments, harder validation discipline, and multiple report rewrites.

**Participants:** User and gpt-5.5-xhigh.

The session opened with a push of 98 pending commits, followed by a strong
rejection of the J-lens section in the ValueGraft draft as too compressed and
unreadable. The assistant repaired it with concrete quoted examples and
side-by-side write-time/fresh readout tables for named anchors (Vacuum, Dex,
Maple, B-410, Crane) plus a coding example, committing in small increments.

A methodological question then reframed the entire investigation: since J-lens
readouts are vocabulary-ranked like next-token logits, how much of the signal is
just next-token prediction? The team designed and ran a next-token-vs-J-lens
control on Qwen3.6-27B (RunPod A100), finding real but layer-dependent
divergence — mean top-20 Jaccard 0.256 overall, 0.076 at layer 48 versus 0.436
at layer 62 — establishing layer 48 as more distinctively "semantic" and layer
62 as closer to ordinary continuation behavior. This was written up as a
standalone control report and integrated as a caveat into both the
semantic-readout draft and the main ValueGraft draft.

A follow-up question about layer selection revealed the layers (16/32/48/62, a
"quarter" heuristic) were never scientifically chosen, and a further exchange
clarified that the team had actually wanted a full summary-token × layer sweep
ranked by change/divergence, not hand-picked anchors — a broader-than-planned
scope that had quietly narrowed. This was corrected by building and running a
full-layer sweep (1,175 summary tokens × 63 layers, 74,025 token-layer rows) on
a second A100 run (after a first artifact was lost to a premature pod
termination), producing a `full_layer_sweep_report.md` and a persistent
`WORKLOG.md` handoff document recording rationale, artifacts, and pitfalls for
future pickup — a project convention the user explicitly requested going forward
(durable notes on decisions/state/mistakes, not just final conclusions).

A process failure followed: when the sweep's JSON structure didn't match query
assumptions, the assistant patched around it with ad hoc jq queries rather than
treating the surprise as a signal to stop and validate the schema against the
producing code. This was called out sharply as scientifically irresponsible. The
correction was to build a strict schema validator
(`validate_full_layer_sweep.py`), confirm the artifact was actually internally
consistent (the assistant's queries, not the data, were wrong), and record the
failure and the new rule ("if structure surprises you, stop and validate before
interpreting") explicitly in `WORKLOG.md` — establishing a standing project
practice of recording process failures as state, not just fixing and moving on.

The team then commissioned a polished, standalone, shareable explainer of
ValueGraft using J-lens evidence, produced via multiple subagent review passes
(structure/scope, technical overclaim guardrails, fresh-eyes prose) and
iteration, resulting in `interpreting_valuegraft_examples.md`, later expanded
with a detailed prior-art appendix (interpretability and KV/cache literature)
per user request. A user-supplied OpenRouter API key was saved to a git-ignored
`.openrouter_key` file at the repo root.

A second, more consequential correction followed: after multiple readability
complaints, the user identified that the document's evidence design was invalid
for its stated purpose — it only compared old (write-time) versus fresh
(post-compaction) context, which shows there is a state gap, but never included
a third condition showing the graft's actual effect. Investigation confirmed the
grafted-state capture had silently failed in the underlying probe script
(`boundary_probe.py`) due to Qwen3.6's non-standard KV-cache structure, and the
document had been built and iterated on without anyone noticing the core
intervention data was missing. This was treated as a significant methodological
failure requiring: (1) fixing the cache-snapshot/rebuild code to handle
hybrid/linear-attention cache layers, (2) building a genuine
three/four-condition probe (full context, fresh compacted, alpha-0 sanity
control, and grafted compacted) with teacher-forced target sequences, (3) a
hardened validator enforcing these acceptance gates, (4) independent subagent
audits confirming the fix, and (5) a full report rewrite
(`valuegraft_four_sequence_intervention_probe.md`) built strictly from the
corrected artifact. Key result from the corrected probe: alpha-0 exactly
reproduces the fresh-context baseline (sanity check passes); aligned
alpha_V=0.75 graft changes 16 value-cache layers across 96 aligned summary-token
pairs, with mixed effects across layers (layer 48 closure improved, layer 62
worsened slightly); a shifted/misaligned control produced a spurious argmax
rescue but clearly worse layer-48 closure, serving as a useful negative control.

Further user feedback pushed on document quality and content balance: an initial
rewrite over-corrected into a metrics-only report, stripping out the J-lens
readouts that were the actual point; this was restored with both next-token and
J-lens tables per example. The user then flagged that the examples were too
weak/similar to be compelling and that the document didn't clearly state what
context was being truncated in each condition — prompting an honest revision
acknowledging the primary probe case (`rivermark`, a coding scenario) was a weak
qualitative demonstration because the compacted summary and retained tail
already contained nearly the full answer, leaving little room for the graft to
show a visible effect. The document was revised to state this limitation plainly
rather than oversell the tables, and to add an explicit description of what was
compacted, retained as tail, and available to the graft in each case.

A formal fact-checking gate (independent read-only subagent comparing report
claims against raw artifacts and code) was adopted from this point onward as a
standing quality step, alongside existing structure/prose reviewers, following
user request.

The final phase was a deliberately planned second experiment cycle to obtain
more compelling or falsifying evidence: (1) an "ordinary" batch of ten stronger
scenarios (block-party, checkout/coding, household, Pokémon anchors) reusing the
validated cache-surgery code via a new batch runner, run on an H100 pod, and (2)
a stricter "sparse challenge" batch with label-only summaries and no retained
tail (forcing the model to recover deliberately omitted details), which required
a harness fix after Qwen's chat template rejected a prefix with no user turn.
Results: the ordinary batch showed small, alpha-sensitive positive effects at
lower alpha values (rather than the default 0.75) and confirmed the
shifted-control-is-worse pattern in aggregate; the sparse challenge was a clean,
scoped negative — V-only, summary-token-only grafting at the sampled layers did
not recover facts omitted from the compacted summary. Both artifacts were
validated with a corrected schema validator (raw JSONs kept local/git-ignored
due to size, compact summaries committed), reviewed by fact-check and
readability subagents, and written up in `intervention_probe_findings.md`, with
claims scoped narrowly (V-graft only, summary-token positions, sampled layers)
rather than generalized. Both RunPod GPU instances used in this phase were
terminated after artifacts were pulled and validated, per instruction to shut
down compute when work concludes.

Standing project/process rules established or reinforced during this chunk:
don't let internal analysis shorthand substitute for reader-facing evidence in
shareable documents; treat any surprising artifact/schema structure as a signal
to stop and validate against the producing code rather than patch queries around
it; record process failures and resulting rule changes in the probe's
`WORKLOG.md`, not just in chat; keep the public-facing report free of narration
about rejected approaches or internal debugging history (that belongs in
`WORKLOG.md`); require matched/consistent evidence columns (next-token and
J-lens) across all examples in a report; use ShellCheck (not just `bash -n`) as
a preflight check on scripts before launching pod jobs; and treat two-condition
(old-vs-fresh) J-lens comparisons as diagnostic-only, never as evidence of the
ValueGraft intervention's effect, which requires the explicit third/fourth
grafted and alpha-0-control conditions.
