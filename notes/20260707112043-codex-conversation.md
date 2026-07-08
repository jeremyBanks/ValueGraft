_This shard traces the J-lens interpretability side-investigation from a
shell-tooling aside through a major methodological correction: an initial
two-state (write-time vs. fresh) sweep was mistaken for evidence about the
ValueGraft intervention itself, when no actual grafted condition had ever been
captured, followed by repeated rounds of report rewrites, fact-checking, and a
new stronger-scenario experiment plan._

**ShellCheck decision.** Early on, the assistant clarified that ShellCheck
should only gate the team's own pod-launch job scripts (alongside existing
`bash -n` checks in `scripts/launch_pod.sh`), not generated SWE-Gym agent shell
commands, which should only be logged as diagnostic columns since valid
one-liners can trigger false positives.

**J-lens layer selection was heuristic, not systematic.** The assistant admitted
the `16/32/48/62` (`n_layers//4, //2, 3*//4, n-2`) layer sample used across
probes was a pragmatic depth heuristic, not derived from the workspace paper.
The next-token control had narrowed to `48,62` specifically because layer 48
showed low overlap with next-token logits (mean top-20 Jaccard 0.076) versus
layer 62's high overlap (0.436), motivating those as the reporting pair
(semantic-readout layer vs. continuation-pressure layer).

**Full-layer sweep design and mishap.** The user pushed back that they'd
expected a fuller sweep (every summary token × every fitted layer, ranked by
write-time/fresh divergence, not hand-picked anchors) — this became the "broad
sweep, then top-k by change" design, built by extending `multi_demo_scan.py`
into a new script plus pod job wrapper. On the first A100 pod run, the assistant
made an operational error: it fired the artifact pull (`rsync`) and pod
termination in parallel, and termination cut off the transfer, losing the 64MB
output. The run had to be fully repeated on a second pod (using `setsid` for
cleaner detachment this time), successfully producing and validating a 63MB raw
JSON (1,175 summary tokens × 63 layers = 74,025 rows) plus a compact 1.9MB
derived summary (the raw file exceeds the repo's 4MB commit guard and was
gitignored). Analysis showed divergence peaking around layers 47–59, with layer
62 much more next-token-like — supporting layer 48 as the best default display
layer.

**Process-discipline correction on schema handling.** When jq queries against
the sweep JSON produced unexpected nulls (from the `demos` object being keyed by
name rather than array-indexed, plus jq operator-precedence bugs), the
assistant's initial reflex was to work around the "surprising" structure rather
than validate it. The correction was methodological: self-produced data should
trigger producer/schema inspection, not ad hoc filtering. The assistant wrote
`validate_full_layer_sweep.py`, a strict schema validator, and — per further
user instruction — recorded the failure and the new rule ("if artifact structure
surprises us, stop, inspect the producer, validate schema, record the
resolution") directly in `jlens_boundary_probe/WORKLOG.md`, not just as an
apology in conversation. This WORKLOG became the standing internal handoff
document for the whole J-lens effort, distinct from public-facing reports.

**Repeated document-quality cycles.** The user requested a polished, standalone,
human-readable explainer of ValueGraft using J-lens examples, to be iterated
with subagent reviewers for clarity, scope, and prose (explicitly invoking a
rigor bar: "if a PhD student did this in a paper I would fire them"). Multiple
review passes (structure/scope, technical-overclaim guardrails, fresh-eyes
prose) produced `interpreting_valuegraft_examples.md`, later expanded with a
detailed prior-art appendix (interpretability + KV/cache literature) per user
request, while web searches were minimized since most prior art was already
collected in-repo. A tangential task: an OpenRouter API key was saved to
`.openrouter_key` (mode 600, gitignored, never committed).

**Critical methodology failure: missing intervention data.** After several
rounds of user complaints about clarity, the user identified the core problem
was not wording but content: the document only ever compared two states
(write-time vs. fresh), never the actual grafted condition, meaning the
intervention itself was never evaluated. Investigation confirmed the boundary
probe's graft path had silently failed (`grafted_post_token` was absent from the
artifact) because the graft sampler didn't know how to snapshot Qwen3.6's actual
cache structure (a hybrid cache with `LinearAttentionLayer` entries, not plain
K/V `DynamicCache`). This was treated as a severe process failure requiring: (1)
fixing the cache snapshot/rebuild code to handle hybrid linear-attention layers,
(2) redesigning the probe as a four-sequence comparison (full context, fresh
compacted, alpha-0 sanity control, aligned grafted, later also alpha-sweep and
shifted-value negative control), (3) an explicit worklog failure record, and (4)
independent subagent audits before any report rewrite. The corrected artifact
(`qwen36_boundary_three_state_probe.json`) was validated via a hardened
`validate_three_state_probe.py` requiring graft availability, matched
forced-target alignment, and alpha-0-equals-fresh sanity checks. Results:
aligned alpha=0.75 gave one next-token argmax rescue and improved layer-48
closure (+0.0337); alpha=1.0 gave two rescues but worsened layer-48 closure; a
shifted-value control also produced one incidental rescue but substantially
worsened layer-48 closure (−0.0903), showing argmax rescue alone is an
insufficient/misleading success criterion.

**Report iteration and prose standards.** Multiple further corrections followed:
the user objected to public-facing prose narrating rejected approaches/internal
history (that belongs in WORKLOG, not the shareable report); to code-fence
formatting disabling word-wrap for prose sentences; to the rewritten report
being too short/shallow after a metrics-focused overcorrection stripped out the
actual J-lens readout tables that made the original compelling; and finally,
after inspection, to the finding that the chosen example scenario (`rivermark`,
a coding bugfix scenario) was a weak qualitative demo because the compacted
summary and retained tail already contained nearly the entire answer, so all
conditions' readouts collapsed into near-identical top-k lists. The assistant
added an explicit "what context is truncated" section and reframed the artifact
honestly as a sanity/control probe rather than a compelling qualitative
showcase, while flagging that stronger candidate scenarios (`B-410`, `Maple`,
`Falcon`/`Patch 17`, `Ghost2`, `Dex`) should be used instead. A fact-checking
gate using independent read-only subagents was formalized as a required step
before each report push, and it caught real issues (a
capitalization/table-normalization bug, and unlabeled filtered-excerpt tables
presented as literal data).

**New experiment plan at shard boundary.** The user requested a full new cycle:
carefully plan, execute stronger experiments (using known higher-contrast
scenarios), then repeat the full report/revision/fact-check process — explicitly
asking for a considered plan before acting, and for any spun-up pod to be shut
down when the work concludes (leaving it running between steps is fine). The
assistant built a batch-probe harness (`intervention_batch_probe.py`,
`analyze_intervention_batch.py`, `job_qwen36_intervention_batch.sh`) reusing the
existing `boundary_probe.py` cache-surgery functions rather than reimplementing
them, designed to run multiple stronger scenarios in one model load and rank
examples by focus-span closure rather than raw token-level noise. After an A100
capacity shortage, an H100 80GB pod (`u513d5hb7s4m3j`) was launched and the
batch job started (remote PID 587); at the shard's end the assistant is
mid-monitoring, having interrupted only the local hung launcher shell (not the
remote job) and preparing to check the remote log directly via a separate
connection. Pod termination is expected once artifacts are pulled and validated,
per the user's standing instruction.

---
