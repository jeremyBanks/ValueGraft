_This shard covers the SWE-bench real-task pivot's collapse and recovery: after
finally launching real coding-agent episodes, the program discovers zero
completions, diagnoses the target model as incapable of the task (validated by a
Sonnet control and by the vendor's own benchmark card), pivots through several
fallback tiers (oracle retrieval, easier datasets, a custom "chain-of-exercises"
tier), and ends with a completed but scientifically null chain-tier result —
confirming a tuning fix but finding no compaction damage to measure — leaving a
stop-and-decide point with pods shut down to halt idle billing._

**Participants in this Conversation.**

User; `claude-fable-5` (Claude Code `2.1.200`); `claude-opus-4-8` (Claude Code
`2.1.200`).

Assistant model sequence: `claude-fable-5` (Claude Code `2.1.200`) ->
`claude-opus-4-8` (Claude Code `2.1.200`).

## Incremental caching withdrawal and rebuild (icache v1→v2)

The morning opened with a fresh incident: the incremental-KV-cache speed
optimization (icache v1) caused repeated shim OOM deaths. Root cause, worked out
in exact arithmetic: `rebuild_cache` cloned the stored per-session KV snapshot
on every call, so serving held two copies simultaneously (61GB model + ~5-8GB
stored + ~5-8GB clone + activations ≈ 73-81GB against an 80GB card) — a
peak-VRAM bug distinct from the earlier "unbounded accumulation" bug already
patched that morning. The assistant's local validation had only proven bit-exact
_correctness_ on a 0.6B model at 2K tokens (CPU-scale), never checked memory at
deployment scale — a lapse explicitly named as repeating incident #11's failure
shape. At user prompting, icache v2 was built and validated properly: in-place
`DynamicCache` extension (eliminating the duplicate-copy design flaw), a VRAM
peak-predictor formula validated against measured allocator peaks on the small
model, and a guard that drops cache under memory pressure rather than crashing.
v2 was canary-deployed to one lane behind an admission gate (live probe before
serving), which incidentally caught a pre-existing mis-wired tunnel bug (lane r2
pointing at r3's pod). v2 is validated as "output-stable, not bit-exact"
(attention isn't decomposition-invariant at ~1e-4 fp32) and stays flag-off
fleet-wide pending a later confirm-phase soak.

## Monitoring failures recurred despite hardening

Even after the prior day's audit and single-strike alarms, monitoring kept
failing on a new axis: **liveness vs. progress**. Watchers confirmed processes
were running but not that they were producing scored rows, so two lanes sat idle
for 3-4 hours behind dead tunnels while looking "healthy." The user caught this
by direct questioning three times in a row. Fixes applied: a global
throughput-floor alarm (zero scores across all lanes in 60 min), then refined
per the user's explicit request to a **per-lane** floor (any single lane silent
for 100 minutes alarms by name, covering both "stalled" and "producing only
errors" cases). A full coverage-map exercise (previously specified but never
executed) was finally run, surfacing two additional unwatched gaps (silent
repo-sync failures, no low-balance alarm). A separate operational bug (follower
shell scripts created with spaces in filenames from an unquoted loop, silently
un-launchable) caused a multi-hour grid stall later found by direct inspection
rather than any monitor. The recurring theme, named directly by the user and
accepted by the assistant: fixes have been reactively triggered by the user's
questions rather than proactively caught, and the pattern itself (not any single
gap) is the standing risk.

## The real-SWE-bench collapse and diagnosis

Three humane-tier lanes (production-calibrated compaction: 12K threshold / 6K
tail / production-faithful summary prompt, replacing an inadvertently
adversarial detail-free summarizer found and fixed the prior day) ran real
SWE-bench-Lite instances across five arms. The first-ever completed real episode
was a clean, non-timeout FAIL (flask-5063, Compacted arm, agent self-terminated
after ~30 min, patch rejected by tests) — establishing the instrument worked
end-to-end. But subsequent validation runs (Original arm, full context, no
compaction — meant to establish a solvability ceiling) also failed, repeatedly:
0 for 2, then continuing to 0 for 7+ across a mixed pool of SWE-bench-Lite and
SWE-bench-Verified "easy" instances, even under oracle retrieval (naming the
fix's files in the prompt) and with generous timeouts. The user did the cost
math live: signal cost scales as 1/p (solve rate), and near-zero p makes the
dataset unaffordable regardless of budget.

Two independent diagnostics resolved the ambiguity between "task too hard" and
"our rig broken":

- A single-sample Sonnet-subagent control, given identical oracle-mode
  materials, solved the easiest queued instance in 52 seconds — proving the
  tasks were solvable in principle and that the subject 30B model itself was the
  bottleneck, not the dataset or scaffold.
- A scaffold-configuration audit found two real defects handicapping the model
  below its trained capability: native tool-calling had been disabled (forcing
  agentic-coding-trained Qwen3 through prompt-text tool conventions instead of
  its native format), and the shim's decode loop was hardcoded to greedy argmax
  even though a `temperature` field was silently accepted and ignored — despite
  Qwen3's model card explicitly warning against greedy decoding for long
  generations. Both were logged as high-severity incidents (rule 17: parameters
  must reach their effect or be rejected, "accepted but ignored" configuration
  is itself a bug class).

The user's suggestion to consult the model vendor's own published benchmark
tables ("model card as difficulty certificate," now rule 18) was treated as a
breakthrough: it would have certified capability boundaries before any spend,
and post hoc it corroborated the empirically-discovered floor (repo-level
SWE-bench-style fixing: no; short agentic exercises and tool-use tasks: yes).

## Pivot sequence and priority-queue design

Per explicit user direction, task assignment was restructured into a
**validated-first dynamic priority queue**: any instance/config must pass a
full-capability (Original-arm) validation episode before constrained arms are
run on it, and the moment something validates, its constrained-arm rows jump
ahead of remaining validation work — all lanes acting as promotable consumers
rather than one dedicated lane. Multiple correction cycles were needed before
the implementation matched this spec exactly (assistant initially queued
promoted rows behind remaining validation rows; then failed to make all lanes
promotable). The user separately mandated interleaving additional task sources
(SWE-bench Verified's human-rated "<15 min fix" tier) into validation to
diversify difficulty, and tightened episode timeouts from 60 to 30 minutes given
near-zero expected solve times on tasks the model can actually do.

When the SWE-bench floor became conclusive (multiple sources, multiple repos,
all failing even under oracle mode), the user ordered an immediate hard pivot:
purge all remaining SWE rows from every queue (16 rows dropped) rather than let
them keep running as "boundary documentation," which the assistant had been
doing on its own judgment without re-checking against the user's actual
abandonment decision.

## The chain-of-exercises tier

The fallback built and adopted: chained sequences of 3-4 Aider-Polyglot-style
Exercism tasks solved sequentially in one conversation, chosen specifically
because published benchmarks certify this model class can do them, with context
growing across exercises so compaction triggers mid-chain (giving
position-resolved signal — does the agent degrade on exercise 3-4 where pressure
peaks, does it forget exercise-1 constraints).

Results: 4 of 6 candidate chains validated at full capability (Original arm
passing all 4 exercises), consistent with the vendor-certified expectation. The
five-arm comparison ran to completion (20/20 cells) across four validated seeds
(Original, Compacted, graft α=0.75, graft α=1.0, champion/per-layer-tuned). Key
findings:

- **Confirmed:** the per-layer-tuned "champion" config cures a real instability
  — full-strength graft (α=1.0) collapsed to 0/4 on one seed (s1) while champion
  scored 4/4 on every seed including that one. This directly validated the
  user's hypothesis that naive full-replacement grafting is dangerous and that
  per-layer tuning is the fix.
- **Confirmed null:** Compacted arm passed 4/4 on every measured seed — no
  compaction damage was detectable at binary pass/fail resolution. The chains
  are too compaction-robust for this comparison, realizing the "ceiling problem"
  flagged earlier in the session.
- **Void:** the recall-probe metric (meant to catch subtler degradation) was
  found to still be querying constants from the old synthetic (t1/t2) task type
  rather than the actual chain exercises' content — another inherited-component
  mismatch bug, invalidating that column for chain-tier data.
- Effort proxy (event counts) showed no separation between arms either.

The honest verdict delivered to the user: one genuine tuning result banked, but
no evidence of compaction damage for grafting to repair on this task tier — a
scientifically null outcome on the core question, distinct from an operational
failure. The assistant recommended against continuing to spend on the current
chain design and proposed either harder chains (cross-exercise dependencies
forcing genuine reliance on evicted content) or moving directly to the
previously-scouted tau2-bench confirm-phase design (which has a built-in
evictable-vs-system-message policy dissociation, considered the more principled
fix for the same gap) as the next real investments — explicitly deferring to the
user's decision rather than proceeding autonomously, and shutting down the three
billing pods to stop idle spend while awaiting that call.

## Other standing decisions and context

- Total budget arc: $200 loaded across four installments; roughly
  $163.50 spent before the day's continued burn, decomposed as ~1/3 durable results, ~1/3 costly-but-now-documented lessons (22+ incidents, 20+ rules), ~1/3 ordinary infrastructure cost. Balance fluctuated back up via top-ups (reaching ~$63-65
  after a top-up mid-chain-tier run).
- The user authorized, but did not yet trigger, a bounded ($6, 2-pod-hour)
  GLM-4.5-Air trial as the next model-swap candidate if the current model's
  task-capability floor can't be worked around — with an explicit demand for
  tight, active supervision during that trial (inverting the normal economy-mode
  monitoring posture) and a locked sequence (validate once, then immediately run
  arm variations on success).
- GLM suitability was partially checked: tokenizer/template layer passed a full
  dry-run for GLM-4.5-Air and GLM-4-9B; a serving-layer smoke test on GLM-4-9B
  locally OOM'd (margin-arithmetic miscalculation, logged) and was deferred in
  favor of doing the representative smoke directly on GLM-4.5-Air when/if the
  trial is funded.
- A serving-architecture discussion catalogued why the custom Python/PyTorch
  shim (rather than vLLM/SGLang) is ~3-10x slower — full re-prefill every turn,
  no flash-attention kernels, single-threaded serial serving — but is necessary
  because production servers' paged KV layouts make the value-graft surgery
  infeasible without deep plugin work; a vLLM KV-connector-based implementation
  was noted as a plausible long-term/write-up "deployment path" idea, and a
  same-stack optimization path (flash-attention, compiled decode, async serving,
  ~2-4x recovery) was logged as the first infrastructure investment due at the
  next phase boundary, not before.
- tau2-bench (banking_knowledge domain) was identified via the
  vendor-benchmark-card idea as a strong confirm-phase candidate: its policy is
  delivered through evictable conversation turns (unlike classic tau domains
  where policy lives in the untouched system message), giving a built-in
  positive/negative control pair on a standard, licensed, harness-compatible
  benchmark. It remains gated behind a cheap pilot (verify token lengths cross
  the compaction threshold, reward determinism, user-simulator cost) and behind
  the user's results-then-budget approval process — not yet started.
- Standing rule set now includes: model card as difficulty certificate before
  task-source commitment (rule 18); serving parameters must be honored or
  rejected, never silently ignored (rule 17); implausible results (too clean or
  too catastrophic) get inspected before being believed; capability-smoke any
  new task source before comparison experiments; re-derive inherited parameters
  at every purpose boundary; error-detection nets must be built from real
  observed failure phrasings, not assumed ones.

---
