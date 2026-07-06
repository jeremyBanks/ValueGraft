# AGENTS.md — orientation for any agent working in this repo

## What this repo is

A self-contained research project run by an autonomous coding agent on a
32 GB Apple-Silicon MacBook: measuring whether KV-cache state written at
generation time carries semantic meaning that re-encoding the same text
loses, across LLM conversation-compaction boundaries.

## Read in this order

1. `STATE.md` — where things stand right now and what's queued. Start here.
2. `DECISIONS.md` — every methodology decision, deviation, and verified
   runtime fact. Non-negotiable reading before touching cache-surgery code.
3. `semantic-continuity-experiment-brief.md` — the original experiment design
   (arms A–F, probe suite, build ladder).
4. `amendments-from-external-review.md` — controls added after review
   (B-causal, negative grafts, leakage classes, metric hierarchy).
5. `followup-explorations-arms-GH.md`, `phase2-scaleup-and-coding-extension.md`
   — future work, only after the main analysis.

## House rules (from the user + hard-won)

- **Never skip the build ladder.** Any new model/config must pass L0/L1/L3
  identity tests (`src/l0_identity.py`, `src/l1_l2_surgery.py`,
  `src/l3_l4_identity.py`) before its results count. Silent position bugs
  masquerade as findings.
- **Serial GPU jobs.** Two MLX processes halve each other; run one at a time.
- **Detach long jobs** (`nohup ... & disown`, PID file in scratchpad) —
  harness-tracked background shells have been killed mid-run. Use `python -u`,
  `tee` full output to a scratchpad log, monitor the log.
- **Timestamps around long commands** (`date` before/after) to spot
  pathological runtimes.
- **Commit often** (snapshot-style, fine-grained). No large binaries — model
  weights live in the HF cache, never in the repo.
- **Local subject model generates conversation text only.** All meta-work
  (authoring scenarios, judging answers, analysis review) goes to Claude
  subagents — the local model is the experimental subject, not a collaborator.
- **Subagents: small numbers, bounded batches** (≤2 concurrent).
- **Honest reporting.** Nulls are reportable; contaminated probes get
  reclassified, not deleted; negative results in DECISIONS.md, not buried.
- Temperature 0 for all evaluation; corpus generation is seeded sampling.

## Quick technical map

- Python via `uv run python src/<script>.py`; deps pinned in pyproject
  (mlx-lm 0.31.3, transformers 5.0.0 — do not upgrade casually).
- `src/kvlib.py` — cache serialize/rebuild, GappedKVCache (position counter
  decoupled from storage), teacher-forcing (batched; NEVER compare batched
  logits to stepwise logits — different kernels).
- `src/arms.py` + `src/run_arms.py` — the experimental arms and driver.
- Qwen3 chat-template trap: the final assistant message (and the assistant
  message before a generation prompt) grows an empty `<think>` block, so
  token prefixes are unstable across re-renders. Always use the canonical
  non-final rendering (`canonical_ids` — dummy-user trick); locate message
  boundaries by scanning `<|im_start|>` positions, never by re-tokenizing
  prefixes.
- Data: `data/scenarios.json` (authored plants), `data/synthetic/`,
  `data/natural/` (composed conversations), `results/raw*/` (per-conversation
  arm outputs), `results/scores.json` (probe scoring).

## Source-control policy (07-05)

- Committed: all docs, src/, small JSON experiment artifacts in results/
  (the experiment record — keep them versioned). NOT committed: datasets
  (*.parquet, HF caches), model weights, credentials (.gitignored), venv.
- A local pre-commit hook rejects staged files >4MB (.git/hooks/pre-commit —
  hooks do NOT travel with clones; recreate it from this note if absent).

## When stuck: invoke a different model family (user directive, 07-05)

Both Claude and (per user) OpenAI Codex CLIs should be available on this
machine (`claude` is on PATH; `codex` was NOT found on PATH as of 07-05 —
check again / ask the user if needed). If you are hitting repeated
obstacles — many attempts, little progress, or a diagnosis that keeps not
paying off — you are STRONGLY ENCOURAGED to invoke an agent from a
different model family for a fresh perspective:

- Use the highest available model + reasoning-effort settings.
- Point it at the relevant documents (STATE.md, DECISIONS.md, the
  experiment briefs, the failing code/logs) so it builds real context, and
  ask it to (a) explain what might be going wrong and (b) suggest what else
  to try.
- If the CLI supports session resumption, resume the same session for an
  extended back-and-forth rather than one-shot queries.
- Rationale (user): differently-trained models make different connections;
  the value is the independent perspective, not raw capability. Getting
  stuck is natural — treat cross-model consultation as a normal tool, not
  a last resort.
