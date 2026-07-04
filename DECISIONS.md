# DECISIONS.md — running log of deviations and choices

Format: date — decision — reason.

- 2026-07-04 — **Environment:** uv project, Python 3.12, `mlx-lm==0.31.3`, `mlx==0.31.2`, `transformers==5.0.0`. transformers pinned to 5.0.0 because mlx-lm 0.31.3 requires ≥5 but crashes on 5.13 (`AutoTokenizer.register` API change).
- 2026-07-04 — **Dev model:** `mlx-community/Qwen3-4B-Instruct-2507-4bit` rather than plain Qwen3-4B/8B. Same 2507-instruct family as the 30B-A3B target, instruct-tuned without thinking-mode preamble, small enough to iterate fast. Brief allowed either.
- 2026-07-04 — **Repo layout:** per brief §7 — `src/` (surgery, arms, probes), `data/` (manifests + conversations), `results/` (tables, plots, logs).
