# Champion validation experiment — placebo-controlled, per-quant, CI'd

**Author:** design + implementation agent (2026-07-09). Implements the owner's
plan exactly (refined for runnability; no redesign). Coordinator (Claude) runs the
pods; this doc is the runnable spec + the committed harness change.

---

## TOPLINE (4–5 sentences + first command)

We validate whether the **per-layer-tuned CHAMPION value-graft** gives a *genuine,
content-specific* positive on post-compaction recovery, or merely amplifies the
already-known content-*independent* generic effect. The decisive contrast is
**E_champion − placebo_champion** (real tuned graft vs. the SAME champion
layers/positions/α but corrupted-source values), plus E_champion − B, on the
held-out recovery plants (c07–c24, 18 conversation clusters, ~200 paired plants),
with **conversation-clustered bootstrap 95% CIs**. We run it at **4-bit first**
(cheap) then **bf16**, deriving the champion config **per quant** (the 4-bit champion
is re-derived from a 4-bit per-layer profile; bf16 reuses the existing
`layers_30b_bf16` champion). CONFIRM if the E_champion − placebo CI lower bound > 0
(content-specific champion); REFUTE (null stands) if it spans 0. The harness change
is committed (`src/cross_arch_probe.py` now accepts the per-layer/per-head champion
config and reports the paired content-specificity CI).

**First command to run (4-bit, derive+validate, one unattended pod):**
```bash
cd /Users/jeb/experimentation
MODELS="JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4" \
SC_HF_MODEL="JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4" \
SC_TUNE_TAG="30b_4bit" SC_LOAD_DTYPE="auto" \
SC_CONV_START="6" SC_CONV_LIMIT="18" \
  scripts/launch_pod.sh champ4bit scripts/job_champion_4bit.sh
```
(`MODELS=` is required so the fail-closed pre-flight gate can verify the 4-bit repo
on the HF hub before launch.)

---

## THE QUESTION AND THE GAP (verified against the harness)

- The bf16 **null** (`effect_bound`, placebo-controlled, CI'd) ran only **scalar
  α=0.75**. Its placebo showed the raw improvement is **NOT content-specific**.
  NB: `effect_bound_probe.py` lives only on the pod (`jlens_boundary_probe/`); it is
  **not in the repo** and takes only `--alpha` (scalar). So it is *not* the vehicle
  for a per-layer champion — see "harness wiring".
- The **per-layer tuned config** was never subjected to the *current* robust
  placebo-CI estimator. Grounding the numbers:
  - `tune_configs.json → layers.alpha_map` = the per-layer coarse {0, 0.75, 1.0}
    map over 48 layers. It **PASSED** the 2026-07-06 wrong-conversation guard
    (content-dependent; DECISIONS 07-06 03:15) → the surviving champion candidate.
    **This is the primary CHAMPION.**
  - `tune_rules.json → posslots_a1` (57 (layer,head) slots @α=1) scored biggest on
    holdout (I recomputed `results/tune_eval_30b_bf16`: **+0.0384, 10/10**) but
    **FAILED** the wrong-conversation guard (content-INDEPENDENT; DECISIONS 07-06
    03:20) → DEAD as a content-carrying finding.
  - `tune_rules.json → top16_a1` (13-layer head mask @α=1): **+0.0209, 10/10**
    recomputed; never guard-tested.
- So: the wrong-conversation guard was a *coarse* content control on the *old*
  estimator. The **rigorous** control — corrupted-source placebo through the exact
  champion slots, robust raw_EB metric, conversation-clustered bootstrap CI, paired
  E−placebo — was **never run on the champion**. That gap is this experiment.

---

## DESIGN

### Core contrast (requirement 1)
Per scored recovery plant, on the compacted baseline convs:
- **B** — compacted baseline logprob of the gold continuation (`lp_B`).
- **E_champion** — value-only graft using the champion per-layer α-map (`lp_E`).
- **placebo_champion** — graft through the **SAME champion layers/positions/α** but
  with corrupted **source** values (`lp_E_placebo`).

Two quantities, both CI'd (conversation-clustered bootstrap, 10 000 resamples):
- **E_champion − B** = `raw_EB` (does the champion recover meaning at all).
- **E_champion − placebo_champion** = per plant `(lp_E − lp_B) − (lp_E_placebo −
  lp_B) = lp_E − lp_E_placebo`, **paired within plant** then resampled over
  conversations. *This is the decisive content-specificity number.*

### Placebo (requirement 1 & gotchas)
All placebos go through the identical champion config (same slots/α); only the
source values differ. `PLACEBO_MODES` in `cross_arch_probe.py`:
- **`shuffle_pos`** (PRIMARY, "deranged"): permute the source values *across the
  grafted positions* — same value set, destroyed position↔content correspondence.
  This is the owner's "random/deranged values at the same slots/α".
- **`shuffle_probe`** (STRONGEST content control): source values drawn from ANOTHER
  conversation's summary snapshot — the rigorous version of the wrong-conversation
  guard that posslots failed.
- **`gauss`**: per-row L2-norm-matched Gaussian noise — random values, matched
  energy (rules out "any injected energy of this magnitude helps").

The job runs all three; `shuffle_pos` is the headline, `shuffle_probe`/`gauss` are
robustness. A real content-specific champion must beat ALL three.

### Metric & confirm/refute criteria (requirements 1 & 5)
Primary metric = **robust raw_EB** (bounded logprob lift; the corrected metric, not
the CI-inflated ratio). Everything is CI'd; **no bare point estimates**.

- **CONFIRM (real content-specific champion):** the pooled `E_champion −
  placebo_champion` conversation-clustered 95% CI **lower bound > 0** for the
  primary placebo (`shuffle_pos`), and directionally consistent (CI>0 or at least
  positive point + shuffle-probe/gauss agreeing). Then the tuning buys genuine
  content-specific recovery, not just amplified generic lift.
- **REFUTE (null stands):** the `E_champion − placebo_champion` CI **spans 0** — the
  champion's larger E−B is the *generic, content-independent* effect amplified by
  grafting more slots. Report honestly; the scalar-α null generalises to the
  champion.
- Report E_champion − B alongside so "recovers meaning" and "content-specific" are
  distinguished (a champion can be positive vs B yet indistinguishable from placebo
  — that is exactly the null outcome).

### n / power (requirement 5)
- Recovery plants: `data/synthetic/c01…c24`, 10 plants each (266 usable across the 6
  CATS = sense/referent/stance/ruled_out/evicted_fact/strong_prior).
- **Derive** the champion on the tuning VAL set (c01–c06,n01–n04). **Validate** on
  the **held-out c07–c24** → **18 conversation clusters, ~200 paired plants**
  (dry-run confirmed 203 usable in just sense+referent). Out-of-sample →
  content-specificity is not a tuning artifact.
- 18 clusters is a sound clustered-bootstrap n; the paired E−placebo design cancels
  plant/gold difficulty and tightens the CI. **Recommendation:** run c07–c24 (n=18
  clusters) as the pre-registered set. If the decisive CI is *marginal*, extend to
  the natural convs or pool c01–c24 (the paired contrast tolerates in-sample
  derivation convs because the placebo passes through the same possibly-overfit
  config) — but pre-register that as a secondary, not a fishing step.

### Secondary arms (minor addition — see refinements)
`posslots_a1` and `top16_a1` are wired as optional secondary champion configs
(`data/champion_configs/`). Validating `posslots` under the rigorous placebo is a
built-in consistency check: it FAILED the old guard, so it should FAIL the E−placebo
CI here. Primary claim rests on the per-layer `layers` champion only.

---

## HARNESS WIRING — the change I implemented (requirement 2)

**Chosen path: `src/cross_arch_probe.py`** (not `effect_bound_probe.py`, which is
off-repo and scalar-only). It already had (a) the placebo modes, (b) robust raw_EB,
(c) `bootstrap_ci_95_cluster` conversation-clustered CIs, and (d) `blend_values`
that *already accepts a per-layer alpha dict AND a per-head `head_map`*
(`src/kvlib_hf.py:130`). So the champion maps in cleanly and **value-only is
guaranteed by construction** — `blend_values` only ever rewrites V rows; keys are
never touched → **α_K = 0** regardless of config.

Committed changes (all in `src/cross_arch_probe.py` unless noted):
- **`load_champion_graft_cfg(path)` — `:622`.** Loads a canonical champion config
  (`{"alpha_map": {...}}` for per-layer α, or `{"alpha":x, "head_map": {...}}` for
  per-head slots). Fails loud on both-or-neither.
- **`run_model(..., champion_cfg=…)` — resolve + apply, `:2124`–`:2146`.** Resolves
  the config once into `graft_alpha` (dict or scalar) + `graft_head_map`. When no
  config is given, behaviour is the historical scalar-α path (fully backward
  compatible; the whole existing cross-arch sweep is unchanged).
- **Real graft E — `:2559`** now `blend_values(..., graft_alpha,
  head_map=graft_head_map)` (was scalar `alpha_v`).
- **Placebo graft — `:2593`** uses the **same** `graft_alpha` + `graft_head_map`, so
  E and placebo differ ONLY in source values (the point of the control).
- **Decisive paired CI — `doc["content_specificity"]`, `:3085`–`:3125`.** Computes
  `E − placebo` per plant from the resume-safe `traces` (both `raw_EB` and
  `placebo_raw_EB` are stored index-aligned per plant with `conversation_id`),
  grouped by conversation, clustered bootstrap; overall + by-category.
- **Checkpoint fingerprint — `run_fingerprint(..., champion_cfg=…)`.** The champion
  config is added to the **SCORE** fingerprint (canonicalised as sorted-key JSON so
  it is stable across the save/reload round-trip) but NOT the render fingerprint —
  so a champion run **reuses the expensive render** yet never pools uniform-α scored
  rows.
- **CLI/env — `--champion-config` / `SC_CHAMPION_CONFIG`, `:4548`; parsed `:4681`;
  passed to run_model `:4763`.**
- **`SC_LOAD_DTYPE` env-gated load dtype — `cross_arch_probe.py:1892`,
  `run_tune_hf.py`.** Default `bfloat16` (unchanged); set `auto` for a pre-quantized
  4-bit repo so transformers honours its `quantization_config`.
- **`scripts/launch_pod.sh`:** forwards `SC_CHAMPION_CONFIG`, `SC_LOAD_DTYPE`,
  `SC_TUNE_TAG`; syncs `tune_rules.json` + `data/champion_configs/` (and moves the
  latter under `data/` on the pod).

Verified on CPU (no GPU touched):
- controls self-test (`--self-test`) still passes (placebo index plans, checkpoint
  fingerprint gating, native/pure parts).
- `load_champion_graft_cfg` unit tests (layer-map, head-map, None passthrough).
- fingerprint stability: same file → same fp run-to-run; None ≠ champion ≠ distinct
  configs (a champion run cannot silently reuse a uniform-α checkpoint).
- dry-run of the validation path (`SC_NATIVE_RENDER=0`, c07–c24) → 203 usable plants;
  `qwen3_moe` geometry = 4 KV heads (head-map heads 0–3 valid).

### Champion config files (committed, `data/champion_configs/`)
- `layers_30b_bf16.json` — **PRIMARY**; the per-layer α-map from
  `tune_configs.json` (27 nonzero layers: 10 @1.0, 17 @0.75). Passed the old guard.
- `posslots_30b_bf16.json` — secondary/near-negative (57 head slots @1.0).
- `top16_30b_bf16.json` — secondary (13-layer head mask @1.0).
- `layers_30b_4bit.json` — **built on-pod** by STAGE 1b of `job_champion_4bit.sh`.

**Derivation rule (reproduces the bf16 map; applied identically at 4-bit):** rank
the 48 layers by VAL marginal dEB (mean over VAL convs of `score(graft layer L
alone @α=1) − B`); **top ~21% → α=1.0, next ~35% (if marginal>0) → α=0.75, rest →
0.0**. Checked against the bf16 profile: this yields exactly the shipped 10 @1.0 /
17 @0.75 / 21 @0.0. Fractions are `SC_FRAC_HIGH=0.21` / `SC_FRAC_MID=0.35`.

---

## 4-BIT MODEL CHOICE + POD DEPS (requirement 4)

**Primary:** `JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4` — GPTQ Int4 is the
best-supported 4-bit path for a Qwen3 MoE in transformers ≥4.57 via **`gptqmodel`**
(auto-gptq is deprecated). Pod dep: `pip install gptqmodel optimum` (already in the
job). This is the closest loadable analogue to the local MLX 4-bit (group-wise
affine int4).

**Fallbacks (swap `SC_HF_MODEL` + uncomment the matching pip line in the job):**
- `Intel/Qwen3-30B-A3B-Instruct-2507-int4-AutoRound` → `pip install auto-round`.
- `DevQuasar/Qwen3-30B-A3B-Instruct-2507-W4A16-GPTQ` → `gptqmodel` (same as primary).
- `cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit` → `pip install autoawq` (AWQ MoE
  support is historically the flakiest — last resort).

The value graft is **quant-agnostic**: it operates on the KV cache (activations),
which stay bf16/fp16 regardless of int4 weights. Load with `SC_LOAD_DTYPE=auto`.

---

## EXACT ORDERED RUNNABLE COMMANDS (requirement / deliverable)

All launched via `scripts/launch_pod.sh <name> <job>`. `MODELS=` must be set so the
fail-closed pre-flight verifies the repo on the hub. Jobs detach; pull results with
the standard watchdog/auto-pull. **4-bit first.**

### 1) 4-BIT — derive champion + placebo-validate (one pod, unattended)
```bash
cd /Users/jeb/experimentation
MODELS="JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4" \
SC_HF_MODEL="JunHowie/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4" \
SC_TUNE_TAG="30b_4bit" SC_LOAD_DTYPE="auto" \
SC_CONV_START="6" SC_CONV_LIMIT="18" \
  scripts/launch_pod.sh champ4bit scripts/job_champion_4bit.sh
```
Produces on the pod → pulled to repo:
- `data/champion_configs/layers_30b_4bit.json` (the derived 4-bit champion — commit
  it).
- `results/champion_validate/champion_30b_4bit_{shuffle_pos,shuffle_probe,gauss}_<UTC>.json`
  each with `raw_EB` (E−B), `placebo`, and **`content_specificity.e_minus_placebo`**
  (the decisive clustered CI). Job prints the SUMMARY block at the end.

### 2) bf16 — placebo-validate the existing champion (one pod)
```bash
cd /Users/jeb/experimentation
MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" \
SC_HF_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507" \
SC_TUNE_TAG="30b_bf16" SC_LOAD_DTYPE="bfloat16" \
SC_CONV_START="6" SC_CONV_LIMIT="18" \
SC_CHAMPION_CONFIG="data/champion_configs/layers_30b_bf16.json" \
  scripts/launch_pod.sh champbf16 scripts/job_champion_bf16.sh
```
Produces `results/champion_validate/champion_30b_bf16_{…}_<UTC>.json`.

### 3) (optional) secondary arms — posslots / top16 under the same placebo
Point the champion config at a secondary and re-run (bf16 example):
```bash
MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" \
SC_HF_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507" SC_TUNE_TAG="30b_bf16_posslots" \
SC_CHAMPION_CONFIG="data/champion_configs/posslots_30b_bf16.json" \
SC_CONV_START="6" SC_CONV_LIMIT="18" \
  scripts/launch_pod.sh champposs scripts/job_champion_bf16.sh
```
(Expect posslots to FAIL the E−placebo CI — the built-in consistency check.)

### 4) (optional) bf16 RE-DERIVE instead of reuse
Run `job_champion_4bit.sh` with the bf16 model + `SC_LOAD_DTYPE=bfloat16
SC_TUNE_TAG=30b_bf16_rederive` to regenerate the layer profile + champion and
validate — confirms the shipped map reproduces under the current setup.

**Reading the result:** in each JSON, `content_specificity.e_minus_placebo` =
`{mean, lo, hi, n, n_clusters}`. CONFIRM if `lo > 0`; REFUTE if `lo ≤ 0 ≤ hi`.
Cross-check `raw_EB` (E−B) > 0 to confirm the champion recovers meaning at all.

---

## CORRECTNESS GOTCHAS (flagged)

1. **α_K = 0 (value-only) is structural, not a flag.** `blend_values` rewrites only
   V; keys are passed through untouched. No key graft is possible via this path.
2. **Corpus must match derivation.** The champion is tuned on the **pre-rendered**
   `data/synthetic` convs with **self-gen** summaries (`run_tune_hf`). Validation
   therefore uses `SC_NATIVE_RENDER=0 SC_SELFGEN=1` — NOT the native-render default
   (that renders a different corpus from the scenarios scaffold and would test a
   different distribution). Both jobs hard-set this.
3. **Checkpoint fingerprint.** The champion config is in the SCORE fingerprint only
   → a champion run reuses the render but never pools uniform-α scored rows.
   Canonicalised as sorted-key JSON so it survives the save/reload round-trip
   (in-memory int map keys → str on disk).
4. **Placebo seeding & render reuse.** All three placebo modes share the
   `--out-dir` render checkpoints, so the ~expensive render is generated once
   (first mode) and reused; each mode's result is copied to a UNIQUE
   `champion_<tag>_<mode>_<UTC>.json` (never a shared `summary.json`, per the
   output-naming rule). `shuffle_probe` disables *scored*-resume (needs live prev
   snapshots) but still reuses the render.
5. **MoE 4-bit loading.** Set `SC_LOAD_DTYPE=auto` for the quantized repo. Some GPTQ
   MoE exports quantize only the expert MLPs and keep attention in fp16 — fine (the
   graft is on attention V). **Positive-control the load:** the job prints
   `nvidia-smi` (VRAM ~½ of bf16 ⇒ genuinely 4-bit) and cross_arch's smoke gates
   (`alpha0_ok`, `graft_changes_output`) fire before any headline number — verify
   them, don't trust `status=OK` (house rule: read the number).
6. **Cache format.** The KV snapshot/rebuild uses `DynamicCache` (bf16/fp16 V rows)
   independent of weight quant. If a quant integration rejects the explicit dtype,
   `SC_LOAD_DTYPE=auto` is the escape hatch (already the 4-bit default).
7. **Per-quant champion is mandatory.** Do NOT reuse the bf16 α-map at 4-bit — the
   best layers differ by quant. STAGE 1 of the 4-bit job re-derives it; commit the
   resulting `layers_30b_4bit.json`.
8. **KV-head count.** `qwen3_moe` has 4 KV heads (0–3); the head-map configs
   reference exactly these — verified. A different arch/head count would need the
   head_map regenerated (not relevant for these Qwen repos).

---

## MINOR REFINEMENTS TO THE OWNER'S PLAN (no high-level change)

- **Estimator vehicle:** used `cross_arch_probe.py` (in-repo, already
  placebo+cluster-CI capable, `blend_values` already per-layer/per-head) rather than
  wiring the off-repo scalar-only `effect_bound_probe.py`. Cleaner + correct; same
  metric.
- **Added a *paired* E−placebo clustered CI** (`content_specificity`) instead of
  only comparing the two side-by-side CIs — a strictly stronger, less-noisy decisive
  statistic (cancels plant difficulty).
- **Validate out-of-sample (c07–c24)**, derive on VAL (c01–c06) — makes
  content-specificity un-confoundable with tuning overfit. (Owner said "recovery
  plants c01–c12 … c13–c24 too"; I recommend the held-out 18-cluster split and note
  the pooled fallback if the CI is marginal.)
- **Primary champion = `layers` per-layer α-map** (passed the old guard). `posslots`
  (biggest raw number but FAILED the old guard) + `top16` wired as **secondary
  arms** — `posslots` doubles as an internal consistency control (should fail the
  rigorous placebo). This directly tests the owner's cited +0.054/+0.030 numbers
  without letting the dead-but-larger config become the headline.
- **Rank-based derivation rule** (top 21% / next 35%) makes the 4-bit champion
  derivation scale-adaptive and exactly reproduces the shipped bf16 map — apples to
  apples across quants.
