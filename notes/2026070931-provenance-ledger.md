# Provenance ledger — key existing artifacts audited (2026-07-09)

**Why:** agents kept answering "yes, we have data for X" when the data was the wrong
model / dtype / intervention / metric / condition, because result files self-document
almost nothing and provenance was **inferred from directory names**. This ledger
reconstructs the provenance of the key existing artifacts from launch settings + code
+ git, marking each controlled variable **PROVEN** (recorded in the artifact itself)
vs **INFERRED** (reconstructed from a dir name / job default / code literal), and
lists the gaps. The instrumentation added this session (`src/provenance.py` + manifest
stamping in the four harnesses) closes these going forward; this ledger covers data
already on disk, which was written BEFORE the manifest existed.

Legend: **PROVEN** = the value is written inside the result file / a committed
manifest, read from the runtime. **INFERRED** = reconstructed after the fact from the
dir name, the job-script default, or a hardcoded code literal — the failure mode we
are eliminating.

---

## 1. `results/swegym_30b_bf16/` — the "+0.0156 SWE-Gym anchor" (75 trajectory files)

Recompute (this session): mean(E-tuned − B) tf_mean = **+0.01564**, 45/75 wins,
95% CI [+0.0047, +0.0266] (CLAIMS REC-6). Audited fields of the on-disk files
(`t*.json`, top-level keys: `idx, meta, model, summary_tokens,
failed_evicted_commands, n_target_tokens, arms`):

| Controlled variable | Value | Status | Source of truth |
|---|---|---|---|
| model repo id | `Qwen/Qwen3-30B-A3B-Instruct-2507` | **PROVEN** | `model` field in every file |
| resolved weights revision/commit | unknown | **GAP** | not recorded anywhere |
| load dtype | bf16 | **INFERRED** | dir name `swegym_30b_bf16` + code path; **the `dtype` field is ABSENT from these 75 files** (they predate even the code's `"dtype":"bfloat16"` literal, which itself was not a runtime read) |
| quantization | none (full bf16) | **INFERRED** | code loads `dtype=torch.bfloat16`, no quant config; not recorded |
| intervention arm | `E-tuned` | **PROVEN (name only)** | `arms."E-tuned"` key in file |
| graft type / α_K | value-graft, keys neutral | **INFERRED** | `blend_values` only rewrites V; not recorded |
| alpha | **scalar α=0.75** (NOT the tuned champion) | **INFERRED** | `E_ALPHA` default `0.75` in `run_swegym_hf.py`; the name "E-tuned" is **misleading** — it is a flat scalar graft, not `data/champion_configs/layers_30b_bf16.json` |
| alignment mode | difflib positional-within-region | **INFERRED** | code (`build_alignment`); not recorded |
| metric | teacher-forced mean logprob of true next action (**PROXY, not resolve-rate**) | **PROVEN (values) / INFERRED (framing)** | `arms.*.tf_mean` present; the "proxy, not test-pass" nature is not stated in-file |
| condition: summary kind | **brief = handicapped** | **INFERRED** | job default `SC_SUMMARY=brief`; **not in the file AND not in the dir name** (the dir has no `_brief` suffix — the current code writes `swegym_<tag>_<summ>` but this data predates that) |
| summary-request hash | — | **GAP** | not recorded |
| corpus / split | SWE-Gym/OpenHands parquet, shard 0/1 | **INFERRED** | code default `PARQUET=swegym.parquet`, `SC_SHARD=0/1` |
| N + instance ids | N=75; idxs | **PROVEN** | 75 files, each `idx` field |
| code git commit | unknown | **GAP** | not recorded |
| timestamp / GPU | unknown | **GAP** | not recorded |

**Verdict:** the anchor is real and recomputes, but of the six controlled variables
the owner flagged, only **model** and **metric-values** are PROVEN in-artifact. **dtype,
intervention-alpha, and the brief condition are all INFERRED** — precisely the class of
error that started this. The headline must be stated as: *teacher-forced logprob proxy,
scalar value-graft α=0.75 (not the tuned champion), bf16, brief (handicapped) summary.*

---

## 2. `data/champion_configs/` — the tuned champions (content-hashed here for the first time)

| File | sha256 (first 16) | What it is | Provenance |
|---|---|---|---|
| `layers_30b_bf16.json` | `3d7d244162554522` | per-layer coarse {0.75,1.0} α-map (27 layers), **PRIMARY** champion, passed the 07-06 wrong-conversation guard | derived by `run_tune_hf.py` PHASE=layer on VAL (c01-c06,n01-n04); label + note in file **PROVEN**; derivation run's model/dtype **INFERRED** (tuning outputs carried no manifest) |
| `posslots_30b_bf16.json` | `ee553fcc3971f352` | 57 (layer,kv-head) slot mask @ α=1; **FAILED** the wrong-conversation guard, kept as near-negative control | same; the note self-documents its negative-control role **PROVEN** |
| `top16_30b_bf16.json` | `8d80749e14d9d18d` | top-16 slot variant | derivation manifest **GAP** |

These configs pin the tuned map by content (the value-graft is only as reproducible as
the exact α-map), but the **run that derived them recorded no manifest** — the model id /
dtype / VAL split behind each map is INFERRED from the tag `30b_bf16` and
`run_tune_hf.py` defaults. The new `run_tune_hf.py` instrumentation records this on the
NEXT derivation.

---

## 3. Currently-running jobs (bf16, primary model) — provenance reconstructed from launch env

Owner's launch env for both: `SC_HF_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`,
`SC_LOAD_DTYPE=bfloat16`, `SC_TUNE_TAG=30b_bf16`, `SC_CONV_START=6 SC_CONV_LIMIT=18`
(held-out c07..c24). **NOT touched** (per constraint): the pod state files.

### 3a. Per-layer champion validation — pod `champbf16` (`.pod_champbf16_state.json`, A100 80GB, CA-MTL-3)
`scripts/job_champion_bf16.sh` → `cross_arch_probe.py --champion-config
data/champion_configs/layers_30b_bf16.json --placebo {shuffle_pos,shuffle_probe,gauss}`,
`SC_NATIVE_RENDER=0 SC_SELFGEN=1`. Decisive quantity = `content_specificity.e_minus_placebo`
(held-out, conversation-clustered CI).

| Variable | Value | Status (for the OUTPUT once written) |
|---|---|---|
| model / dtype | Qwen3-30B-A3B-Instruct-2507 / bf16 | **PROVEN** (new manifest reads param dtype at runtime) |
| intervention | per-layer champion `layers_30b_bf16.json` (sha `3d7d24…`) vs placebo | **PROVEN** (manifest records path + content hash + label) |
| metric | raw_EB = lp_E − lp_B, teacher-forced (proxy) | **PROVEN** |
| condition | self-gen prod summary (`SUMMARY_REQUEST`), pre-rendered corpus | **PROVEN** |
| corpus/split | c07..c24 held-out (start=6, limit=18) | **PROVEN** (manifest `instance_ids`) |

### 3b. Per-head champion scan — pod `headscan` (`.pod_headscan_state.json`, A100 80GB, EU-RO-1)
`scripts/job_headscan_bf16.sh`: STAGE1 `run_tune_hf.py` PHASE=head (derive per-(layer,kv-head)
profile on VAL) → STAGE1b `head_select.py` → STAGE2 `cross_arch_probe.py` placebo-validate
{heads, layers, intersection, union} on c07..c24. Same model/dtype/condition provenance
status as 3a; every STAGE2 output now carries the champion config path+hash of the specific
config being validated.

**Note:** both jobs were launched with the NOW-instrumented harnesses only if the pods
pulled the current commit. If they were launched before this commit, their outputs will
lack `_manifest`; re-pool or re-run after syncing to stamp them. The reconstruction above
is the authoritative provenance in the interim.

---

## 4. Gaps summary (what is still not captured, even with the new manifest)

- **Existing 75 swegym files + existing champion configs cannot be retro-stamped** with a
  runtime dtype/revision — they were produced without a live model in hand now. Their
  provenance lives in THIS ledger; do not re-derive dtype from the dir name in prose.
- **Weights `resolved_revision`** is only captured when transformers records
  `config._commit_hash` (hub load). A local-path load leaves it None — acceptable, but note
  it in the paper's reproducibility section (pin by repo id + the date-stamped checkpoint).
- **prod-summary SWE-Gym cell = NONE on disk.** See the controlled matrix (§ deliverable 4)
  and the born-annotated launch command (returned to the owner).

---

## 5. Self-gen vs fixed summaries — authoritative design history + compression regime (2026-07-09 follow-up)

**Trigger:** the bf16 per-layer champion validation (pod champbf16) came back null
(raw_EB CI [-0.041,+0.043], "null/underpowered") and its result recorded
`summary_source: "PER-MODEL fallback (NOT cross-model comparable)"`, which read like an
accidental missing-file fallback. Authoritative reconstruction from the design docs +
code follows.

### Q1 — Was PER-MODEL SELF-GEN the FINAL intended design? YES, decisively.
Fixed/shared summaries were abandoned ON PURPOSE for this mechanism/champion work.
- **FINDINGS.md "✅✅ RESOLVED + MECHANISTIC FINDING: graft needs the model's OWN summary
  (07-08)"** (lines 483-503): same model + harness, only summary source differs —
  FIXED Sonnet summary → referent **+0.004 (null)**, POSITIVE CONTROL FAILED; SELF-GEN →
  referent **+0.136 CI[+0.034,+0.23]**, SIGNIFICANT. Explicit conclusions: *"(2)
  FIXED-SUMMARY DESIGN BROKEN → sweep switches to SELF-GEN summaries. (3) MECHANISTIC
  FINDING … ValueGraft re-injects the write-time state of the model's OWN summarization
  ACT — a summary the model merely READ doesn't carry the recoverable continuity."*
- **DECISIONS.md 2026-07-08** (line 187): *"Do NOT change the metric mid-cross-arch
  (raw_EB, shared gold, bootstrap-over-convs, **native+self-gen**) — changing it loses
  comparability to the validated +0.10."* → native + self-gen is the LOCKED design.
- **INCIDENTS.md #32/#33** (lines 425-464): self-gen is named *"the redesign's PRODUCTION
  path (fixed summary suppresses the graft)."*
- History (why fixed ever existed): DECISIONS 2026-07-07 21:00 (line 144) introduced ONE
  shared Sonnet summary purely as a **cross-model confound control** (different models
  write different-quality summaries → baseline gap varies). That control turned out to
  BREAK the mechanism, so it was retired. It was never the intended final condition — and
  for a **single-model** champion validation the cross-model-comparability rationale does
  not even apply, so self-gen is doubly correct here.

**Answer:** self-gen is the final intended design; there is NO reason a shared/fixed
summary should have been used for the champion validation. The `"NOT cross-model
comparable"` clause in the label is irrelevant to a within-model E-vs-placebo test.

### Q2 — Was self-gen reached correctly, or via a fragile missing-file path?
**Reached CORRECTLY, by explicit declaration — the coordinator's missing-file hypothesis
is NOT what happened.** The code DOES read `SC_SELFGEN` (`cross_arch_probe.py:4765`), and
BOTH jobs set `SC_SELFGEN=1` (`job_champion_bf16.sh:47`, `job_headscan_bf16.sh:112`). The
`force_selfgen` branch (line 4793) short-circuits BEFORE the fixed-file load — so whether
`data/fixed_summaries.json` was rsynced to the pod is IRRELEVANT; self-gen was declared.

The real defect was **PROVENANCE LABELING**: `summary_source` was computed only from
`native_render` + `fixed_summaries is not None`, so a DECLARED self-gen run
(fixed_summaries forced None) fell through to the last branch and mislabeled itself
`"PER-MODEL fallback (NOT cross-model comparable)"` — indistinguishable from the genuine
missing-file accident. That mislabel is what made a correct, intended run look accidental.

**Hardening applied (committed):** `run_model` now takes `selfgen_declared` (threaded from
`force_selfgen`) and labels declared self-gen as *"PER-MODEL SELF-GEN summary (DECLARED
SC_SELFGEN=1; redesign default)"*, reserving the "ACCIDENTAL … NOT comparable" wording for
the true `native_render=0 AND SC_SELFGEN unset AND no fixed file` path (which already emits
a loud stderr warning). The provenance manifest now records
`condition.summary_selfgen_declared: true` and `condition.summary_compression_regime`. The
jobs ALREADY declare `SC_SELFGEN=1` (no job edit needed); the fix makes the DECLARATION
visible in every result + manifest instead of masquerading as a fallback.

### Q3 — Which compression regime is the null in? REALISTIC-LENGTH (moderate), not aggressive.
The cross_arch self-gen summary uses **`SUMMARY_REQUEST`** (`cross_arch_probe.py:1815`) =
*"a thorough context note … roughly 300-500 words"* — the **realistic-length** regime. The
champion-validation null therefore lives under MODERATE compression, NOT the aggressive
lossy regime. This is a first-class axis because the POSITIVE headlines live in a DIFFERENT
regime:

| Result | Summary request | Compression regime | Summary source | Outcome |
|---|---|---|---|---|
| SWE-Gym +0.0156 anchor | `SUMMARY_REQUEST_BRIEF` (3-5 sentences, no details) | **AGGRESSIVE / lossy (handicapped)** | self-gen | POSITIVE |
| Judged +12pp (sense) | BRIEF | AGGRESSIVE / lossy | self-gen (MLX 4-bit) | POSITIVE (render-fragile, CLAIMS REC-5) |
| Logprob referent +0.125/+0.136 | `SUMMARY_REQUEST` (~300-500w) | realistic-length | self-gen | POSITIVE (referent), sense underpowered |
| **Per-layer champion validation (bf16)** | `SUMMARY_REQUEST` (~300-500w) | **realistic-length** | self-gen (**declared**) | **NULL / underpowered** (agg raw_EB CI spans 0, held-out c07-24) |

Caveat against over-reading: the +0.136 referent effect was ALSO under realistic-length
`SUMMARY_REQUEST` self-gen, so "realistic summary" alone does not kill the raw effect. The
champion null is a HARDER, aggregate, held-out content-specificity test (E-champion vs
placebo on c07-24), and is underpowered — distinct from the raw referent lift. The clean
open question worth fleshing out (separate thread): does the graft benefit CONCENTRATE
under aggressive compaction (BRIEF), where more meaning is evicted (larger A-B headroom to
recover)? A `summary_request × compression-level` sweep (BRIEF vs SUMMARY_REQUEST vs PROD)
at matched N would answer it. Recorded here so the matrix carries the axis.

### Controlled-matrix addition — summary source × compression regime
| Summary source \\ regime | AGGRESSIVE (BRIEF) | REALISTIC (SUMMARY_REQUEST ~300-500w) | FAITHFUL (PROD, OpenHands-style) |
|---|---|---|---|
| **self-gen (own)** — REQUIRED by mechanism | HAVE: swegym +0.0156, judged +12pp (POSITIVE) | HAVE: logprob referent +0.125/+0.136 POSITIVE; **champion validation NULL/underpowered** | **NEED** (prod SWE-Gym cell; born-annotated cmd in report) |
| **fixed/foreign (Sonnet)** — retired, SUPPRESSES graft | (isolation test) | referent +0.004 null (FINDINGS 07-08) | n/a — abandoned by design |

**Bottom line for the owner:** the champion validation used the RIGHT condition
(self-gen, declared) in the REALISTIC-length regime; the "fallback NOT comparable" label
was a provenance bug, now fixed. The null is a real, honestly-reportable
underpowered-aggregate result in the moderate-compression regime; the positive headlines
sit under aggressive-compaction (BRIEF). Whether the effect is compaction-severity-gated is
a clean, worthwhile follow-up, not a contradiction.

---

## 6. Champion-on-SWE-Gym — the tuned map now testable on the coding task (2026-07-09)

**Gap closed:** `run_swegym_hf.py` previously grafted ONLY at scalar α=0.75 (its
"E-tuned" arm is a misnomer — flat α, not the champion), so the conversation-tuned
champion was never tested on SWE-Gym. Added `SC_CHAMPION_CONFIG` support: when set, the
runner loads the config via `load_champion_graft_cfg` (the same loader cross_arch uses) and
adds a distinct **`E-champion`** arm applying the tuned per-LAYER `alpha_map` (or per-HEAD
`head_map`), value-only (α_K=0), ALONGSIDE the scalar `E-tuned` arm — so one run yields both
numbers vs B for a direct paired comparison.

**Born-annotated recording (no ambiguity about which intervention made which number):**
- Each result's `arms.E-tuned.intervention` = `{kind implicit scalar, alpha:0.75, champion:null}`;
  `arms.E-champion.intervention` = `{alpha:null (per-layer map) or scalar, champion:{config_path,
  config_sha256, label, family}}`.
- The run manifest's `intervention.grafted_arms` lists both arms with the champion path +
  content sha256 + label; the startup log prints `CHAMPION arm ACTIVE/INACTIVE`.

**Honest framing (report either way):** this tests whether the champion tuned on the
SYNTHETIC CONVERSATION corpus (VAL c01-c06,n01-n04) TRANSFERS OUT-OF-DOMAIN to the coding
task, and whether it beats the naive scalar α=0.75 **where the effect appears** (the BRIEF /
aggressive-compaction regime — see §5). Plausible outcomes, all reportable:
1. Champion transfers and beats scalar under BRIEF → the tuned per-layer profile is
   domain-general (strong result).
2. Champion ties/underperforms scalar → the tuning is corpus-specific / doesn't transfer to
   code (also informative — argues the scalar graft is the robust shippable form).
3. Both null under PROD (faithful) → consistent with the compression-severity story (§5):
   little evicted meaning to recover under a faithful summary.
The head-to-head `E-champion − E-tuned` paired CI (printed by the job) is the arbiter; do
NOT assume transfer — the champion was tuned on conversations, SWE-Gym is a different domain.
