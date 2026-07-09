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
