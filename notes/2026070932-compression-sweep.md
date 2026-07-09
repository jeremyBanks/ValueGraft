# Compression sweep — does the value-graft effect concentrate under aggressive compaction?

**Status:** BUILT + queued, NOT launched (owner/coordinator fires it when a pod frees).
**Owner funding:** ~$25 top-up. One A100 80GB. Est. wall ≈ 4-6 h for the 4-level ×
3-placebo grid (well under budget); ~+1.5 h if `prod` is added as a 5th level.

## The question (honest, pre-registered)
The per-layer champion is **NULL** under the realistic self-gen summary
(`raw_EB` CI [-0.041, +0.043], "null/underpowered"). But the project's positive
headlines — judged **+12pp**, SWE-Gym **+0.0156** — lived under **aggressive/BRIEF**
summaries. Open hypothesis: is there a **monotonic** relationship — does `raw_EB`
(and content-specificity `E − placebo`) go positive as the summary gets more lossy?

Both outcomes are legitimate findings, reported with CIs:
- **Concentrates under aggressive compaction** → the effect is compaction-severity-gated
  (a real, interesting scope result worth fleshing out).
- **Flat null across all compression levels** → also a real result; the champion doesn't
  recover meaning at this held-out set regardless of how much the summary evicts.

No p-hacking: held-out c07–c24 only, full placebo battery, conversation-clustered CIs,
honest about small per-model n.

## Design — vary ONLY the summary compression level; hold everything else fixed
| Held fixed | Value |
|---|---|
| model / dtype | `Qwen/Qwen3-30B-A3B-Instruct-2507`, **bf16** (`SC_LOAD_DTYPE=bfloat16`) |
| summary source | **DECLARED self-gen** (`SC_SELFGEN=1`) — the graft needs the model's OWN summarization act (FINDINGS 07-08). `SC_NATIVE_RENDER=0` (pre-rendered corpus; matches how the champion was derived) |
| corpus / split | held-out recovery plants **c07..c24** (`SC_CONV_START=6 SC_CONV_LIMIT=18`), disjoint from tuning VAL |
| champion | per-layer `data/champion_configs/layers_30b_bf16.json` (overridable — see swap note) |
| placebo | **full battery**: `shuffle_pos`, `shuffle_probe`, `gauss` |
| metric | `raw_EB` (PRIMARY) + `content_specificity.e_minus_placebo`, both conversation-clustered bootstrap CIs, **per level** |

### The compression axis (`SC_SUMMARY_LEVEL`, monotonic: least → most summary text)
Implemented in `src/cross_arch_probe.py` (`compression_levels()` / `resolve_summary_level()`);
`SC_SUMMARY_LEVEL` overrides the single `_REQ` used by every self-gen call site. Unset
or `realistic` = byte-unchanged current behavior.

| Level | Request | Target words | Regime | Expected ratio* |
|---|---|---|---|---|
| `ultra` | constructed 1-sentence | ≤25 | very-aggressive / most lossy | ~0.003–0.006 |
| `brief` | `SUMMARY_REQUEST_BRIEF` (existing) | ~70 | aggressive (the +12pp/+0.0156 regime) | ~0.008–0.015 |
| `medium` | constructed | ~150 | moderate | ~0.02–0.04 |
| `realistic` | `SUMMARY_REQUEST` (existing) | ~300–500 | realistic — **the current NULL** | ~0.04–0.09 |
| `prod` (opt) | `SUMMARY_REQUEST_PROD` (existing) | ~300–500 | faithful, OpenHands-style | ~0.04–0.09 (faithfulness contrast, not a distinct ratio) |

\* **Expected only.** The harness **MEASURES** the realized ratio per conv
(`clean-summary tokens / full-context tokens`, reasoning block stripped — what actually
lands in the compacted B context) and reports `doc.compression.ratio_mean` +
`_manifest.condition.measured_compression_ratio_mean`. So "aggressive vs realistic" is a
NUMBER; `raw_EB` and `E−placebo` are reported AS A FUNCTION of that ratio. Full-context
tokens for c07..c24 are on the order of a few thousand, so the ratios above are estimates
— trust the run's measured values.

`ultra` + `medium` are constructed intermediates that vary ONLY the length/detail budget
in the same instruction style as the existing requests, so the axis is length, not wording
style. `prod` is the same length as `realistic` (a faithfulness contrast), so it is
optional and not part of the core monotonic axis — include it to check whether a
*faithful* realistic summary differs from the *generic* realistic one.

## Correctness / born-annotated provenance
- **No cross-level summary reuse.** The self-gen summary IS part of the render; a level
  change changes the summary. The summary-request sha256 is now in BOTH the render and the
  score fingerprints (`render_fingerprint` / `run_fingerprint`), AND each level writes to
  its own out-dir (`_work_compsweep/<level>/`). Belt and suspenders — a level can never
  pool another level's summary.
- **Render reuse WITHIN a level.** The 3 placebo modes at one level share the out-dir, so
  the self-gen summary is generated ONCE per level and reused across placebos (champion +
  placebo live in the SCORE fingerprint only). This is what keeps the sweep cheap.
- **Manifest.** Every result + per-conv checkpoint carries `_manifest` with
  `condition.summary_compression_level`, `.summary_compression_regime`,
  `.summary_target_words`, `.summary_request_sha256`, and (backfilled after the loop)
  `.measured_compression_ratio_mean`. `doc.compression` carries the per-conv rows +
  ratio_min/mean/max + mean summary/context tokens.

## Swapping in a per-head champion (coordinate, don't block)
If the running per-head scan (pod headscan) produces a champion that BEATS the per-layer
one on held-out `content_specificity`, re-run the sweep with that config — everything else
identical — by setting `SC_CHAMPION_CONFIG`:
```
SC_CHAMPION_CONFIG=data/champion_configs/heads_30b_bf16.json   # or heads_x_layers_..._union.json
```
Do this as a SECOND sweep (keep the per-layer sweep as the registered primary); don't block
the primary sweep on the head-scan finishing.

## How to run — born-annotated launch command(s)
The pod is provisioned + the job launched by `scripts/launch_pod.sh <name> <job.sh>` (it
`nohup`s the job and the ssh returns). Forward the controlled variables as leading env
assignments so they are DECLARED, not defaulted:

**Primary (4-level core axis, per-layer champion):**
```bash
SC_HF_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
SC_LOAD_DTYPE=bfloat16 \
SC_TUNE_TAG=30b_bf16 \
SC_CONV_START=6 SC_CONV_LIMIT=18 \
SC_CHAMPION_CONFIG=data/champion_configs/layers_30b_bf16.json \
SC_LEVELS="ultra brief medium realistic" \
scripts/launch_pod.sh compsweep scripts/job_compression_sweep_bf16.sh
```
(Writes `.pod_compsweep_state.json`; results → `results/champion_validate/compsweep_30b_bf16_<level>_<mode>_<STAMP>.json`; the job prints the compression-vs-effect curve at the end.)

**Optional — add the faithful `prod` point (5 levels):** set `SC_LEVELS="ultra brief medium realistic prod"`.

**Optional — per-head champion sweep (after head-scan lands a winner):** same command with
`SC_CHAMPION_CONFIG=data/champion_configs/heads_30b_bf16.json` and a distinct pod name
(e.g. `compsweep-heads`).

## How to read the curve
The job's final block prints, per level: measured `ratio`, mean summary tokens, `raw_EB`
[lo,hi], and `E−placebo` [lo,hi] (primary placebo = `shuffle_pos`), flagged `*CI>0*` when a
lower bound clears 0; then sorts by measured ratio (most-aggressive first) and reports the
trend.
- **Compaction-severity-gated:** `E−placebo` (and `raw_EB`) rise monotonically as the ratio
  shrinks, AND a CI lower bound clears 0 at the aggressive end (`ultra`/`brief`).
- **Flat null:** effect ~0 with CIs spanning 0 at every level → the champion does not
  recover meaning on held-out c07–c24 regardless of compression severity.
- Read `raw_EB` together with `pre_graft_gap` (lp_A − lp_B): a bigger available gap at
  aggressive compression is EXPECTED (more meaning evicted) — the interesting signal is
  whether the graft *recovers* more of it, placebo-controlled, not just that the gap grows.

## Files
- `scripts/job_compression_sweep_bf16.sh` — the sweep job (loops levels × placebos).
- `src/cross_arch_probe.py` — `compression_levels()` / `resolve_summary_level()`;
  `SC_SUMMARY_LEVEL` override of `_REQ`; summary-request sha in both fingerprints;
  per-conv compression measurement + `doc.compression` + manifest backfill.
- This note.
