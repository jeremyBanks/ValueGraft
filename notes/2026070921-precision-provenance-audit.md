# DEEP precision / data-provenance audit (2026-07-09) — where do we ACTUALLY stand on 4-bit vs bf16, and is the honesty "cornerstone" even the method we claim?

*Author: Fable, invoked as an independent integrity auditor. Method: read the on-disk
result `model`/`dtype` fields, the arm-construction code (`src/arms.py`, `src/arms_hf.py`),
the design + reframing notes, and the scored verdicts — no GPU/pods/MLX run. Where I am not
certain I say so explicitly. I independently re-derived every number below from the result
files; I did not take the invoking agent's summary on trust (and I caught one of its likely
errors — see §1 note on effect_bound "Qwen3.6-27B", which is real, not a mislabel).*

---

## TOPLINE (read this first)

1. **The precision truth is worse than the paper implies.** BOTH behavioral cornerstones that
   have been reported as headline — the **recovery** metric (judged sense/referent) AND the
   **honesty/fabrication** metric — are **4-bit MLX only** in every SCORED form. A bf16 honesty
   answer set exists (`results/honesty_30b_bf16`, `phase2_30b_bf16`) but has **never been
   scored**. So as of now there is **no scored bf16 behavioral honesty or recovery result at all**.
2. **The honesty result does not test ValueGraft.** The honesty experiment's intervention arm is
   **H-pack**, which retains **write-time keys (re-rotated) AND write-time values** for the summary
   tokens in a **packed sinks+summary layout with no conversation tail** (`arm_h_pack_snapshot`,
   `src/arms.py:285`). That is a **coupled KV-graft (α_K=1, α_V=1) in a non-production layout** —
   the reframing note itself calls it "Packed KV-Graft, auxiliary layout experiment"
   (`2026070646-controlled-key-graft-reframing.md:170`). **ValueGraft = value-only (α_K=0, α_V tuned)
   in the production summary+tail layout** (arm E, `arm_e_snapshot`, `src/arms.py:369`). **H-pack ≠
   ValueGraft.** They differ on BOTH the key axis and the layout axis.
3. **The honesty effect is not even isolable to write-time content.** The `H-pack-wrongS` control —
   H-pack built from *a different conversation's* summary state — suppresses fabrication just as much
   as real H-pack (24/24 ADMITTED on decoys, both 4B and 30B). Fabrication is suppressed by the
   **packed regime**, not by content-carrying value state. `B-min-pack` (fresh keys, same packed
   layout) already captures most of the admission gain. So the honesty result is a claim about a
   **packed-layout retention regime**, not about value grafting and not even about matching content.
4. **The only bf16 value-only ValueGraft signals are (a) small and (b) null-or-untested.** The
   SWE-Gym `E-tuned` effect is **+0.0156 nats** teacher-forcing logprob (Δ vs B, n=75), ~9.5% of the
   full-context gap (A−B=+0.164), and it does **not** change greedy generations (render-identical).
   The placebo-controlled bounding test (`effect_bound`) of the exact value-only graft on the
   recovery plants is **NULL on TWO bf16 models** (E−B CI includes 0 on both Qwen3.6-27B and
   Qwen3-30B-A3B).
5. **Where we stand:** Under the owner's rule (*cornerstone must be bf16 AND the current value-only
   method*), **this paper currently has no valid positive cornerstone.** The recovery cornerstone is
   4-bit and its placebo-controlled bf16 version is null. The honesty cornerstone is 4-bit AND a
   different intervention (packed coupled-KV, not value-only) AND not content-specific. The one bf16
   value-only positive (SWE-Gym +0.0156 nats) is small, TF-only, and un-CI'd.
6. **Recommendation:** Treat the paper as **honestly a negative / cautionary / bounding result**
   unless new bf16 value-only data lands. Scoring the existing bf16 honesty answers is **worth doing
   (cheap, CPU-only, and it either confirms the 4-bit honesty pattern at bf16 or reveals it as a
   quant artifact)** — but even a positive bf16 honesty result stays **SUPPORTING**, not a
   ValueGraft cornerstone, because H-pack is not the value-only method. See §4–§5.

---

## §1. Complete precision-provenance table

Precision/hardware read from each result's on-disk `model` (and `dtype` where present) field.
Convention verified across the tree: **local MLX runs carry `mlx-community/…-4bit` model ids**;
**pod/HF bf16 runs carry the bare HF repo id** (`Qwen/…`, `mistralai/…`) and, where the writer
included it, `"dtype":"bfloat16"`. Tuning/guard dirs omit the model field and rely on the
`_bf16` / `_4b` dir-name convention (flagged LOWER-CONFIDENCE below).

| Result (dir) | Metric / role | Model + quant | HW | Scored? | Evidence (file : field) |
|---|---|---|---|---|---|
| `raw` , `raw_brief` | recovery (judged sense/referent), 4B pilot | **Qwen3-4B-Instruct-2507 4-bit MLX** | local | scored (`judge_verdicts.json`) | `raw/c01.json` `model=mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| `raw_30b`, `raw_30b_brief`, `raw_brief_repro` | **recovery — the 30B "core" recovery data** | **Qwen3-30B-A3B-Instruct-2507 4-bit MLX** | local | scored (`judge_verdicts_30b.json`) | `raw_30b/c01.json` `model=mlx-community/…-2507-4bit` |
| `phase2_4b` | **honesty (fabrication/admission), 4B** | **Qwen3-4B-Instruct-2507 4-bit MLX** | local | scored (`phase2_4b_verdicts.json`) | `phase2_4b/c01.json` `model=mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| `phase2_30b` | **honesty — the 30B "cornerstone" honesty data** | **Qwen3-30B-A3B-Instruct-2507 4-bit MLX** | local | scored (`phase2_30b_verdicts.json`) | `phase2_30b/c01.json` `model=mlx-community/…-2507-4bit` |
| `honesty_30b_bf16` | honesty answers, bf16 (raw generations) | **Qwen3-30B-A3B-Instruct-2507 bf16 HF** | pod | **RAW, NOT SCORED** | `honesty_30b_bf16/c01.json` `model=Qwen/…-2507`, `dtype=bfloat16` |
| `phase2_30b_bf16` | reformat of the above into phase2 layout | **Qwen3-30B-A3B-Instruct-2507 bf16 HF** | pod | **RAW, NOT SCORED** (judge queue built, not judged) | `phase2_30b_bf16/c01.json` `dtype=bfloat16`; `phase2_30b_bf16_judge_queue.json` exists |
| `swegym_30b_bf16` | **SWE-Gym coding; E-tuned = value-only graft** | **Qwen3-30B-A3B-Instruct-2507 bf16** | pod | scored (tf_mean per task in-file) | `swegym_30b_bf16/t0001.json` `model=Qwen/…-2507`; runner sets `dtype=torch.bfloat16` `run_swegym_hf.py:111` |
| `cross_arch_done` | cross-arch value-only E-post sweep | **Qwen3-32B / Qwen2.5-32B / phi-4 / Mistral-Small-24B, bf16 HF** | pod | scored (in-file) | `cross_arch_done/Qwen__Qwen3-32B.json` `model=Qwen/Qwen3-32B` (no `-4bit`) |
| `cross_arch_wide` | cross-arch value-only, wide | **Mistral-Small-3.2-24B / Qwen3-30B-A3B, bf16 HF** | pod | scored | `cross_arch_wide/…2506.json` `model=mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| `effect_bound` | **placebo-controlled bounding of the VALUE-ONLY graft on recovery plants** | **Qwen3.6-27B (summary.json) AND Qwen3-30B-A3B (fullwin file), bf16** | pod | scored (bootstrap CIs in-file) | `effect_bound/summary.json` `model=Qwen/Qwen3.6-27B` + job `--model Qwen/Qwen3.6-27B` (`job_effect_bound.sh:46`); `effect_bound_qwen3-30b-a3b_fullwin_*.json` `model=Qwen/…-2507` |
| `longmemeval_4b` | LongMemEval | **Qwen3-4B 4-bit MLX** | local | scored (`longmemeval_4b_verdicts.json`) | `longmemeval_4b/001be529.json` `model=mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| `longmemeval_30b` | LongMemEval | **Qwen3-30B-A3B 4-bit MLX** | local | scored (`longmemeval_30b_verdicts.json`) | `longmemeval_30b/07741c44.json` `model=mlx-community/…-2507-4bit` |
| `longmemeval_30b_bf16` | LongMemEval | **Qwen3-30B-A3B bf16 HF** | pod | partially (judge batches present) | `longmemeval_30b_bf16/01493427.json` `model=Qwen/…-2507` |
| `tune_{sweep,layer,head}_30b_bf16`, `tune_eval_30b_bf16` | ValueGraft α tuning (value-only) | **30B bf16 (name convention; NO in-file model field)** | pod | scored (in-file metrics) | dir name `_bf16`; **model field ABSENT — LOWER CONFIDENCE** |
| `tune_*_4b_bf16`, `tune_*_gemma4b_bf16` | α tuning, 4B / gemma | **bf16 (name convention; field absent)** | pod | scored | dir name; **field ABSENT — LOWER CONFIDENCE** |
| `guard_layers_30b_bf16`, `guard_posslots_30b_bf16` | guard/robustness scans | **30B bf16 (name convention; field absent)** | pod | scored | dir name; **field ABSENT — LOWER CONFIDENCE** |
| `alpha_sweep_30b`, `alpha_sweep_4b` | early α sweeps | **4-bit MLX (no `_bf16`; local sweep pattern; field absent)** | local | scored | dir name convention; **field ABSENT — LOWER CONFIDENCE, but pattern-consistent 4-bit** |
| `alpha_sweep_qwen31.7b4bit`, `…8b4bit` | α sweeps | **4-bit MLX (name)** | local | scored | dir name `…4bit` |
| `gap_closure_cat*`, `concept_readout`, `head_profile_4b`, `layer_profile_4b`, `kv_sweep`, `kv_layer_probe` | mechanism/exploratory | mixed (`kv_layer_probe`/`kv_sweep` = 30B bf16 per field; `*_4b` = 4-bit; `gap_closure_cat` field absent) | mixed | scored | `kv_layer_probe/c01.json` `model=Qwen/…-2507` |
| `micro_sense.json`, `champion_harvest`, `redraw_harvest` | supporting | not re-verified here | — | — | (out of scope; flag if promoted) |

### 4-BIT-ONLY flags (cannot be a cornerstone under the owner's rule)
- **RECOVERY (judged sense/referent), all sizes:** `raw`, `raw_brief`, `raw_30b`, `raw_30b_brief`,
  `raw_brief_repro` → **4-bit MLX only.** No scored bf16 recovery exists. (The bf16 *placebo-
  controlled* recovery test is `effect_bound`, and it is NULL — see §3.)
- **HONESTY (fabrication/admission), all SCORED forms:** `phase2_4b`, `phase2_30b` → **4-bit MLX
  only.** The bf16 answers (`honesty_30b_bf16` / `phase2_30b_bf16`) exist but are **unscored**.
- **LongMemEval scored headline:** `longmemeval_4b`, `longmemeval_30b` → **4-bit MLX** (bf16 exists,
  scoring incomplete).

### Provenance corrections this audit establishes
- `effect_bound/summary.json`'s `model="Qwen/Qwen3.6-27B"` is **NOT a mislabel** — the job actually
  ran `--model Qwen/Qwen3.6-27B` (`job_effect_bound.sh:46`; commit `679145c` "effect-bound 27B
  result"). There are **two** effect_bound runs on **two different models** (27B and 30B-A3B), both
  bf16, both value-only, both NULL on E−B. The invoking brief's "Qwen3.6-27B?" suspicion is thus
  resolved: it's a real second model, not a field error.
- The paper/CLAIMS/FINDINGS labeling the 30B honesty result "bf16" is **wrong** — it is 4-bit
  (`phase2_30b/c01.json`). Already flagged by the invoking agent; independently confirmed here.

---

## §2. The H-pack question, resolved

**What H-pack actually does** (`arm_h_pack_snapshot`, `src/arms.py:285-305`;
HF twin `arm_h_pack_snapshot_hf`, `src/arms_hf.py:77-87`):
retains the summary tokens' **write-time keys, RE-ROTATED to packed positions** (`rotate_keys(...)`)
**and their write-time values**, concatenated after the 4 sink tokens — a contiguous
`[sinks][summary]` cache with **no conversation tail**. In (α_K, α_V) terms this is
**α_K = 1 (write-time keys) and α_V = 1 (write-time values)** — a **coupled/full KV-graft**, in a
**packed minimal layout**.

**What ValueGraft (arm E) actually does** (`arm_e_snapshot`/`arm_e_inter_build`, `src/arms.py:355-385`;
`blend_values` used by `run_swegym_hf.py:196`): takes a **freshly prefilled production compacted
context** ([system][summary-as-context-note][verbatim recent tail], `build_b_messages`, `src/arms.py:133`),
keeps **fresh keys untouched (α_K = 0)**, and blends **only values** at aligned summary-token
positions: `V ← (1−α_V)·V_fresh + α_V·V_write_time`. This is **value-only graft in the production
layout** — the method the paper is named after.

**Verdict: H-pack is a related cousin, NOT the same intervention, and on the confounded side of two
axes at once.** It differs from ValueGraft on:
1. **Key axis:** H-pack uses write-time keys (α_K=1, re-rotated); ValueGraft uses fresh keys (α_K=0).
2. **Layout axis:** H-pack is `[sinks][summary]` with **no tail**; ValueGraft keeps the production
   `[system][summary][recent tail]`. The reframing note explicitly names this layout/tail difference
   a **confound** (`2026070646-controlled-key-graft-reframing.md:73-77`) and instructs that packed
   arms "should not be used as if they isolate the effect of fresh keys versus write-time keys."

The design note is candid that H-pack was chosen as a *deployability* demo (contiguous,
prefix-cache-friendly), with `B-min-pack` as the within-layout fresh-KV control
(`2026070511-phase2-design.md:5-13`). The **canonical name mapping** is unambiguous
(`…reframing.md:163-174`): `H-pack → "Packed KV-Graft, auxiliary layout experiment"`;
`E → "V-only Graft"`. They are different rows of the table.

**Can the honesty result support "write-time KV state broadly"?** Only for a **packed-KV retention
regime**, and even then **not content-specifically**:
- **The layout, not the graft, drives it.** Both H-pack AND B-min-pack (fresh keys, same packed
  layout) collapse fabrication relative to production compaction B. On 4B decoys: B fabricates 18/24,
  B-min-pack 4/24, H-pack 3/24 (`phase2_4b_verdicts.json`, this audit's aggregation). The *packed
  regime* is doing the work; H-pack vs B-min-pack is a small, same-direction, noisy residual (on 30B
  decoys H-pack 3/24 FAB vs B-min-pack 10/24 — a real but single-contrast gap on 12 convs).
- **Content is not required.** `H-pack-wrongS` (write-time state from a *different* conversation)
  admits **24/24** decoys on both 4B and 30B — as honest as, or more honest than, real H-pack. So the
  admission behavior is triggered by the **presence of a packed write-time-state regime**, not by the
  values carrying the *right* content. This directly undercuts any "the retained value state carries
  the semantic knowledge that makes the model honest" reading.

**So: the honesty result is a claim about "packed keys+values retention induces admission over
fabrication," dominated by layout and not content-specific. It is NOT a claim about ValueGraft
(value-only, production layout) at all.**

---

## §3. Is there ANY robust positive that is BOTH bf16 AND value-only ValueGraft?

Enumerated candidates:

**(a) SWE-Gym `E-tuned` (bf16, value-only, production layout) — the main candidate.**
- Δ(E-tuned − B) = **+0.0156 nats** teacher-forcing logprob, n=75 tasks (this audit's paired
  aggregation of `swegym_30b_bf16/t*.json`; matches owner's +0.0156). Full-context headroom
  A−B = **+0.164 nats**, so E-tuned closes **~9.5%** of the gap.
- **Arm identity: correct.** `E-tuned` = `blend_values(b_snap, summary snapshot, pairs, α=0.75)`,
  fresh keys, value-only, production compacted context (`run_swegym_hf.py:196`). This **is** the
  method the paper claims to test. ✓ bf16 (`dtype=torch.bfloat16`, `run_swegym_hf.py:111`). ✓
- **NOT render-robustness tested.** The metric is `tf_mean` = mean teacher-forced logprob of the
  gold next-assistant turn. It is **not** a generation/behavioral outcome. For t0001 the greedy
  `gen` strings for A, B, and E-tuned are **byte-identical** — the +0.0156 nats does not cross the
  threshold to change decoded output. No task-solve / pass-rate / behavioral delta is measured.
- **No CI in-file.** It's a raw mean over 75 tasks; I found no bootstrap/CI for the SWE-Gym Δ. Given
  the effect is ~1/10 the size of the A−B gap and TF-only, its robustness is **unestablished**.
- **Assessment:** real, correctly-armed, bf16, value-only — but **small, TF-only, un-CI'd, and
  render-null.** Not cornerstone-grade on its own.

**(b) `effect_bound` — placebo-controlled bounding of the value-only graft on the recovery plants
(bf16, value-only). THE decisive bf16 recovery test.**
- Design: A=full, B=fresh-compacted, **E = B + aligned summary-token VALUES grafted (α_V=0.75, keys
  bit-identical)**, P = same positions with a norm-matched **deranged/random** value graft; paired
  bootstrap 95% CIs (`effect_bound/summary.json:design`). This is the cleanest, most honest test of
  value-only ValueGraft on the sense/referent metric.
- **Result — NULL on both bf16 models:**
  - Qwen3.6-27B (k=12): Δ(E−B) mean **+0.0166**, CI **[−0.033, +0.065]** → includes 0 → **NULL**.
  - Qwen3-30B-A3B (k=48 full window): Δ(E−B) mean **−0.0485**, CI **[−0.113, +0.014]** → includes 0
    → **NULL** (point estimate negative).
  - What IS significant: E ≫ placebo (content-specificity, +0.44 CI excl. 0) and placebo ≪ B
    (a random value graft actively harms). So value grafting is **content-specific and non-destructive
    relative to a random graft, but not distinguishable from doing nothing (plain compaction).**
- **Assessment:** the best bf16, value-only, placebo-controlled, CI'd recovery test in the repo is a
  **preregistered NULL** on E−B (`verdict:"NULL"`, `effect_bound/summary.json:summary.verdict`). This
  is the single most important number in this audit for the recovery cornerstone.

**(c) `cross_arch_done` / `cross_arch_wide` (bf16, value-only E-post).** Scale/diversity evidence
that value-only graft behaves consistently across architectures. Supporting/motivation, not a
cornerstone effect-size claim; not placebo-controlled or CI'd here in a headline form.

**Conclusion for §3:** The only bf16 value-only *positive* is SWE-Gym +0.0156 nats — small, TF-only,
un-CI'd, render-null. The one bf16 value-only *placebo-controlled recovery* test is **NULL** (twice).
**There is no robust, cornerstone-grade positive that is both bf16 and value-only ValueGraft.**

---

## §4. Should we score the existing bf16 honesty answers (`phase2_30b_bf16`)?

**Yes — score them. It is cheap (CPU/LLM-judge only, no GPU), the answers already exist, and the
judge queue is built (`phase2_30b_bf16_judge_queue.json`).** It resolves a real open question and is
strictly informative either way:
- If the bf16 arms reproduce the 4-bit pattern (B fabricates; packed arms admit; wrongS admits) →
  the honesty phenomenon is **not a quantization artifact**, strengthening it as **supporting**
  evidence and removing the "it's only 4-bit" objection.
- If they do NOT reproduce → the 4-bit honesty result is partly a **quant artifact**, which is
  itself a critical finding that must gate any use of the honesty data.

**But even a fully-positive bf16 honesty result is NOT a valid ValueGraft cornerstone**, for the §2
reasons: the arm is **H-pack (packed coupled-KV, no tail)**, not value-only ValueGraft, and the
effect is **layout-driven and content-non-specific** (`H-pack-wrongS` admits everything). Scoring
bf16 honesty upgrades the honesty result's *precision provenance*; it does **not** convert it into a
test of the paper's named method. **Recommendation: score it, report it as SUPPORTING (a packed-KV
retention-regime behavioral phenomenon, precision-robust if it holds), never as the ValueGraft
cornerstone.**

---

## §5. WHERE WE ACTUALLY STAND — honest verdict

Applying the owner's rule literally — **the cornerstone must be BOTH bf16 AND the current value-only
ValueGraft method** — the paper as currently built **does not have a valid positive cornerstone:**

- The **recovery** cornerstone (judged sense/referent) is **4-bit MLX**, and its **bf16
  placebo-controlled version is NULL** (`effect_bound`, twice). It cannot be the cornerstone.
- The **honesty** cornerstone is **4-bit MLX**, tests a **different intervention** (H-pack = packed
  coupled-KV, not value-only), and is **layout-driven and not content-specific**. It cannot be the
  ValueGraft cornerstone regardless of precision.
- The **only bf16 value-only positive** is **SWE-Gym +0.0156 nats** — small (~9.5% of the A−B gap),
  **teacher-forcing-only, render-null, un-CI'd.** Too thin to be a cornerstone.

**Honest characterization: this is, right now, a negative / cautionary / bounding paper** with a
mechanistic sub-story, not a positive "ValueGraft recovers lost continuity" paper. What it can
*honestly* claim:
1. **A tight, preregistered, placebo-controlled bound on value-only ValueGraft for recovery: the
   effect is content-specific (E ≫ random-value placebo) and non-destructive, but NOT distinguishable
   from plain summary compaction (E−B null) at bf16.** This is a genuine, publishable, honest result —
   as a *bound*, not a *win*.
2. **A behavioral phenomenon in the packed-KV retention regime:** packing write-time summary
   keys+values (with sinks, no tail) shifts the model from fabrication toward admission — but this is
   **layout-driven and content-agnostic** (wrongS reproduces it), so it is a finding about
   *compaction regime and calibration*, not about value-carried semantics. 4-bit today; score bf16 to
   de-risk the quant confound.
3. **A small bf16 value-only TF signal on real coding trajectories (SWE-Gym +0.0156 nats)** — enough
   to motivate that value-only graft is directionally non-harmful and slightly helpful on
   next-token prediction, not enough to claim behavioral recovery (renders unchanged).
4. **Scale/diversity/mechanism supporting evidence** (cross-arch consistency, α/layer/head tuning,
   own-summary mechanism) — motivation and trajectory, explicitly 4-bit where local.

**Recommended next steps (no GPU needed for the first two):**
1. **Score `phase2_30b_bf16`** (CPU/judge) to settle the honesty quant-artifact question; report as
   supporting.
2. **Re-frame the paper around the honest bound + the packed-regime honesty phenomenon**, demote
   "recovery cornerstone" language, and state the precision provenance table (§1) explicitly per
   METHODS-PROVENANCE-REQUIREMENTS.md.
3. **If a positive bf16 value-only cornerstone is wanted, it must be NEW data:** a
   render/behavior-scored, CI'd, value-only (α_K=0) test in the production layout that beats B with
   CI excluding zero. Nothing on disk clears that bar today. (Requires GPU — owner decision.)

---

## INVALID vs VALID (by status)

**INVALID as a cornerstone (do not present as the core positive claim):**
- ❌ **Recovery (judged sense/referent), any size** — 4-bit MLX only; bf16 placebo-controlled version
  (`effect_bound`) is NULL. Demote to *supporting/motivation* at most, with the 4-bit label and the
  bf16-null bound stated alongside.
- ❌ **Honesty / fabrication-vs-admission (H-pack)** as a ValueGraft claim — it is a *different*
  (packed coupled-KV) intervention, 4-bit, layout-driven, content-non-specific. Not a ValueGraft
  cornerstone at any precision.
- ❌ **LongMemEval 4-bit headline** as a precision-clean cornerstone — 4-bit; bf16 scoring incomplete.

**VALID, at the stated status:**
- ✅ **`effect_bound` NULL bound (bf16, value-only, placebo-controlled, CI'd)** — VALID as the paper's
  honest **headline result IF the paper is framed as a bound/negative**: value-only ValueGraft is
  content-specific and non-destructive but not distinguishable from plain compaction on recovery.
- ✅ **SWE-Gym `E-tuned` +0.0156 nats (bf16, value-only, correct arm)** — VALID as **supporting**
  (small, TF-only, render-null, un-CI'd; do not call it behavioral recovery).
- ✅ **Honesty / packed-KV admission phenomenon** — VALID as **supporting** (a compaction-regime
  calibration finding), *conditional on scoring the bf16 set*; label 4-bit until then; never as
  value-only ValueGraft.
- ✅ **Cross-arch consistency, α/layer/head tuning, own-summary mechanism** — VALID as
  **motivation / mechanism / trajectory** (mark 4-bit vs bf16 per §1; tuning dirs are bf16 by
  name-convention — LOWER confidence, no in-file model field).

*Certainty notes: precision/arm/scored-status claims above are cited to on-disk fields or code lines
and I am confident in them. The three LOWER-CONFIDENCE items are the `tune_*`/`guard_*`/`alpha_sweep_*`
dirs whose JSONs omit a `model` field — their precision is inferred from the `_bf16`/`_4b` dir-name
convention, not a stored field; verify against their launch scripts before citing precision in the
paper. Everything else is field- or code-grounded.*
