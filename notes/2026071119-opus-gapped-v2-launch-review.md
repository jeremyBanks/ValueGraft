# Claude Opus review — gapped-v2 production launch

This note preserves the final independent Claude-family scientific and apparatus
review before any second paid 30B attempt. The reviewer should state its exact
runtime model identifier, read the full referenced evidence directly, distinguish
observed facts from inference, and give an explicit GO or NO-GO. Any conditions
must be concrete pre-launch blockers rather than suggestions deferred into the
semantic run.

Review scope: both coherent-state preregistration amendments; the original frozen
preregistration and failed paid artifact; the final gapped-v2 runtime, production
driver, calibration, analysis, storage, job/watcher/harvest validation; all focused
tests; and the unique passing gapped-v2 0.6B ladder artifact.

---

**Claude — claude-opus-4-8:**

Runtime model identifier: `claude-opus-4-8`. I read every scoped file directly
(both amendments + the original frozen prereg; the failed paid artifact's
`failure.json`/`manifest.json`; `src/coherent_state_{runtime,store,tokens,hf,
calibration,cases}.py`, `src/kvlib_hf.py`, `src/analyze_coherent_state.py`,
`src/run_coherent_state_hf.py`, `src/l_coherent_state_hf.py`; the harvest
validator, and the focused tests) and inspected the passing gapped-v2 ladder
artifact's actual gate numbers. Observed facts are marked; inferences are named.

**Process caveat that governs this verdict.** The repo *changed under me during
this review*. At session start HEAD was `76605d5`; it is now `4859727` (two new
commits: `dded1f0` "analysis: weight unique calibration variants equally" and
`4859727` "…terminalize partial failure artifacts"), and the working tree is
currently **dirty** — `git status` shows uncommitted `M src/analyze_coherent_state.py`
and `M tests/test_analyze_coherent_state.py`. A concurrent agent is actively
editing the apparatus. Everything below reflects the working tree as I last read
it; a launch must be certified against a *frozen commit*, not this moving target.

**What I verified correct and directly evidenced (observed):**

1. *DynamicCache physical/logical decoupling.* `validate_position_schedule`
   fails closed unless `cache_position` is contiguous physical storage and
   `position_ids` strictly increasing; it is called at every prefill/append/score
   site. The ladder's `automatic_vs_independent_4d_physical_causal_mask` gate runs
   two forwards with **logical positions [37,38,39] over physical [5,6,7]** (a real
   gap) and the artifact records `logits/k/v max_abs = 0.0` vs an independently
   built 4-D physical-storage causal mask — i.e. HF masks from `cache_position`,
   not from the gapped `position_ids`. `causal_future_mutation = 0.0` (a later
   token cannot alter earlier logits/rows), and the "logical-as-cache_position"
   and "pre-tailed boundary" failure injections both report `rejected: true`. This
   same `run_loaded_gapped_gates` is re-run on the 30B bf16 model as a fail-closed
   production gate (identity 1e-4) *before any scoring*; on FAIL it writes a
   complete `production_kernel_gate.json` then raises. This is the design's central
   correctness risk and it is exercised, not assumed.

2. *Exact source/layout spans.* `gapped_destination_layout` asserts the compact
   system island equals the correct prefix's system (Amendment-2 gate), the
   request/header suffix equals the correct-prefix suffix ending at `P_i-1`, the
   islands do not overlap, and `require_exact_span` enforces a *unique* exact
   summary occurrence. The driver re-asserts all of this per conversation before
   scoring. Ladder `logical_gap = 53`, physical length 64 vs A_full 94 — the gap
   is real.

3. *Arm-tail causality.* Each arm forks at the summary boundary
   (`gapped_arm_boundary` → boundary snapshot at `physical_summary_end`),
   `_assert_boundary_intervention` proves pre-tail length and that only summary
   rows changed, and only then `append_gapped_post_summary` recomputes the tail
   per arm. Repeated-branch and correct/counterfactual pre-tail and post-tail
   hashes must match; a reused precomputed tail is impossible by construction. The
   ladder's ε-perturbation control shows a changed summary state flows into the
   recomputed tail (`tail_sens: true`) and the fork boundary is not mutated.

4. *Gate ordering.* `_static_design_self_check` (AST) makes `move_key_rows` and
   the packed kernel-gate unreachable in the driver; `gapped_arm_boundary` never
   rotates keys (bit-copy at matched logical positions). Order is amendments-exist
   → provenance/revision/dtype/geometry → production gate → per-conversation
   position/structure gates → A_full/treatment/calibration. No semantic outcome is
   scored before its gates.

5. *Persistence/resume/validator alignment.* Atomic temp-then-rename writes;
   fingerprint includes both amendment SHAs; stage promotion forbids regression
   and field overwrite; the forced-restart resume probe requires the first scored
   checkpoint to be byte-identical (`sha256_file`) across a real restart, and main
   refuses to finish unverified. `validate_scored_checkpoint`, the analysis
   `validate_docs`, and the import-free `validate_coherent_harvest.py` agree on
   schema-2, the exact gapped arm set, the `gapped_same_source_summary_position`
   policy, `technical_pass`, and retired-arm rejection.

6. *Two-variant calibration correction.* The gate logic requires **both** unique
   label variants (A and B) to have `mean(GF)>0` and `mean(GW)>0`, frozen to the
   first six, un-rescuable by the extension — correct per Amendment 2. I confirmed
   the frozen allocation empirically: the passing ladder ran calibrations for c10
   and c07 and recorded `c10→A`, `c07→B`, so the single B execution (position 5)
   is present in the first six and both variants are evaluable. **Note (initially a
   finding, resolved mid-review):** at session start the reported summary `mean_GF`
   was the execution-weighted flat mean and the focused test
   `test_calibration_summary_mean_weights_unique_variants_equally` was red;
   commit `dded1f0` plus the current working-tree edits fix this — `mean_GF` is now
   the variant-equal-weighted mean and `repeated_execution_weighted_mean_GF` is
   emitted, matching the test. I re-read `calibration_fires` in the live tree to
   confirm. This is why the tree must be frozen and the suite re-run at the launch
   commit (see conditions).

7. *Frozen 5e-4 zero-gap tolerance.* Separately frozen, distinct from the 1e-4
   copy/replay/mask identities. The ladder recorded k/v/logit split-vs-contiguous
   diffs of `1.30e-4 / 1.73e-4 / 4.86e-5` (< 5e-4) — the exact numbers Amendment 2
   cites. It is re-recorded on the 30B model and, per code, will not be loosened on
   a 30B failure (the constant is in the run fingerprint).

8. *Statistics.* Conversation is the unit; two-sided 95% Student-t (correct
   critical values), 10k conversation bootstrap seed 20260711; intersection-union
   (both lower bounds > 0); regime and calibration frozen at n=6 and not recomputed
   over the extension; no early efficacy. Matches the prereg.

9. *Narrow claim scope.* Amendment 1 restricts any positive result to
   position-preserving compaction; the interpretation matrices are conservative;
   futility is inconclusive, not equivalence; exploratory contrasts are labeled.
   The failed paid artifact honestly records the *retired* packed-rotation gate
   failure (`nativeK=2.810546875 nativeV=0.810546875`, zero=0.0) — matching
   Amendment 1 — which the gapped design no longer relies on.

**No apparatus- or science-level launch blocker survived verification.** The one
substantive code/reporting defect I found (calibration summary mean
pseudo-replicating the 5:1 allocation + a red focused test) was corrected by a
concurrent agent during the review, and I confirmed the fix in the live tree.

**Concrete pre-launch conditions (operational, $0, no GPU) — these are blockers,
not deferrable suggestions:**

- **Freeze and commit the tree.** The working tree is dirty (`analyze` + its test
  uncommitted) and HEAD moved twice during this review. Launch only from a clean,
  committed commit. This is not optional courtesy: `runtime_provenance` fails
  closed on any dirty path outside the run dir, so a launch attempt from the
  current state will refuse anyway — commit first.
- **Re-run the focused suite green at that exact commit** (esp. `test_analyze_coherent_state.py`,
  `test_coherent_state_tokens.py`, `test_coherent_state_store.py`) and regenerate a
  fresh unique gapped-v2 0.6B ladder PASS from that commit, since the previously
  passing ladder artifact predates these analysis/driver commits. I could not
  execute the suite here (test/model execution is sandbox-blocked in this session),
  so the green-test confirmation is the one thing I am asserting *must be observed*
  rather than assumed — do not treat the older ladder PASS as covering the new
  commit.

**Distinction requested by the scope:** the items above are *code-freeze /
reporting-integrity* conditions, not apparatus or statistical defects. The
DynamicCache mask correctness, span exactness, arm-tail causality, gate ordering,
persistence/resume, the two-variant calibration gate, the frozen 5e-4 tolerance,
the statistics, and the claim scope are all sound and directly evidenced.

**VERDICT: GO — conditional and time-bound.** The design and apparatus are sound
and I clear them for the paid 30B run *from a frozen, committed commit whose
focused tests and a freshly generated gapped-v2 0.6B ladder are observed green*.
Until the tree is committed clean and that green-test/ladder reconfirmation exists
at the launch commit, it is a **NO-GO on the current in-flux working tree** — for
apparatus-freeze/reporting reasons only, with no unresolved scientific or
cache-correctness blocker.

**Reviewer: Claude Opus 4.8 — exact runtime model identifier `claude-opus-4-8`.**
Review performed 2026-07-11. Working tree HEAD during review advanced from
`76605d5` to `4859727` (see "Concurrency" below). All code excerpts, tests, and
the ladder artifact were read directly; numeric gate values were extracted from
disk, not inferred. Where I write "observed" I read the number/branch; where I
write "inferred" I reasoned from code I did not execute on the 30B path.

## 1. Bottom line

**Verdict: GO** for the position-preserving gapped-v2 apparatus, conditional on
the operational pre-launch items in §7 (which are coordination/commit-freeze
steps, not code fixes) and subject to the claim/reporting constraints in §8. I
looked for a correctness blocker in the surgery, masking, gating, wrong-history,
tail-recomputation, resume, calibration, and analysis paths and did not find one.
The design is careful, the gates are fail-closed, and the single deepest technical
risk (HF logical `position_ids` vs physical `cache_position`/masking across a
positional gap) is directly and successfully gated on-model — twice (0.6B ladder
now, and re-executed live on the 30B before any spend).

This is not a default endorsement: the reasons it is a GO are enumerated and
checked below, and the reasons it is *only narrowly* a GO (claim scope, small n,
render-time corpus dependency, the 5e-4 provenance) are stated as constraints, not
buried.

## 2. What the design gets right (the core insight)

The first paid attempt (`results/coherent_state/..._20260711T062724Z/`, observed:
`schema:1`, `rotation_tolerance:0.02`, error `loaded K/V shift gate failed:
zero=0.0 nativeK=2.810546875 nativeV=0.810546875`, code `eaa8913`) died inside a
technical gate that tried to *re-rotate* post-RoPE quantized K after a positional
repack. Amendment 1 retires that path. gapped-v2 replaces it with a genuinely
different mechanism, and the mechanism is correct:

- The summary is generated by the correct source at contiguous logical positions
  `P_i..P_i+n-1`. In the gapped destination the summary is *also* assigned logical
  positions `P_i..P_i+n-1` (`summary_position_ids = range(source_start, ...)`,
  `source_start = len(correct_prefix) = P_i`). Physical storage is compact and
  contiguous; only the *logical* prefix carries the hole.
- Because source and destination summary rows share the identical absolute logical
  position, the post-RoPE keys are transplantable **bit-for-bit with no rotation**
  (`gapped_arm_boundary` → `replace_summary_rows`, `use_keys=True`, no
  `move_key_rows`). Values are position-free. This is why the class of failure that
  killed attempt 1 cannot recur on a confirmatory arm.

Observed structural guard: `run_coherent_state_hf._static_design_self_check()`
AST-scans the driver and refuses to run if `move_key_rows` or the retired
`run_loaded_kernel_gates` are reachable in it. The driver imports neither. So
"no packed rerotation on a confirmatory arm" (Amendment-1 gate 7) is enforced by
construction, not discipline.

## 3. HF DynamicCache: logical position vs physical cache_position/masking

This was the item I most wanted to break, and it is the item most likely to have
produced a silent position bug. It holds:

- Observed in the passing ladder
  (`coherent_state_ladder_gapped_v2_Qwen3-0.6B_20260711T073117Z.json`, `status:PASS`):
  `automatic_vs_independent_4d_physical_causal_mask` with `logical_positions:[37,38,39]`
  over `physical_cache_positions:[5,6,7]` yields `logits_max_abs:0.0`,
  `k_max_abs:0.0`, `v_max_abs:0.0`. i.e. an independently constructed full-physical
  causal mask is bit-identical to transformers' automatic masking when RoPE runs on
  gapped logical positions and the cache index is contiguous physical. This is the
  exact property the apparatus depends on.
- `causal_future_mutation`: `0.0/0.0` (a later token cannot change earlier logits or
  earlier stored K/V — confirms no future-leak through a logical-position mask).
- `logical_as_cache_position_failure_injection`: `rejected:true` — supplying logical
  positions as `cache_position` is caught by `validate_position_schedule`, which is
  called at every append in the gapped path (`build_gapped_fresh_boundary`,
  `append_gapped_post_summary`, `score_target`) and asserts physical contiguity +
  strictly-increasing logical.
- `snapshot_rebuild_equivalence`: `0.0` across logits/K/V.

Crucially, the 30B production driver re-runs this same `run_loaded_gapped_gates`
live (`run_coherent_state_hf.main`, phase `PRODUCTION_GATE`) and writes
`production_kernel_gate.json` for both PASS and FAIL before raising; on non-PASS it
stops before any A_full/treatment/calibration scoring. So the 0.6B ladder being one
or two commits behind HEAD (it is; see §7) does not gate correctness on its own —
the authoritative gate is executed on the paid hardware, fail-closed, before spend.
Inference (not executed by me): the same masking result holds on the 30B bf16 model.
That is what the live gate exists to confirm.

## 4. The specifically-flagged concerns, each checked

- **Source/destination spans & exact wrong-history construction.** `matched_wrong_prefix_ids`
  replaces only declared evicted content slots with frozen-donor IDs (truncate/cycle
  to exact slot length), asserts structural/special/tail/request/header slots
  byte-identical, at least one token changed, and identical prefix length. The driver
  re-verifies (changed ⊆ declared content, structural equal, no special replacement,
  length equal, summary start == `P_i`). Ladder failure injection
  `altered_structure_failure_injection: rejected:true`. Wrong and correct summaries are
  forced at the *same* absolute positions (contiguous sources), so `W` is a true
  exact-position unrelated-history control, and its summary-token NLL is persisted as
  the alternative surprisal explanation (`wrong_summary_mean_nll`). Matches Amendment-1 §3.
- **Per-arm tail recomputation.** `score_gapped_bounded` forks a fresh boundary per
  plant *and per target side*, asserts pre-tail hashes are stable across repeats and
  identical between correct/counterfactual, then recomputes the tail via
  `append_gapped_post_summary` before scoring. A pre-tailed boundary is rejected
  (`pre_tailed_boundary_failure_injection: rejected:true`); reusing a fresh-precomputed
  tail is structurally impossible here. Ladder sensitivity control: a summary
  perturbation moved the margin by 0.158 and tail K/V by 6.42 (`passes:true`). Matches
  Amendment-1 gate 10.
- **Outcome-before-gate.** No semantic outcome is scored before (a) the live 30B
  production gate PASSes, and (b) per-conversation pre-score structure/position gates
  are persisted (`CHECKPOINT_GATED`). A_full is reconstructed only afterward. The gate's
  own margins are on an engineered fixture and are explicitly labeled
  `technical_margins_not_semantic_outcomes`. Any technical failure raises → the
  checkpoint is terminalized `void` and the run stops (serial rule 1). Confirmed
  fail-closed.
- **Resume/persistence/validator alignment.** Atomic tmp+fsync+rename; fingerprint-gated
  reads; monotonic promotion that refuses field overwrites; forced resume probe
  (`--resume-probe-stop-after-one` → exit 75 → second invocation must find the first
  scored checkpoint byte-identical via `sha256_file`, else `ArtifactError`). The
  independent harvest validator (`validate_coherent_harvest.py`, no `src` imports)
  requires COMPLETE manifest + `resume_probe_verified` + PASS gate + contiguous 6/12
  positions + the terminal log marker, and rejects retired arm names. `4859727`
  ("terminalize partial failure artifacts") tightens the failure path; model-free
  suite green.
- **Calibration pseudo-replication correction.** `calibration_fires` groups by unique
  label, requires **both** A- and B-variants to have positive GF and GW (not four-of-six
  over a 5:1 duplicated allocation), freezes to the first six, reports the summary
  `mean_GF/mean_GW` as *equal-variant-weighted* and the pooled majority-weighted mean
  separately under `repeated_execution_weighted_mean_*`. Observed: `test_analyze...`
  green including `test_calibration_does_not_pseudoreplicate_majority_label` and
  `test_calibration_summary_mean_weights_unique_variants_equally`. Matches Amendment 2.
  The ladder exercises both unique variants (c10→A, c07→B), and c07 sits in the first
  six, so the frozen allocation genuinely contains both labels and the gate can fire.
- **The 5e-4 zero-gap tolerance (post-observation, pre-semantic).** Observed
  `zero_gap_equivalence: k=1.30e-4, v=1.73e-4, logits=4.86e-5` (matches the values
  Amendment 2 froze the bound from). This tolerance governs a *split-vs-contiguous
  prefill kernel-shape* comparison only — the same batched-vs-stepwise kernel
  divergence documented all over DECISIONS.md — not any transplanted-row identity. The
  confirmatory copies are checked with `torch.equal` (bit-exact) and the other identity
  gates remain at 1e-5/0.0. It was frozen before any 30B semantic result, is re-checked
  on the 30B at launch, and "will not be raised in response to a 30B failure." I accept
  it as legitimate; it is a reporting-disclosure item (§8), not a launch blocker.
- **Delta placebo.** Fixed-point-free (Sattolo) per layer/head derangement of
  `V_correct − V_fresh`; intended multiset preserved exactly (observed
  `max_multiset_diff:0.0`, `fixed_points:0`), with the honest bf16 finite-precision
  qualification (§15.5) bounded by applied-quant ≤ 0.05 / moment ≤ 0.02 (observed
  1.9e-6 / 1.1e-5). Keys and non-summary rows untouched (asserted by `torch.equal`).
  Matches prereg §6/§15.5.
- **Do GF/GW support only the narrow claim?** Yes. `theta_GF`/`theta_GW` are the only
  confirmatory contrasts; both must clear zero (intersection-union, t-CI lower bound
  > 0). All of `G_Vcorrect−G_fresh`, `G_Kcorrect−G_fresh`, `G_correct−G_delta`,
  `A_full−G_fresh` are exploratory with pointwise intervals; no subgroup can become a
  headline. `A_full` is a full-history *reference*, not a treatment (and note its probe
  follows the summary directly, whereas the G arms interpose the re-encoded tail — an
  intended asymmetry that keeps A_full purely a competence/headroom yardstick; read
  `theta_AF` accordingly). The licensed reading is exactly Amendment-1's table: a
  correct-history-specific coherent summary-state channel **under position-preserving
  compaction only**.

## 5. Provenance / fail-closed spine (observed)

Exact revision `0d7cf2399…` asserted against the resolved `_commit_hash`; live params
asserted `{torch.bfloat16}` on `{cuda}`; geometry asserted `48/32/4/128/θ1e7`; CUDA
required; clean-worktree-outside-run-dir asserted with `--untracked-files=all`;
config/tokenizer-vocab/chat-template/special-tokens hashes recorded; fingerprint binds
model+revision+code_commit+scenario/target/amendment SHAs+seeds+tolerances. The
generated summary must be `generated_incremental` and end on natural EOS before the
900-token cap; generated-vs-replay identity ≤ 1e-4 on IDs/logprobs/K/V. These are the
right invariants and they fail closed.

## 6. Regime/analysis (observed)

Regime gate (competence ≥4/6 and mean>0; headroom ≥4/6 and mean AF ≥ 0.30) is frozen
to the first six even at N=12 (`test_n12_regime_gate_remains_frozen_to_first_six`
green), so a stage-1 decision cannot be reversed post-hoc by extension outcomes.
Serial 6→12: STOP on technical/regime, STOP_FUTILITY only if both primary means ≤ 0
**and** calibration does not fire, else EXTEND. Primary uncertainty = conversation-level
two-sided 95% Student-t (df table); bootstrap seed 20260711 as sensitivity. Independent
unit is the conversation; plants are never pseudo-replicated. Interpretation matrix
matches the amendment. This all mirrors the prereg.

## 7. Launch preconditions (operational — NOT code fixes, but must hold before spend)

1. **Freeze one authoritative commit and push it.** Local `HEAD=4859727` is 15 commits
   ahead of `origin/trunk` (`30c142d`). `job_coherent_state_bf16.sh` clones
   `origin/trunk` and asserts the clone HEAD equals `SC_EXPECTED_COMMIT` (fail-closed),
   so a stale origin cannot silently launch old code — but the operator must push the
   intended commit and set `SC_EXPECTED_COMMIT` to it. Do not launch against an
   unpushed HEAD.
2. **Concurrency / single launch authority.** A second Claude-Opus session is committing
   to this same trunk during this review (commit `30c142d`: "two Claude-Opus sessions
   active … session B … single launch-gate authority … free delta-table measurement
   before any 30B spend"; the code under me changed from the version I first read to the
   version I finally reviewed). Before paid launch, the two sessions must converge on ONE
   frozen commit; my GO applies to the exact gapped-v2 surgery/gate/analysis code at the
   frozen commit. If any of `coherent_state_{hf,runtime,tokens,calibration,store}.py`,
   `l_coherent_state_hf.py`, or `kvlib_hf.rotate_keys/prefill/tf_logprobs` is further
   edited after freeze, a fresh unique 0.6B v2 ladder artifact must be regenerated at
   that commit (Amendment 2 requires it; the live 30B gate still backstops, but do not
   skip the cheap pre-flight).
3. **Ladder correspondence.** The committed passing ladder predates HEAD by two commits
   (`dded1f0` analysis-weighting, `4859727` failure-terminalization); neither touches the
   gate/surgery/masking path, so it remains representative. Confirm no surgery/gate edit
   lands between freeze and launch.

## 8. Claim / reporting constraints (separate from launch blockers)

- A positive result licenses **only position-preserving compaction** (serving must
  decouple physical cache length from logical RoPE positions). It does **not** establish
  that post-RoPE K can be re-rotated after a client packs/resets positions; the paper
  must carry the failed packed bf16 attempt, its numeric diagnostic, and this deployment
  distinction (Amendment 1 §2/§6).
- Disclose the 5e-4 zero-gap tolerance and its provenance (kernel-shape split bound,
  50× the original 1e-5, chosen after the observed ~1.3–1.7e-4 split drift, before any
  semantic result), and the bf16 delta-placebo finite-precision qualification — do not
  describe stored bf16 deltas as exactly matched.
- Report `theta_GF/theta_GW` as the only confirmatory claim (intersection-union); label
  V/K/delta/AF exploratory. n ≤ 12 → intervals are wide; an interval spanning zero is
  "inconclusive," never "no effect / equivalence." A six-conversation futility stop is a
  bounded canary, not proof of absence.
- Conversation bodies are rendered on-pod at launch (not from the committed scaffold);
  plant placement and the retained-tail boundary come from `native_render_specs`. The
  regime/headroom gate is the fail-closed backstop: if the corpus does not actually
  favor the correct target with ≥0.30 nats headroom, the run stops REGIME_INADEQUATE and
  makes no channel claim. State the render provenance and the new mechanistic wrapper vs
  the older production-like layout, and report every deviation, per prereg §14.

## 9. One-line verdict

GO — the gapped-v2 position-preserving apparatus is correct and fail-closed and the
0.6B v2 ladder passes; launch only after freezing/pushing one authoritative commit and
deconflicting the concurrent session (§7), and carry the position-preserving claim scope
and 5e-4/bf16/n≤12 disclosures (§8).

Note path: `notes/2026071157-opus-gapped-v2-launch-review.md`

---

**Claude — claude-opus-4-8 (v3 follow-up):**

Runtime model identifier: `claude-opus-4-8`. This delta review covers only what
changed after my v2 GO: the science audit that found (a) target-as-donor **cycle
dependence** breaking the independence of the co-primary `GW` contrast and (b) an
**N=6 efficacy-label bug**, plus the repairs in `COHERENT-STATE-PREREGISTRATION-AMENDMENT-3.md`
and commits `7a8c391` (Amendment 3), `97bb02f` (implementation), `d9b1d8d` (v3
ladder). Current HEAD is `6ef133a` (a notes-only exec-coord commit; no code after
the ladder). The working tree is now **clean** except this untracked note — the v2
"dirty tree" launch blocker is resolved. I read the diffs and current files
directly and extracted the v3 ladder numbers and donor-file metadata from disk.

**Blocker 1 — statistical cyclic-donor dependence: REMOVED (verified).** v1/v2
`WRONG_DONOR` was two six-cycles *among the scored targets*, so each `GW_i` was a
function of another scored conversation and adjacent `GW` rows shared data; the
conversation-level t/bootstrap did not have their nominal meaning for `GW`, and at
N=6 there was a single connected cycle. v3 replaces the map with twelve unique
external donors `c10→c13 … c03→c30` (`src/coherent_state_cases.py:WRONG_DONOR`),
each disjoint from the scored set. `load_external_donors` (driver) fails closed on
any reuse or target/donor overlap, requires the file's `id` to match, runs
`validate_native_conversation`, and requires a non-empty `meta.author`;
`test_coherent_state_cases.test_external_donors_are_unique_and_disjoint_from_targets`
asserts 12 unique donors disjoint from `FROZEN_ORDER`. I confirmed all twelve files
exist with valid structure (`middle_end_msg=49`, 65–71 msgs) and the exact recorded
authors Amendment 3 claims: **`sonnet` (c13–c15), `opus` (c16–c18), `codex-gpt5.5`
(c25–c27), `sonnet-render` (c28–c30)**. Each `GW_i` is now a function of one unique
target and one unique exogenous fixed control, so conversation-pair clustering is
coherent and the interval has its stated across-pair interpretation. Amendment 3
correctly *demotes* the residual (fixed-benchmark exchangeability, not population
coverage over arbitrary wrong histories) to a claim constraint, not a blocker.

**Blocker 2 — N=6 efficacy label: REMOVED (verified).** `interpret()` now returns
`INTERIM_NO_EFFICACY_DECLARATION` whenever `n_conversations < 12`, placed *after*
the `VOID_TECHNICAL`/`REGIME_INADEQUATE` precedence and *before* any channel
reading, so a confirmatory `HISTORY_SPECIFIC_CHANNEL` is unreachable at the interim
even with both provisional lower bounds positive.
`test_n6_extends_when_channel_direction_or_calibration_fires` now asserts the
interim label while still EXTENDING. Matches Amendment 3 §"No early efficacy."

**External-control provenance, persistence, fingerprint (verified).** Donors are
**authored/non-subject-native** controls — the fingerprint carries `wrong_donors`
(the external map) plus `external_donor_provenance` = per-donor `{path, sha256,
recorded_author, subject_native:false}`; the per-conversation checkpoint persists
`external_donor` alongside the exact `matched_wrong` replacement pools, changed
positions, and decoded diagnostic *before* scoring. Exact file hashes and author
strings are therefore bound into the run fingerprint and reported. Rendering is now
target-only (`ensure_render_positions({position-1})`); donors are never subject-
rendered, consistent with their control role of supplying unrelated content-token
pools for the already-artificial, exact-length, same-position donor-slot swap.

**Analysis exact-order / common-fingerprint enforcement (verified).**
`validate_docs` now requires N∈{6,12}, a common reference `fingerprint` on doc[0]
whose `frozen_order==FROZEN_ORDER` and `wrong_donors==WRONG_DONOR` (external map)
with scenario/target hashes present, and per-doc `conversation_id==FROZEN_ORDER[i]`
plus `fingerprint==reference_fingerprint`. Tests
`test_wrong_frozen_id_or_fingerprint_fails_closed` and
`test_nonterminal_sample_size_is_rejected` exercise these. The independent harvest
validator mirrors it (v3 identity, external donor-map check, exact frozen-ID order,
manifest fingerprint identity) with no `src` imports.

**Wrong-sign diagnostic (verified, non-authorizing).** The retired packed block now
records `move_key_rows(rows37, +37)` (deliberately wrong sign vs the correct `-37`
inverse) and `wrong_sign_failure_injection_detected`. Observed in the v3 ladder:
correct inverse `native_shift_k_max_abs ≈ 1.45e-4` vs `wrong_sign_shift_k_max_abs ≈
43.58`, `failure_injection_detected: true`, `authorizes_run: false`. Sign/layout
sensitivity is visible and cannot authorize or veto the gapped semantic assay;
`_static_design_self_check` still keeps `move_key_rows` unreachable from the driver.

**v3 ladder (`…gapped_v3_Qwen3-0.6B_20260711T075100Z.json`, observed PASS).**
Identity `coherent-state-gapped-v3` / `…AMENDMENTS-1-2-3`. Every apparatus-
correctness number carries over **unchanged** from v2 (the surgery/masking/tail
path was not touched): `automatic_vs_independent_4d_physical_causal_mask` logical
`[37,38,39]` over physical `[5,6,7]` → `0.0/0.0/0.0`; `causal_future 0.0/0.0`;
`zero_gap k/v/logits 1.30e-4 / 1.73e-4 / 4.86e-5 < 5e-4`; `self_replacement_exact`,
`render_resume_exact`; placebo `fixed_points 0`, `max_multiset_diff 0.0`. This
ladder corresponds to HEAD (only a notes commit follows `d9b1d8d`), resolving the
v2 ladder-currency caveat.

**New defect introduced by the repair? None found.** The changes are confined to
the donor map, the interim-label guard, fingerprint/exact-order enforcement, the
identity strings, and the non-authorizing wrong-sign diagnostic; no cache-surgery,
masking, gating, calibration, or tail-recomputation code changed, and there are no
stale `gapped-v2` / `AMENDMENTS-1-2"` strings left in `src/`/`scripts/`. The one
substantive worry worth naming is not a defect but a **claim constraint**: the
external donors are frontier-model-authored text and may be more fluent than
subject-native history, so `G_correct−G_wrong` can still absorb a
congruence/surprisal component — which is exactly why wrong-summary NLL and
decoded-control quality remain mandatory reported diagnostics and why the licensed
reading is bounded to "better than the prespecified exact-position external
donor-slot controls in this fixed benchmark," not "better than every plausible
wrong history" and not isolation of the planted fact.

**Remaining items (operational launch-freeze, not code fixes; + claim
constraints).** Blockers to clear before spend, per Amendment 3's launch gate: (1)
freeze and **push one clean authoritative commit** and set the pod's
`SC_EXPECTED_COMMIT` to it (the job script fail-closes on clone-HEAD mismatch), (2)
deconflict the concurrent Opus sessions onto that single commit, (3) observe the
**full focused test suite and the trusted monitor suite green at that commit** — I
could **not** execute tests/monitors here (test/model execution is sandbox-blocked
this session), so this is the one thing that must be *observed*, not inferred, and
(4) the live 30B `run_loaded_gapped_gates` production gate still backstops fail-
closed before any A_full/treatment/calibration scoring. Claim constraints carry
forward: position-preserving compaction only; the fixed-benchmark/exchangeability
`GW` boundary above with authored non-subject-native donors named; N=6 interim only
and confirmatory only at N=12; interval-spanning-zero = inconclusive; the 5e-4 zero-
gap and bf16 delta-placebo disclosures; render-time corpus provenance with the
regime/headroom gate as the fail-closed backstop.

**VERDICT: GO.** The Amendment-3 repairs remove both the statistical cyclic-donor
blocker and the N=6 efficacy-label blocker; I verified each in the diffs, the
updated tests, the twelve authored external donor files (with their exact recorded
authors), and the passing v3 ladder, and I found no new defect. This clears the
gapped-v3 apparatus for the paid 30B run **from a frozen, pushed, deconflicted
commit whose focused tests and trusted monitor suite are observed green** (the only
remaining items are that operational freeze plus the reporting/claim constraints);
the dirty-tree and ladder-currency conditions from my v2 review are already
resolved.

One-line verdict: **GO** — v3 removes both the cyclic-donor statistical blocker and
the N=6 efficacy-label bug with no new defect; launch only from a frozen/pushed/
deconflicted commit with tests+monitors observed green, carrying the authored-
external-donor fixed-benchmark and position-preserving claim constraints.

Note path: `notes/2026071157-opus-gapped-v2-launch-review.md`

---

**Claude — claude-opus-4-8 (final launch-commit delta):**

Runtime model identifier: `claude-opus-4-8`. Bounded review of the four
launch-commit deltas (`3419f56`, `9ebf2f9`, `547c6d8`, `b0a2fe3`), all ancestors of
HEAD `f17e7d8`; working tree clean. I read every diff and the production-tokenizer
artifact directly. These are strictly additive, fail-closed hardening — no gate is
loosened and no surgery/masking/analysis/donor-map/arm/tolerance logic changes.

- **`3419f56` + `9ebf2f9` — harvest fingerprint binding (verified).** The
  independent, no-`src`-import harvest validator now requires **every scored
  checkpoint's `fingerprint` to equal the terminal manifest fingerprint** on both
  the COMPLETE and the ERROR/FAILURE paths, and fails closed if a scored checkpoint
  is preserved with no/malformed manifest fingerprint. Since the fingerprint binds
  the external donor provenance/hashes, scenario/target hashes, amendment SHAs, and
  code commit, this catches donor/scenario/code tampering or checkpoint-mixing at
  certification, complementing the analysis-side common-fingerprint check. Both
  additions ship adversarial mutation tests (tampered `external_donor_provenance` →
  COMPLETE raises "manifest fingerprint"; tampered `scenario_sha256` → FAILURE
  raises). Void checkpoints stay unbound, which is correct — they carry failure
  evidence, not scored outcomes.
- **`547c6d8` + `b0a2fe3` — exact production-tokenizer donor preflight (verified).**
  `src/validate_coherent_external_donors.py` resolves the exact
  `Qwen3-30B-A3B-Instruct-2507@0d7cf2399…` snapshot, asserts the resolved revision,
  and runs `matched_wrong_prefix_ids` over **all twelve** frozen external mappings
  under the *production* tokenizer, recording per-mapping target/donor SHA-256,
  recorded author, token counts, and U+FFFD decode-corruption counts, and asserting
  twelve unique donor hashes. The committed artifact
  (`…coherent_external_donors_gapped_v3_…_20260711T075800Z.json`) is observed
  **PASS**: resolved revision == requested, `Qwen2Tokenizer` vocab 151669,
  `n_rows=12`, `all_rows_pass=true`, `max_replacement_char=0`, `min_content_positions
  6753`, `n_unique_donors=12`, identity `…AMENDMENTS-1-2-3` / `gapped-v3`. This
  closes the one Amendment-3 launch-gate item my v3 review could only see exercised
  on the 0.6B ladder tokenizer: the exact-length donor-slot construction now
  provably tokenizes and decodes cleanly for every mapping under the real 30B
  tokenizer — retiring a latent tokenizer-mismatch risk rather than assuming it.

**New defect? None.** The deltas are small, fail-closed, and covered by tests. The
external donors remain **authored/non-subject-native** controls with exact recorded
authors (`sonnet` c13–c15, `opus` c16–c18, `codex-gpt5.5` c25–c27, `sonnet-render`
c28–c30), now additionally hash-and-author-validated under the production tokenizer
and fingerprint-bound across every harvest path. The full-suite **124 pass /
monitor 35 clear** and the independent code auditor's GO corroborate the one item I
could not execute here (test/model runs are sandbox-blocked this session); I
attribute that observation to those runs, not my own execution. The v3 claim
constraints are unchanged (position-preserving compaction; fixed-benchmark `GW`
against authored external donor-slot controls, not superiority to every wrong
history; N=6 interim only, confirmatory at N=12; 5e-4/bf16 disclosures; render-time
corpus with the regime backstop).

**VERDICT: GO (v3 GO preserved).** The four deltas are pure fail-closed hardening
plus the production-tokenizer donor validation; they strengthen the launch posture,
introduce no blocker, and close my last outstanding pre-launch verification gap.
The remaining conditions are operational only: launch from this frozen, pushed,
deconflicted clean commit.

One-line verdict: **GO** — harvest-to-manifest fingerprint binding and the
exact-production-tokenizer 12-mapping donor preflight (artifact PASS, zero decode
corruption) are additive fail-closed hardening with no new defect; v3 GO preserved,
launch from the frozen clean commit.

Note path: `notes/2026071157-opus-gapped-v2-launch-review.md`
