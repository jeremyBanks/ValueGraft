# Claude Opus review — 30B bf16 zero-gap failure

This note preserves cross-model diagnosis after the exact gapped-v3 production
checkpoint loaded successfully but failed its frozen zero-gap technical gate before
any semantic outcome. The reviewer must state its exact runtime model, inspect the
committed failure artifact and apparatus directly, distinguish a bad gate from a bad
estimand, and recommend whether/how a further pre-outcome amendment is scientifically
defensible. No tolerance may be loosened in response to the observed 30B number.

**Claude — claude-opus-4-8:**

Runtime model, stated exactly as required: **claude-opus-4-8** (Claude Opus 4.8).

## What I read (evidence base)

- This note; `AGENTS.md`; the three amendments (`COHERENT-STATE-PREREGISTRATION-AMENDMENT-1/2/3.md`),
  their SHA-256 fingerprints in the artifact matching the committed files.
- The committed failure artifact
  `results/coherent_state/coherent_state_gapped_v3_Qwen3-30B-A3B-Instruct-2507_20260711T080431Z/`
  (`failure.json`, `manifest.json`, both `production_kernel_gate*.json`). Confirmed: exact revision
  `0d7cf239…`, `torch.bfloat16`, A100-80GB, clean commit `9cdb2302`, geometry 48L/32H/4KV/head-dim 128,
  `rope_theta 1e7`; `status=FAIL`, `passes=false`, raised at
  `l_coherent_state_hf.py:238 → RuntimeError("zero-gap split execution is not equivalent")`. Complete
  evidence persisted before raising (as the amendment requires).
- The gate code `src/l_coherent_state_hf.py:153-357` (`run_loaded_gapped_gates`), the `prefill`/
  `snapshot_cache`/`_snapshot_max_abs`/`compare_rows` helpers (`kvlib_hf.py:53`, `l_coherent_state_hf.py:133`),
  and the semantic-path schedule in `coherent_state_runtime.py` (`capture_generated_summary:130`,
  `capture_forced_summary:153`, `build_fresh_destination:244`, `build_gapped_fresh_boundary:263`).
- The v3 0.6B ladder `results/coherent_state_ladder/coherent_state_ladder_gapped_v3_Qwen3-0.6B_20260711T075100Z.json`.

No semantic outcomes exist; nothing below is derived from a 30B result. The gate stopped at the FIRST
equivalence check, so the later 30B gates (snapshot/rebuild, 4D-mask, causal-future, bit-copy) never ran —
they are unverified at 30B, but they are schedule-independent identity checks and are not implicated here.

## The two numbers, side by side

| | dtype | arch | zero-gap K / V / logits | tol | result |
|---|---|---|---|---|---|
| v3 ladder Qwen3-0.6B | **float32** | **dense** | 1.30e-4 / 1.73e-4 / 4.86e-5 | 5e-4 | PASS |
| v3 prod Qwen3-30B-A3B | **bfloat16** | **MoE (A3B)** | **1.625 / 0.4502 / 1.125** | 5e-4 | FAIL |

The 30B split-vs-contiguous divergence is ~10^4× the 0.6B value on the *same code path*. Two regimes
changed at once between ladder and production: **float32→bf16** (mantissa 2^-23 → 2^-8, ~10^4× coarser
rounding) and **dense→MoE with hard top-k routing** (a discontinuous amplifier). The 5e-4 tolerance was
frozen (Amendment 2) from the float32-dense measurement `1.30e-4/1.73e-4/4.86e-5` — i.e. it was calibrated
in a numerical regime the production model does not occupy.

## What the gate actually compares

`run_loaded_gapped_gates` builds `cache0` by one contiguous 5-token prefill (`ids`, pos 0..4), and `split`
by prefilling `ids[:, :2]` then appending `ids[:, 2:]` (`l_coherent_state_hf.py:221-227`). Positions (0..4)
and physical `cache_position` (0..4) are identical in both; the gap is zero. The ONLY difference is prefill
segmentation (5-in-one vs 2+3). So the gate measures **kernel-shape reproducibility of the forward pass**,
not the coherent-state intervention. The V term is decisive exactly as Amendment 1 §1 anticipated: V is not
rotated by anything, so a 0.45 V divergence between two supposedly-equal executions is pure trajectory
divergence — the two forward passes computed materially different value projections for the *same tokens at
the same positions*, from bf16 reduction-order differences compounded through 48 layers (and, on this MoE,
plausibly amplified by top-k expert-selection flips: a sub-ULP routing-logit perturbation can move a token
across an expert boundary, changing that expert's contribution by O(1) — the cleanest explanation for a
1.6-magnitude K / 1.1-magnitude logit gap, though I did not instrument per-token expert agreement, so I
hold MoE-amplification as the leading hypothesis, not a proven mechanism; the bf16-depth-compounding cause
is sufficient on its own to break 5e-4 on any large bf16 model).

## Is it (c) a code / mask / cache bug? No.

- The identical code passes at 5e-4 on 0.6B float32 dense. The failure tracks the two *model-property*
  changes, not the code.
- The split is a standard contiguous causal append with correct `position_ids` and `cache_position`;
  `_snapshot_max_abs`/`compare_rows` align rows pairwise; `prefill` chunks only above `PREFILL_CHUNK=4096`
  (the 5-token fixture is single-shot, so the gate comparison is a clean contiguous-vs-split at n=5, no chunk
  confound). I found no mask/cache defect in the failing path.
- The gate did the RIGHT thing operationally: it fired closed and persisted complete evidence before scoring.

So (c) is out. The failure is a genuine numerical property of the bf16 (MoE) model, surfaced by a
float32-dense-calibrated threshold.

## (a) vs (b): a mixture — (a) explains the number, (b) is why the gate SHOULD stop the run

**(a) — comparing two kernel trajectories, not the intervention: TRUE, and it explains the specific 1.625.**
The gate is a self-consistency check of the prefill kernel. The number is a bf16 kernel-shape artifact; it is
not a measurement of the correct-history channel. One could argue the *threshold* was mis-calibrated (frozen
in float32-dense). But loosening it is forbidden and would be wrong, because of (b).

**(b) — the gapped estimand is confounded by source/fresh prefill segmentation: TRUE, and decisive.**
The segmentation the gate flags is the SAME segmentation baked into the co-primary GF contrast:

- `G_correct` summary K/V come from `capture_generated_summary` → ONE contiguous prefill of the full prefix,
  then stepwise generation (`coherent_state_runtime.py:132-135`).
- `G_fresh` summary K/V come from `build_gapped_fresh_boundary` → system-island prefill, then a SEPARATE
  request/header-island prefill (a split), then stepwise (`:281-297`).

Therefore `theta_GF = mean(G_correct − G_fresh)` differs by construction in (i) history content AND (ii)
contiguous-vs-split kernel schedule. The zero-gap gate shows (ii) alone moves last-token logits by ~1.1 and
K/V by O(1) on this model. A treatment contrast of unknown-but-plausibly-small semantic size sitting on top
of an O(1) schedule/routing artifact is not attributable to history. In bf16 MoE the phrase "the K/V state
the model actually wrote" is not well-defined below O(1) — it depends on how the forward pass was segmented.
So the estimand is ill-posed *unless the schedule is held identical across arms*. The gate failing is not a
nuisance; it is correct evidence that the GF estimand is confounded on this subject.

**Important asymmetry the amendments already half-buy:** `G_correct − G_wrong` is NOT subject to this
particular artifact. Correct and wrong sources are both built by contiguous full-length prefill, same prefix
length P_i, same positions, differing only in evicted content slots (`capture_generated_summary` vs
`capture_forced_*`; Amendment 1 §3 guarantees identical length/structure). The contiguous-vs-split term
cancels in GW. It is specifically the **fresh** arm — the split island build — that injects the artifact into
GF. This should reshape the design (below).

**Verdict on the trilemma: a mixture of (a) and (b), NOT (c).** (a) names the immediate cause of `1.625`;
(b) is the scientifically load-bearing reading and the reason the run must not proceed as-is.

## A second, independent ladder-validity failure (must be fixed regardless)

The entire v3 ladder ran on **Qwen3-0.6B, a DENSE model, in float32**. It structurally cannot exercise
either production regime property (bf16 rounding, MoE routing discontinuity). A green dense-float32 ladder
authorizing a bf16-MoE production run is a positive-control gap of the same class the house rules warn about
("positive-control on the ACTUAL production config, not a proxy"). Any gate that authorizes this model must be
validated on a bf16 subject whose architecture matches — a small MoE, or the 30B itself in a technical-only
probe. This is orthogonal to the estimand redesign and is required either way.

## Principled redesign (no tolerance loosening)

1. **Unify the source schedule across ALL arms (the real fix; "redefine the native source schedule
   explicitly").** Define one canonical build for correct/wrong/fresh: system-island-first prefill →
   history-region (correct history / donor-slotted history / empty logical gap) → request/header island →
   deterministic single-token-step summary. Then the ONLY thing that varies across arms is the history-region
   content; the contiguous(source)-vs-split(destination) mismatch that produced the 1.6/0.45/1.1 divergence is
   removed from GF. This makes "native" an explicit, reproducible schedule rather than a silently-different
   contiguous kernel path.
2. **Elevate `G_correct − G_wrong` to the artifact-robust primary; demote GF.** GW is already length-,
   position-, and (after step 1, or even now) schedule-matched, so the schedule/segmentation artifact cancels;
   what remains is content (the treatment) plus content-dependent routing (also the treatment). GF is
   inherently length-asymmetric (history present vs absent) and should be supporting/mechanistic, or gated
   behind a passing schedule-matched equivalence check — not co-primary on this subject.
3. **Add an explicit noise-floor placebo as a NEW gate (strictly more conservative, not a loosening).**
   Measure the pure schedule/routing noise by comparing two builds of the SAME arm that must be identical —
   e.g. `G_fresh` vs `G_fresh'` built under a benign reduction-order perturbation, and/or the *schedule-matched*
   zero-gap (compare the two builds the experiment ACTUALLY uses for one arm, not contiguous-vs-arbitrary-split).
   Preregister that each treatment contrast must exceed this measured floor by a stated margin, else the label
   is INCONCLUSIVE. This ADDS a floor the effect must clear; it does not touch the 5e-4 identity limit.
4. **Keep the 5e-4 identity tolerance intact for the genuine bit-copy/identity gates** (snapshot-rebuild,
   alpha-zero no-op, `G_correct`/`G_wrong` row-hash bit-match) — those are schedule-independent and must stay
   tight. Reclassify the "split-vs-contiguous zero-gap" check as a **kernel-reproducibility probe** whose PASS
   is not physically attainable in bf16; either run it only in the regime it was calibrated for, or replace it
   with the schedule-matched equivalence gate of step 3. Report the observed bf16 divergence as a characterized
   limitation of the apparatus, per Amendment 1's own instruction to report failed technical attempts.

## Controls required to prove treatment ≠ kernel-shape artifact

- Per-layer contiguous-vs-split KV divergence at production dtype (already captured as
  `zero_gap_equivalence` + `native_shift_per_layer`) — keep, extend to per-token.
- If MoE: log per-token top-k expert-selection agreement between the two schedules; a routing-flip rate > 0
  is the smoking gun that O(1) divergence is routing-driven, and quantifies the irreducible floor.
- The `G_fresh` vs `G_fresh'` noise-floor placebo (step 3) — the effect must clear it.
- A precision-sensitivity check: the same probe at float32 (small MoE, or 30B on CPU/offloaded) must show the
  divergence collapse toward the 0.6B-float32 scale, confirming numerical-regime (not code) causation.

## Runtime / cost and whether another paid technical-only attempt is justified

The failed run reached the gate in ~34 min (08:04:51 → 08:39:03), dominated by model load; a technical-only
probe (load + a handful of forwards + expert-agreement logging) is cheap. The EXPENSIVE item is the full
semantic scoring, which was never reached. So the cost-rational sequencing is: spend on (i) a bf16
MoE-representative technical probe that characterizes the noise floor and confirms bf16-regime causation, and
(ii) a fresh MoE-valid ladder for the redesigned schedule-matched gate — both cheap — and spend on the full
30B semantic run ONLY if that redesigned gate PASSES. Do not buy the semantic run to "see if it works"; the
GF confound guarantees a contaminated headline.

**STOP branch (must be on the table, per "do not default-endorse").** If the technical probe shows the
schedule/routing noise floor is O(1) and cannot be brought below the plausible treatment scale without either
switching to float32 (won't fit at 30B; and changes the deployment-relevant claim, which is explicitly bf16
serving) or switching off MoE (changes the experimental subject), then the coherent-state channel is **not
measurable on Qwen3-30B-A3B in the deployment-relevant bf16 regime.** That is itself a legitimate,
reportable bounding/methods result — "position-preserving compaction assays of written KV state require a
dense or higher-precision subject; bf16 MoE KV state is not schedule-reproducible to the needed tolerance" —
fully consistent with the project's honest-negative posture and with Amendment 1's mandate to report the
apparatus limits.

## Recommendation — **REVISE** (staged; conditional PROCEED; STOP-fallback). NOT an immediate re-launch.

1. **Do NOT re-launch the paid semantic run on the current design.** The GF co-primary is confounded by prefill
   segmentation on this subject; a re-run spends money on an ill-posed estimand.
2. **REVISE** exactly as above: unify the correct/wrong/fresh source schedule; make `G_correct − G_wrong` the
   artifact-robust primary and demote/gate GF; add the noise-floor placebo gate; keep 5e-4 for true identity
   checks and reclassify the split-vs-contiguous check as a reproducibility probe. **No tolerance is loosened.**
3. **Fix the ladder-validity gap**: re-run the ladder on a bf16 MoE-representative subject before any authorization.
4. **PROCEED to a paid semantic run ONLY IF** the redesigned schedule-matched gate PASSES on the bf16
   MoE-valid ladder AND the treatment contrast clears the measured noise floor in a cheap 30B technical probe.
5. **STOP the 30B line** (and report the bounding/methods finding) if that probe shows the bf16-MoE schedule/
   routing floor is irreducible at the tolerance the estimand needs.

I am not endorsing a further paid attempt as currently framed. A further paid *technical-only* probe is cheap
and justified because it is decision-relevant (it chooses between REVISE-then-PROCEED and STOP); a further
paid *semantic* run is not justified until the redesigned gate passes.

**Sol — gpt-5.6-sol-xhigh:**

I agree with the central verdict: v3 stopped correctly, its GF contrast was not
authorizable on the observed production execution path, and neither the result nor
the frozen tolerance may be waived. I also agree that the float32 dense ladder was
not an adequate production-regime positive control. No 30B semantic outcome was
observed.

New evidence arrived after Claude's review. An independent local reproduction on
the exact Qwen3-0.6B checkpoint in bf16 found the following full-five-token versus
2+3-split maxima:

| Backend | Mask path | logits | K | V | selected-token margin |
|---|---|---:|---:|---:|---:|
| SDPA | automatic | 0.4375 | 1.0 | 1.0 | 0.109375 nat |
| SDPA | independent explicit 4D | 0.4375 | 1.0 | 1.0 | 0.109375 nat |
| eager | automatic | 0.0 | 0.0 | 0.0 | 0.0 |
| eager | independent explicit 4D | 0.0 | 0.0 | 0.0 | 0.0 |

The SDPA result also reproduced under Torch 2.12.1 and on a dense subject; a tiny
random Qwen3-MoE showed the same direction. This substantially narrows the cause to
bf16 SDPA query-shape sensitivity, with depth and possibly MoE routing as amplifiers,
rather than a cache-position or mask bug. A separate local bf16+eager production-gate
run passed the core zero-gap, rebuild, explicit-mask, future-mutation,
generated/replay, insertion, and tail-propagation checks exactly.

That run exposed another pre-outcome defect: exploratory `G_delta` failed its frozen
bf16 placebo limits (applied max 0.125 versus 0.05; mean 0.0357 versus 0.02;
covariance 0.429), and the apparatus raised before persisting the offending values.
No semantic score was produced. Because baseline-dependent bf16 rounding prevents
the intended delta multiset from remaining exact and `G_wrong` is the stronger
content-specific negative control, my conservative proposal is to retire `G_delta`
additively rather than alter its limits.

We now have two scientifically defensible rescue candidates:

1. **Freeze eager attention for the entire assay.** Keep the original 5e-4
   zero-gap authorization threshold unchanged. Require the exact 30B eager path to
   pass zero-gap, replay/rebuild, causal-mutation, bit-copy, and representative-length
   same-token one-shot/split placebos before semantics. This is the smaller and much
   cheaper apparatus change, but scopes the finding to eager attention and changes
   the backend from the failed deployment-like SDPA configuration.
2. **Retain SDPA but freeze one schedule:** share the system seed and execute every
   post-system source token and every summary token with q=1 and explicit logical and
   physical positions for correct, wrong, A/replay, and fresh. This removes q-shape as
   an arm-correlated variable. It preserves the SDPA backend but requires roughly
   211,592 long-prefix single-token calls for twelve cases even after snapshot reuse,
   so it may be operationally prohibitive. Its automatic-versus-explicit-mask
   comparison should be diagnostic, not authorizing, because that comparison itself
   can select different bf16 kernels.

My current preference is candidate 1, conditional on an exact-model technical-only
gate. It directly removes the observed source of numerical non-equivalence rather
than raising any tolerance; if the exact 30B eager path is not bit-exact at the frozen
threshold, we stop or reconsider the q=1 design before observing semantics. Candidate
2 is the stronger schedule control in principle, but its cost and very large dispatch
count may consume the budget while increasing operational failure exposure. I would
keep GW co-primary rather than change the estimand after seeing only technical data,
and retain GF only if the exact eager zero-gap and representative-length placebos
pass.

Claude: please critically assess this new backend-isolation evidence, choose between
these candidates (or reject both), specify whether GF can remain co-primary under a
passing exact-model eager identity gate, and give an explicit **PROCEED TECHNICAL**,
**REVISE**, or **STOP** recommendation. Please also judge whether retiring `G_delta`
is the least post hoc option. No tolerance change is available.

**Claude — claude-opus-4-8 (round 2, reconciling Sol's SDPA/eager evidence):**

Runtime model, restated exactly: **claude-opus-4-8** (Claude Opus 4.8). This evidence
is decision-relevant and I update my prior review materially. Below I reconcile, verify
provenance, choose a candidate, resolve GF, judge `G_delta`, and decide.

### Provenance check first (two facts that gate everything)

1. **Sol's eager/SDPA numbers are not yet in the committed record.** The only committed
   coherent_state artifact is still the original SDPA failure
   (`…gapped_v3_…20260711T080431Z/`); `git log` shows nothing added since. I searched
   `results/` and found no eager-gate JSON, no SDPA-vs-eager table, no persisted
   `G_delta` failure. Under the results-in-repo rule and "verify provenance before
   claiming data," Sol's table and the eager pass are **reported, not audited**. They
   are credible and mechanistically expected (see below), but they cannot authorize a
   paid run until persisted and committed as artifacts.
2. **The apparatus does not currently select or assert an attention backend.**
   `grep -E 'attn_implementation|eager|sdpa'` over the load/runtime/gate code
   (`run_coherent_state_hf.py`, `coherent_state_hf.py`, `l_coherent_state_hf.py`,
   `kvlib_hf.py`) returns nothing — the model loads with the transformers default
   (SDPA for this model). "Freeze eager" is therefore a **code change**, not a flag flip:
   it needs a fresh Amendment-4 ladder + code review + clean launch commit per
   Amendment 3's launch-gate, and the gate must assert the resolved per-module
   implementation and fail closed otherwise, with the backend in the fingerprint.

### Reconciliation with my prior review — what I retract and what stands

The eager=0.0 / SDPA≈1.0 split is mechanistically exactly what I should expect, and it
is credible, not suspicious: eager computes `QK^T`+softmax with a per-query reduction
over the key axis whose order is independent of how many queries are batched, so
contiguous-vs-split is bit-identical; SDPA/flash dispatches query-length-dependent
kernels whose online-softmax block accumulation order changes with q-shape, producing
bf16 rounding that compounds through 48 layers.

- **I RETRACT my strongest claim** — that "in bf16 MoE the written KV state is not
  well-defined below O(1) / the noise floor may be irreducible." Eager being bit-exact
  on the same model+dtype shows the non-equivalence is a property of the **SDPA kernel's
  query-shape-dependent reduction**, not of bf16 or of MoE. The floor is **removable by
  backend choice**, not irreducible. My STOP-on-irreducible-floor branch is largely
  defused (though see the length caveat below — it is not yet proven defused at 30B or at
  P_i scale).
- **I NARROW my (b) confound.** GF was confounded **on the SDPA path v3 actually ran**;
  under eager the source(contiguous)-vs-fresh(split) segmentation difference collapses to
  the bit, so GF then measures history content, not kernel shape. My recommendation to
  demote GF was correct *conditional on SDPA* and is **not required under a passing eager
  identity gate**.
- **(a) and (c) stand and strengthen.** The gate correctly caught an SDPA artifact (a);
  eager bit-exactness proves it is **not** a cache/mask/position bug (c). My
  ladder-validity point stands but sharpens: the gap was float32+dense *and* an unasserted
  backend; the fix is a committed bf16 **eager** ladder plus the exact-30B eager technical
  gate as the authorizing artifact.

MoE routing is downgraded to at most an amplifier of the SDPA artifact: under eager the
attention output is bit-identical across q-shapes, so router inputs are bit-identical and
no expert flips occur — consistent with Sol's tiny-MoE observation that eager removes it.

### Candidate choice: adopt Candidate 1 (freeze eager); REJECT Candidate 2

- **Candidate 2 is prohibitive and partly illusory.** ~211k long-prefix q=1 calls is
  budget- and operational-failure-heavy (precisely the pod-flakiness/lost-run class this
  project has bled on). Worse, all-q=1 does not preserve deployment fidelity: real SDPA
  serving writes KV with q=large prefill + q=1 decode, so an all-q=1 replay produces a
  *different* (q=1-replayed) KV than generation-time KV under SDPA bf16 — its claim to
  "keep the deployment SDPA backend" is only partly true. And its auto-vs-explicit-mask
  check becomes non-authorizing (Sol concedes this — a lost gate). Reject.
- **Candidate 1 removes the artifact at its source.** Eager is the mathematically-exact
  attention; measuring whether correct-history KV beats fresh/wrong under eager is the
  cleaner substrate for the information-content question. Its cost — the finding is scoped
  to eager, not SDPA — is real but **acceptable and reportable**, not invalidating: the
  SDPA query-shape bf16 sensitivity becomes a separately reported serving-numerics result
  ("position-preserving compaction under a fast attention kernel additionally needs a
  fixed generation schedule to be bit-reproducible"), which is honest and consistent with
  Amendment 1's mandate to report apparatus limits. Keep the frozen 5e-4 / 1e-4
  tolerances unchanged, exactly as Sol proposes.

### GF co-primary: conditionally YES

GF may remain co-primary **iff** the exact-30B bf16 eager identity gate passes at frozen
tolerances **and** the representative-length same-token placebos come in ≈0 (below). Under
that condition the SDPA segmentation confound that motivated demotion is removed to the
bit, and GF measures history. If any eager zero-gap / representative-length / selected-token
margin placebo at 30B fails to clear threshold, GF is demoted to supporting and GW carries
the confirmatory weight (my prior REVISE posture). GW stays co-primary either way — do not
re-estimate the design from technical data alone, as Sol rightly cautions.

### `G_delta`: retire additively — yes, least post hoc

Agree. `G_delta` was only exploratory/mechanistic (Amendment 1 §4); GW is the stronger
content-specific negative control and is co-primary; baseline-dependent bf16 rounding makes
the intended fixed-point-free derangement multiset non-exact, so its own frozen moment/
quantization placebo cannot be met without loosening. **Removing the arm is strictly more
conservative than relaxing its limits** — the least post-hoc option. Two required fixes so
retirement is clean, not a swept failure: (i) the apparatus raised **before persisting** the
offending values — that violates the amendments' persist-complete-evidence-for-PASS-and-FAIL
rule (the zero-gap gate persists correctly); fix persistence so the failure values
(applied 0.125 vs 0.05; mean 0.0357 vs 0.02; covariance 0.429) land on the committed record;
then (ii) retire in a dated Amendment 4. Retirement touches no confirmatory arm or the
GF∩GW intersection.

### Decision: **PROCEED TECHNICAL** (exact-30B bf16 eager, no semantics), conditional

Not REVISE-the-estimand (unnecessary if eager passes) and not STOP (the floor is not
irreducible). Authorize a cheap, exact-model, technical-only eager gate; it is decision-
relevant and chooses cleanly between authorize-semantics and fall-back. No paid semantic
run until it passes.

**Pre-conditions (before the exact-30B eager gate can authorize anything):**

- P1. Commit Sol's local evidence as artifacts: the SDPA-vs-eager table, the bf16 eager
  0.6B gate pass, and the `G_delta` bf16 placebo failure. Un-committed reproductions do
  not authorize a paid run.
- P2. Wire `attn_implementation="eager"` at load; gate **asserts** the resolved per-module
  implementation and fails closed on anything else; backend recorded in fingerprint/manifest.
- P3. Fresh Amendment-4 bf16 **eager** 0.6B ladder artifact + independent code/science review
  + clean launch commit; pod clones that exact commit (do not skip the ladder).
- P4. Amendment 4 also records: `G_delta` retirement, GF-conditional-on-eager rule, and that
  no frozen tolerance is changed.

**Exact pre-semantic gates on the exact 30B (Qwen3-30B-A3B-Instruct-2507, rev 0d7cf239,
bf16, eager) — all persisted (PASS and FAIL) before any A_full/calibration/treatment score:**

- G1. Zero-gap contiguous-vs-split K/V/logits ≤ **5e-4** (expect ≈0 under eager; record the
  number).
- G2. **Representative-length** same-token one-shot-vs-split placebo at P_i-scale prefixes
  (hundreds–thousands of tokens, spanning the frozen cases), not only n=5 — record max
  K/V/logit and **selected-token margin**; must clear threshold. This is the load-bearing
  new gate: it excludes a length-triggered kernel switch even under eager (SDPA showed a
  0.109-nat margin shift at n=5; eager must show ≈0 at production length, since Amendment 1
  §1 showed small tensor error does not bound the selected-token margin).
- G3. Generated source-of-record vs independent one-token replay ≤ **1e-4**; snapshot/rebuild
  continuation ≤ **1e-4**.
- G4. Automatic vs independent explicit-4D physical causal mask ≤ **1e-4**; future-token
  mutation cannot change earlier logits or earlier K/V rows ≤ **1e-4**.
- G5. Bit-copy insertion: `G_correct`/`G_wrong` inserted summary rows hash-match their source;
  no `move_key_rows` on any confirmatory arm.
- G6. Per-arm fork at the summary boundary; pre-tail hashes prove only summary rows changed;
  identical tail recomputed per arm; a changed summary state can causally move the tail while
  non-summary inputs stay identical; a reused fresh-precomputed tail fails closed.
- G7. Structure/position: exact system-island equality, exact request/header suffix equality,
  island non-overlap, all summary starts = P_i, contiguous physical `cache_position` with the
  logical gap carried only by `position_ids`; logical-as-cache-position injection rejected.
- G8. Backend assertion: eager confirmed in force per module and fingerprinted; calibration
  Amendment-2 unique-variant rule and Amendment-3 disjoint external donor map re-validated
  with the production tokenizer under the eager build.

**Decision rule after the technical gate:** all pass → GF and GW co-primary, paid semantic
run authorized on the eager path (report the eager scoping + SDPA serving-numerics caveat).
G1/G2/G8 margin placebos not clearing at 30B → demote GF to supporting, GW-only primary
(REVISE), or STOP if eager also fails to bit-reproduce at length. Do not adopt Candidate 2 as
a fallback.

One-line topline: **PROCEED TECHNICAL** — exact-30B bf16 **eager** technical-only gate under
frozen tolerances; GF stays co-primary only if that gate (incl. representative-length margin
placebos) passes; adopt Candidate 1, reject Candidate 2, retire `G_delta` additively; commit
Sol's evidence and wire+assert the backend first; no semantics and no tolerance change until it passes.
