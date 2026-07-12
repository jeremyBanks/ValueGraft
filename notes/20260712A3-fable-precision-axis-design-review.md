# Precision-axis (NF4 vs bf16) design review — independent adviser assessment

**Author (exact runtime identity):** Claude Fable 5, model id `claude-fable-5`, running as an
independent scientific-design adviser in a fresh session.
**Date:** 2026-07-12
**Inputs read:** AGENTS.md; STATE.md (post-e01 current-truth); the owner-request note
(`notes/owner-request-precision-axis-quantization-dependence.md`); Sol's e01 final interpretation
(`notes/2026071287`); Sol's quantization-axis disposition (`notes/2026071296`); PAPER.md structure
(§6–8, §12); targeted inspection of `src/coherent_canary_loader.py` (pinned 40-hex revision
resolution, per-tensor `BF16` weight assertion, explicit quantization-absent check).
**Scope:** whether a bounded NF4 observation can answer anything useful, and if so the minimum
apparatus. Settled questions (v12 stop, e01 non-confirmatory status) are out of scope and are
treated as fixed constraints, not re-litigated.

---

## (a) Go/no-go verdict

**Conditional GO — but only after reframing what the primary endpoint is.**

GO for: a bounded, exploratory, *new-protocol* diagnostic that asks whether the **large,
well-powered observables** — the ~22.3-nat oracle-vs-fresh compaction damage and the total
behavioral null (every cell generating the same wrong answers) — are robust to NF4 weight
quantization on the same host, with a fresh same-host bf16 replication as the paired arm.

NO-GO for: (1) framing this as a quantization×graft **interaction test** — there is no cleared bf16
treatment estimand to interact with (formal v12 stopped pre-treatment; e01 is N=1, placebo-less,
schedule-fragile, diagnostic-only); (2) treating it as adjudicating the early MLX +10–12 pt result
(different stack, different estimand — judged answers vs forced logprob — and a VOID apparatus);
(3) running anything through the frozen v12 protocol or relabeling e01 confirmatory.

The decisive design insight: **power dictates the endpoint.** A 0.17-nat N=1 contrast cannot
support a precision-dependence claim in either direction — if it appears at NF4 you learn almost
nothing beyond e01, and if it vanishes you cannot distinguish "precision-dependent effect" from
"fragile numerical residue," exactly because e01 had no placebo and already collapsed 8.3× under a
schedule change alone. But the 22-nat damage and the all-cells-wrong-answer behavioral null are
enormous relative to any plausible noise floor, so testing *their* NF4 robustness is well-powered
even at one case. That is also what the owner actually needs for the paper: it converts "whether
anything differs at 4-bit is untested" into "the headline negative result (compaction destroys the
channel; the graft changes no behavior) reproduces under 4-bit weight quantization," which is a
genuinely bounded exploratory precision-axis result.

## (b) Minimal design

**Protocol identity.** A new, small, explicitly exploratory protocol (suggested name:
`precision_probe_p01`), with its own output namespace under `results/` following the
unique-naming rule. It reuses v12 *components* (loader, gap/cache machinery, scoring) but is not
v12, does not share its preregistration, and edits none of the frozen v12 files — quantization
support goes in a parallel loader path, since the current loader hard-asserts per-tensor `BF16`
weights and quantization-absence.

**Axis, named correctly.** The manipulated axis is **weight representation + linear kernels**
(bf16 weights/matmuls vs bitsandbytes NF4 weights/dequant kernels), with **KV cache dtype held at
bf16 in both arms and explicitly recorded**. This matches the historical MLX condition (4-bit
weights, fp16 KV) and is the only axis a bnb load can vary. No claim about KV-cache precision is
licensed by any outcome.

**Subject.** The exact pinned `Qwen/Qwen3-30B-A3B-Instruct-2507` snapshot at the same 40-hex
revision the v12 loader binds (`REVISION_30B`) — never a re-resolved `main`.

**Arms and cells.** One case: the existing frozen e01 fixture (reuse buys direct comparability and
avoids new-case authoring spend; the adverse resolved-answer calibration is a named limitation).
Both runtimes run **fresh on the same pod, sequentially** (bf16 ~60 GB and NF4 ~20 GB both fit an
A100-80GB; do not co-resident them). Archived e01 numbers are *not* the comparison arm for the
small contrasts — host/driver/time confounds; the fresh bf16 arm also doubles as the first e01
replication attempt on a new host, which is independently informative. Cell set, trimmed to the
minimum that supports the endpoints:

- **Primary:** oracle (full history), fresh compaction, FC and FW grafts for {value-only, full
  K+V} at R2 under schedule N — in both runtimes. Endpoints: (P1) oracle-vs-fresh focal damage
  reproduces in sign and order of magnitude at NF4; (P2) generated answers per cell — does any
  graft cell change any generated answer at NF4 (at bf16 in e01: none did).
- **Secondary (descriptive only):** the D/SEL/Hplus/U/Uplus contrast table for those cells, plus
  the P-schedule replay of the value-only R2 cell in both runtimes (schedule-sensitivity check).
  Report signs and magnitudes; prespecify no threshold that converts these into a claim.
- **Explicitly dropped** (log the drop per the no-silent-caps rule): R1/R3 regions, key-only and
  crossed K/V families, additional cases. FW serves as the defensible negative graft / active
  control; the matched orthogonal placebo is known-infeasible at bf16 casting (1,024-attempt
  failure in e01) and NF4 does not change the cache dtype, so it will not become feasible —
  do not spend attempts re-deriving it.

**Analysis pre-declaration (written before launch, in the protocol note):** what counts as
"pattern reproduced" for P1 (same sign, damage within, say, 0.5–2× the bf16 fresh-arm value) and
P2 (identical yes/no on any-answer-change), and that all secondary numbers are descriptive.

## (c) Gates and stopping rules

Ordered, fail-closed; each gate failure is itself a reportable result and stops spend.

- **G0 (local, ~$0):** desk-check bitsandbytes/transformers support for this *packed MoE*
  checkpoint at the pinned library versions. Qwen3-MoE experts may be fused 3-D parameter tensors
  rather than `nn.Linear` modules; bnb quantizes only `nn.Linear`. If the packed layout is not
  quantizable, outcome is "NF4 infeasible at bounded effort" — stop before any pod.
- **G1 (pod load/compat gate, ≤20 min):** load NF4 (`BitsAndBytesConfig`, NF4, double-quant,
  bf16 compute dtype); assert expert linears actually quantized (not silently skipped — a
  skipped-experts load would be a bf16-in-disguise arm and the worst silent failure available);
  record KV dtype; short greedy generation sanity; measure tokens/sec and project total cost —
  if projection exceeds the ceiling in (e), stop.
- **G2 (identity ladder under NF4):** L0/L1-style — serialize/rebuild cache → identical greedy
  continuation and bounded max-logit delta; gap/position machinery test. The ladder is
  non-negotiable per house rules; a new runtime is a new subject.
- **G3 (positive control):** oracle-vs-fresh focal damage at NF4 is large and correctly signed.
  If the ~22-nat class of damage does not reproduce, the apparatus is not measuring the same
  thing at NF4 — stop; do not proceed to graft cells.
- **G4 (determinism repeat):** score one cell twice; byte-compare score objects. If
  nondeterministic (bnb kernels are not guaranteed deterministic), switch secondary reporting to
  ranges from ≥2 repeats or drop secondaries entirely; P1/P2 survive nondeterminism.
- **Run-phase stopping:** complete the pre-declared cell set and stop. No chasing — if a
  surprising NF4 number appears, the disposition is "report and propose a follow-up
  preregistration," never "add cells now."
- **Hard ceiling:** one pod, ≤6 pod-hours or ≤$15 total, whichever first; single reprovision
  allowed on a degraded pod per existing ops rules. Persist every render/score per the
  save-every-render rule; commit results before teardown; pod off before any writing.

## (d) What each outcome would and would not mean

- **G0/G1 failure:** "A bounded NF4 load of this checkpoint is not feasible with the standard
  stack." Paper keeps the open-question caveat, now with a concrete recorded reason for not
  running the axis. Means nothing about the science.
- **G3 failure:** the measurement apparatus does not transfer to NF4 — reportable as a
  methodological result; licenses no precision claim in either direction.
- **P1+P2 reproduce (expected):** licenses: "the compaction damage and the behavioral null of the
  graft are not specific to bf16 weights; both reproduce under NF4 weight quantization on the
  same host and case." Does **not** license: any statement about 4-bit KV caches, other
  quantization schemes (GPTQ/AWQ/MLX), other cases, or the cause of the early MLX result — though
  it does make "the early positive was a weight-precision effect" *less* plausible.
- **Secondary contrasts reproduce at e01 scale:** a descriptive sentence that the ~0.17-nat
  value-only hint is not bf16-specific. Still N=1, still placebo-less — no upgrade in
  evidentiary status.
- **Secondary contrasts vanish or reverse at NF4:** descriptive only; indistinguishable from
  kernel-level numerical fragility given e01's own 8.3× schedule collapse. Not evidence of
  precision dependence.
- **NF4 shows a large focal-selective effect or a changed generated answer where bf16 does not:**
  the only outcome that would be genuinely surprising. Licenses exactly one thing: a new
  preregistered multi-case study with placebos designed for that regime. It is not itself a
  finding; it is a reason to design one.
- **Fresh bf16 arm fails to replicate e01's favorable cell on the new host:** important adverse
  robustness evidence about e01 itself, independent of the precision question — report it.

## (e) Rough runtime/cost logic

A100-80GB secure ≈ $1.6–2.5/hr. Loads: bf16 ~10–15 min, NF4 similar. Gates G1–G4 ≈ 30–45 min.
The trimmed primary+secondary cell set is far smaller than e01's 31 cells — order ~10 cells per
runtime, dominated by teacher-forced forward passes; estimate ~45–90 min at bf16. **The wildcard
is NF4 MoE throughput:** bnb dequant on many expert linears is commonly 2–5× slower than bf16, so
budget the NF4 arm at 1.5–4 h. Realistic total: **3–6 pod-hours ≈ $6–15.** That is more than "a
few dollars"; the honest options are (i) accept ~$10–15, or (ii) cut the secondary cells and
P-schedule replay, running only G-gates + P1/P2, landing nearer $4–7. The G1 tokens/sec
measurement converts this from a guess into a projection before the money is spent.

## (f) Likely implementation hazards

1. **Fused-expert non-quantization (top hazard):** bnb silently skipping packed expert tensors
   yields an arm that is ~bf16 with quantized attention only. G1 must positively verify expert
   quantization by module inspection, not by "load succeeded."
2. **Loader/preflight collisions:** the existing loader asserts per-tensor `BF16` and
   quantization-absence; the preflight fingerprint binds dtype/runtime. Fork a parallel loader
   path + new fingerprint schema; never loosen the frozen v12 assertions in place.
3. **Path-control semantics:** the ULP-based path-control machinery is defined on bf16 cache
   values; under NF4 the cache is still bf16 so it may run, but source-execution paths differ
   numerically and it may fail differently. Treat path-control outputs as recorded diagnostics in
   p01, not as gates — and say so in the protocol note before launch.
4. **Throughput blowup / cost overrun:** mitigated by the G1 projection gate.
5. **Nondeterministic bnb kernels:** mitigated by G4.
6. **Version drift:** pin bitsandbytes/transformers/torch exactly in the fingerprint; a bnb
   version change is a different runtime.
7. **Estimand creep:** the strongest social hazard — the temptation to read a secondary-contrast
   divergence as "we found quantization dependence." The pre-declared analysis note is the
   defense; write it before the pod exists.
8. **Chat-template/token-prefix traps** transfer unchanged (canonical rendering, `<|im_start|>`
   scanning) — reuse the existing token machinery verbatim, do not reimplement.

## (g) Corrections to the owner-relayed proposal

1. **"Run the identical corrected apparatus (… matched placebos)":** the matched placebos do not
   exist — construction failed at bf16 after 1,024 attempts, and NF4 leaves the cache dtype
   unchanged, so they will not exist here either. Any NF4 run inherits e01's placebo gap; the
   design must be built around endpoints that do not need the placebo (P1/P2).
2. **"A quantization×graft interaction is itself a publishable finding":** overclaims. With no
   cleared bf16 treatment estimand and an N=1 uncontrolled diagnostic, no interaction is
   identified. What is publishable is the bounded robustness statement in (d).
3. **"The clean test is the corrected apparatus, run at 4-bit" (re: the early +10–12 pt):** no
   run in this design adjudicates the early MLX result. Different stack, different estimand,
   void apparatus. At most, a reproduced behavioral null at NF4 lowers the posterior that weight
   precision explains it.
4. **"The identity/technical gates need a 4-bit variant … bounded work, not a rebuild":**
   directionally right but understated — it is a parallel protocol (loader fork, new fingerprint,
   new namespace, its own ladder), not an edit. Still bounded (~a day of apparatus work), but call
   it what it is.
5. **Axis naming (confirming Sol's correction):** this tests *weight/runtime* quantization with
   bf16 KV in both arms — which happens to be the correct analog of the historical MLX condition —
   and licenses no KV-precision claim. A KV-dtype experiment is a separate, different design.
6. **"Mostly local/cheap":** it is pod work end-to-end (a 30B MoE does not fit the local machine
   usefully in either dtype), and "a few dollars" holds only for the trimmed variant; see (e).

**Bottom line:** build the small p01 apparatus, run the gate ladder, and buy the well-powered
robustness result (P1/P2) with the descriptive contrast table as a free rider. Do not buy an
interaction test — at this N, with no placebo, it is not for sale at any price.
