# STATE.md — session handoff / current state

## CURRENT TRUTH (2026-07-12, post-e01) — formal v12 stopped; diagnostic complete

No pod is active. Formal v12 is terminal. The literal written §14.1 rule stops
at ULP 2, where both edits changed the readout but the minus edit moved in the
wrong direction; that branch is a technical failure. The sealed implementation
continued to ULP 4 and observed bidirectional path sensitivity, retained only as
implementation-defined plumbing evidence. It cannot rescue the written branch.

Exact e01 Phase A returned `PRETREATMENT_PASS`. Because the stop-rule conflict
was discovered and dispositioned before treatment, exactly one unchanged e01
treatment ran as a post-ambiguity diagnostic. Its receipt records
`formal_v12_decision_eligible=false`; it authorizes no e02--e06, aggregate,
confirmation, conversation stratum, or live-agent evaluation.

At primary R2/N, value-only produced a narrow favorable pattern:
`D_focal=+0.174545`, `D_nonfocal=-0.000070`, `SEL=+0.174476`,
`Hplus=+0.710260`, `U=+0.181650`, and `Uplus=+0.945986`. Full K+V did not:
`D_focal=+0.097237`, `D_nonfocal=+0.245486`, `SEL=-0.148249`, and
`Hplus=-0.131035`. Value-only `D_focal` fell to `+0.021046` under P; both
families failed the singleton descriptive component of the 3x schedule
yardstick. R1/R3 did not reproduce a coherent value-only pattern.

Fresh compaction caused `22.298429` nats of margin damage and `22.290991` nats
of correct-target-logprob damage. The best value-only cell recovered only 0.81%
and 4.24%, respectively. Every one of the 31 primary cells retained the same
wrong/other greedy focal and nonfocal answers. All three placebo constructions
were `PLACEBO_UNAVAILABLE`, so available-placebo count is zero. Natural
calibration was adverse (`rho_green=-0.035714`, `rho_amber=0.297872`).

Artifact integrity passed: treatment FF matched Phase A canonically, raw
reconstruction was byte-exact, and local/pod harvests matched after only
allowlisted path/timestamp normalization. The treatment pod was deleted and
independently observed absent. Immediate provider balance was `$61.5865176995`;
the conservative canary delta from `$63.3160022124` was `$1.7294845129`.
Provider settlement and the final end-to-end money/token audit remain pending.

The current interpretation is one weak, uncontrolled, schedule-sensitive
value-only directional hint in one explicitly resolved engineered fixture — no
established semantic channel or useful mitigation. V11 remains paused. Any
further mechanism test requires a fresh preregistration, a bf16-feasible
placebo validated on actual state geometry, multiple independent cases, both
N/P schedules, and a less saturated calibration regime. The final disposition
is `notes/2026071287-sol-e01-diagnostic-final-interpretation.md`.

## SUPERSEDED CURRENT TRUTH (2026-07-12) — exact-v12 attempt three awaiting A7/B7 freeze

No pod is active. Two exact technical rentals occurred and both stopped before
any subject forward or semantic observation. Attempt 1 used host driver
`580.159.03`; pinned Torch/CUDA 13 initialized and loaded the exact 30B model,
then the loader incorrectly compared raw per-expert checkpoint topology to
Transformers 5's packed MoE runtime topology. The corrected loader now derives
all 531 runtime tensors from all 18,867 checkpoint rows, conserves the exact
parameter count, attests the retained conversion recipe, and bit-checks nine
complete tensors at three cross-shard sentinels.

Attempt 2 used the same Secure A100 type and image but driver `550.90.12`;
pinned CUDA 13 could not initialize, so it stopped before model download. This
proved that the container image did not pin the host driver. The correction
preserves Torch 2.12.1 / CUDA 13 / Transformers 5.0.0 / bf16 / eager, asks
RunPod for CUDA 13, and admits only the exact A100-80GB PCIe with driver
`>=580.65.06` and at least 80,000 MiB before bootstrap.

A fresh code audit then found that the first admission wrapper could swallow a
failed deletion, retry bootstrap failures as bad hosts, suppress required-token
deployment, leak a billing pod on several failures, and race a moving trunk.
Commits `f7c72e5` and `5496b94` closed those paths with explicit 85/86/87
taxonomy, one-pod pre-job cleanup, fail-closed credential transfer, bounded API
calls, exact-commit checkout, local pre-spend frozen verification, and mocked
lifecycle tests. Sol observed `24/24` provisioning tests and the complete
focused suite at `189/189`; Bash syntax, ShellCheck, and Python compilation were
clean. Two fresh Codex audit contexts returned conditional GO after the new
freeze. A capped Fable post-fix review series consumed nominal subsidized usage
but produced no verdict; none is attributed.

Authorization 6 is invalid for the current history. The immediate next step is
an additive A7 apparatus commit and immediate B7 authorization-only child, then
one result-only launch-authorization record, push, frozen verification,
preflight, and one bounded exact technical attempt. Only exits 85/86 may retry,
within three total admissions; exit 87 or any other/new failure class stops.
No Phase A or treatment is authorized by this state.

The two rentals' observed balance deltas total `$0.0971406935`; their combined
conservative bound is `$0.2023333333`. A fresh read observed zero active RunPod
pods and balance `$63.1124828884`; the difference from the last termination-time
balance may include delayed settlement and is not assigned without billing
evidence. The owner additionally requires a final end-to-end audit of money and
tokens across RunPod, Claude, Codex, and other providers, with resumed-session
deduplication and exact/lower-bound/upper-bound/unknown labels. That read-only
inventory has begun in parallel and must be reconciled after the last run.

## CURRENT TRUTH (2026-07-11, ultra-depth regroup) — v11 paused; exploratory canary next

The twelve-case paired-v11 corpus is **paused before any semantic model forward**.
Six committed files remain non-executable discovery/engineering checkpoints:
c01/c04/c05 are mechanically complete but unreviewed drafts, c02/c03 are
authoring-source WIP, and c06 is an unvalidated full draft. Do not expand, score,
or silently promote them. The frozen v11 corpus contract remains a possible
future confirmatory design, not the current execution plan.

An ultra-depth review by Sol, two independent Codex subagents, and a focused
`claude-fable-5` consultation converged that building all twelve long,
token-exact triple histories before observing the exact model has poor decision
value. The next scientific version is a separate exploratory exact-model
decision canary. Its stimuli are permanently excluded from any later
confirmation. The strategy and closed corrections are:

- `notes/2026071190-trajectory-red-team-ultra-regroup.md`;
- `notes/2026071191-causal-statistical-blackboard-ultra-regroup.md`;
- `notes/2026071192-sol-ultra-regroup-decision.md`;
- `notes/2026071193-fable-ultra-regroup-review.md`;
- `notes/2026071192-sol-fable-review-disposition-and-canary-closure.md`.

The canary first runs exact-stack gates and a same-path downstream-note positive
control, then four engineered matched-history cases (three short, one
mid-length). Conversation discovery is conditional on a full-KV engineered
channel. The primary source protocol is a q=1 role-native stepwise replay, not
the current whole-assistant-message P prefill. The primary retained region is
summary content plus the canonical close/boundary; content-only and a fixed
anchor extension are descriptive. All regions contain identical visible anchor
text. Clean semantic evidence comes from same-schedule correct-versus-minimally-
counterfactual state with focal selectivity; fresh comparisons measure utility.
Full-KV and value-only claims remain separate.

Local end-to-end gates must pass before a pod. A critical subset must then rerun
on the exact bf16 30B pod before semantic scoring. Paid order is gates ->
same-path positive control -> one full-arm engineered case -> observed forecast
-> remaining engineered cases -> conditional conversation cells. Initial paid
authorization is `$2`; the full canary has a hard `$8` ceiling only after the
forecast, with at least `$15` reserved for failures and final synthesis/review.

No paid coherent-state compute has run and no pod is active. The latest focused
Fable review plus two capped Claude Sonnet 5 reserve-stimulus authoring attempts
bring conservative provider-usage estimates to `$36.863842` against the owner's
approximate `$60` ceiling; this is not verified as incremental cash billing. The
second Sonnet attempt left complete authored text and exact-width choices in a
temporary constructor; Sol executed that tokenizer-only constructor without
creative changes. E01--e06 now all have committed mechanically passing drafts,
but all remain non-executable pending blind, paired, and diversity review. The
prior RunPod balance observation remains `$63.3160022124`.

The retired local v10 ladder processes (PIDs 55251/55253) and accidental
historical v6 ladder processes (70711/70713) were observed stopped/paused, then
terminated with SIGTERM after their useful durable evidence was committed. A
follow-up process listing contained none of the four PIDs. No authorizing work
was discarded; both designs are nonauthorizing.

Revision-1 e01--e06 is now additively rejected. Personal event tracing found
that the initial planner first fragmented production turn additions, then placed
the carrier after rather than before the retained tail. Those two defects were
corrected in `b51db97` and `7cb98b4`. A further validator audit found that the
length bands had been applied to the whole authored conversation instead of the
literal pre-carrier source prefix: observed pre-carrier lengths were only
670/574/812/1866/665/947 for e01--e06, so every case fails its frozen band. The
formal revision-1 target-aware/diversity review independently returned FAIL
(only e02 passed case-level review; shared scaffold rejected), and the blind
review required revisions to both e04 and e06 histories. Full correction:
`notes/2026071199-sol-v12-precarrier-length-and-review-failure.md`.

Revision-4 is the current authored candidate set. E01/e02/e03/e05/e06 retain
their revision-2 literal histories; e04 revision-3 repaired a late chronology
defect and revision-4 corrected only stale cabinet-control provenance indices.
Fresh independent review passed the carrier and all 12 histories, all six paired
causal contrasts, and cross-case diversity. The complete tokenizer-only manifest
at `results/coherent_canary_validation/coherent_canary_revision4_full_manifest_Qwen3-30B-A3B-Instruct-2507_20260711T205951Z.json`
persists the literal token/event/logical/physical/source arrays and observed
decode round trips. It remains explicitly nonauthorizing.

The execution-gap audit observed that v12 initially had planners but no model
runner, bounded store, release layer, harvester, or budget wrapper. The first
additive runtime core now executes frozen N/fresh events, enforces packed versus
logical positions and cache bounds, transplants selected K/V rows with confinement
checks, causally recomputes suffix events, and scores q=1 targets. Its tests used
a deterministic fake cache model only. The differentiable path control, bounded
persistence/release modes, independent harvester, exact preflight, and local 0.6B
full-apparatus gate remain incomplete. No subject-model forward is authorized;
paid coherent-state compute remains `$0` and no pod is active.

The immediate work is to complete and independently validate that apparatus,
then run the local 0.6B gate. Do not launch exact-stack compute merely because
the stimulus review passed.

## CURRENT TRUTH (2026-07-11) — coherent-state v10 execution

The active experiment is the additive Amendment-1-through-10
`coherent-state-gapped-v10` assay. It asks whether the exact incremental K/V of a
model-generated summary carries correct-history-specific information that a fresh
same-token encoding loses under position-preserving compaction. It does **not** test
every possible write-time channel elsewhere in the cache.

- Exact production checkpoint: `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision
  `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, bf16, eager attention.
- No paid pod is running, paid experiment compute remains `$0`, and no v10
  semantic outcome exists.
- RunPod balance was last observed at `$63.3160022124`; the owner authorized an
  approximately `$60` total ceiling. Completed and failed Fable CLI calls have
  reported a conservative combined `$31.176448` provider usage estimate. This is
  not verified as an incremental cash charge; paid experiment compute remains `$0`.
- The v10 production-tokenizer donor artifact mechanically reconstructed 12/12
  frozen pairs and is committed at
  `results/coherent_state_ladder/coherent_external_donors_gapped_v10_Qwen3-30B-A3B-Instruct-2507_20260711T122320Z.json`.
- The exact local 0.6B bf16 eager CPU ladder is paused as PID `55253` (`T+`), output prefix
  `results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z`.
  Static provenance passed 1/1, attention-backend attestation passed 28/28, and the
  synthetic schedule stage passed seven lengths of one five-token periodic stream
  with aggregate discrepancy exactly `0.0`. That is a smoke test, not seven
  independent fixtures and not natural-content equivalence evidence. The first
  realistic generated-conversation fixture, `c10`, then decisively failed: identical
  8,430 tokens/positions under partitions
  `[4096,4096,238]` and `[23,4096,4096,92,123]` produced cache K/V maxima
  `16.125/5.125`, final-logit maximum `0.59375`, selected-margin shift
  `0.060546875`, and continuation-logit maximum `0.84375`. Layer-0 stored K/V are
  exact; divergence begins at layer 1. The sealed sidecar is committed at
  `4ad714f`. The second realistic fixture, `c02`, independently failed with K/V
  `6.5/5.6875`, final-logit `0.5`, selected-margin shift `0.1318359375`, and
  continuation logits/K/V `0.46875/0.5/1.1875`; layers 0–3 were exact and the first
  stored-cache divergence appeared at layer 4. The two-case sidecar is committed at
  `cfde9bc`. C01 only entered `RUNNING` before the reversible pause and has no durable result.
- The accidental historical v6 CPU ladder remains paused and cannot authorize v10.

Amendment 11 separated scheduling from authorization, but the observed `c10` FAIL now
makes a frozen v10 local `L=PASS` impossible. Therefore no paid v10 technical attempt will
launch: even a paid `T=PASS` would be rejected by the machine-enforced `L AND T` resolver.
V10 is permanently non-authorizing unless a separately preregistered scientific version
replaces it; no gate will be waived or reinterpreted after the failure.

The preregistered three-branch c10 origin diagnostic completed at `f19c4df` with
non-authorizing outcome `QUERY_SHAPE_ROUNDING`. Every A/B/C branch repeated bit-exactly.
The same first 23 tokens diverged between a 23-token and 4,096-token query beginning at
layer 1, row 22 (A/B K/V maxima `0.75/0.875`), while equal-shaped B/C calls with only
causally future rows changed remained exactly equal on protected rows at all layers.
The changed-tail positive control diverged from layer 0, row 23 with K/V
`374.0/118.5`. Thus the tested mask is causally correct and query shape changes
deterministic bf16 arithmetic; the exact low-level kernel path and 30B magnitude remain
unmeasured. The sealed artifact is under `results/c10_schedule_origin/`; interpretation
is in `notes/2026071185-sol-c10-schedule-origin-result.md`.

The raw-input audit found two additional blocking design defects before any semantic
outcome. First, the schedule repeatedly called `message_block`/`message-aligned` is
actually coarse system + whole-history 4096 chunks + request/header; c10 has 47 message
starts, and a true turn-aligned replay would use 46 calls. Second, the frozen `G_wrong`
fills exact-length content slots by cyclically repeating short donor messages; 21–25 of
32 slots per case cycle, with maxima 20–51. It therefore contrasts coherent history with
repetitive corruption and cannot support history specificity. The old `GW` co-primary is
invalid. The proposed replacement is a plant-specific minimally counterfactual coherent
history, renamed `GMC`, with exact per-message token lengths, all downstream references
repaired, frozen-target alignment, and blind decoded-coherence review. Omission is a
secondary diagnostic. No semantic implementation or outcome may use the cyclic arm.

Four c02/c10 minimally counterfactual feasibility witnesses now pass the exact
production-tokenizer mechanical contract: unchanged message count/roles and
non-allowlisted content, exact per-message widths, identical full-prefix lengths,
message starts, and P/O schedules. The review-bound validator passed 113/113 focused
tests and emitted a sealed `MECHANICAL_PASS` artifact at `4ef5393`, while explicitly
recording `semantic_authorized=false`, `execution_authorized=false`, blind review
`FAIL`, and target-aware verdict `REVISE_ALL_FOUR`. The blind review found pervasive
clipped/nonresponsive turns and broken chains inherited from the c02/c10 bases; the
paired factual audit also found edit-specific contradictions and invented details.
None of the four candidates may be scored. They remain immutable feasibility
witnesses; any repair requires a new symmetrically repaired correct/counterfactual
pair and both reviews again.

The repository also contains duplicate banks of all twelve Qwen3-30B model-authored
canonical conversations, but they are one corpus rather than independent renders.
Their old fingerprints omit resolved revision/dtype/backend/code provenance, their
reply records omit raw token IDs/text, and 226/264 replies hit the 320-token cap before
canonical trimming. They may reduce cost only as frozen text candidates after decoded
review; they are not preserved native incremental state and cannot be described as a
fully provenance-bound exact-revision render. See
`notes/2026071184-sol-banked-30b-render-reuse-audit.md`.

The replacement design will not repair c10/c02 or promote the capped 30B bank as its
primary corpus. It will author and freeze twelve new concise, diverse matched
correct/referent-counterfactual/sense-counterfactual conversations before any semantic
outcome. A new additive apparatus must use true turn-aligned replay `P`, ordinary chunks
`O`, fixed gapped destination `D`, and decoded-valid counterfactual sources to measure
`GF`, `GMC`, focal selectivity, and P/O interactions. The first two already-frozen
members form a local 0.6B technical-completeness diagnostic only; they are not an
efficacy or equivalence sample. The exact subject's summary is freely generated under
correct-history P, then the same IDs are forced through all matched histories and
schedules. Imported body text remains explicit replay, not live subject-native state.

The release layer was frozen and implemented without changing any of the 35 inventoried v10
apparatus files; the aggregate remains
`818a60623c4858f0796865a124d255897325136f147ad611c1810026c9352715`, byte-identical to
the ladder-launch commit. Independent code and science re-audits returned GO after the
resolver gained deep 28-layer raw/numeric/source recomputation, exact path/inventory checks,
remote pre-inference closure, exact technical binding, outer receipt, and post-run release
closure. The targeted suite passed 34/34; the prior full suite passed 230 tests and monitor
self-test passed 59/59. A fresh full-suite run on the final exact candidate remains required
before paid launch.

Claude Opus 4.8 raised a recent-prior-art concern that history-conditioned notes may
reside on downstream tokens. After direct geometry inspection, two independent audits,
and a deep `claude-fable-5` consultation, the decision is to keep v10 frozen. Native
retained-tail tokens occur before the summary request in the source but after the summary
in the destination, so the proposed one-arm tail transplant was neither same-position
nor rotation-free and lacked a wrong-history control. Claude accepted and withdrew it.
The binding null boundary is: **no detected downstream-usable channel carried by the
generated summary rows under this fixed assay**, never “no write-time state exists
elsewhere.” See `notes/2026071170-fable-tail-channel-design-review.md` and the concluding
turns of `notes/2026071156-sol-fable-execution-coordination.md`. A same-position
request/header or immediate-post-summary correct-versus-wrong assay is reserved as a
separately preregistered follow-up reusing saved renders.

After data collection, Fable leads a fresh paper draft; Sol owns factual/methodological
truth-checking. The final review stack remains the required multi-angle Fable passes,
critics, terminology and methods/provenance checks, and an independent Codex review.
Only a fully reviewed in-repo paper may be promoted to `README.md` and pushed.

*Updated 2026-07-09. PIVOTED: QK-norm/ablation thesis DROPPED; now testing FRESH-CONVERSATION
REPRODUCTION of the headline. Everything below the "SUPERSEDED" marker is historical.*

## CURRENT TRUTH (2026-07-09) — read first

### ⚠⚠⚠ 2026-07-09 LATE — NO VALID POSITIVE CORNERSTONE (deep audit: notes/2026070976; postmortem: notes/2026070977)
The paper had NO valid positive cornerstone under the owner's rule (**cornerstone must be bf16 AND the
value-only ValueGraft method**). Verdict = honestly a **NEGATIVE / CAUTIONARY / BOUNDING** paper.
- **Precision truth:** BOTH behavioral headlines are **4-bit MLX only** in every SCORED form: recovery
  (judged sense/referent: raw_30b, raw_brief_repro) AND honesty (phase2_30b/4b). Local=MLX-4bit; pods=bf16.
  A bf16 honesty answer set exists (honesty_30b_bf16) but was NEVER SCORED → no scored bf16 behavioral result.
- **Honesty ≠ ValueGraft:** its arm H-pack = coupled KV-graft (α_K=1,α_V=1) in a PACKED non-production
  layout — NOT value-only ValueGraft (E-post, α_K=0, aligned, production layout). AND not content-specific:
  H-pack-wrongS (a DIFFERENT conv's state) suppresses fabrication 24/24 too → a packed-layout calibration
  phenomenon, not value-carried meaning. Supporting at best.
- **The bf16 value-only tests:** effect_bound (placebo-controlled, value-only, on recovery plants) = NULL
  on TWO bf16 models (Qwen3.6-27B + Qwen3-30B-A3B). SWE-Gym E-tuned +0.0156 nats (bf16, value-only) but
  tiny/TF-only/render-null/un-CI'd. So no robust positive at bf16+value-only.
- **STATUS: HALTED.** Provisional honesty-led draft committed (e017631) is SUPERSEDED. Reframing to the
  honest negative/bounding result OR seeking fresh bf16 value-only data — owner deciding; Fable advising
  on how to continue (notes/2026070978). CLAIMS/FINDINGS need a full precision-provenance correction pass.
  4-bit = SUPPLEMENTAL only (weaker/more-artifacts); primary target = bf16 ~30B.

### ⭐ STANDING GUIDANCE + OPEN DIRECTIVES (owner, 2026-07-09) — DO NOT LOSE THESE
1. **Fable writes to a notes/ file for EVERY serious consult** — I pre-create the empty file, pass rich
   context, Fable writes its own assessment there, returns the path. (AGENTS.md "FABLE WRITES TO A NOTES FILE").
   Don't blindly follow Fable — get its perspective, owner decides.
2. **CLAIMS.md is the audited single-source-of-truth** — every shippable number recomputed from disk with
   file + exact command + commit + status. Assemble the paper FROM it. Never second-guess a claim ad-hoc again.
3. **NATIVE/SELF-RENDER PREMISE must be VERIFIED before the paper implies it's a requirement.** Readers will
   infer self-rendering is required; it's NOT a problem if it turns out only helpful (still more robust/interesting)
   — but we must interrogate it, not assert it. Verification experiment being designed (Fable). Capture + reflect.
4. **RE-RENDER + BANK every key model's conversations** for reproducibility, each tagged with model ID +
   quantization + size. Fleet banks 32B/Mistral/Qwen2.5/phi-4 (never banked before). 30B already banked.
5. **Champion scans** (owner override of Fable's hold): mechanism map per model; bank the renders.
6. **BRIEF-CONDITION CAVEAT is load-bearing:** the judged +12pp AND SWE-Gym +0.0156 both ran under the
   brief (mechanism-isolation, baseline-handicapping) summary. Fable's #1 recommendation: get a
   **production-faithful (std-summary) arm** for the headline — currently NO positive result exists under std.
7. **Push-notify major results.** SWE-Gym cross-model = cheap warm-pod add-on (LOWER priority).
8. Fable holistic assessment (2026-07-09): notes/2026070972-fable-holistic-assessment.md — argues honesty-led
   may be safer than sense-led (degrades gracefully); std-summary arm is highest-leverage missing piece.

**THE PIVOT (incident #37, from re-asking Fable UN-ANCHORED):** the cross-arch QK-norm thesis (H1) is
DROPPED — empirically FALSIFIED (Mistral no-QK-norm referent +0.035, CI [+0.005,+0.066] excludes 0 =
the OPPOSITE of H1) and the within-model ablation BROKE the model (removing QK-norm → dead generation).
H1 is reported as a PRE-REGISTERED NULL ($0). The real threat, surfaced only by the un-anchored
consult: the headline +0.10 rests on 12 HAND-AUTHORED convs (c01-c12); the fresh augment (c13-c54) did
NOT carry it on the old render (~+0.009); "native render fixes it" was ASSERTED but never shown on disk.

**CURRENT EXPERIMENT — fresh-conversation reproduction (BLOCK design):**
- ONE apparatus over c01-c36 (Qwen3-30B-A3B native render, greedy self-gen, per-token), analyzed by
  BLOCK: c01-c12 = BASELINE (positive control, MUST reproduce +0.10), c13-c36 = FRESH (held-out test).
- Assembled across pods: canary renders c01-c24, expm renders c25-c36; pooled CENTRALLY (ONE relative
  competence floor over all lp_A, conv-clustered CI) via scripts/block_analysis.py.
- Selector SC_CONV_START (validated: start=12,limit=24 → exactly c13-c36). Pre-committed stopping rule:
  extend to c37-c54 ONLY if the fresh referent CI spans 0 at n=24.

**RESULTS BANKED (2026-07-09):**
- SENSE judged +12pp HOLDS under conv-clustered bootstrap: CI [+2.2,+22.9], excludes 0 (task #30).
- REFERENT judged +9.7pp does NOT (CI [-6.9,+26.2]). SIGNIFICANCE FLIPS BY METRIC (logprob:
  referent-sig/sense-weak; judged: sense-sig/referent-weak) — dissociation real but metric-dependent.
- Cross-arch (preliminary, self-native): Mistral referent+ / sense- / stance- (DIFFERENT signature than
  Qwen +/+/null); Qwen2.5 referent-; phi-4 null. Graft effect is ARCHITECTURE-SPECIFIC and can reverse
  — THAT (not QK-norm) is the novel finding to lead with.
- Methodology: CI now CONVERSATION-clustered (was plant = anti-conservative); competence floor now
  per-model RELATIVE median-3*MADN (pre-registered) — OLMo "empty alignment" was a floor misdiagnosis.

**LIVE (2026-07-09, balance ~$25/$80 cap):**
- LOCAL: brief-condition judged decider generating (results/raw_brief_repro/ c01-24; the +12pp reproduction
  + held-out test). $0.
- PODS (5, champion scans — OWNER OVERRIDE of Fable's hold): cs30b (Qwen3-30B MoE, REUSE banked renders —
  validated: 12/12 reused, scoring champion configs), cs32b (Qwen3-32B), csmis (Mistral-24B), csq25
  (Qwen2.5-32B), csphi (phi-4) — the four fresh ones RENDER+BANK c01-12 (fills the reproducibility gap:
  we never banked renders for these) then champion-scan. Env-parse crash (empty SC_CONV_START) fixed f128712.
- CLAIMS.md = audited ledger (committed): 16✅/5⚠/1❌/1⏳. F2 "38/48 accurate" = ❌ overclaim (corrected).
- OPEN CRITICAL QUESTION (owner, 2026-07-09): the NATIVE-RENDER premise — we motivated huge work (native
  re-rendering) on the assumption it's the valid measurement; native referent +0.012 vs authored +0.125 shows
  nativeness changes the result enormously, but we have NOT rigorously verified the premise. Fable designing a
  moderate verification experiment; capture + reflect required. Also planned: SWE-Gym eval across the fleet
  models (cheap on warm pods).

*(Superseded note: earlier today all pods were briefly off after the block reproduction resolved — the Fable
verdict [referent headline FRAGILE, not a regression] still stands; renders banked to results/redraw_harvest/.)*

**CURRENT DECIDER (running locally, brief condition):** Fable's un-anchored verdict = the whole positive
story rests on the judged SENSE +12pp, which exists ONLY for c01-c12 and was NEVER tested held-out.
Reproducing it under a clean current-code pipeline for c01-c12 (baseline: does +12pp reproduce?) AND
c13-c24 (held-out: does it survive?), then judge+bootstrap+compare. PROVENANCE CORRECTION (see FINDINGS
banner): judged answers are 4-bit MLX (not bf16) under the BRIEF summary condition (mechanism-isolation,
B-handicapped) — restored the dropped SC_SUMMARY=brief knob in run_arms.py (c01 positive control: 543-char
brief summary reproduced exactly). Builder: scripts/build_judge_batches.py (validated vs committed).
Std/production-faithful held-out = planned second arm. Paper draft (framing-independent) in paper/DRAFT.md.

**PLAN (ordered):** block read (baseline reproduces? fresh carries?) → full n=24 fresh → arch poles →
FABLE UN-ANCHORED trajectory review → pick honest framing → robust-metric audit (#29) → PODS OFF →
write paper FRESH (heavy Fable, current requirements) → repo + README when confident (external internet
publish NEEDS owner). Opportunistic tail only: Wave3 / OLMo-relative-floor / breadth.

**KEY FILES:** FINDINGS.md=results; DECISIONS.md=pivot+policies; PREREGISTRATION.md=H1-null +
relative-floor amendment; scripts/block_analysis.py + judged_bootstrap.py; notes/README.md=summaries.
Paper=task #38, written FRESH (NOT the archived REPORT.md).

## ---- (everything below SUPERSEDED — historical 07-08 and earlier) ----

*(Historical state layers 07-04..07-08 removed 2026-07-09 for clarity — full history in git + notes/ summaries.)*
