# Day plan (07-06) — pending user wake

Board: 2 pods (e1 shim w/ session-isolation fix reloading + anomaly
re-runs auto-firing; p4 idle-ready). Balance ~$48.8. Night results:
MORNING-REPORT.md.

## Pre-approved / in flight
- α=0.75 anomaly re-runs (6×E + 2×B, fresh seeds s20-s22, isolated
  sessions) — decides whether the 27% cell was a shim state-leak artifact.

## User decisions needed
1. **E2 / champion confirmation** (~$25-35): harder multi-constraint t3
   (5+ checked constants across files — must NOT fit a brief summary),
   n≥20/arm on A / B / E:cfg=layers / E:a1.0. This is the headline
   experiment. t3 AUTHORING IS NOT DONE — write it fresh with full
   attention, self-test fail-pre/pass-post, and dry-run before pod time.
2. **Mistral variety** (~$5, p4): pre-tune sweep via template_ops
   (system-less; Mistral-Small-2409). Machinery ready.
3. **Blog integration day**: fold night results into blog-draft.md —
   agent-table + guards story + honesty-bf16 table; scenario framing per
   writeup-guidelines.
4. Gemma hybrid scoring path (~2h code) — optional.
5. Sealed final eval (s50-s99): stays locked until champion stable.

## Notes for whoever executes
- Champion candidate = cfg=layers (9/9 night, guard-passed); E:a1.0 backup.
- All arms MUST re-run under the fixed shim (per-mode sessions) — night
  arm-vs-arm comparisons may carry state-leak noise; the fix is committed.
- Recall/dissociation probe now captures final assistant message — usable
  going forward.

## Infra: network-volume model cache (user, 07-06 morning)
15-min pod spin-up is ~12 min HF download + ~3 min load. FIX: RunPod
network volume (~70GB ≈ $5/mo, per-datacenter, secure-cloud only): create
in current pods' DC, pre-warm /workspace/hf from a live pod, add
networkVolumeId + dataCenterId to pod.py create (env SC_POD_VOLUME).
Payoff: cold pod → serving in ~3 min; biggest win for spot churn.
Implement opportunistically behind gate-watching.

## icache v2 validation plan (user-requested; build at confirm boundary, NOT before)
(1) bit-exact CPU test of in-place extension (no per-call clone);
(2) VRAM predictor formula validated against measured allocator peaks on
0.6B at 1K/4K/16K (±15%) → same formula, 30B params, drives a live
headroom guard (refuse-to-cache, serve fresh, never die);
(3) fake-budget local soak: interleaved A/B/E crossing the cap — asserts
cache-drop logging, correct service, process survival;
(4) pod-side: measured-vs-predicted check on first real calls gates the
flag. Checklist question added: "peak VRAM at max realistic input, shown
as arithmetic."

## Fallback ladder (validated, 07-06 afternoon)
1. SWE-bench Verified "<15min" x docker-free repos (48 inst; 2 validated;
   sympy id fix REQUIRED in swebench_tasks._clean_ids first)
2. BugsInPy lightweight subset (~74; license check needed)
3. synthetics (floor)
Trigger per DECISIONS autonomy rule + difficulty screening.

## Model-swap options if capability floor persists (user asked; NO action)
1. Qwen3-Coder-30B-A3B: drop-in (same arch/template/size/pods), coder-tuned;
   ~$3 recalibration; first resort.
2. Qwen3-235B-A22B: same family, big capability jump; multi-GPU ($12-25/hr),
   conditional-budget scale; episode math must be redone.
3. Cross-family (70B dense etc.): new adapter+calibration+continuity break;
   last resort.
GLM scaffold-validation plan (if that rung is reached): layer 1 = full
template dry-run vs the TARGET model's tokenizer (KBs, exact); layer 2 =
serving-path smoke on GLM-4-9B 4-bit locally (~6GB). Zero pod spend
before commitment.
SWAP-LADDER SCAFFOLD STATUS (validated 07-06 night, tokenizer-level, 6/20
convs): Qwen3-Coder-30B = qwen family, PASS (drop-in); GLM-4.5-Air +
GLM-4-9B = prefix-stable via generic path, PASS. Remaining before any
pivot: full 20-conv dry-run (~5 min), local serving smoke (0.6B qwen /
GLM-4-9B), calibration pass (~$3), gold-patch scorer smoke.
Disk: 96G free; HF cache 34G (no pressure).
GLM current-family fact (checked 07-06 night): NO small member exists —
GLM-4.5-Air (106B-A12B) IS the smallest of the current line (4.6 adds no
small variant). 9B-0414 = prev-gen dense: code-path smoke only, NOT
representative. Representative tooling smoke = 1-hour single pass of
Air-FP8 itself on 2xA100 (~$4): mini-ladder + one A-mode episode; doubles
as capability evidence + first onboarding step if swap proceeds.
SCAFFOLD-CONFIG SUSPECTS (user incredulity → audit, 07-07): (1)
native_tool_calling=False in e1_agent.py (bring-up choice; Qwen3 is
trained for native calls); (2) temperature 0.0 (Qwen card recommends
~0.7; documents greedy degradation in long generations). LADDER INSERT:
if chain smokes fail → scaffold-config A/B (native calls + recommended
sampling on one failed instance, ~$1) BEFORE any model swap. If chain
smokes pass → config adequate, difficulty is real.

## Serving-stack upgrade ideas (captured 07-07, NOT scheduled)
1. vLLM KV-connector graft plugin (days-weeks): paged-KV blend via the
   cache-transfer/LMCache interfaces; version-brittle; the right EVENTUAL
   home — also the write-up's "deployment path" section (opaque handle as
   a serving-layer feature; our shim = reference implementation).
2. Own-stack hot-path upgrade (~1 day, 2-4x): FA2 kernels + compiled
   decode + async server (replace wsgiref) + icache-v2 (validated,
   benched). Keeps surgery freedom + test discipline. The investment if PRIORITY RAISED (user 07-07):
   FIRST infra investment at next phase boundary, before confirm/E2
   spend; ships with full validation ritual.
   the program continues at scale.
