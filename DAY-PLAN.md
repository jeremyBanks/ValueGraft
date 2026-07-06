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
