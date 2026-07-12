# V12 e01 Phase A attempt 2 — one-host authorization

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

Attempt 1 allocated Secure pod `otv9b2lyptkmh2` but it never acquired a
runtime, public IP, or port mapping. It was deleted after 189 seconds, before
bootstrap or any scientific/model action. The active-pod list is now empty and
direct lookup returns HTTP 404. This is the known degraded-host class, so one
fresh provisioning attempt is scientifically unchanged and operationally
proportionate.

This file authorizes **one** new Secure A100-80GB Phase-A host for e01, using
the exact scope, frozen bindings, ignored job/puller hashes, gates, and terminal
rules in
`results/coherent_canary_v12_budget/coherent_canary_v12_phase_a_e01_launch_authorization_20260712T0216Z.md`.
It does not authorize treatment or a third automatic allocation.

- Immediate attempt-1 balance delta: `$0.0306665676`.
- Conservative attempt-1 full-window bound: `$0.072975`.
- Phase-A unit cap: `$0.50` across attempts.
- Conservative remaining unit allowance: `$0.427025`.
- Current balance before this authorization: `$62.8432299995`.
- Active pods before this authorization: `0`.
- At `$1.39/hour`, the conservative remaining allowance corresponds to about
  18.4 rental minutes. Terminate earlier on a terminal artifact, platform
  rejection, setup/error marker, or missing progress.

The exact commit adding this result-only authorization is the second launch
head. It must remain clean, pushed, pass the production frozen verifier, and be
checked out literally on the pod. Any incompatible/degraded second host is
terminated and stops provisioning pending a new decision; it is never silently
replaced.
