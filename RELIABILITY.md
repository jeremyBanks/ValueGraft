# RELIABILITY.md — successor prevention map

The complete pre-takeover manual is archived in `notes/` under the slug
`legacy-reliability-through-20260712`. This root guide contains only mechanisms
that must be active for the powered successor.

| If we do this | It must be caught by this |
|---|---|
| Launch paid compute | Frozen preflight verifies exact commit, clean explicit inventory, budget, active-pod count, credentials, model/config, host admission, output path, and checkpoint sink |
| Load a model | Runtime attests exact model/revision/dtype/backend/geometry/parameter count and expected CUDA VRAM; wrong checkpoint and CPU/offload injections fail |
| Use a new config/model | Required build ladder plus actual production-path positive/identity controls pass before outcomes |
| Score a case | Mechanical geometry, decoded review, oracle competence, fresh damage, exact arm lineage, and applied placebo availability are independently validated |
| Run many cases | One measured short and mid case project time/disk/$; atomic per-case artifacts bound loss to one case |
| Detach a job | Captured parent pipe reaches EOF, wanted child survives under PID 1/supervisor, GPU memory rises, and log reaches real work |
| Trust a monitor | Fault-injection and happy-path self-test pass; expected set includes absent/never-launched state; monitor itself is independently supervised |
| Report progress | Count complete independently valid case artifacts and spend, not processes or log markers |
| Stop/delete a pod | Wanted artifacts are pulled, hash-checked, committed, pushed; DELETE returns; direct lookup is 404; active inventory is empty |
| Retry allocation | Only explicit no-allocation or positively deleted rejected host may retry within a frozen attempt bound |
| Change retained state | Commit message answers the five-part stateful-change checklist and production-path probe gate |

## Preflight sequence

1. **Repository:** `trunk`; explicit clean/allowed-dirty state; exact commit
   pushed; no credential or >4-MB file staged; unique outputs cannot overwrite.
2. **Budget/lifecycle:** fresh balance and active-pod inventory; session and
   per-case ceilings; reserve; deletion owner; retry taxonomy; provider timeout.
3. **Inputs:** all case/control hashes, author/review provenance, sampling order,
   eligibility rule, independent-unit count, analysis/stopping code frozen.
4. **Runtime:** provider CUDA filter; actual A100-80GB name/memory; driver floor;
   CUDA; disk; network; exact package lock; no unbounded/moving dependency.
5. **Model:** exact ID/revision; bf16 or declared precision; eager or declared
   backend at every layer; no offload; tokenizer/template hashes; correct loaded
   tensor topology and parameter count.
6. **Apparatus:** identities, signed bidirectional path control, can-it-fail
   fixture, applied-dtype placebo on real geometry, independent harvester, and
   per-case resume equivalence.
7. **Real work:** within two minutes of launch verification, observe expected
   VRAM and a durable first real checkpoint—not only a PID or setup marker.

## During a paid session

- Source one pure classifier for pod/job state; monitors never reimplement it.
- Every expected pod/job is one of: allocating, rejected-and-deleted, booting,
  real-work, checkpointed-progress, terminal-success, terminal-failure, or
  cleanup-failure. Unknown/missing is an alert.
- Monitor per-case output rate, last durable checkpoint age, GPU utilization and
  memory, log error delta, disk headroom, provider deadline, and balance.
- Pull each completed case while the pod continues. Local committed bytes are
  the audit record; pod storage is not.
- The live gap ledger shows valid independent N / next look / max N and observed
  phase spend / verified runway.
- Any code repair stops scientific scoring. Preserve state, terminate or stop as
  the frozen lifecycle says, patch locally, rerun allowed gates, and relaunch
  only under a new recorded attempt.

## Terminal conditions

- **Scientific stop:** the frozen sequential decision resolves, max N is
  reached, or a predeclared validity/control rule fails.
- **Operational stop:** driver/model mismatch, CUDA/offload, persistence or
  harvest corruption, ambiguous allocation, cleanup failure, budget ceiling,
  or monitor loss without independent coverage.
- **Success is retrospective:** only independently validated committed numbers
  count. A launch, PASS label, exact repeat, or favorable point estimate is not
  completion.

All long-lived operational state is bounded and has a named owner/eviction
path. If that sentence cannot be made literal in code, do not deploy.
