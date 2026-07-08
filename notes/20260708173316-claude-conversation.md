_This shard covers a cross-architecture sweep operational incident: a validation
crisis on the w1 (Qwen3-30B-A3B) worker caused by racing/stacked job launches,
its diagnosis, and the subsequent decision to stop micromanaging and let
automated monitoring drive the sweep._

## Science-side confirmation

The aggregator confirmed that the previously-applied QK-norm fix works:
Qwen3-30B-A3B now correctly reports `qk_norm=True (module:q_norm)`, meaning the
pre-registered H1 (QK-norm) predictor can be reliably measured across the sweep.
This validates earlier infra fixes (env-forward, timeout scaling) as
functionally correct at the harness level.

## The w1 incident

After the sweep was declared "fully set up" (per-token re-run, correct 6.5h
timeout, monitor armed, aggregator ready, pre-reg frozen), w1 reported
UNSUPPORTED. Initial hypothesis was a stale result file (the re-run had only
started ~10 minutes prior against a ~5hr runtime), but direct investigation of
the pod found the real cause: multiple w1 processes were racing on the same GPU.
Repeated manual re-launches had stacked rather than replaced each other — one
process (pid 4233) was still running under the old 3600s timeout — and three
processes sharing one 80GB GPU caused a collective OOM. This reconfirms the
earlier lesson that per-token rendering itself does not OOM; concurrent
duplicate processes do.

Fix applied: `job_sweep.sh` now includes a kill-guard so relaunches replace any
prior process on the same worker rather than stacking (already reflected in the
repo's committed state per git log: "job_sweep: pkill prior cross_arch_probe
before start").

## Operational grinding pattern (self-corrected)

Attempting to clean up w1 in place led to a repeated failure loop: SSH into the
reused pod (11793) intermittently returned empty/dropped connections, and each
retry to check-and-relaunch raced against the previous attempt's launch,
repeatedly reintroducing 2-3 duplicate processes despite the kill-guard fix.
This was diagnosed as pod degradation under heavy reuse combined with
launch-timing races that the guard couldn't fully close.

Decision made: stop grinding on the degraded, heavily-reused pod. Terminated pod
11793 and relaunched w1 on a fresh pod instead of continuing to fight the flaky
one — fresh pods only ever receive one launch each, so they don't hit the same
race condition. This is flagged explicitly as a pattern to avoid in future infra
work: don't repeatedly retry against a single degrading resource when a clean
replacement is available.

## State at shard boundary

- 3 pods running cleanly (community-capacity retry loop provisioned them, one
  job each, no racing).
- w1 relaunching fresh after pod termination.
- All known bugs (env-forward, timeout scaling, kill-guard) are fixed and
  deployed, so any new pod the retry loop brings up runs correctly from launch.
- Monitor and aggregator remain armed to surface results and compose
  significance summaries (sign-map, H1/QK-norm test, de-confound pair check) as
  models land.
- Bottleneck is community GPU availability, which is outside direct control and
  trickles in via the existing retry loop.
- Explicit handoff: further work should be monitor-driven rather than manual
  per-pod intervention; science-side problems should route to Fable; STATE file
  should be kept current as results arrive.

---
