# RunPod storage and session economics for the powered successor

**Date:** 2026-07-12. **Scope:** operational planning only; no provider resource
was created or changed during this review.

## Observed local facts

- The live account read at `2026-07-12T14:39:21Z` returned balance
  `$57.1287946692`, spend limit `$80`, and zero active pods.
- Prior exact A100 allocations reported `$1.39/hour`.
- A 20-minute cold start at that rate costs about `$0.4633`, comparable to the
  complete observed e01 treatment allocation delta (`$0.4342`).
- Current `src/pod.py` requests `containerDiskInGb=200` and `volumeInGb=0`.
  Previous HF caches under `/workspace/hf` therefore did not survive pod
  deletion and are not presently available to a new pod.

## Provider documentation checked

- RunPod’s current [Pod pricing page](https://docs.runpod.io/pods/pricing) says
  Pod compute and storage are billed by the second. Its higher-level overview
  still says “by the minute,” so accounting should retain actual provider
  timestamps/balance evidence and conservatively tolerate minute rounding.
- [Storage types](https://docs.runpod.io/pods/storage/types): container disk is
  erased on stop/restart; local volume disk at `/workspace` survives stop but is
  deleted with the pod; a network volume survives pod deletion and can be
  attached to future pods.
- [Pod management](https://docs.runpod.io/pods/manage-pods) and the
  [zero-GPU restart note](https://docs.runpod.io/pods/troubleshooting/zero-gpus)
  explain that a stopped local-volume pod remains tied to its physical machine
  and may restart without a GPU if that slot was rented.
- [Network-volume documentation](https://docs.runpod.io/storage/network-volumes)
  gives standard storage as `$0.07/GB/month`, Secure Cloud only for Pods, and
  notes that a volume constrains pod deployment to its datacenter. It can reduce
  model re-downloads but can also reduce GPU/driver failover options.
- The [Pod creation API](https://docs.runpod.io/api-reference/pods/POST/pods)
  exposes `networkVolumeId`, `volumeInGb`, datacenter filters, and
  `allowedCudaVersions`; actual host driver/GPU admission remains necessary.

At 100 GB, standard network storage is about `$7/month`, or `$0.0097/hour`.
Two days cost about `$0.47`: almost exactly one observed 20-minute cold start.
It pays economically only when it avoids more than one such cold start during
the volume’s lifetime; its main value would be resilience, not large dollar
savings.

## Decision for the core run

1. Keep one compatible admitted A100 host warm through each Goal-A case batch
   and local sequential-look pause. Cases and analysis must be ready before
   allocation so the host is never waiting on authoring or code repair.
2. Do not add network-volume creation to the first validation/core critical
   path. The planned core needs one main host and one independent replication
   host; saving only the second download does not clearly repay the new
   datacenter/driver constraint and lifecycle code.
3. Use a second fresh acquisition for host portability evidence; its cold start
   is part of the replication budget rather than population N.
4. Reconsider a 100-GB network volume only if the frozen exploration plan needs
   at least three additional acquisitions on the same checkpoint, or if an
   observed first-session failure makes cache portability worth more than its
   scheduling constraint. Any such change gets its own lifecycle test and cost
   record before creation.
5. Pod-local results remain checkpoints, never the sole copy. Pull, hash-check,
   commit, and push every completed case while the warm pod continues.

This choice is reversible before launch and does not alter the scientific
design. No storage resource currently accrues charges.
