# Provider storage and ephemeral-pod economics for v13

**Date:** 2026-07-12  
**Status:** official-documentation review and current main-agent disposition;
no provider resource was created or changed.

## Verified provider rules

RunPod's current official documentation says:

- Pod compute and ordinary Pod storage are billed by the second. On-demand
  deployment requires at least one hour of credit, but billing is not rounded
  to an hour.
- Container disk is temporary and is erased when a Pod stops. It costs
  `$0.10/GB/month` while running and is not charged while stopped.
- Local volume disk is mounted at `/workspace`, survives stops/restarts for the
  same Pod lease, and is deleted when the Pod is terminated. It costs
  `$0.10/GB/month` while running and `$0.20/GB/month` while stopped.
- Stopping a Pod releases its specific GPU but keeps the Pod tied to the same
  physical machine that holds the local volume. Restart can therefore return
  zero GPUs if that machine's GPUs were rented; migration is a separate beta
  path.
- Network volumes survive Pod termination, are portable among Pods within
  their datacenter, replace the local `/workspace` volume, and are available to
  Pods only in Secure Cloud. They cost `$0.07/GB/month` below 1 TB, billed
  hourly, and constrain deployment to the volume's datacenter. The documented
  nominal transfer range is `200–400 MB/s`, with higher peaks.
- A Pod with a network volume cannot be stopped; it is terminated while the
  separate volume persists.
- RunPod templates can pin an image/environment, and a custom image can bake a
  Hugging Face model. RunPod's automatic cached-model feature is documented for
  Serverless endpoints, not as a general Pod guarantee.

Sources:

- <https://docs.runpod.io/pods/pricing>
- <https://docs.runpod.io/pods/storage/types>
- <https://docs.runpod.io/pods/manage-pods>
- <https://docs.runpod.io/pods/troubleshooting/pod-migration>
- <https://docs.runpod.io/storage/network-volumes>
- <https://docs.runpod.io/pods/templates/create-custom-template>

## Economic implication

Short-lived compute is genuinely economical because compute is per-second.
The obstacle is not a billing-minute floor; it is setup/replay time and host
compatibility. At the previously observed effective A100 rate near `$1.42/h`,
twenty minutes of download/setup costs about `$0.47`, whereas keeping the GPU
warm costs about `$0.024` per minute. A 100 GB stopped local volume costs about
`$0.66` per day and a 100 GB network volume about `$0.23` per day. Those
charges are modest for a short handoff window but are not free long-term.

The reliability tradeoff dominates:

1. A stopped local-volume Pod preserves the cached environment on the same
   physical machine and therefore has the best chance of preserving the host
   driver, but the released GPU may be unavailable at restart.
2. A network volume avoids another model download after termination, but it
   narrows datacenter supply and does not preserve GPU/driver identity. A
   60–70 GB model read at the documented ordinary throughput can still take
   several minutes before model initialization.
3. Baking this exact large model into a custom image would create a large image
   build/pull/distribution project and does not remove model-load time. It is
   not justified for the remaining one primary batch plus one host audit.

## v13 disposition

Do not create a network volume before the primary Phase-A admission. The
current scarcity/driver risk and datacenter constraint are more consequential
than the small possible cold-start saving, and the full v13 workflow has only
one planned core batch. Keep all important artifacts locally/committed; RunPod
storage is not archival storage.

Place the Hugging Face cache and environment under a sufficiently sized local
`/workspace` volume on the admitted Pod so a bounded stop/resume remains a
technical fallback. It is not automatically authorized for decision-bearing
reuse: a resumed Pod must re-attest machine/GPU UUID, driver, CUDA, dependency,
model, tokenizer, and exact production gates. If the runtime fingerprint
changes, old tensor bundles remain audit evidence and the new runtime follows
the preregistered rebuild/reverification path.

For the Stage-A-to-Stage-B review, the first choice remains a rapid automated
manifest review while the admitted core host is warm, bounded by the existing
45-minute/`$1.10` ceiling. Stopping solely to save a few warm minutes adds GPU
reacquisition risk and is not worthwhile. If review cannot finish within that
bound, harvest and terminate as preregistered; reconsider a network volume only
if an observed cold pilot shows setup time materially larger than the current
projection and a literal additive protocol/budget change passes before any
unseen treatment.
