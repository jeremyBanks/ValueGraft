# Cloud scale-up plan (for review — nothing here is started)

Goal (revised 07-05 after user discussion): SCALE THE EVIDENCE on the model
we already know — Qwen3-30B-A3B in bf16 — against established benchmarks and
real agent/coding traces at high n (the Phase-2 doc's Strategy P: our effects
are real-but-conditional, so power > new models). Model diversity is a
secondary stage, not the point of renting hardware.
Everything below is designed so **you do ~10 minutes of account setup and I
do the rest over an API key + SSH**, with spending capped by construction.

## Recommendation: RunPod, prepaid credits (not AWS)

Why not AWS first: new/low-history AWS accounts must file quota-increase
requests for GPU instance families (often 1-3 day turnaround), the GPU
instance names are opaque, and billing is post-hoc (easy to overspend).

Why RunPod: **prepaid credits are the spend cap** — you load $50, we can
never spend $51. Per-second billing, one-click Ubuntu+CUDA+PyTorch pods,
simple REST API I can drive end-to-end, community A100/H100 pricing among
the cheapest. (Lambda Cloud is a fine alternative — cleaner machines,
slightly simpler — but bills to a card without a hard cap, so RunPod's
prepaid model wins for the guardrail.)

## What you do (~10 minutes, when you're ready)

1. Create an account at runpod.io (email + password, verify).
2. Billing → add credits: **start with $25** (enough for all of stage 1 and
   most of stage 2; we can top up deliberately later). Do NOT enable
   auto-top-up — its absence is our hard cap.
3. Settings → API Keys → Create key (READ & WRITE scope).
4. Give me the key (paste it in chat, or put it in a file like
   `~/.runpod-key` and tell me). That's everything.

## What I then do (no further input needed)

1. Generate a dedicated SSH keypair locally; register the public key on the
   account via API.
2. Provision the cheapest suitable pod (see stages), SSH in, `rsync` the
   repo, set up Python env.
3. Run, monitor via detached logs exactly like the local runs, pull results
   back with rsync, **terminate the pod** (I terminate — never leave a pod
   idle; idle pods bill).
4. Report per-stage cost from the API before starting the next stage.

Safety rails I will follow (written here so they're auditable):
- Confirm with you (or a written go-ahead in advance) before the FIRST pod
  launch and before any stage projected >$30.
- One pod at a time; terminate before switching stages; check the credit
  balance via API before and after every stage and log it in DECISIONS.md.
- If a run stalls, terminate first, debug locally second.

## The technical prerequisite: a transformers port (mostly free, done locally)

MLX is Apple-only; cloud GPUs mean CUDA + HuggingFace transformers. The
surgery is ~all "manipulate per-layer K/V tensors + position offsets," which
maps to transformers' `DynamicCache`. Plan: port `kvlib.py`/`arms.py` cores
and **re-run the L0–L4 + LH identity ladder locally** (Qwen3-4B on MPS —
slow but sufficient for identity tests) BEFORE renting anything. Cloud hours
then buy experiments, not debugging at $3/hr. **STATUS: core port DONE and identity-validated locally** (src/kvlib_hf.py +
src/l_hf_ladder.py, all-pass on Qwen3-0.6B/CPU, keys post-RoPE confirmed,
re-rotation exact to 7e-5 in fp32). Remaining port work: arms/runner
adaptation (~an hour), best done on the pod against the actual target model.

## Stages and cost estimates (RunPod secure-cloud prices, ±30%)

Primary model throughout: Qwen3-30B-A3B-Instruct-2507 **bf16** (fits 1×
A100-80GB; fast — 3B active params). Also removes the 4-bit-quant confound
from every local result.

| stage | what | time | est. cost |
|---|---|---|---|
| 0 | Pod setup + HF-port identity ladder on the 30B bf16 | ~1-2 h | ~$3 |
| 1 | **LongMemEval, full 500 questions** (vs our n=36-48 sample), all 6 arms, incl. the multi-session + temporal-reasoning types we skipped locally | ~8-12 h | ~$15-25 |
| 2 | **Coding-agent traces (SWE-Gym OpenHands trajectories)**: filter to ≤16K tokens, n≈50-100; offline next-action prediction under compaction arms + behavioral checks (does the compacted agent re-run already-failed commands?) | ~6-10 h | ~$12-20 |
| 3 | **Second standard benchmark**: LoCoMo (long-conversation memory, ACL 2024) or SCBench (Microsoft, KV-lifecycle-aware — closest published eval framing to ours); pick whichever fits ≤16K instances better on inspection | ~4-6 h | ~$8-12 |
| 4 | (Only if a gap needs it) powered synthetic probes n≈50 — our controlled instrument for leakage-audited fabrication decoys, which standard benchmarks lack; small and clearly labeled as ours | ~3-4 h | ~$6-10 |
| 5 | (Bonus, budget permitting) model variety: Mistral Small 3.2 24B on stages 1/3 subsets; Llama-3.3-70B or others only if budget clearly allows | ~4-8 h | ~$10-25 |
| 6 | (Contingent) hybrid testbed: Gemma 3 27B profile-first | ~3-4 h | ~$6-8 |

Stages 0-3 ≈ $40-60 — the core, all industry-standard data. 4-6 from remainder.

## Decision points for you (defaults chosen, change freely)

- Provider: RunPod (default) vs Lambda vs AWS (only if you specifically want
  to stay in AWS — then: request quota for `g6e.xlarge`/`p4d` now, since
  approval latency dominates).
- Second family: **Mistral Small 3.2 (24B)** (default — modern, dense, standard
  attention; Ministral-8B as the cheaper fallback) vs OLMo-2-32B (fully-open
  reproducibility pick). NOTE most 2025-26 frontier open models are
  architecture-incompatible with the surgery (Llama-4 iRoPE/NoPE layers,
  DeepSeek/GLM MLA latent caches, Gemma sliding-window, Qwen3.5+ linear
  hybrids) — verify config on-pod before committing to any model.
- Scale anchor: Llama-3.3-70B stays (newest CLEAN dense 70B: vanilla
  GQA+RoPE); its age is an architectural constraint, noted in write-up.
- Budget (updated 07-05): user loaded $50 + $50 top-up = ~$100 total. Full
  plan incl. Llama-3.3-70B (~$85 worst case) is authorized. A further $100
  is CONDITIONAL: only if results through the current budget are very
  promising, I recommend it at that point, AND the user agrees again.

## Hardware strategy and utilization (agreed 07-05)

- **One pod, one A100 80GB, serial stages.** 30B-A3B bf16 (~61 GB weights)
  fits one card; workload is bandwidth-bound so H100 is worse per dollar;
  multi-GPU only if the (bottom-priority) Llama-70B option ever runs.
- Tier: secure cloud (~$1.6–1.9/hr) for the first pod; community
  (~$1.1–1.4/hr) acceptable for long batches afterward — all runners are
  per-item resumable so preemption is cheap.
- Utilization: provision → detached resumable batch → rsync results →
  TERMINATE → analyze/judge locally off-meter → next stage. GPU billed only
  while a batch runs. Add a small network volume (~$7/mo prorated) to cache
  the model if >2 stages.
- Total core estimate: 25–35 GPU-hours ≈ $40–60.

## Credentials (staged, NOT yet authorized for use)

- `.runpod_key` and `.huggingface_key` exist in repo root (chmod 600,
  gitignored, never committed). DO NOT use either until the user explicitly
  approves the first pod launch. HF gating: everything in the core plan is
  ungated; Gemma approved; Llama-3.3-70B pending Meta approval (dispensable).
