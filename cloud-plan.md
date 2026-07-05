# Cloud scale-up plan (for review — nothing here is started)

Goal: replicate the local findings on (a) a second model family and (b) a
70B-class model, at higher n, per the Phase-2 doc's Strategy S/P discussion.
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

| stage | hardware | time | est. cost |
|---|---|---|---|
| 0. Port validation on cloud (ladder on 8B, bf16) | 1× A100 80GB (~$1.6/hr) | ~2 h | ~$4 |
| 1. Second family, high n: Llama-3.1-8B-Instruct, full arm set, synthetic+LongMemEval subsets, n≈100 questions | 1× A100 80GB | ~8–12 h | ~$15–20 |
| 2. Scale anchor: Llama-3.3-70B-Instruct bf16, trimmed arms (A/B/B-min-pack/H-pack/E-tuned), n≈48 | 2× A100 80GB (~$3.2/hr) or 1× H100 (~$2.8/hr) | ~10–15 h | ~$30–45 |
| 3. (Optional) contingency/reruns | — | — | remainder |

Total for stages 0–2: **roughly $50–70**, inside a $100 top-up with margin;
$25 initial credit fully covers stages 0–1. Everything is resumable
(per-item output files, same as local), so an interrupted pod wastes at most
one item.

## Decision points for you (defaults chosen, change freely)

- Provider: RunPod (default) vs Lambda vs AWS (only if you specifically want
  to stay in AWS — then: request quota for `g6e.xlarge`/`p4d` now, since
  approval latency dominates).
- Second family: Llama-3.1-8B (default; different pretraining + RoPE config)
  vs Mistral-Nemo/Small.
- Budget: $25 initial / $100 ceiling (default).
