_This conversation shifted the project from chasing positive headlines to
provenance-clean, controlled evaluation of the primary bf16 model, with the
paper now expected to report whatever the completed data supports—including a
null or bounding result. The remaining work is an overnight spend-down of funded
experiments, followed by Fable-led valuation, writing, review, and repository
promotion._

**Participants:** User and claude-opus-4-8.

**Handoff State.** The primary per-layer champion validation on
Qwen3-30B-A3B-Instruct-2507 bf16 produced a held-out raw graft-versus-baseline
result of −0.0026, CI [−0.041, +0.043], i.e. null/underpowered for one model.
The positive shuffle-position contrast (+0.1925, CI [0.118, 0.266]) indicates
that scrambling content harms performance, but does not establish that correct
grafting improves over compaction. The validation used the intentionally
selected per-model self-generated, realistic-length summary condition;
shared/fixed summaries had previously suppressed the mechanism and were
abandoned by design. The “fallback” label was misleading, not evidence of an
accidental condition, and was corrected in commit `057767a`. Remaining placebo
modes were stopped after the decisive raw-E/B result was harvested; gauss, mean,
and shuffle-probe contrasts can be rerun later.

The placebo battery is defined as: position-shuffling real values across graft
slots; values from another probe; norm-matched Gaussian noise; and per-layer
mean values. Only the first is complete in the interrupted validation. The
interpretation must distinguish mechanism-level content specificity from net
performance lift, and must not call the teacher-forced log-prob proxy a
resolve/test-pass result.

**Provenance and methodology.** Repeated earlier errors involved treating data
as interchangeable despite differences in model, dtype, summary condition,
intervention, and metric. The corrected matrix identifies existing N=75 SWE-Gym
data as bf16 primary-model, scalar α=0.75, brief-summary, teacher-forced-logprob
proxy data—not champion validation, not placebo-controlled, and not resolve
rate. The missing comparable cell is production-faithful summaries at the same α
and metric. Fable’s provenance infrastructure (commit `312e438`) makes manifests
record live model/dtype/quantization/alpha/condition/metric; launcher forwarding
was fixed so requested settings cannot silently revert to defaults. The
production SWE-Gym run was verified from its live manifest as bf16, unquantized,
`summary=prod`, α=0.75, N=75.

Fable’s per-head design was committed and pushed as `fe65795`: derive 192
layer/KV-head marginals on VAL, select a `head_map` under one preregistered
rule, then held-out placebo-validate heads, layers, intersection, and union on
c07–c24. The per-head derive completed and validation was running; this is the
principal unresolved champion question. The existing 4-bit effort was abandoned:
quantization libraries repeatedly altered the torch/CUDA environment, and
compressed-tensors decompressed weights to roughly 48 GB at runtime, causing
A6000 OOM; it was supplemental and not worth further priority.

**Queued experiments and operations.** A compression sweep was designed and
committed as `ed8cf19`, varying ultra/brief/medium/realistic (and optionally
prod) self-generated summaries while holding bf16 model, champion, held-out
data, and placebo design fixed; actual summary/full-context token ratios are to
be recorded. Its purpose is to test whether any effect concentrates under
aggressive compaction. The balance was topped up to approximately $37, so the
intended order is: finish per-head comparison and prod-SWE-Gym N=75; run the
compression sweep; restore the interrupted placebo battery and optionally obtain
matched-scale brief SWE-Gym. At the latest checkpoint, prod SWE-Gym was 50/75
and the per-head scan was still validating. Earlier ETA estimates were
approximately 3–4 hours for the four-way per-head result, 2.5–3 hours for prod
SWE-Gym, and 8–12 hours for the full spend-down.

The user explicitly requested that all project memory files be preserved; 16
files, including `MEMORY.md` and `verify-boring-before-clever.md`, were copied
unchanged into `notes/`.

**Paper direction.** Do not begin final writing or valuation until the remaining
research is complete. Fable is expected to assess the complete dataset, then
write a coherent academic paper with extensive adversarial, proofreading,
readability, flow, and focus review. The paper should be rigorous and
non-triumphal, with its shape determined by the evidence: either a validated
effect localized to a condition or a carefully bounded/null result. Do not coin
“ValueGraft” or any capitalized/camelCase technique name in the title; use a
plain lowercase descriptive phrase, define it once as an internal glossary
label, and avoid presenting it as persistent vocabulary or a recommended
technique. The explicit commitment was to continue autonomously overnight and
aim for a fully polished paper by the following morning, with repository README
promotion and push authorized after confidence is earned.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `adbda149278468b95`
