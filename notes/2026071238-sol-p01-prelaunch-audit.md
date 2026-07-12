# P01 prelaunch audit

**Author:** Sol — gpt-5.6-sol-xhigh

**Recorded:** 2026-07-12T07:50:51Z

**Apparatus commit audited:** `f5353b3bb75c5cf318246ffb68301152045fc6ab`

**Frozen preregistration SHA-256:** `5619c5ced93f2e60564fcc2c98e24fd9bbf687f93b6cab6132f5be2105cda374`

## Decision

Proceed with the bounded P01 pod run. This is a narrow, matched-runtime precision probe requested by the owner after we established that the earlier Transformers 5 apparatus did not actually quantize the Qwen3 MoE expert weights. It is not a reopening of formal v12, a powered efficacy study, or a test of 4-bit KV-cache storage.

The run compares the same exact Qwen/Qwen3-30B-A3B-Instruct-2507 checkpoint revision, tokenizer, frozen e01 case, code commit, host, and isolated Transformers 4.57.6 runtime under two weight-storage regimes: bitsandbytes NF4 weights with bfloat16 compute and KV cache, then bfloat16 weights and KV cache. The fixed NF4-first order, two full e01 repeats per regime, 34 attempted arms per repeat, outcome-blind extension rule, and interpretation boundaries are frozen in `PRECISION-PROBE-P01-PREREGISTRATION.md`.

## Evidence observed before launch

- The actual Transformers 4.57.6 metadata gate passed locally against the pinned checkpoint: 18,867 checkpoint/meta keys matched exactly; 18,672 eligible linear modules were found, including all 18,432 expert linears, 192 attention linears, and 48 router linears; the required cache snapshot/rebuild API was present.
- The focused P01 stack passed: **53 tests passed**, with only the already-known SWIG deprecation warnings. This covered the loader, runner, lifecycle operations, and independent analyzer.
- The broader reused coherent-canary regression passed with the required import path: **113 tests passed and 7 subtests passed**, with the same two SWIG deprecation warnings.
- Python compilation, Bash parsing, and ShellCheck passed in the final adversarial review.
- Two independent code reviews re-read the committed loader, runner, analyzer, job, launcher, and puller. Neither found a remaining P0/P1 or launch-blocking defect.
- Before adding this audit note, the worktree was clean and local `trunk`, `HEAD`, and `origin/trunk` all resolved to the apparatus commit above. The launch source tree adds only this note on top of that audited code; the exact checkout commit is recorded by the launcher and run artifacts.

The first attempted broader regression without `PYTHONPATH=src` failed during test collection because the repository's reused coherent-canary tests import modules from `src` directly. Repeating the command with `PYTHONPATH=src` produced the passing result above. This was a test-invocation error, not an observed apparatus failure.

## Defects found and resolved during adversarial review

Before this audit, review found and the committed apparatus corrected several real issues:

- ShellCheck's trap-only-function warning could abort the launcher before allocation under `set -e`.
- Phase-A checkpoint package names could collide with final outcome-package discovery.
- The matched-runtime check could accept a non-PASS/non-evaluable state.
- The scientific extension calculation did not initially receive the exact full provider cap.
- Two independent SIGINT mechanisms could fire near the operational deadline.
- The first analyzer version did not independently validate the whole technical package, source bindings, receipts, and outcome-blind extension calculation.

The current committed version uses one graceful interrupt mechanism, receipt-driven artifact recovery, exact provider-cap binding, strict PASS gates, disjoint checkpoint/final naming, and independent reconstruction and validation by the analyzer.

## Budget and lifecycle bounds

- Provider wall time is capped at `min(7200 seconds, floor($4.00 * 3600 / hourly_rate))`, measured from the provider's creation timestamp.
- The runner's scientific extension decision uses the frozen exact rule `elapsed + 1.20 * (6*t + 900) <= min(7200, provider_cap, $4.00*3600/rate)`, where `t` is the slower of the two completed NF4 e01 timing receipts.
- The job's sole graceful interrupt is GNU `timeout` at provider cap minus 180 seconds. A local watchdog independently owns final pull and termination at the hard provider deadline.
- Only a secure A100 80 GB PCIe host with the declared driver/CUDA/runtime gates is admissible. One degraded-host reprovision is allowed; further infrastructure failure stops the attempt rather than expanding spend.
- Any scientific claim requires the complete receipt-bound artifact set and analyzer PASS. A G1/G2 or operational failure counts only as apparatus evidence.

## Known residuals accepted for this run

- Full 30B NF4 CUDA load, bitsandbytes state, sentinels, and forward execution have not yet been observed. Those are deliberately G1/G2 pod gates; a failure stops scientific interpretation.
- `run_treatment_case()` is monolithic. If execution dies midway through a 34-arm treatment, completed earlier treatment arms still held in memory can be lost, although Phase A and fully completed outcomes are persisted. Fixing this would alter a frozen reused component and was judged disproportionate for the bounded probe.
- The analyzer has synthetic end-to-end coverage but has not yet seen a real P01 artifact tree.
- Cross-runtime equality of model-facing plans and target-token IDs follows from the exact shared commit, case hash, tokenizer attestation, and runtime checks and is recorded in the raw packages, but the analyzer does not add a second normalized equality assertion over every such field.
- Checkpoint shards are bound through the exact Hugging Face revision and index plus load/topology/sentinel gates, not by independently hashing every downloaded shard.

These residuals limit confidence if triggered; they do not justify more prelaunch engineering. The run remains conditional on every observed technical gate and on complete, durable artifacts.

## Interpretation boundary

P01 can say whether this one fixed diagnostic behaves differently under matched NF4 versus bfloat16 **weight storage** in the specified runtime. It cannot establish general quantization dependence, production-agent efficacy, statistical generalization, 4-bit KV-cache behavior, or a renewed formal v12 result. Any result—positive, null, failed gate, or interrupted run—will be recorded without deleting or relabeling the prior terminal v12 record.
