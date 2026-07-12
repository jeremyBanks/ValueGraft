# P02 final interpretation and Fable-review disposition

**Author:** Sol — OpenAI GPT-5.6 Sol, extra-high reasoning

**Date:** 2026-07-12

**Status:** final scientific interpretation of P02; no further paid run authorized

## Decision

P02 completed its frozen 2×2 fixed-fixture protocol and reproduced every
predeclared P01 common-field observation exactly. This upgrades P01 from a
post-run analysis of an interrupted experiment to a prospectively specified,
second-host reproduction of the same deterministic fixed-fixture computation.
It does **not** add an independent semantic fixture, a population sample, a
working placebo, a behavioral recovery, or a causal quantization contrast.

The scientific headline therefore does not become more positive. On this one
engineered case, compaction severely damaged the planted answer; correct- and
wrong-source grafts produced small, nonselective or schedule-sensitive score
movements; every graft generation remained wrong; and the one conspicuous NF4
literal flip was exactly reproducible but not semantically specific. The result
is strong evidence that the apparatus can reproduce this fixed computation
across the two observed hosts. It is not evidence that the graft recovers
meaning.

No P03 or other paid run is warranted for the current paper. A third execution
of the same fixture has very low expected information value, and the frozen
protocol independently forbids result-driven extension. Informative future
work would require a new preregistration with independent fixtures, a control
that is demonstrably constructible at the target dtype/geometry, and—if the
precision axis matters—a design that unbundles weight representation from
linear-kernel implementation.

## What passed operationally

- P02 runner status: `COMPLETE`; outer pod receipt: terminal `PASS`.
- Exact launch commit:
  `a40b1dffe8d7bf5e310d308e22d26fef126a73b7`.
- Forty-two lossless packages and all four final outcome packages verified;
  compact and render reconstructions passed; zero recovery files existed.
- Both NF4 and bfloat16 technical gates passed, as did the matched-runtime
  gate. P02 completed NF4 and bfloat16 repeats 1 and 2.
- The pull returned status 0 before the pod was deleted. A later provider query
  returned HTTP 404 for the pod and the pod list was empty.
- P02 ran on an A100 80GB PCIe host with GPU UUID
  `GPU-8a42830e-71ab-fb23-351d-125ccbd5bdb2`, driver `580.159.03`, CUDA 13.0,
  and 81,920 MiB. P01 used a distinct A100 UUID and driver `580.159.04`.
- The local creation-rate estimate through DELETE return was
  `$1.647922222222222`; the later quiescent prelaunch-to-postrun balance delta
  was `$1.6823556332`. The latter remains a nonadditive cross-check until final
  provider-billing reconciliation.

`PASS` above describes artifact/operations gates. It is not a positive verdict
on semantic transfer or the hypothesis.

## Exact common-field result

The independent P01↔P02 comparator bound the P01 analysis SHA-256
`c370d1cc2dbe2a9ee5db64a19ce8837f9879f3b8fc008210756d2985d41aac7e`
and P02 analysis SHA-256
`7b4b77e178f0b50d532459512dd028a044877627d4490197510eba135d5ab551`.
All 20 regime-specific estimands below were exactly equal between P01 and P02;
every P02-minus-P01 difference was `0.0`.

| Matched-repeat-1 estimand | NF4 | bfloat16 |
|---|---:|---:|
| Oracle→fresh focal margin damage | +22.1007595 | +23.6558170 |
| Oracle→fresh nonfocal margin damage | +14.5325801 | +15.4574559 |
| N/R2 value-only `D` focal | +0.1031094 | +0.0394001 |
| N/R2 value-only `D` nonfocal | +0.1613944 | +0.0415351 |
| N/R2 full-KV `D` focal | -0.3457737 | +0.2329350 |
| N/R2 full-KV `D` nonfocal | -0.0778708 | -0.1132853 |
| P/R2 value-only `D` focal | +0.0328388 | +0.0121040 |
| P/R2 value-only `D` nonfocal | +0.0044076 | +0.1552229 |
| N-minus-P value-only shift, focal | +0.0702705 | +0.0272961 |
| N-minus-P value-only shift, nonfocal | +0.1569867 | -0.1136878 |

The common-field comparison also found exact equality for every focal and
nonfocal five-cell generation payload: decoded strings, content-token IDs,
generation hashes, stop reasons, cap flags, and change vectors. The focal
tuples remained:

- NF4: `[Atlas sentence, Ring 3, Ring 3, Ring 3, Ring 3]` for
  `[FF, FC, FW, CC, WW]`;
- bfloat16: `[Ring 3, Ring 3, Ring 3, Ring 3, Ring 3]`.

The literal NF4 fresh sentence was “Atlas 4.8 was not selected under the
recorded mandatory selection rule.” Every nonfocal cell in both regimes
generated `30 days`. No focal cell generated the correct target `partner beta`.

The complete placebo diagnostic payloads also matched exactly. Both regimes
again returned `PLACEBO_UNAVAILABLE`, at layer 1 / destination row 76 for NF4
and layer 1 / row 77 for bfloat16. This is repeated missing control evidence,
not a null placebo effect.

Within P02, NF4 repeat 1 and repeat 2 had identical normalized payload hashes
and zero differing JSON pointers; bfloat16 did too. All available scalar repeat
deltas were exactly zero. These repetitions test execution stability of one
fixed computation. They are not independent cases, and a cross-runtime delta
exceeding a zero repeat delta is not a statistical noise comparison.

## Reading the movements

Gross compaction damage is the dominant fact. Fresh state lost 24.0006 nats of
correct-target log probability in NF4 and 22.7969 in bfloat16, alongside the
22.10/23.66 margin damage above. The graft contrasts are roughly two orders of
magnitude smaller and did not restore the answer.

The superficially positive N/R2 value-only focal `D` is not focal-selective:
nonfocal movement is larger in both regimes. The signed selectivity values are
`-0.0582850` (NF4) and `-0.0021350` (bfloat16). Correct-target movement also
differs across the bundled runtime axis: `Hplus=-0.2478943` for NF4 and
`+0.1935272` for bfloat16. Thus the NF4 focal contrast is favorable only because
the countertarget worsens more; the correct target itself worsens. The
bfloat16 correct target improves slightly, but without selectivity, a working
placebo, or any change in the wrong generated answer.

N→P schedule changes reduce the focal value-only contrast in both regimes.
Those differences are deterministic descriptions of two computation schedules,
not estimates of random variability. The exact full-KV sign split
(`-0.3458` NF4 versus `+0.2329` bfloat16) is likewise a stable one-fixture
runtime signature, not evidence for quantization causality.

The most economical licensed interpretation is generic, deterministic
intervention sensitivity in an already badly damaged fixture. Correct and
wrong source state lead to the same literal answers, and nonfocal movement is
at least as large as focal value movement. The data therefore do not separate
semantic source information from ordinary perturbation effects.

The saved generation records contain content IDs, decoded text, stop reasons,
and hashes—but not the alternate `Atlas`-versus-`Ring` first-token logits.
Consequently the proposed “near decision boundary” explanation for the NF4
flip cannot be tested locally from preserved artifacts. It remains an
unnecessary hypothesis and will not appear as a finding.

## What exact reproduction establishes—and does not

It was directly observed that the normalized common scientific fields were
identical across:

1. P01 and P02 on two different physical A100 UUIDs and adjacent driver patch
   versions; and
2. both repeats of both regimes inside P02.

This is valuable apparatus and portability evidence within the sampled
conditions. It rules against ordinary run-to-run instability or one particular
host instance as explanations of the P01 pattern under these conditions.

It does not prove universal determinism, guarantee that a third execution or a
different GPU/driver major returns the same bits, or show whole-artifact byte
identity. P01 and P02 have different protocol metadata and arm inventories, so
their whole-package canonical hashes appropriately differ. The comparator
established exact equality only after normalizing to the specified common
scientific fields.

P02 also does not increase the independent semantic-case count: the study
still has one engineered fixture. It contributes independent reproducibility
evidence, not independent population information. Cross-fixture variability,
the quantity needed for generality or effect inference, remains unmeasured.

Finally, the NF4/bfloat16 contrast bundles checkpoint weight representation and
linear-kernel implementation. KV-cache storage is bfloat16 in both regimes.
Nothing here isolates quantization as the cause, and no term stronger than
“bundled weight/runtime-axis difference” is licensed.

## Frozen A8 expectations

Every outcome-bearing A8 expectation was realized:

- gross focal damage stayed positive in both regimes;
- the NF4 fresh anomaly and the bfloat16 five-`Ring 3` tuple reproduced exactly;
- every graft cell failed the exact target;
- both placebos remained unavailable at the same rows;
- N/R2 and P/R2 value-only focal directions remained positive;
- the full-KV focal sign split reproduced;
- both P02 repeats were exactly stable; and
- all formal, semantic, efficacy, population, and quantization claim boundaries
  remained closed.

The no-result-driven-extension rule was also followed: one P02 ran, no case or
arm was added, and no P03 was launched.

## Disposition of Fable's advisory review

I accept Fable's principal recommendations in
`notes/20260712A9-fable-p02-result-interpretation-and-paper-disposition.md`:

- P02 changes the provenance and reproducibility status of the precision-axis
  paragraph, not the paper's headline;
- P01 and P02 should appear in one common table, not as two independent
  estimates;
- terminal `PASS` must remain an operational label;
- the repeat guard must not be described as statistical evidence;
- no further paid execution of this protocol is useful for the present paper;
  and
- the missing placebo blocks semantic attribution.

I narrow four formulations:

1. Two exact executions do not “guarantee” a third; they make another identical
   run low-value.
2. The evidence is concordance across the two observed A100 hosts and adjacent
   driver patches, not general “hardware/driver invariance.”
3. Whole protocol payloads are not bit-identical because their inventories and
   metadata differ; the normalized common scientific fields are exactly equal.
4. A knife-edge/near-tie explanation is not supported by saved logits and is
   omitted rather than promoted.

Fable also correctly notes that exact reproduction adds no independent fixture
and no estimate of cross-case variability. I phrase this as “independent
reproducibility evidence but no independent population information,” rather
than the unqualified claim that P02 adds zero information.

## Paper language

The paper should state, in substance:

> A prospectively specified conditional reproduction repeated the same one-case
> NF4/bfloat16 screen on a second A100 host. Every normalized common scalar,
> generation payload, and unavailable-placebo diagnostic matched the earlier
> run exactly, and both within-regime repeats were exact. This demonstrates
> reproducibility of the fixed computation under the two observed hosts, not a
> larger sample. Compaction damage remained about 22–24 nats, every graft
> generation remained wrong, value-only source contrasts were nonselective, and
> no matched placebo could be constructed. The result licenses no semantic
> transfer, efficacy, quantization-causality, population, or agent claim.

P01 must be labeled a post-run partial descriptive analysis of an interrupted
run. P02 must be labeled a prospectively specified **conditional** reproduction:
its apparatus and analysis were frozen before P01 values were opened, but the
decision to spend on P02 was made after P01 looked interesting. Formal v12
remains stopped; neither P01 nor P02 reopens it.

## Canonical evidence

- `results/precision_probe_p02/` — receipt-bound raw packages, renders, runtime,
  logs, and provider settlement.
- `results/precision_probe_p02_analysis/precision-probe-p02-independent-analysis_Qwen3-30B-A3B-Instruct-2507_20260712T114104839688Z.json`
  (SHA-256 `7b4b77e178f0b50d532459512dd028a044877627d4490197510eba135d5ab551`).
- `results/precision_probe_p01_p02_comparison/precision-probe-p01-p02-exact-comparison_Qwen3-30B-A3B-Instruct-2507_20260712T120014514457Z.json`
  (SHA-256 `588229df5f4ca5c8613dc4f21564043214bfe889fc4dba176300ace0435442be`).
- `notes/20260712A8-sol-p02-conditional-replication-expectations-and-launch-decision.md`
  — frozen expectations and reading.
- `notes/20260712A9-fable-p02-result-interpretation-and-paper-disposition.md`
  — independent advisory review, with its own disclosed reading/cost limits.
