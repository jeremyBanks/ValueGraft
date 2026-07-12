Observed runtime model identity: `claude-fable-5` — the session harness reports: "You are powered by the model named Fable 5. The exact model ID is claude-fable-5." (Anthropic Claude Fable 5.)

# P02 → PAPER.md integration: Fable assessment and ready-to-paste patches

**Author:** Anthropic Claude Fable 5, acting as narrative coauthor
**Date:** 2026-07-12
**Status:** advisory integration draft; PAPER.md deliberately untouched
**Inputs read (exactly these, nothing else):** `PAPER.md`; `notes/20260712AA-sol-p02-final-interpretation-and-fable-disposition.md`; `METHODS-PROVENANCE-REQUIREMENTS.md`
**Constraint compliance:** only this file was written. Sol's 20260712AA note is treated as the controlling scientific interpretation; every patch below preserves its boundaries and his four narrowings of my earlier A9 review. Nothing here reopens formal v12, adds a fixture or N, or upgrades the paper's headline.

---

## 1. Topline assessment

The completed P01/P02 precision screen belongs in the paper as a **descriptive extension of the fourth (diagnostic) evidence stratum** — a new **§8.4** plus small, surgical touches to the abstract, the §1 roadmap, the §3 conventions sentence and strata table, §10, §11, §12, Appendix A (new A.6), and Appendix B. It changes the **provenance and reproducibility status of the precision-axis material** — §12's claim that the NF4 arm "was not run" is now false and must be replaced — and it adds exactly one genuinely new establishable fact: **exact normalized-common-field reproduction of one fixed computation across the two observed A100 hosts and within P02's repeats.** It changes no headline, ascends no rung of the §3 claim ladder, and relaxes no negative boundary. Title, §2, §4, §5, §6, §7, §9.1, §9.3, Author contributions, and Acknowledgments need no change.

On the §3 claim ladder the screen adds no rung: it strengthens confidence that the stratum-4 apparatus computes what it computes (execution validity and portability within sampled conditions), while leaving rungs 2–4 exactly as open as §8 left them. That is why it folds into stratum 4 rather than becoming a fifth stratum — it contributes independent reproducibility evidence, not independent population information, and a fifth table row would visually manufacture a new body of evidence that does not exist.

## 2. Placement decision and rationale

**Decision: new subsection §8.4, inside §8 ("The e01 diagnostic"), after §8.3.**

- The screen re-executes the same engineered e01 fixture in the same descriptive, no-p-value, no-interval evidence class as §8. It is a sibling of the diagnostic, not a new experiment family. Placing it in §8 keeps all one-fixture diagnostic-class evidence in one place, so a reader encounters the reproduction immediately after the result it reproduces the *status* of.
- It avoids renumbering §§9–12 and every internal cross-reference — consistent with the instruction not to rewrite unrelated sections.
- It preserves the abstract's "four evidence strata" framing unchanged; only the stratum-4 row of the §3 table is extended.
- Naming: the paper prose should say **"precision screen (P01/P02)"**; artifact paths use `precision_probe_*`. Appendix B rows carry the literal paths, so both names are discoverable.

**Not chosen:** a fifth stratum (manufactures apparent independent evidence); a new top-level §9 (forces renumbering of the failure catalogue, §10–12, and the abstract's section pointers); a Limitations-only mention (the exact cross-host reproduction is an observed result and belongs in the evidence sections, not only in caveats).

## 3. Status and sequencing honesty — three latent contradictions the patches must defuse

1. **§12 is stale.** It currently reads "A bitsandbytes NF4 arm would … It was not run because …". The NF4 arm has now run, as the descriptive precision screen. Patch P11 replaces this passage with the true present state while keeping the quantization question open.
2. **§11 says "Paid data collection for this project is finished."** The screen ran under its own recorded launch decision (`notes/20260712A8-sol-p02-conditional-replication-expectations-and-launch-decision.md`). Patch P10 adds a closure sentence that cites that launch decision and states the screen is itself closed (no-result-driven-extension rule followed; one P02; no P03). I deliberately do **not** assert the temporal order of P01 relative to the data-collection stop note, because my inputs do not establish it; the patch is worded to be true either way. If the editor wants to state the order, verify it first (checklist item 6).
3. **§8's opening says "Exactly one unchanged e01 treatment execution was permitted after the stop."** True within the v12 authorization, but a reader who then meets P01/P02 could count them against it. Patch P5 adds one forward-pointing parenthetical so the sentence stays exact.

**Cost figures stay out of the paper body.** The Sol note records a `$1.647922222222222` creation-rate estimate and a `$1.6823556332` quiescent balance delta (explicitly a nonadditive cross-check pending provider reconciliation). §10 already deliberately defers *all* cost reporting to the open end-to-end audit; piecemeal per-run dollar figures would break that policy and invite pooling of non-comparable accounting bases. A.6 instead points at the settlement records in `results/precision_probe_p02/` as inputs to that audit.

## 4. Boundary-compliance map

| Required boundary (owner brief + Sol 20260712AA) | Where enforced below |
|---|---|
| P01 = post-run **partial** descriptive analysis of an **interrupted** run | P7 status paragraph; P12 |
| P02 = **prospectively specified conditional reproduction** (apparatus/analysis frozen before P01 values opened; spend decision after P01 looked interesting) | P7; P12 |
| Exactness = **normalized common scientific fields**, across the **two observed A100 hosts** (adjacent driver patches) and **within P02 repeats**; not whole-artifact byte identity; not "hardware/driver invariance" | P7 reproduction paragraph; P4; P9 |
| **No new semantic fixture, no new statistical N**; repeats test execution stability, not independence; cross-runtime delta vs zero repeat delta is not a noise comparison | P7 final + reproduction paragraphs; P3; P9 |
| **No placebo** — `PLACEBO_UNAVAILABLE` again in both regimes; repeated missing-control evidence, not a null placebo effect | P1; P7; P8; P9; P11 |
| **No correct generation** — no cell generated `partner beta`; focal `Ring 3` (NF4 fresh: the Atlas sentence), nonfocal `30 days`; near-boundary explanation unsupported by saved logits and **omitted, not promoted** | P7 generations paragraph; P11 |
| **Nonselective value contrast** — signed selectivity −0.0582850 (NF4), −0.0021350 (bfloat16); NF4 `H+` negative (correct target worsens) | P7 reading paragraph; P1 |
| **Bundled weight-representation + kernel axis; KV-cache storage bfloat16 in both arms**; no term stronger than "bundled weight/runtime-axis difference" | P7; P9; P11; P12 |
| **No universal-determinism, quantization-causality, efficacy, population, or agent claim**; two exact runs make a third *low-value*, not guaranteed | P7 final paragraph; P9; P10 |
| **Cross-stack fact:** the Transformers 4.57.6 bfloat16 screen is descriptive context, **not a same-stack replication** of the Transformers 5.0.0 e01 diagnostic | P7 status paragraph; P12; checklist item 1 |
| Formal v12 remains stopped; the diagnostic's authorization is unaffected | P5; P7; P10 |
| Terminal `PASS` / `COMPLETE` are **operational labels**, not scientific verdicts | P7; P12 |
| Headline unchanged | §1 above; no title patch exists |

## 5. The patches

Anchors quote the current PAPER.md text exactly. "Insert after" means append immediately following the anchor; "Replace" means substitute the anchor.

### P1 — Abstract (insert one passage)

**Anchor:** `The final exact-state mechanism experiment formally stopped at a preregistered technical gate; a single permitted post-stop diagnostic produced a weak, schedule-sensitive, placebo-uncontrolled value-only trace with no behavioral recovery.`

**Insert after:**

> A subsequent one-fixture precision screen re-ran that diagnostic's engineered case under bundled NF4 and bfloat16 weight/runtime regimes (KV cache bfloat16 in both) on a different software stack; its prospectively specified conditional reproduction matched the first, interrupted run's normalized common scientific fields exactly on a second A100 host, with exact within-regime repeats — apparatus reproducibility of one fixed computation, not new evidence of recovery: grafts remained nonselective, every graft generation stayed wrong, and no matched placebo could be constructed.

### P2 — §1 roadmap sentence

**Anchor:** `We then report the mechanism redesign that was meant to resolve the question cleanly, why it formally stopped before its treatment arm (§7), what the one permitted diagnostic showed and failed to show (§8), and the failure catalogue (§9).`

**Replace with:**

> We then report the mechanism redesign that was meant to resolve the question cleanly, why it formally stopped before its treatment arm (§7), what the one permitted diagnostic showed and failed to show (§8), a later two-regime precision screen of the same fixture whose prospectively specified second run reproduced the first exactly without moving any scientific boundary (§8.4), and the failure catalogue (§9).

### P3 — §3 statistical-conventions sentence

**Anchor:** `and the v12/e01 material is N=1 descriptive evidence carrying no p-value, interval, or population claim.`

**Replace with:**

> and the v12/e01 material — including the P01/P02 precision screen that re-executed the same fixture — is N=1 descriptive evidence carrying no p-value, interval, or population claim.

### P4 — §3 strata table, fourth row

**Anchor (current row):**

`| Formal v12 + e01 (§6–8) | Exact 30B bf16 canary; fixed authored correct/wrong histories; same fixed carrier text; N/P forced replay | Formal technical stop; one later non-authorizing N=1 diagnostic gives a weak, schedule-sensitive, placebo-uncontrolled value-only hint |`

**Replace with:**

`| Formal v12 + e01 + precision screen (§6–8) | Exact 30B bf16 canary; fixed authored correct/wrong histories; same fixed carrier text; N/P forced replay; later bundled NF4/bfloat16 re-execution of the same fixture on a second stack (P01/P02) | Formal technical stop; one later non-authorizing N=1 diagnostic gives a weak, schedule-sensitive, placebo-uncontrolled value-only hint; the screen's conditional reproduction matched all normalized common fields exactly across two hosts — apparatus reproducibility, no added selectivity, recovery, or sample |`

### P5 — §8 opening paragraph, forward pointer

**Anchor:** `It is one engineered case. Nothing below carries a p-value, an interval, or a population claim.`

**Replace with:**

> It is one engineered case. Nothing below carries a p-value, an interval, or a population claim. (A later, separately frozen precision screen re-executed the same fixture under two bundled weight/runtime regimes on a different software stack; it is reported in §8.4 and does not alter this diagnostic's authorization, its status, or the formal stop.)

### P6 — §8.3, noise-floor sentence (recommended, optional)

**Anchor:** `Forward-run repeatability was not tested, so the same-condition 30B/A100 noise floor is unknown.`

**Replace with:**

> Forward-run repeatability was not tested, so the same-condition 30B/A100 noise floor is unknown. (The later precision screen observed exact within-run and cross-host repeatability of its own fixed computation on a different software stack, §8.4 — adjacent apparatus evidence, not a measurement of this stack's noise floor.)

### P7 — NEW §8.4 (the core block; insert after §8.3, before §9)

> ### 8.4 The precision screen (P01/P02): a bundled NF4/bfloat16 re-execution, exactly reproduced
>
> After the diagnostic, the same engineered e01 fixture was re-executed under a separately frozen fixed-fixture protocol as a one-case **precision screen**, crossing two bundled weight-representation/runtime regimes — NF4-quantized weights versus bfloat16 weights, with KV-cache storage bfloat16 in **both** — with two repeats per regime. The compared common fields span oracle→fresh damage, N/R2 value-only and full-K+V contrasts, and P/R2 value-only contrasts, for the focal and nonfocal probes in both regimes, plus five free-generation cells (FF, FC, FW, CC, WW) per probe. Status labels come first, because they bound everything that follows. **P01** is a post-run partial descriptive analysis of an interrupted run. **P02** is a prospectively specified **conditional** reproduction: its apparatus and analysis were frozen before P01's values were opened, but the decision to spend on P02 was made after P01 looked interesting. Neither run reopens formal v12, which remains stopped. P02's runner status `COMPLETE` and terminal `PASS` receipt are artifact/operations labels, not verdicts on semantic transfer. Like the rest of this section, the screen is one-fixture descriptive evidence carrying no p-value, interval, or population claim. Finally, the screen ran on a Transformers 4.57.6 stack rather than the Transformers 5.0.0 stack of §§6–8 (Appendix A.3, A.6), so its bfloat16 regime is descriptive cross-stack context for the e01 diagnostic, not a same-stack replication of it; numerical differences from §8.1's cells are apparatus differences and license no stability or instability conclusion about either stack.
>
> **Exact common-field reproduction.** An independent comparator bound the P01 and P02 analysis artifacts by SHA-256 and found all twenty regime-specific common estimands exactly equal: every P02-minus-P01 difference was 0.0. It also found exact equality of every focal and nonfocal five-cell generation payload — decoded strings, content-token IDs, generation hashes, stop reasons, cap flags, and change vectors — and of the complete placebo diagnostic payloads. Within P02, the two repeats of each regime had identical normalized payload hashes and zero differing JSON pointers, and all available scalar repeat deltas were exactly zero. P01 and P02 ran on two different physical A100 UUIDs under adjacent driver patch versions (580.159.04 and 580.159.03). This is apparatus and portability evidence within the sampled conditions: it rules against ordinary run-to-run instability, or one particular host instance, as explanations of the P01 pattern under these conditions. It does not prove universal determinism, does not guarantee that a third execution or a different GPU/driver major would return the same bits, and is not whole-artifact byte identity: the two runs' protocol metadata and arm inventories differ, their whole-package canonical hashes appropriately differ, and exact equality holds after normalization to the specified common scientific fields. The repeats test execution stability of one fixed computation; they are not independent cases, and a cross-runtime delta exceeding a zero repeat delta is not a statistical noise comparison. Two exact executions do not guarantee a third; they make another identical run low-value.
>
> Matched-repeat-1 estimands (definitions as in §8.1; the "N-minus-P value-only shift" is the between-schedule difference of the value-only `D`):
>
> | Matched-repeat-1 estimand | NF4 | bfloat16 |
> |---|---:|---:|
> | Oracle→fresh focal margin damage | +22.1007595 | +23.6558170 |
> | Oracle→fresh nonfocal margin damage | +14.5325801 | +15.4574559 |
> | N/R2 value-only `D` focal | +0.1031094 | +0.0394001 |
> | N/R2 value-only `D` nonfocal | +0.1613944 | +0.0415351 |
> | N/R2 full-KV `D` focal | −0.3457737 | +0.2329350 |
> | N/R2 full-KV `D` nonfocal | −0.0778708 | −0.1132853 |
> | P/R2 value-only `D` focal | +0.0328388 | +0.0121040 |
> | P/R2 value-only `D` nonfocal | +0.0044076 | +0.1552229 |
> | N-minus-P value-only shift, focal | +0.0702705 | +0.0272961 |
> | N-minus-P value-only shift, nonfocal | +0.1569867 | −0.1136878 |
>
> **Reading the movements.** Gross compaction damage is the dominant fact: fresh state lost 24.0006 nats of correct-target log probability in NF4 and 22.7969 in bfloat16, alongside the +22.10 and +23.66 focal margin damage above. The graft contrasts are roughly two orders of magnitude smaller, and no graft restored the answer. The superficially positive N/R2 value-only focal `D` is not focal-selective: nonfocal movement is larger in both regimes, with signed selectivity −0.0582850 (NF4) and −0.0021350 (bfloat16). Correct-target movement splits across the bundled runtime axis: `H+` is −0.2478943 in NF4 — the favorable focal contrast exists only because the countertarget worsens more, while the correct target itself worsens — and +0.1935272 in bfloat16, but without selectivity, a working placebo, or any change in the wrong generated answer. Moving from schedule N to P reduces the focal value-only contrast in both regimes; those differences are deterministic descriptions of two computation schedules, not estimates of random variability. The exact full-K+V focal sign split — −0.3457737 in NF4 versus +0.2329350 in bfloat16 — is a stable one-fixture runtime signature, not evidence for quantization causality. The most economical licensed interpretation is generic, deterministic intervention sensitivity in an already badly damaged fixture: correct and wrong source state lead to the same literal answers, and nonfocal movement is at least as large as focal value movement, so the data do not separate semantic source information from ordinary perturbation effects.
>
> **Generations and controls.** No focal cell in either regime generated the correct target `partner beta`; every nonfocal cell in both regimes generated `30 days`. The bfloat16 focal tuple was `Ring 3` in all five cells; the NF4 tuple differed only in its fresh cell, which generated the literal sentence "Atlas 4.8 was not selected under the recorded mandatory selection rule." That NF4 literal flip was exactly reproducible and not semantically specific. The saved generation records preserve content-token IDs, decoded text, stop reasons, and hashes — but not the alternate first-token logits — so a near-decision-boundary explanation for the flip cannot be tested from preserved artifacts; it is omitted rather than promoted. The norm-matched placebo constructor again returned `PLACEBO_UNAVAILABLE` in both regimes, at layer 1, destination row 76 (NF4) and row 77 (bfloat16): repeated missing-control evidence, not a null placebo effect.
>
> **What the screen changes, and what it cannot.** A prospectively specified conditional reproduction repeated the same one-case NF4/bfloat16 screen on a second A100 host; every normalized common scalar, generation payload, and unavailable-placebo diagnostic matched the earlier run exactly, and both within-regime repeats were exact. This upgrades P01 from a post-run analysis of an interrupted experiment to a reproduced fixed-fixture computation under the two observed hosts — independent reproducibility evidence, not independent population information. The study still has one engineered fixture; cross-fixture variability, the quantity needed for generality or effect inference, remains unmeasured. The NF4/bfloat16 contrast bundles checkpoint weight representation with linear-kernel implementation, while KV-cache storage is bfloat16 in both regimes; nothing here isolates quantization as a cause, and no term stronger than "bundled weight/runtime-axis difference" is licensed. Every outcome-bearing expectation frozen before the P02 launch was realized, and the protocol's no-result-driven-extension rule was followed: one P02 ran, no case or arm was added, and no P03 was launched. The screen licenses no semantic-transfer, efficacy, quantization-causality, population, or agent claim.

### P8 — §9.2, failure item 6, failure cell (optional)

**Anchor (within the item-6 failure cell):** `For each of three e01 regions, attempts 0–1023 yielded no applied-bf16 perturbation satisfying the frozen norm and cosine tolerances at the shared first nonzero row.`

**Replace with:**

> For each of three e01 regions, attempts 0–1023 yielded no applied-bf16 perturbation satisfying the frozen norm and cosine tolerances at the shared first nonzero row. The precision screen's constructor later returned `PLACEBO_UNAVAILABLE` again in both of its regimes (§8.4).

### P9 — §10, both lists

**(a) Insert a bullet after:** `- One N=1, diagnostic-only, placebo-missing, schedule-sensitive value-only trace under identical visible text, with no behavioral recovery (§8).`

> - Exact reproduction of the one-fixture NF4/bfloat16 precision screen: a prospectively specified conditional second run matched every normalized common scientific field of the interrupted first run across two observed A100 hosts and adjacent driver patch versions, with exact within-regime repeats (§8.4) — apparatus and portability evidence within the sampled conditions, carrying no selectivity, no behavioral recovery, and no available placebo.

**(b) Anchor:** `- Whether weight quantization or KV-cache dtype changes the effect.`

**Replace with:**

> - Whether weight quantization or KV-cache dtype changes the effect. The precision screen's exactly reproducible regime differences on one fixture do not answer this: its NF4/bfloat16 contrast bundles weight representation with kernel implementation, and KV-cache storage was bfloat16 in both regimes (§8.4).

### P10 — §11 closing paragraph

**Anchor:** `The formal re-entry requirements are recorded in ` `notes/2026071288-sol-data-collection-stop-and-future-reentry.md` `. Paid data collection for this project is finished.`

**Replace with (append one sentence):**

> The formal re-entry requirements are recorded in `notes/2026071288-sol-data-collection-stop-and-future-reentry.md`. Paid data collection for this project is finished. The one-fixture precision screen (§8.4) ran under its own recorded launch decision and frozen expectations (`notes/20260712A8-sol-p02-conditional-replication-expectations-and-launch-decision.md`) and is likewise closed: its protocol forbade result-driven extension, exactly one conditional reproduction ran, and no further execution of it is warranted for this paper.

### P11 — §12 Limitations

**(a) Anchor:** `The exact redesign and diagnostic were bf16 only. The early 4-bit-weight MLX hint used fp16 KV cache and a now-retired, confounded apparatus, so neither weight-quantization dependence nor KV-cache-dtype dependence has been tested cleanly. A bitsandbytes NF4 arm would change weight representation and kernels while ordinarily leaving KV state floating-point; it would be a useful matched runtime/weight-quantization axis, not an isolated test of “4-bit KV.” It was not run because formal v12 stopped before eligible treatment, its diagnostic lacked placebos, and paid collection had closed; a new precision axis belongs behind the same re-entry gates rather than being attached post hoc to an N=1 diagnostic.`

**Replace with:**

> The exact redesign and diagnostic were bf16 only. The early 4-bit-weight MLX hint used fp16 KV cache and a now-retired, confounded apparatus, so neither weight-quantization dependence nor KV-cache-dtype dependence has been tested cleanly. The later precision screen (§8.4) added an NF4 weight/runtime regime alongside bfloat16 on the same single engineered fixture, with KV-cache storage bfloat16 in both regimes; it is a bundled weight-representation-plus-kernel axis on a different software stack, not an isolated test of quantization or of “4-bit KV,” and its exactly reproducible regime differences describe one fixed computation rather than a causal precision effect. The screen ran and reproduced without a working placebo, without any correct graft generation, and without adding a fixture or sample, so the quantization question remains open; unbundling weight representation from kernel implementation — with a control demonstrably constructible at the target dtype and geometry — belongs behind the recorded re-entry gates.

**(b) Anchor:** `per-trajectory summary text, and all K/V tensor values from e01.`

**Replace with:**

> per-trajectory summary text, all K/V tensor values from e01, and the precision screen's alternate first-token logits (so the NF4 literal-flip mechanism cannot be probed from preserved artifacts).

**(c) Optional. Anchor:** `Neither v12 replay schedule is native continuous generation, and the one suggestive cell was schedule-sensitive — the construct validity of forced replay for live-agent state is untested.`

**Replace with:**

> Neither v12 replay schedule is native continuous generation, and the one suggestive cell was schedule-sensitive — a pattern the precision screen reproduced exactly in both of its regimes (§8.4) — so the construct validity of forced replay for live-agent state is untested.

### P12 — NEW Appendix A.6 (insert after A.5)

> ### A.6 Precision screen P01/P02
>
> The screen re-executed the literal e01 fixture (A.3) under a separately frozen fixed-fixture protocol: two bundled weight-representation/runtime regimes (NF4-quantized weights; bfloat16 weights), KV-cache storage bfloat16 in both, two repeats per regime, N and P replay schedules at region R2 in value-only and full-K+V families, and five free-generation cells (FF, FC, FW, CC, WW) per probe. The analysis artifacts bind subject `Qwen/Qwen3-30B-A3B-Instruct-2507`; the receipt-bound runtime and log records in `results/precision_probe_p02/` are the binding source for the exact checkpoint revision, quantization backend, and library versions. The screen's software stack is Transformers 4.57.6 — not the Transformers 5.0.0 stack of A.3 — so its bfloat16 regime is cross-stack descriptive context for §8, not a same-stack replication.
>
> P01 ran on a distinct A100 host (distinct GPU UUID, driver 580.159.04) and was interrupted; its analysis is post-run and partial. P02 ran on an A100 80GB PCIe host with GPU UUID `GPU-8a42830e-71ab-fb23-351d-125ccbd5bdb2`, driver 580.159.03, CUDA 13.0, and 81,920 MiB, from launch commit `a40b1dffe8d7bf5e310d308e22d26fef126a73b7`, with apparatus, analysis, and outcome-bearing expectations frozen before P01 values were opened (`notes/20260712A8-sol-p02-conditional-replication-expectations-and-launch-decision.md`). The runner completed with status `COMPLETE` and a terminal `PASS` outer pod receipt — operational labels only. Forty-two lossless packages and all four final outcome packages verified; compact and render reconstructions passed; zero recovery files existed; the pull returned status 0 before the pod was deleted, and a later provider query returned HTTP 404 with an empty pod list. The independent comparator bound P01 analysis SHA-256 `c370d1cc2dbe2a9ee5db64a19ce8837f9879f3b8fc008210756d2985d41aac7e` and P02 analysis SHA-256 `7b4b77e178f0b50d532459512dd028a044877627d4490197510eba135d5ab551` and established exact equality of every specified normalized common scientific field; whole-package canonical hashes appropriately differ because protocol metadata and arm inventories differ. Saved generation records preserve content-token IDs, decoded text, stop reasons, and hashes, but not alternate first-token logits. Provider-settlement records feeding the still-open end-to-end cost audit are retained in `results/precision_probe_p02/`.

### P13 — Appendix B, new artifact-map rows (insert after the "Quantization-axis disposition" row)

```
| Precision screen P02: receipt-bound packages, renders, runtime, logs, settlement | `results/precision_probe_p02/` |
| Precision screen P02 independent analysis | `results/precision_probe_p02_analysis/precision-probe-p02-independent-analysis_Qwen3-30B-A3B-Instruct-2507_20260712T114104839688Z.json` (SHA-256 `7b4b77e178f0b50d532459512dd028a044877627d4490197510eba135d5ab551`) |
| P01↔P02 exact common-field comparison | `results/precision_probe_p01_p02_comparison/precision-probe-p01-p02-exact-comparison_Qwen3-30B-A3B-Instruct-2507_20260712T120014514457Z.json` (SHA-256 `588229df5f4ca5c8613dc4f21564043214bfe889fc4dba176300ace0435442be`) |
| P02 frozen expectations and launch decision | `notes/20260712A8-sol-p02-conditional-replication-expectations-and-launch-decision.md` |
| P02 advisory review and final interpretation | `notes/20260712A9-fable-p02-result-interpretation-and-paper-disposition.md`; `notes/20260712AA-sol-p02-final-interpretation-and-fable-disposition.md` |
| P01 packages and analysis | [ADD EXACT PATH BEFORE PUBLISHING — not recorded in the integration inputs; the P01 analysis SHA-256 is `c370d1cc2dbe2a9ee5db64a19ce8837f9879f3b8fc008210756d2985d41aac7e`] |
```

## 6. Pre-paste verification checklist (do these before applying patches; none require a paid run)

1. **Transformers 4.57.6 for the screen** — owner-supplied fact in this integration brief. Confirm it against the runtime record inside `results/precision_probe_p02/` before pasting the sentences in P7 and P12 that state it; while there, capture Torch/CUDA/driver library fields if the editor wants A.6 at full A.3 parity.
2. **P01 artifact path** for the Appendix B row — not present in my inputs; do not guess from the `precision_probe_p02` naming pattern.
3. **Checkpoint revision for the screen** — A.3's `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` is expected but must be bound from the P02 runtime record, not inherited.
4. **Quantization backend identity** — old §12 hypothesized bitsandbytes; my prose says only "NF4" to avoid asserting an unverified implementation. If the runtime record names the backend, A.6 may state it.
5. **"2×2" reading** — I rendered Sol's "frozen 2×2 fixed-fixture protocol" as two regimes × two repeats (it matches "P02 completed NF4 and bfloat16 repeats 1 and 2"); confirm against the A8 note, and cite any canonical frozen-protocol document/SHA in A.6 if one exists beyond the A8 note.
6. **P01 timing relative to the data-collection stop note** — only needed if the editor wants §11 to state the sequence explicitly; P10 is deliberately order-agnostic.

## 7. Methods-provenance requirements: how the new material satisfies the binding items

Checked against `METHODS-PROVENANCE-REQUIREMENTS.md` as a checklist. Items 1 (per-token provenance): the screen introduces no new authored tokens — the fixture is e01, whose complete authorship lineage (GPT-5.6 Sol session-authored histories/probes/targets; experimenter-authored frozen carrier; none subject-generated) is already documented in A.3, which A.6 and §8.4 cross-reference rather than restate; the only new model-generated tokens are the screen's free generations, which are preserved with IDs, text, stop reasons, and hashes. Item 3 (exact checkpoint IDs): the full HF id `Qwen/Qwen3-30B-A3B-Instruct-2507` is stated, with revision bound via the runtime record (checklist item 3); attention-geometry parameters are already tabulated in A.3 for this checkpoint. Item 5 (reproducibility): launch commit, analysis and comparison SHA-256s, receipt-bound lossless packages, and the independent comparator are all cited; the demonstrated exact repeats are themselves the reproducibility evidence, with the boundary (normalized common fields, two observed hosts, no universal claim) stated wherever the evidence is. The file's per-model-native design items (its §§1–2 and nuances 1–2, 5–9) describe an earlier program design that the final paper's strata deliberately document deviations from (A.1, A.3); they impose no new obligation on the screen beyond what §8.4/A.6 state. No provenance claim in my patches goes beyond the Sol note or PAPER.md; unpreserved fields (first-token logits; P01 path) are declared rather than papered over.

## 8. Deliberately unchanged, and residual disagreements

Unchanged: title and headline framing (Sol: P02 changes the precision-axis paragraph's provenance/reproducibility status, "not the paper's headline"); §2 related work; §§4–7 (other strata and the stop, which the screen does not touch); §9.1/§9.3; §11's evaluation-design requirements; Author contributions (P02 execution/interpretation falls under Sol's existing attribution; my A9 advisory review falls under my existing notes-archive attribution); Acknowledgments; References.

Residual disagreements with the Sol note: none. I accept his four narrowings of my A9 review and have applied them throughout — "low-value" not "guaranteed" for a third run; "two observed A100 hosts and adjacent driver patches," never "hardware/driver invariance"; normalized-common-field equality, never whole-payload bit identity; the near-tie hypothesis omitted, not promoted. Arithmetic spot-checks pass: SEL = D focal − |D nonfocal| reproduces −0.0582850 (NF4) and −0.0021350 (bfloat16) from the table rows, and the ten estimand rows × two regimes account for exactly the twenty compared estimands. All numeric values in the patches are copied verbatim from the Sol note (including his printed N-minus-P shifts, which I did not re-derive beyond consistency checks); none are invented.
