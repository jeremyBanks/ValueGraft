---
name: verify-provenance-before-claiming-data
description: "Before saying \"we have data for X,\" verify every experimental variable (model, dtype/quant, intervention, metric, condition, split) against the question — proven from the data/runtime, not inferred from a dir name."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

Systemic, repeated failure the owner flagged (2026-07-09, sharply): when asked "do we have data for X?", agents keep answering "yes" when the data is actually the WRONG type — wrong model, wrong dtype/quant, wrong settings, or measuring a different intervention/metric than the question implies. It corrupts a science project. It happened again live in the SWE-Gym "+0.0156": pitched as a "SWE-Gym result on the right model," but precisely it is (a) a teacher-forced **log-prob proxy** of the true next action, NOT resolve/test-pass rate; (b) value-graft at **scalar α=0.75**, NOT the tuned champion; (c) **bf16 inferred from the directory name** `swegym_30b_bf16`, not recorded in the result JSON; (d) the **brief (handicapped) summary**, not production-faithful. Every one of those is a controlled variable that was glossed.

**Why:** This is science. Data measured under the wrong variable masquerading as an answer is worse than no data — it produces false confidence and bad claims. The paper's integrity depends on knowing exactly what each number measured.

**How to apply:** Before claiming any dataset answers a question, enumerate the controlled variables and mark each PROVEN (read from the data or the runtime model object) vs INFERRED (from a dir name, memory, or a launch command): model repo id + weights revision; load dtype + quantization (read from the *loaded model*, e.g. a parameter's dtype / quant_config — never a folder name); exact intervention (arm, graft type value/key, α scalar vs champion config + content hash, alignment/positions); metric definition (proxy vs endpoint — log-prob ≠ task success); summary/condition (brief=handicapped vs prod=faithful); corpus + split + N; code commit. Then report the HEDGED truth, not "yes." Results must self-document this (born-annotated), so provenance never rests on a directory name. See [[mitigation-framing-was-owner-intent]], [[validate-before-trusting]], [[fable-full-context]].
