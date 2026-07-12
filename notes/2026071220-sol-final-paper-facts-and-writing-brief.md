# Facts and writing brief for the fresh final paper

**Prepared by:** Sol — OpenAI GPT-5.6 Sol, extra-high reasoning  
**Date:** 2026-07-12  
**Purpose:** factual/narrative brief for Claude Fable 5's initial fresh draft

## 1. Assignment and non-negotiable conclusion

Write a genuinely fresh paper. Do not patch, continue, or imitate the current
`README.md`; it is polished but scientifically superseded. Its blocking audit is
`notes/2026071115-sol-final-paper-review.md`. Historical `report2-plan` and
`report-synthesis` are style references only; their positive scientific spine
did not survive later validation.

The paper's safe central conclusion is:

> Generation-time KV state can carry useful information beyond restart text,
> but that broad premise is already established by prior work. This repository
> did not establish that a naive, training-free post-hoc old-value/fresh-key
> transplant reliably mitigates conversation compaction. Trustworthy
> performance evidence was mostly null, harmful, or control-incomplete; the
> strongest coding result was a small likelihood proxy without behavioral or
> matched-placebo confirmation; and the final exact-state canary formally
> stopped at a technical gate, leaving one diagnostic case with only a tiny,
> schedule-sensitive, uncontrolled value-only trace and no behavioral recovery.
> The strongest original contribution is methodological: finite-precision
> execution trajectory, source-state provenance, decoded control validity,
> control representability, and literal agreement between prose and code can
> move or invalidate a KV-transplant measurement at the scale of the hoped-for
> effect.

Do not claim that the channel is absent, that grafting can never work, that a
confidence interval spanning zero proves equivalence, or that this repository
established a practical agent effect. Paid data collection is finished; no pod
or live-agent evaluation is running or authorized.

## 2. What the paper contributes

The contribution is not a branded technique and not the broad discovery that
state can preserve information across compaction. It is three narrower things:

1. A transparent negative/bounding evaluation of one family of **training-free,
   post-hoc activation transplants** on an ordinary instruction model.
2. Two tightly bounded leads that did not clear the causal/practical ladder:
   one coding-domain likelihood proxy and one single-case exact-state
   diagnostic.
3. A concrete failure catalogue for activation-state experiments, including
   several failures that passed elaborate hash, test, review, and release
   machinery because those checks validated the wrong literal assumption.

The final paper should be science first and process second. The process material
is valuable, but present it as failure → consequence → guardrail, not as a long
chronology or a redemption narrative.

## 3. Evidence hierarchy — keep these strata separate

Never pool sample sizes, state procedures, interventions, controls, or status
across these four strata.

| Stratum | What actually ran | Strongest licensed reading |
|---|---|---|
| Legacy synthetic | Foreign/mixed-source conversation bodies; 30B self-generated summaries; prefill-reconstructed source; combined summary+tail value graft | No detected average lift in that exact exploratory apparatus; no equivalence or clean generation-summary-state null |
| SWE-Gym/OpenHands proxy | Historical successful trajectories; 30B self-generated brief summary; actual generation-mutated source snapshot; combined summary+tail value graft | Small out-of-fitting demonstrated-next-action likelihood effect for a selected layer map; no matched selected-map placebo, behavior, executed action, or task-success result |
| Coherent-state v10/v11 | Local technical fixtures, failed controls, paused draft corpus | Methodological evidence only; no semantic treatment result and no sample |
| Formal v12 + e01 | Exact 30B bf16 canary with fixed authored correct/wrong histories, same fixed carrier text, N/P forced replay | Formal technical stop; one later nonauthorizing N=1 diagnostic gives a weak, schedule-sensitive, placebo-uncontrolled value-only hint |

## 4. Legacy synthetic evidence — what is safe

### 4.1 Held-out average

Four selected variants on 18 held-out conversations / 203 plants had these
graft-minus-compacted estimates (conversation-clustered 95% intervals):

| Variant | Mean nats/token | 95% CI |
|---|---:|---:|
| Per-head | +0.017 | [-0.027, +0.062] |
| Per-layer | -0.003 | [-0.041, +0.043] |
| Intersection | +0.008 | [-0.032, +0.052] |
| Union | -0.012 | [-0.053, +0.034] |

This supports “no detected average lift in the apparatus that ran.” It does not
support “zero,” “definitive null,” “refuted,” or equivalence; the upper bounds
still permit modest effects.

The apparatus must travel with every claim:

- subject/scorer: `Qwen/Qwen3-30B-A3B-Instruct-2507`, resolved revision
  `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, Torch bf16 on A100;
- c07--c12 bodies were generated by
  `mlx-community/Qwen3-4B-Instruct-2507-4bit`, temperature 0.7, seeds
  1006--1011, with truncation/tail repairs in metadata;
- c13--c24 bodies were externally authored by assistant sessions labeled
  Sonnet, Opus, and Fable; exact runtime receipts are not always exposed;
- the 30B subject did **not** generate these conversation bodies;
- the 30B subject generated their compaction summaries;
- the synthetic harness discarded the actual incremental generation snapshot
  and reconstructed old state by prefill;
- the graft covered retained-tail **plus** summary regions, not summary only;
- alignment was difflib positional-within-region; and
- headline manifests often contain `git_commit:null`.

System prompts, user turns, plants, probes, and golds were AI-assisted synthetic
authorship under human direction — not simply “hand-authored” and not generated
by the experimental subject.

### 4.2 Placebos

The old correct graft often beat position-shuffled or Gaussian sources. The
safe interpretation is only that aligned source values were non-exchangeable
and often less disruptive than grossly mismatched full vectors. The controls
did not match the actual treatment delta `V_old - V_fresh`; Gaussian rows
matched source energy, shuffles were not proven strict derangements, and correct
old rows may simply have been closer to fresh. Do not use “content-specific
semantic information” or “the mechanism is real.”

### 4.3 Invalid or retired legacy claims

- The four-level compression treatment conclusion is void. Ultra/brief/medium
  generated summary text under their level-specific request but reconstructed
  old state under the default realistic request. Only the realistic cell was
  internally consistent. Compression ratios remain descriptions of text; no
  flat-across-30x or severity conclusion is licensed.
- The old +10--12 percentage-point judged recovery headline was local 4-bit MLX,
  brief/adversarial summary, c01--c12, and render-fragile. A clean rerender moved
  judged sense from +8.7 to +1.0 points and referent from +9.7 to +5.2, with
  intervals spanning zero. Do not revive it.
- Packed H-pack changed fabrication/admission behavior but combined layout,
  transformed K, and V; it restored 0/24 evicted facts. The old “38/48 accurate”
  claim was unreproducible. It is not evidence for value-only semantic recovery.
- Cross-architecture rows were mixed and mostly adverse, but heterogeneous gates
  and provenance do not support a simple QK-norm, dense/MoE, or keys-neutral law.

## 5. The narrow SWE-Gym/OpenHands lead

The strongest surviving performance lead is a **proxy**, not an agent result.
The layer map used alpha 1 on layers 12--17 and 30--35. It was selected on 41
fresh-pool trajectories, evaluated on 57 disjoint fresh trajectories, and
partially confirmed on 45 of the planned 75 original-pool trajectories.

Clean out-of-fitting results:

- fresh evaluation 57: selected map minus baseline `+0.0117`, 95% bootstrap CI
  `[+0.0063,+0.0172]` nats/token;
- original partial confirmation 45: about `+0.0158`, interval above zero;
- pooled wholly out-of-fitting 102: `+0.0135 [+0.0083,+0.0190]`.

Required caveats:

- the target is one historically demonstrated next action, not necessarily the
  unique correct action;
- the metric is teacher-forced mean token log probability;
- no action was applied, patch tested, or repository task solved;
- the selected map had no matched placebo restricted to its selected layers;
- on the fresh 57, selected minus fixed graft was inconclusive (`+0.0041`, CI
  spanning zero);
- the coarse tool/path/command structural-match metric was 53% selected versus
  54% baseline on the fresh evaluation set; across all 102 out-of-fitting rows,
  three baseline misses were fixed and three baseline successes were broken;
- confirmation stopped at 45/75;
- trajectories were bootstrapped as independent without a completed
  repository/task-cluster sensitivity analysis;
- the local ignored `swegym.parquet` currently hashes
  `ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1`,
  but the immutable upstream revision, historical
  trajectory-generating model identities, and per-trajectory generated summary
  text/hash were not preserved; and
- the fixed scalar graft was sample-heterogeneous: positive on the original 75,
  absent on the disjoint 98, pooled interval spanning zero.

Call this “a selected-map likelihood lead” or “demonstrated-action likelihood
effect,” never coding-agent improvement or task success. A fixed scalar under
realistic summaries was null; the selected map was not tested there.

## 6. The clean mechanism redesign and why it stopped

### 6.1 Design

The final canary asked a cleaner same-visible-text question. Correct and
minimally counterfactual histories each produced state for the same fixed
neutral carrier. Fresh state re-encoded the same carrier text in a
position-preserving gapped destination. The primary semantic contrast was
correct-history versus wrong-history state; correct-history versus fresh was a
utility contrast. K+V and value-only were separate families.

Exact subject/runtime:

- model `Qwen/Qwen3-30B-A3B-Instruct-2507`;
- revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`;
- bf16, eager attention, 48 layers, 32 query heads, 4 KV heads, head dimension
  128, RoPE theta 10,000,000;
- accepted A100 80GB PCIe host, driver `580.159.04`, Linux `6.8.0-100`;
- Python `3.12.11`, Torch `2.12.1+cu130`, Transformers `5.0.0`;
- scientific commit
  `cbdfa481fe08de62bf8178d09ca040716d610f21`;
- runtime fingerprint
  `b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484`.

E01 was an engineered fixed transcript authored by an independent Codex
subagent whose exact runtime model/effort was not exposed. The focal difference
changed only a rollback interval: 6:40 versus 10:40. Under the explicit rule,
the correct target was `partner beta` and countertarget `staff ring`. The
unchanged nonfocal target was `21 days`, countertarget `30 days`.

The carrier was not a naturally generated conversation summary. It was the
externally fixed neutral note:

> The prior discussion established the operating context and relevant decision
> criteria. Continue from this handoff, preserve the existing constraints, and
> answer later questions from the state available here. No unresolved action is
> introduced by this note.

It was followed by a fixed anchor request and `Acknowledged.`. N forced
historical assistant content token by token (`q=1`); P prefilled complete
historical messages. Both forced the identical carrier from that point onward.
Neither schedule is native continuous live-agent generation.

R1 contained carrier content, R2 added the carrier close plus anchor user and
assistant-generation header, and R3 added the fixed acknowledgment content.
R2 alone was primary. The 3x3 N grid was `FF, FC, FW, CF, CC, CW, WF, WC, WW`,
with key source first and value source second; P used the primary R2 subset.
All cells/regions/schedules are correlated views of one case, not added N.

### 6.2 Formal path-control stop

The frozen prose said to stop at the first ULP count where both persisted edits
differed measurably from fresh. The code continued until both edits also moved
in the intended signed directions. Exact cells exposed the ambiguity:

| ULP | Oriented plus movement | Oriented minus movement | Sealed-code verdict |
|---:|---:|---:|---|
| 1 | 0.0 | +0.5 | fail |
| 2 | +0.25 | -0.25 | fail |
| 4 | +0.5 | +0.25 | pass |

The literal written branch stops at ULP 2 and fails. ULP 4 survives only as
implementation-defined intervention/readout sensitivity. The conflict was
discovered and dispositioned before treatment. Formal v12 is terminal and can
produce no four-/six-case aggregate, confirmation, conversation, or agent
claim.

Natural calibration was adverse (`rho_green=-0.035714`,
`rho_amber=0.297872`). A path control never establishes semantic validity.

### 6.3 The one post-stop diagnostic

Exactly one unchanged e01 treatment was permitted as diagnostic-only. Its
Phase A was `PRETREATMENT_PASS`; its receipt explicitly records no formal or
expansion eligibility.

| Schedule / region | Family | D focal | D nonfocal | SEL | Hplus | U | Uplus |
|---|---|---:|---:|---:|---:|---:|---:|
| N / R1 | full K+V | +0.154 | -0.022 | +0.132 | +0.023 | +0.007 | +0.260 |
| N / R1 | value-only | -0.126 | -0.114 | -0.240 | -0.274 | -0.154 | -0.199 |
| **N / R2** | **full K+V** | **+0.097** | **+0.245** | **-0.148** | **-0.131** | **+0.175** | **+0.025** |
| **N / R2** | **value-only** | **+0.175** | **-0.00007** | **+0.174** | **+0.710** | **+0.182** | **+0.946** |
| N / R3 | full K+V | -0.010 | +0.142 | -0.153 | -0.086 | -0.275 | -0.166 |
| N / R3 | value-only | -0.005 | +0.186 | -0.192 | +0.257 | +0.002 | +0.361 |
| P / R2 | full K+V | +0.039 | -0.001 | +0.038 | -0.446 | -0.297 | -0.334 |
| P / R2 | value-only | +0.021 | +0.018 | +0.003 | +0.229 | -0.073 | +0.280 |

`D` is correct-history minus wrong-history state. `SEL` subtracts absolute
nonfocal movement. `Hplus` requires correct-target improvement; `U/Uplus`
compare the correct graft with fresh.

Interpretation:

- full K+V failed: nonfocal movement exceeded focal movement and the correct
  target worsened at N/R2;
- value-only N/R2 was the one internally favorable cell;
- value-only `D_focal` collapsed from +0.1745 under N to +0.0210 under P;
- at token level, P/value-only moved the first `partner` versus `staff` choice
  in the wrong direction (-0.1875); only the conditional second token (+0.2296)
  made the phrase mean slightly positive;
- R1/R3, key-only, and crossed cells did not provide coherent corroboration;
- fresh compaction damage was 22.298 nats of margin and 22.291 nats of
  correct-target log probability; the best value cell recovered 0.81% and
  4.24%, respectively;
- all 31 primary cells freely generated the same unrelated/wrong `Ring 3` and
  `30 days` answers;
- all three bf16 placebos were unavailable at the same first nonzero early row
  after 1,024 attempts; zero available placebos means missing evidence, not a
  placebo null; and
- only hashes/shapes/traces/scores were persisted, not K/V tensor element
  values, so the exact placebo cannot be repaired locally.

The strongest safe positive sentence is:

> One fixed engineered case exhibited deterministic, focal-selective
> forced-logprob movement in the exact N/R2 value-only cell under identical
> visible text.

Classify it as a weak, uncontrolled, schedule-sensitive mechanistic hint. Do
not call it semantic, useful, robust, general, or behaviorally recovered.

## 7. Methodological findings worth keeping

Use a compact table or subsections. These are the paper's most durable original
material.

1. **Query shape/schedule is part of the computed bf16 state.** On local
   `Qwen/Qwen3-0.6B` CPU bf16 eager, the same 8,430-token c10 prefix under
   ordinary versus coarse schedules moved a fixed margin `0.060546875`; c02
   moved `0.1318359375`. A diagnostic held future tokens constant and localized
   first divergence to layer-0 attention when query width changed 23→4096.
   Call this deterministic `QUERY_SHAPE_ROUNDING`; do not invent a specific
   tiling/reduction mechanism or generalize its magnitude to 30B/A100.
2. **A gate must be able to fail for its intended reason.** The apparent 7/7
   zero-difference validation used seven lengths of one five-token periodic
   stream. It was pseudoreplication, not seven representative inputs.
3. **Mechanical geometry is not semantic validity.** A wrong-history control
   cycled short donor text up to 51 times. Later exact-width counterfactuals
   preserved tokenizer geometry but failed full decoded review.
4. **Position correction is numerically consequential.** bf16 re-rotation of
   already quantized keys produced target-margin shifts at the effect scale;
   position-preserving gapped state was required.
5. **Small-model green ladders certify plumbing, not the production numerical
   regime.** Long-context 30B failures need exact-stack gates.
6. **Controls can be mathematically valid before casting and unavailable after
   casting.** The norm-matched e01 placebo failed at bf16 representability.
7. **Prose/code agreement is itself a preregistration gate.** Hash-frozen code
   does not resolve ambiguous stopping semantics.
8. **Container identity is not host identity.** Identical Secure A100 type and
   image exposed different NVIDIA drivers, one incompatible with pinned CUDA.
   This is a reproducibility and operations finding, not an explanation for
   unrelated scientific mistakes.
9. **Preserve the state needed for later controls.** E01's raw JSON is lossless,
   but hashes cannot reconstruct K/V rows. A future bounded tensor bundle would
   have been about 39.84 MiB for the necessary geometry rather than roughly
   480.94 MiB for five full snapshots.
10. **Confirmatory machinery should follow an observed signal.** V11's twelve-
    case corpus build was paused because no exact-model effect had yet justified
    it; e01 did not meet the re-entry standard.

The unifying lesson is not “rigor failed.” It is that rigorous checks of hashes,
partitions, and execution can still validate the wrong estimand when the literal
fixture, dtype, control, or stopping rule does not test the generalization being
claimed.

## 8. Related work and novelty

Use primary citations and narrow the novelty.

- **MEMENTO: Teaching LLMs to Manage Their Own Context**
  ([arXiv:2604.09852](https://arxiv.org/abs/2604.09852)). A trained model uses
  custom in-place block masking and retains full memento KV written with the
  original block visible. On AIME24/Qwen3-8B, normal memento attention was 66.1%
  versus 50.8% after restart/re-prefill, a -15.3-point ablation. Disclose that
  normal used 64 repetitions and restart 8. This establishes the broad dual
  text/state stream in a trained, full-KV setting; it does not validate this
  untrained transplant.
- **Models Take Notes at Prefill: KV Cache Can Be Editable and Composable**
  ([arXiv:2606.17107](https://arxiv.org/abs/2606.17107)). A recent single-author
  preprint causally locates conclusions on downstream aggregator/delimiter
  tokens across model families; the field's own KV drives under 1% in its tasks,
  and downstream recomputation restores the decision. This makes a local
  summary-row transplant plausibly target the wrong locus, but it is not a
  direct test of our carrier or agents.
- **CacheBlend**
  ([arXiv:2405.16444](https://arxiv.org/abs/2405.16444)) reuses cached chunks
  while selectively recomputing tokens to restore cross-context interactions;
  it contrasts with uncompensated transplantation.
- **Learning to Compress Prompts with Gist Tokens**
  ([arXiv:2304.08467](https://arxiv.org/abs/2304.08467)) trains models to encode
  prompts into reusable gist-token state; it is learned latent compression, not
  post-hoc state salvage.

Do not say “three searches found nothing” or imply the broad premise is novel.
If a novelty sentence remains, narrow it to the exact training-free,
post-generation, old-state/fresh-state transplant across text compaction plus
its negative/methodological evaluation.

## 9. Practical agent evaluation

State plainly why it was not run: the mechanistic treatment did not clear its
gate. Running coding agents now would test an unstable, control-incomplete
intervention and produce an uninterpretable task result.

A future practical design, only after a new mechanism study succeeds, is:

1. no-treatment full-context versus ordinary-compaction capability/damage
   canary on tasks the subject can solve and where compaction actually fires;
2. freeze one treatment before seeing agent outcomes;
3. paired forks from identical repository/container/task/seed state, standard
   compaction versus treatment, with full context as reference where feasible;
4. repository-level deterministic tests as the primary endpoint;
5. preserve full transcripts, summaries, patches, tests, tool/token budgets,
   and every compaction event; and
6. cluster inference by task/repository and report feasibility, not a powered
   product claim.

The re-entry requirements are in
`notes/2026071288-sol-data-collection-stop-and-future-reentry.md`.

## 10. Methods and provenance requirements

`METHODS-PROVENANCE-REQUIREMENTS.md` is a mandatory checklist of questions, but
some factual answers embedded in it are stale aspirations. Actual artifacts win.

For every numeric table row, give or link:

- estimand/sign convention and formal/exploratory/diagnostic/void status;
- exact artifact path and analysis command;
- model repo ID, resolved revision, dtype/backend/hardware;
- corpus/case IDs and N, exclusions, and fitting/evaluation role;
- source-state procedure, intervention region, replay schedule, summary/carrier
  condition, and generation/forcing configuration;
- control construction and availability;
- bootstrap unit, seed, repetitions, and multiplicity status; and
- known missing provenance.

Keep source token authorship explicit. In particular, neither the legacy
held-out bodies nor v12 histories were native 30B conversations; v12 used
authored fixed replay and a fixed carrier. Do not let a method-level ideal become
an empirical provenance claim.

E01 integrity facts that may be stated:

- treatment-fresh score objects matched Phase A canonically;
- the 8,897,066-byte raw reconstructed byte-for-byte to SHA-256
  `f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082`;
- independent pod/local harvests matched after only allowlisted path/timestamp
  normalization;
- no unexpected post-run difference was found; and
- exact historical launch, packaging, and audit tools are tracked at
  `scripts/historical/coherent_canary_v12_e01/`.

Also state the limitation: no tensor values were retained, so exact e01 state
cannot be reconstructed without rerunning the model.

## 11. Statistics and language guardrails

- CI spans zero → “no detected average lift,” never “zero” or equivalence.
- Overlapping marginal intervals do not test a paired arm difference.
- Separate fitting, internal evaluation, and external confirmation.
- Disclose multiplicity and exploratory selection.
- V12/e01 is N=1 descriptive evidence: no p-value, CI, or population claim.
- A better correct-minus-countertarget margin is not “less loss” unless the
  correct target itself improves.
- R1/R3/key/crossed cells are correlated descriptions, not replications.
- Use “observed” for measured facts and clearly label inference/speculation.
- Do not put an overall cost figure in the draft. The final money/token audit is
  still open and will separate cash, subscription usage, provider credits,
  list-price equivalents, estimates, lower bounds, and unknowns.

## 12. Authorship and narrative provenance

Top-line contributors must include:

- Jeremy Banks — question, direction, funding, repeated methodological
  corrections, and final scope/taste decisions;
- Anthropic Claude Fable 5 — initial final-paper drafting and synthesis; and
- OpenAI GPT-5.6 Sol, extra-high reasoning — primary final analysis, execution,
  interpretation, and scientific decision authority.

OpenAI GPT-5.5 (extra-high) moves to the acknowledgment/review tier. Attribute
Claude Opus 4.8, Claude Sonnet 5, and other contributors only according to
actual runtime evidence; do not relabel Opus work as Fable or infer an exact
model from an intended alias. Fable has full freedom to improve the title and
opening sentences.

Do not narrate mitigation-first framing as an external reviewer's redirection.
Jeremy Banks intended mitigation from the beginning; agents initially
misunderstood the emphasis. See
`notes/2026070911-framing-provenance-correction.md`.

## 13. Suggested paper spine

1. Title and direct abstract: question, mostly negative answer, narrow leads,
   methodological contribution.
2. Background and related work: broad channel established elsewhere; exact
   intervention remains narrower.
3. Evidence-hierarchy table and causal ladder: channel existence → history
   specificity → component localization → intervention utility → agent utility.
4. Legacy exploratory evidence, honestly reconstructed: no detected synthetic
   average; small SWE proxy; why neither resolved the target question.
5. Coherent-state redesign: exact same-text correct/wrong/fresh estimands,
   positions, schedules, arms, gates, authorship, runtime.
6. Formal path-control stop and adverse natural calibration.
7. E01 diagnostic result, token-level schedule reversal, missing placebo, no
   behavior, integrity evidence.
8. Method-failure catalogue and extracted guardrails.
9. What is and is not established; why no live-agent run; future re-entry.
10. Reproducibility/artifact map, limitations, author contributions, and
    acknowledgments.

Aim for a readable technical paper, not an exhaustive repository diary. Put
large provenance maps and exact commands in appendices/tables. Keep the prose
plain enough for a technically literate reader who is not an interpretability
specialist.

## 14. Minimal source map for drafting

Read these in order:

1. `notes/2026071288-sol-data-collection-stop-and-future-reentry.md`
2. `notes/2026071287-sol-e01-diagnostic-final-interpretation.md`
3. `notes/2026071289-sol-e01-token-level-decomposition.md`
4. `notes/2026071115-sol-final-paper-review.md`
5. `notes/2026071144-opus-paper-spine-and-methodological-postmortem.md`
6. `COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md` §§3--17
7. `results/coherent_canary_v12_budget/coherent_canary_v12_path_control_spec_conflict_disposition_20260712T0334Z.md`
8. `results/coherent_canary_v12_budget/coherent_canary_v12_e01_treatment_lifecycle_and_cost_20260712T0419Z.md`
9. `results/coherent_canary_v12_harvest/coherent-canary-v12-harvest-e01-exact-subject-20260712T040158Z.json`
10. `results/coherent_canary_v12_postrun_audit/coherent-canary-v12-postrun-audit-e01-exact-subject-20260712T0418Z.json`
11. `METHODS-PROVENANCE-REQUIREMENTS.md` as a checklist, not a factual authority
12. current top banners of `FINDINGS.md`, `STATE.md`, and `CLAIMS.md`

For legacy row details only, consult
`notes/2026070936-provenance-ledger.md`, relevant result manifests, and the exact
commands in `notes/2026071115-sol-final-paper-review.md`. Do not import their old
interpretations.

## 15. Drafting and review workflow

Fable's first draft should be preserved verbatim in a new `notes/` file. Sol
will then create a fresh working paper file, verify every number and artifact
mapping, and iterate with Fable. The final shareable revision must pass:

- factual/numeric provenance review;
- the methods/provenance checklist;
- terminology and causal-claim review;
- three independent tool-less Fable angles (generic, skeptical, first-reader);
- adversarial critics and readability review; and
- one independent OpenAI GPT-5.5 extra-high review.

Only after those gates does the working paper replace `README.md`. Repository
publication/commit/push is authorized. Anything outside the repository remains
owner-controlled. The end-to-end token/cost audit follows the last paper review
so its cutoff is complete.
