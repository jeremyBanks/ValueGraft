# V10 gate and semantic-sample input-lineage audit

**Author:** Carver — `gpt-5.6-sol-xhigh`  
**Date:** 2026-07-11  
**Scope:** Independent, literal-input audit of every `coherent-state-gapped-v10`
authorizing stage and the planned N=6/N=12 semantic sample. This audit traces
what each check actually consumes rather than trusting its label or coverage
count. It classifies checks as construction, smoke, stress, or inferential and
examines shared templates, source ancestry, donor reuse and cycling,
plants-per-conversation clustering, fixtures with no substantive failure mode,
and the limits of 0.6B-to-30B transfer.

No model was run for this audit. Repository code, frozen specifications, input
files, committed artifacts, and git provenance were inspected read-only.

## Executive verdict

1. The thirteen-stage v10 ladder is a release/applicability suite, not thirteen
   independent pieces of scientific evidence. It contains construction and
   provenance assertions, single-template plumbing smoke tests, two schedule
   stress panels, and one engineered positive-control smoke test. None is an
   inferential test of the semantic estimand.
2. The semantic endpoint is genuinely N=6 or N=12 at the **conversation** level.
   The two selected plants are correctly averaged within a conversation instead
   of being treated as independent observations. However, the twelve
   conversations are a purposive panel of twelve topic variations from one
   shared scaffold family, not twelve random draws from a defined population.
3. The external donor redesign eliminates literal donor reuse and the original
   donor-cycle graph dependence. It does not make the wrong histories
   representative. Each target has one fixed, artificial donor-slot control;
   short donor message pools are cyclically repeated to fill the target's exact
   token length. In the committed v10 donor artifact, the mean number of cycles
   per replaced slot is 7.34–12.44 and the maximum is 20–51, depending on pair.
4. The 0.6B ladder cannot establish exact-30B numerical behavior. The design
   appropriately requires a separate exact-30B technical pass and actual-render
   source and destination schedule checks before semantic scoring. Those
   conjunctive checks limit the extrapolation, but any substantive result still
   applies only to the exact checkpoint, eager Hugging Face path, and frozen
   mechanistic layout.
5. The observed v10 ladder result demonstrates the representativeness issue
   directly: all seven synthetic schedule rows were exact zero, while the first
   committed natural-prefix row, c10, failed with aggregate discrepancy 16.125.
   The natural row recorded K 16.125, V 5.125, last-logit 0.59375,
   selected-margin shift 0.060546875, continuation-logit 0.84375,
   continuation-K 1.0, and continuation-V 1.59375. The failure sidecar is
   committed in `4ad714f`.

## Classification vocabulary

- **Construction:** verifies identities, hashes, layout, coverage, or a frozen
  intervention recipe. It can detect a malformed apparatus but does not test a
  response of scientific interest.
- **Smoke:** a small, usually single-template execution proving that one code
  path runs and satisfies a narrow invariant. It can catch a plumbing defect but
  is not representative numerical coverage.
- **Stress:** exercises the apparatus over meaningful lengths, partitions,
  positions, or actual production inputs. It validates numerical applicability
  to those inputs, not a semantic effect.
- **Inferential:** supplies observations to an estimand and uncertainty
  calculation. In this design, only the terminal semantic sample is
  inferential.

## Evidence matrix: the thirteen v10 ladder stages

The exact stage order is declared in `src/l_coherent_state_hf.py:195-209`; the
coverage counts and stage schemas are in `src/l_coherent_state_hf.py:232-349`.

| Stage | Literal input and actual coverage | Class | What a pass does and does not establish |
|---|---|---|---|
| `static_provenance` | Model, tokenizer, device, dtype, and inventory declarations. In the local ladder the stage is written as PASS directly after loading (`src/l_coherent_state_hf.py:2097-2127`). | Construction | Detects binding or environment mismatch. It performs no model-mechanism or semantic test. |
| `attention_backend` | Enumerates the local 28 or production 48 attention layers and checks eager resolution, bf16, and context coverage (`src/l_coherent_state_hf.py:1450-1485`; production geometry in `src/run_coherent_state_hf.py:110-119`). | Construction | Establishes applicability to the declared backend and geometry. It is not an independent input sample. |
| `synthetic_schedule_fixtures` | Seven parameterized rows all cycle the same five token IDs obtained from `alpha beta gamma delta epsilon`. The contiguous lengths are 5, 64, 900, 4096, 4097, and 8193; the seventh row uses 64 tokens at logical positions 0–31 and 8192–8223 (`src/l_coherent_state_hf.py:176-187,529-652`). | Stress, low input diversity | Stresses query partitions, the 4096 boundary, long prefixes, and a logical gap. These are seven parameter settings over one literal, not seven independent examples. Amendment 4 explicitly calls them technical rather than sampled conversation lengths (`COHERENT-STATE-PREREGISTRATION-AMENDMENT-4.md:124-153`). |
| `committed_case_schedule_fixtures` | Twelve distinct banked `data/synthetic/c01..c12.json` prefixes, under the production tokenizer, comparing ordinary chunks with separate system/history/request blocks (`src/l_coherent_state_hf.py:674-862,1530-1554`; specification in Amendment 5:108-186). No generated summary, probe, target, calibration margin, or arm score is computed. | Stress | Broadens lexical, positional, and length coverage. It uses banked 4B-rendered bodies and shares scenario ancestry with the semantic sample, so it is neither independent scientific evidence nor exact coverage of the future fresh 30B renders. |
| `generated_replay_identity` | One toy prompt: system `Answer briefly.` and user `Reply with the single word OK and then stop. Do not explain.`, with cap 192 (`src/l_coherent_state_hf.py:1556-1587`). | Smoke | Can catch a generated-versus-q=1 replay mismatch. It does not stress a typical long, contentful summary up to the frozen 900-token cap. |
| `snapshot_rebuild_identity` | The five-token alpha/beta/gamma/delta/epsilon prefix plus the first token from ` z` (`src/l_coherent_state_hf.py:1592-1648`). | Smoke | Checks one short cache snapshot/rebuild continuation. |
| `physical_causal_mask_identity` | The same five-token prefix plus a short ` z A B` extension at logical positions starting at 37 (`src/l_coherent_state_hf.py:1650-1704`). | Smoke / construction | Checks automatic versus explicit physical causal masking for one tiny stream. It does not provide natural schedule coverage. |
| `future_mutation_identity` | Replaces only the final extension token with the first token from ` C` and verifies earlier logits and cache rows remain unchanged (`src/l_coherent_state_hf.py:1706-1755`). | Smoke | A useful causal-leak unit test, but only one short literal configuration. |
| `position_structure` | The names c10 and c13 denote artificial records created by `fake_conv`, not the real c10/c13 files. They contain approved labels A/B and `target-tail`/`donor-tail` markers (`src/l_coherent_state_hf.py:64-82,1757-1850`). | Construction | Validates the exact-length wrong-slot and gapped-position recipe. It does not test the actual cases its labels resemble. |
| `intervention_propagation` | Reuses the same fake record. After self-copy and insertion checks, it adaptively tries alternating positive/negative V perturbations at epsilon 0.1, 0.3, 1.0, and 3.0 until continuation logits and recomputed tail change (`src/l_coherent_state_hf.py:1861-2015`, especially 1921-1956). | Engineered smoke / positive control | Proves the path can respond to a deliberately large perturbation. It does not show sensitivity to natural coherent K/V, the expected effect direction, or a relevant effect size. |
| `calibration_construction` | Exactly two deterministic A/B token templates; no model forwards or scoring (`src/coherent_state_calibration.py:26-67,70-186`; stage code at `src/l_coherent_state_hf.py:2017-2041`). | Construction | Checks token-length matching and frozen calibration layout. It is not a sensitivity result. |
| `external_donor_construction` | Recomputes twelve unique target/donor token constructions and checks hashes, disjoint IDs, structural equality, exact coverage, and special-token exclusion (`src/validate_coherent_external_donors.py:38-178`; stage code at `src/l_coherent_state_hf.py:2042-2067`). | Construction | Establishes that the prespecified artificial intervention was built literally as declared. It does not establish donor representativeness. |
| `retired_G_delta` | A fixed record stating `executed: false` and `authorizes_run: false`, followed by unconditional PASS (`src/l_coherent_state_hf.py:2068-2077`). | Construction / compliance assertion | Contains no model evidence. It cannot substantively fail unless execution or artifact machinery itself breaks. It should never be counted as a scientific or numerical gate. |

### Coverage counts are not independent-case counts

The schema calls the synthetic coverage 7, the committed-case coverage 12, most
identity coverage 1, calibration-construction coverage 2, and donor coverage 12
(`src/l_coherent_state_hf.py:280-333`). These are completeness counts:

- Synthetic 7/7 means seven declared parameter rows over one token literal.
- Committed-case 12/12 means twelve shared-template banked prefixes were checked.
- Calibration 2/2 means the A and B deterministic constructions exist.
- Donor 12/12 means twelve unique fixed pair constructions were validated.

None of those denominators is an inferential sample size.

## Exact-30B and actual-render gates

### 0.6B versus 30B scope

The local ladder loads the **production tokenizer** at the exact 30B tokenizer
revision but loads `Qwen/Qwen3-0.6B` in bf16 eager mode on CPU
(`src/l_coherent_state_hf.py:61-63,2097-2110`). The release resolver expects 28
layers for the local model (`scripts/validate_semantic_release.py:238-284`). The
production subject is `Qwen/Qwen3-30B-A3B-Instruct-2507` at revision
`0d7cf239...`, with 48 layers, 32 attention heads, 4 KV heads, and head dimension
128 (`src/run_coherent_state_hf.py:92-119`). The local ladder has a different
model scale, layer count, KV geometry, architecture/routing regime, device, and
runtime kernel behavior.

Consequently, a local ladder pass can support code-path plausibility and detect
many implementation defects; it cannot authorize a claim about exact-30B
numerics by itself.

Amendment 11 correctly makes semantic authorization the mechanical conjunction
`L AND T`, where L is the eligible committed local ladder and T is a separately
harvested exact-model 30B technical PASS
(`COHERENT-STATE-PREREGISTRATION-AMENDMENT-11.md:70-109`). The exact-30B T suite
still consists of construction, smoke, and stress evidence rather than semantic
evidence, but it closes the direct architecture/device applicability gap.

### Per-render stress gates

The committed-case stress panel uses previously banked bodies, whereas the paid
semantic process freshly renders every conversation with the exact 30B subject.
The design recognizes this difference:

- Before generating a case summary or any semantic outcome, the process runs an
  exact schedule comparison on that case's **actual freshly rendered source
  prefix** (`COHERENT-STATE-PREREGISTRATION-AMENDMENT-7.md:86-114`;
  `src/run_coherent_state_hf.py:868-895`).
- After durably saving the generated summary but before constructing arms or
  scoring outcomes, it runs a second schedule comparison on the **actual
  compacted destination**, including the complete saved summary
  (`COHERENT-STATE-PREREGISTRATION-AMENDMENT-8.md:42-66`).

These are the strongest schedule-applicability checks because they operate on
the literal semantic inputs. They remain technical stress gates: they validate
schedule equivalence, not the semantic estimand.

## Observed synthetic-versus-natural schedule divergence

The current eligible ladder artifact is
`results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z.json`.
Its synthetic stage recorded all seven declared rows passing with aggregate
discrepancy exactly 0.0. The first committed-case row then evaluated the real c10
banked prefix and failed:

| Metric | c10 observed value | Frozen limit |
|---|---:|---:|
| Aggregate | 16.125 | 0.0005 |
| Prefix K maximum | 16.125 | 0.0005 |
| Prefix V maximum | 5.125 | 0.0005 |
| Last-logit maximum | 0.59375 | 0.0005 |
| Selected-margin absolute shift | 0.060546875 | 0.0005 |
| Continuation-logit maximum | 0.84375 | 0.0005 |
| Continuation K maximum | 1.0 | 0.0005 |
| Continuation V maximum | 1.59375 | 0.0005 |

The committed evidence is
`results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z__stage_committed_case_schedule_fixtures.json`
at commit `4ad714f`.

This is not a semantic result. It is direct evidence that the synthetic
five-token-cycle panel did not represent the numerical behavior of at least the
first natural committed prefix, despite exercising similar declared lengths and
partitions. It strongly supports retaining both the committed-case and
actual-render schedule gates and not describing the synthetic 7/7 as broad
input validation.

## Semantic sample ancestry and representativeness

### Sampling frame and frozen order

The planned cases are the first twelve scenarios from a 54-scenario file. The
first twelve were originally added together as one authored corpus in commit
`c2b67ed`. Their starts in `data/scenarios.json` are c01 line 3, c02 line 165,
c03 line 329, c04 line 493, c05 line 656, c06 line 821, c07 line 985, c08 line
1150, c09 line 1313, c10 line 1475, c11 line 1636, and c12 line 1796.

The frozen evaluation order is:

```text
c10, c02, c01, c04, c07, c11, c05, c09, c06, c12, c08, c03
```

It was obtained by sorting a hash of the date and scenario ID
(`COHERENT-STATE-PREREGISTRATION.md:32-57`; literal order at
`src/coherent_state_cases.py:19-29`). Hash ordering prevents discretionary
reordering after outcomes, but it does not make the twelve scenarios a random
sample from a population. N=6 is the first half of this deterministic order;
N=12 is the entire purposively selected panel.

### One common scaffold family

All twelve scenarios have exactly the same structural field counts:

- 2 early user turns;
- 4 middle filler user turns;
- 2 tail filler user turns;
- 10 planted items;
- exactly 2 plants in each of `referent`, `sense`, `stance`, `ruled_out`, and
  `evicted_fact`;
- 4 plants with tail-use fragments.

The common composer deterministically shuffles plants, interleaves about three
plants between each filler, and later shuffles the tail fragments
(`src/compose.py:143-196`). The exact-30B renderer is a port of the same scaffold
algorithm (`src/cross_arch_probe.py:1553-1577,1632-1650`). Topics differ—software
launch, conference, app redesign, academic paper, podcast, garden, board game,
commerce migration, fundraiser, renovation, course, and robotics—but the
conversation and plant-generating template is shared.

This is valuable controlled topic diversity. It is not equivalent to ecological
diversity across real conversations, compaction systems, agent traces, summary
requests, or model families.

### Selected plants and common target authorship

Only the first `referent` plant and first `sense` plant in each scenario are
scored (`src/coherent_state_cases.py:57-71`; original preregistration line 55).
Across all twelve conversations these categories themselves follow closely
shared patterns:

- `referent`: choose one of three named options, then later ask which option was
  selected;
- `sense`: define one codename as referring to one of two entities, then later
  ask which meaning applied.

All 24 correct/counterfactual target pairs were condensed and matched by one
non-subject collaborator and frozen together in
`data/coherent_state_targets.json`. The loader requires exactly those 24 rows
and attaches each source scaffold-gold hash
(`src/coherent_state_cases.py:155-188`). This creates clean matched targets, but
also common authorial and stylistic ancestry.

### Plants-per-conversation clustering

The referent and sense plants in one case share the same rendered conversation,
source summary, retained tail, donor pairing, and arm caches. They are not
independent replicates. The analysis correctly defines the conversation outcome
as their equal-weight mean and names the conversation as the independent unit
(`COHERENT-STATE-PREREGISTRATION.md:121-135`). Therefore:

- interim effective N is 6, not 12 plants;
- endpoint effective N is 12, not 24 plants;
- category and per-plant analyses are descriptive;
- an analysis treating 24 plants as independent would be pseudoreplication.

### Fresh 30B generation does not create population sampling

The exact subject freshly generates assistant elaborations at temperature zero
from the same fixed system and user scaffolds, structural seed, and turn plan
(`src/run_coherent_state_hf.py:824-829`; `src/cross_arch_probe.py:1558-1571`).
This is an important provenance improvement over reusing old model renders, but
it does not turn the fixed scenarios into population draws. It adds one subject
model's deterministic completions to a common authored scaffold family.

## External wrong-control lineage

### What the redesign fixes

The original wrong-history mapping made each scored target another target's
donor, producing connected donor cycles and shared-data dependence. Amendment 3
replaced it with twelve unique, disjoint external donor files. The frozen map is
in `src/coherent_state_cases.py:24-29`; its rationale and claim boundary are in
`COHERENT-STATE-PREREGISTRATION-AMENDMENT-3.md:7-71`.

No donor ID or file hash is reused. No donor is itself a scored target. This
does repair the direct reuse and graph-dependence defect.

The donors nevertheless share batch authorship:

- c13–c15: `sonnet`;
- c16–c18: `opus`;
- c25–c27: `codex-gpt5.5`;
- c28–c30: `sonnet-render`.

They are authored controls, not exact-30B subject-native conversations
(`COHERENT-STATE-PREREGISTRATION-AMENDMENT-3.md:47-53`).

### Literal wrong-slot construction

The wrong-source algorithm retains every target structural token, target system,
target retained tail, request/header, summary positions, and visible destination
token. It replaces only the target's evicted message-content slots. For each
role, it takes the corresponding donor message token pool and repeats it modulo
the pool length until it exactly fills the target slot
(`src/coherent_state_tokens.py:109-186`, particularly lines 145-170). The donor
validator recomputes these slots and confirms exact coverage, structural
equality, changed-position confinement, and unique donor IDs/hashes
(`src/validate_coherent_external_donors.py:38-178`).

This exact matching is causally clean with respect to length, position, wrapper,
and visible destination. It also makes the wrong histories highly artificial:
long target assistant messages may consist of a short donor phrase repeated
many times.

### Measured cycling in the committed v10 donor artifact

The following values were independently recomputed from
`results/coherent_state_ladder/coherent_external_donors_gapped_v10_Qwen3-30B-A3B-Instruct-2507_20260711T122320Z.json`.
Each pair contains 32 replaced content slots.

| Target | Donor | Recorded donor author | Correct-prefix tokens | Replaced-content positions | Changed positions | Mean cycles/slot | Maximum cycles |
|---|---|---|---:|---:|---:|---:|---:|
| c10 | c13 | sonnet | 8,430 | 6,957 | 6,906 | 7.44 | 26 |
| c02 | c14 | sonnet | 8,385 | 6,796 | 6,760 | 7.34 | 20 |
| c01 | c15 | sonnet | 8,855 | 7,051 | 7,019 | 7.81 | 24 |
| c04 | c16 | opus | 8,595 | 7,042 | 7,000 | 8.38 | 29 |
| c07 | c17 | opus | 8,600 | 6,929 | 6,903 | 8.62 | 30 |
| c11 | c18 | opus | 9,381 | 6,917 | 6,875 | 8.56 | 23 |
| c05 | c25 | codex-gpt5.5 | 8,556 | 6,881 | 6,833 | 9.88 | 20 |
| c09 | c26 | codex-gpt5.5 | 9,195 | 6,909 | 6,867 | 10.31 | 25 |
| c06 | c27 | codex-gpt5.5 | 8,876 | 6,753 | 6,709 | 10.03 | 22 |
| c12 | c28 | sonnet-render | 9,509 | 6,880 | 6,844 | 11.88 | 43 |
| c08 | c29 | sonnet-render | 8,525 | 6,828 | 6,798 | 11.94 | 51 |
| c03 | c30 | sonnet-render | 8,913 | 6,896 | 6,859 | 12.44 | 47 |

The difference between replaced-content and changed-position counts reflects
occasional coincident equal token IDs; it does not indicate unchanged slots.

### Consequence for independence and interpretation

Each `GW_i` is now a function of a unique target and a unique fixed donor file,
so direct pair overlap is gone. But there is still only one wrong-history
stimulus per target, and those stimuli come from four three-item author batches
and a highly repetitive exact-length transformation. Student-t or bootstrap
uncertainty therefore describes variability over these twelve fixed target-pair
constructions under an exchangeability assumption; it does not sample the
distribution of plausible wrong histories.

Amendment 3 already states this limitation: its intervals are not guaranteed
population coverage for arbitrary conversations or arbitrary wrong histories,
and congruence or surprisal can remain explanations for `G_correct-G_wrong`
(`COHERENT-STATE-PREREGISTRATION-AMENDMENT-3.md:55-71`). Wrong-summary NLL is an
important diagnostic but does not transform the single fixed donor into a
representative control distribution.

## Inferential scope of N=6 and N=12

N=6 is a competence, headroom, calibration, and futility checkpoint only. It
cannot produce an efficacy declaration. Amendment 3 makes confirmatory
interpretation available only at the frozen N=12 endpoint
(`COHERENT-STATE-PREREGISTRATION-AMENDMENT-3.md:73-78`).

At N=12 the co-primary intersection is inferential for this fixed benchmark:

- `GF = mean_i(Y_i,G_correct - Y_i,G_fresh)`;
- `GW = mean_i(Y_i,G_correct - Y_i,G_wrong)`.

Both intervals must clear zero. The t interval and conversation bootstrap
operate on one contrast per conversation. Within that design they are coherent,
but their broader interpretation depends on treating these purposively authored,
closely templated cases as exchangeable with some target population. No such
population or sampling mechanism is defined.

Accordingly:

- a positive result can support a fixed-benchmark, exact-checkpoint mechanism
  claim under the frozen layout;
- an inconclusive result cannot establish absence of a channel;
- neither sign directly generalizes to real coding-agent transcripts, other
  compaction requests, other model checkpoints, other attention backends, or
  production serving stacks;
- the paper should distinguish uncertainty across the twelve cases from
  uncertainty about external validity.

The frozen documents already license only the exact Qwen3-30B revision, bf16,
eager Hugging Face apparatus, and position-preserving layout, explicitly not
other backends, checkpoints, tasks, or a real coding agent
(`COHERENT-STATE-PREREGISTRATION-AMENDMENT-4.md:295-310`;
`COHERENT-STATE-PREREGISTRATION-AMENDMENT-5.md:431-440`).

## Recommended labels for the paper and release record

Use these groups consistently:

### Release integrity / construction

- static provenance;
- attention-backend attestation;
- calibration construction;
- external donor construction;
- retired-`G_delta` non-execution assertion.

### Plumbing smoke

- generated/replay identity;
- snapshot/rebuild identity;
- explicit physical-mask identity;
- future-mutation causal identity;
- position/source structure;
- engineered intervention propagation.

### Numerical stress

- synthetic schedule panel: **seven parameterized schedule rows from one
  five-token literal**;
- committed-case schedule panel: **twelve banked, shared-template natural-prefix
  stress cases**;
- per-render source schedule fixture: **actual exact-30B source-input stress**;
- per-render destination schedule fixture: **actual exact-30B compacted-input
  stress**.

### Inferential

- N=6: regime/futility canary only;
- N=12: co-primary `GF` and `GW` intersection over twelve conversation-level
  contrasts in the fixed benchmark.

## Phrasing to avoid and preferred corrections

Avoid:

- “seven synthetic cases”;
- “twelve independent donor controls” without qualification;
- “six calibration items”;
- “24 independent probes”;
- “the 0.6B ladder validates production numerics”;
- “representative conversations” or population-general confidence intervals;
- treating the hard-coded retired-arm PASS as model evidence.

Prefer:

- “seven parameterized schedule rows constructed from one repeated token pool”;
- “twelve unique fixed target/donor-pair constructions, with donors drawn from
  four author batches and cyclically expanded into exact-length slots”;
- “two deterministic calibration variants, repeated across conversation IDs”;
- “two plants averaged within each of twelve conversation units”;
- “a 0.6B implementation ladder whose eligibility is conjunctive with a
  separate exact-30B technical pass”;
- “a purposive, tightly templated fixed benchmark with topic diversity but
  limited ecological and population coverage.”

## Final assessment

The experiment's internal causal construction is substantially stronger than a
naive reading of its small sample might suggest: visible tokens and positions
are tightly controlled; the two plants are correctly clustered; literal donor
reuse was removed; exact-30B and actual-render technical checks are required;
and the claim boundary is narrow. The central remaining limitation is not a
hidden arithmetic pseudoreplication in the primary analysis. It is external
validity: twelve closely templated authored cases, two closely templated plant
types, one deterministic subject checkpoint, and one artificial cyclic donor
control per target.

Technical gate coverage should therefore be reported as evidence that the
apparatus was constructed and numerically applicable to the tested inputs, not
as additional sample size or corroborating semantic evidence. The semantic
result, if one becomes technically eligible, should be presented as a
fixed-benchmark exact-apparatus result and as motivation for broader,
independently authored, multi-template, multi-donor, and real-agent follow-up.
