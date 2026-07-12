# Precision probe p01: matched bf16 versus NF4 weight/runtime screen

**Frozen by:** OpenAI GPT-5.6 Sol (extra-high reasoning), 2026-07-12

**Advisory input:** Claude Fable 5 (`claude-fable-5`) and three independent
Codex subagent audits. Sol retains final design, execution, and interpretation
authority.

**Status:** exploratory protocol, frozen before apparatus implementation or any
p01 subject-model outcome. This is not v12, does not reopen v12, and cannot
change v12's terminal stop or e01's diagnostic-only status.

## 1. Question and claim boundary

P01 asks a narrow screening question:

> In a newly matched runtime, does the fixed-case value-only
> correct-history-versus-wrong-history observation differ when the same model
> checkpoint is executed with bf16 weights versus bitsandbytes NF4 weights?

The manipulated runtime axis is bundled:

- bf16 weight values and bf16 linear algebra; versus
- NF4 weight representation, double quantization, and bitsandbytes linear
  kernels with bf16 compute.

KV state must be observed as bf16 in both regimes. P01 therefore does **not**
test 4-bit KV caches, and it does not isolate weight rounding from kernel or
router effects. The historical MLX result likewise used 4-bit weights with an
fp16 KV cache, but MLX remains a different backend and quantizer.

P01 is a fixed-fixture exploratory screen. It has no population interaction
test, equivalence test, p-value, or confirmatory confidence interval. No
outcome can retroactively validate the void early apparatus or the stopped v12
branch.

## 2. A load trap discovered before launch

The repository's pinned Transformers 5.0.0 runtime is ineligible for stock
bitsandbytes NF4. Qwen3-MoE stores 28,991,029,248 expert parameters (94.9526%
of 30,532,122,624 total) as 96 packed three-dimensional `nn.Parameter`
tensors. The stock integration replaces ordinary `nn.Linear`/`Conv1D`
modules; it would quantize only 192 attention linears, or 2.9673% of logical
parameters, while still setting a global “loaded in 4-bit” flag. That would be
a false 4-bit arm and is prohibited.

P01 instead uses one isolated environment for **both** regimes:

- Torch 2.12.1;
- Transformers 4.57.6;
- Accelerate 1.14.0;
- bitsandbytes 0.49.2;
- huggingface-hub 0.36.2;
- safetensors 0.8.0;
- tokenizers 0.22.2; and
- Python 3.12.11.

Transformers 4.57.6 represents the 48 × 128 experts as 18,432 ordinary
projection linears. A zero-GPU meta-model test observed replacement of all
18,432 expert linears, 192 attention linears, and 48 routers: 18,672 modules
and 29,909,581,824 logical weight elements, or 97.9610% of the checkpoint.
Embeddings, normalization weights, and the deliberately skipped `lm_head`
remain full precision. A tiny Qwen3-MoE test also observed compatibility with
the repository's cache snapshot/rebuild, `DynamicCache(ddp_cache_data=...)`,
explicit logical/physical positions, `cache_position`, and `logits_to_keep`
paths.

These local observations authorize only a paid compatibility gate. Full
checkpoint loading and execution remain unverified until observed on the pod.

## 3. Subject, fixtures, and order

The subject is exactly:

- model: `Qwen/Qwen3-30B-A3B-Instruct-2507`;
- revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`;
- attention: eager;
- device: one A100 80 GB PCIe admitted by the existing provider/driver gate;
- same host for both regimes; and
- the same slow tokenizer (`use_fast=False`) and literal case files in both
  regimes.

The mandatory base fixture is e01:

`data/coherent_canary_v12/revision2/session_d/e01.json`

P01 runs NF4 first, then bf16. This fixed order lets the uncertain/possibly
slow runtime establish the budget forecast before the comparator is loaded.
The archived Transformers-5 bf16 e01 result is secondary historical context,
not the paired comparator.

### Outcome-blind extension

The base design runs e01 twice in each regime. After both NF4 e01 executions
are durably saved, the runner may add e02 and e03 once per regime, all-or-none:

- `data/coherent_canary_v12/revision2/session_d/e02.json`;
- `data/coherent_canary_v12/revision2/session_e/e03.json`.

The extension decision may inspect only monotonic wall time, provider rate,
hard-deadline remainder, and completion status. It must not parse, print,
summarize, branch on, or expose any score or generation. The decision and its
inputs are durably recorded before e02/e03 begin.

Let `t` be the slower of the two complete NF4 e01 outcome-execution wall
times, excluding model load and technical gates. Let `elapsed` be provider
wall time already consumed. The extension is allowed only if both NF4 e01
executions completed and:

`elapsed + 1.20 × (6t + 900 seconds) <= min(7,200 seconds, $4/rate)`.

The `6t` conservatively budgets two additional NF4 case executions plus four
bf16 executions (e01 twice, e02 once, e03 once), assuming bf16 is no slower;
900 seconds budgets bf16 load/gates; 20% is a fixed buffer. Otherwise only the
base e01 repeats run. The extension cannot depend on whether e01 looks
positive, negative, null, or surprising.

## 4. Reused fixed components and complete arm grid

P01 reuses these model-facing functions unchanged:

- replay/fresh planning in `coherent_canary_tokens.py`;
- cache snapshot/rebuild, replacement, continuation, q=1 scoring, and greedy
  generation in `coherent_canary_runtime.py`;
- `run_phase_a_case` and the complete `run_treatment_case` in
  `coherent_canary_case.py`; and
- generated-versus-forced identity and fresh self-replacement in
  `coherent_canary_technical.py`.

It must not call the frozen v12 subject loader, release validators, or
harvester. Nested component payloads retain their historical schema labels for
literal reproducibility, but the wrapper must state:

- `protocol_id = precision-probe-p01`;
- `formal_v12_decision_eligible = false`;
- `v12_reentry_authorized = false`; and
- `component_reuse_does_not_inherit_v12_eligibility = true`.

Every outcome execution contains the whole existing grid:

- Phase A: correct-history oracle, wrong-history oracle, and fresh compaction
  for focal and nonfocal probes;
- N schedule: FF/FC/FW/CF/CC/CW/WF/WC/WW at R1, R2, and R3;
- P schedule: CC/WW/FC/FW at primary R2;
- value-only, key-only, full-KV, and crossed-source views inherent in that
  grid;
- all correct/counterfactual forced log-probabilities;
- every raw greedy answer; and
- one bounded attempt at each existing R1/R2/R3 placebo construction.

The placebo is re-attempted in each regime. Keeping KV dtype fixed does not
imply availability is fixed: NF4 can change the correct-minus-wrong activation
geometry. A missing placebo remains missing, not null. A newly available
placebo remains a single deterministic control, not additional sample size.

No result-dependent cell, region, schedule, case, repeat, seed, or control may
be added.

## 5. Primary and secondary estimands

For state source `X`, case `i`, and runtime `r`, define:

`Y_r,i(X) = mean_lp(correct target | X) - mean_lp(counterfactual target | X)`.

The primary within-runtime value-source contrast is the old e01 target:

`D^V_r,i = Y_r,i(FC) - Y_r,i(FW)` under N/R2.

The primary descriptive cross-runtime contrast is:

`Delta_runtime,i = D^V_NF4,i - D^V_bf16,i`.

For both complete e01 repeats, the analysis must also report:

- nonfocal `D` and `SEL = D_focal - D_nonfocal`;
- correct-target movement `H+`;
- correct and counterfactual target log-probabilities separately;
- the per-target-token contribution, especially the first target-choice
  token; and
- literal greedy generations.

Secondary, explicitly descriptive views are:

- oracle-to-fresh damage;
- full-KV `CC-WW`;
- P/R2 value-only `FC-FW` and the N-minus-P shift;
- R1/R3, key-only, and crossed-source patterns;
- fresh comparisons `U` and `U+`;
- placebo availability and movement; and
- the archived Transformers-5 bf16 e01 result.

If e02/e03 are enabled, report all case values, signs, the arithmetic mean,
and the median. With at most three engineered fixtures, no interval or
population claim is licensed.

## 6. Fail-closed gates

### G0 — local compatibility (already observed)

- Reject the Transformers-5 partial-NF4 path.
- Under the isolated 4.57.6 stack, require the checkpoint/meta-model key sets
  to match, all 18,432 expert projections to be replaceable, exact 18,672
  eligible-linear replacement, and cache API compatibility on a tiny model.

### G1 — full pod load, separately for NF4 and bf16

Both regimes must bind the exact model revision, tokenizer attestations,
dependencies, host, driver, CUDA runtime, eager attention, device placement,
and absence of missing/unexpected checkpoint keys.

NF4 additionally requires:

- exactly 18,672 `bnb.nn.Linear4bit` modules;
- exactly 18,432 expert `Linear4bit` modules;
- logical expert coverage 28,991,029,248/28,991,029,248;
- total logical quantized coverage 29,909,581,824/30,532,122,624;
- every expert weight an initialized `Params4bit` with non-null NF4 quantization
  state, double quantization, bf16 compute, and CUDA placement;
- no ordinary linear or packed bf16 parameter below `.mlp.experts`;
- finite, correctly shaped, nonzero-error dequantized sentinel experts at
  (layer, expert) (0,0), (24,64), and (47,127); and
- an observed real forward whose 48 K/V layer pairs are bf16 with shape
  `[1, 4, T, 128]`.

The bf16 regime must contain no quantization module and must observe bf16
weights and bf16 K/V. A global `is_loaded_in_4bit` flag, successful load, or
memory reduction is never sufficient.

### G2 — identity/surgery ladder in each regime

- generated-versus-forced q=1 identity: pass;
- correct-history N replay and fresh destination repeated exactly;
- cache rebuild/continue identity: pass;
- fresh self-replacement at R1/R2/R3 for K-only, V-only, and K+V: all nine
  cache/continuation/logit comparisons pass;
- selected rows equal their source and all outside rows remain unchanged; and
- all observed K/V tensors remain bf16.

Any G1/G2 failure stops outcome work for that regime. P01 deliberately has no
ULP path-control gate: the prior prose/code conflict is settled, and this new
screen does not rescue v12.

### G3 — provider/time ceiling

The extension rule in §3 is evaluated without outcome access. The whole job
stops at the earlier of 7,200 provider seconds or $4.00 of provider rental.
One degraded-host reprovision may occur before subject work; scientific or
performance failure does not authorize a second healthy host.

### G4 — repeat stability

After both regimes finish, compare normalized e01 model-facing payloads across
the two within-runtime repeats. Paths and timing fields are excluded; plans,
scores, target-token records, generations, source/cache hashes, and control
status are included. If either runtime differs, report the exact differing
fields and do not interpret a small cross-runtime contrast. Do not add a third
repeat.

## 7. Persistence and lifecycle

Before another case or regime begins, each completed outcome execution must be
durably written. Every expensive generation is saved as literal text.

For each runtime/case/repeat, preserve:

- a lossless bounded gzip/base64 package of the full raw Phase-A and treatment
  payloads;
- a compact score/generation extraction;
- a plain-text/Markdown render ledger with every decoded C/W/fresh and probe
  generation;
- exact case/model/tokenizer/dependency/host/quantization/KV attestations;
- source file hashes and immutable output names; and
- stage wall times and provider-rate/cost fields.

The final analysis JSON binds every input by SHA-256. Results are pulled,
verified, placed under `results/`, and committed even when failed, partial,
contaminated, or null. The pod is terminated immediately after artifacts are
pulled and verified, before paper writing resumes.

## 8. Interpretation matrix

- **Similar favorable forced-logprob pattern in bf16 and NF4:** in this fixed
  matched apparatus, the directional observation is not unique to bf16.
- **NF4/bf16 forced-logprob divergence:** this fixed-case observable differs
  across the bundled weight/kernel runtimes by the reported amount. Without a
  working placebo and stable repeats, it is not identified as semantic or as a
  population precision interaction.
- **Neither regime reproduces the old hint:** adverse repeatability evidence
  for the archived e01 residue under a matched alternate runtime; it does not
  explain the early MLX result.
- **NF4 changes a generated answer where bf16 does not:** a screening signal
  for a new multi-case, placebo-controlled preregistration, not an efficacy
  result from p01.
- **Placebo movement comparable to FC-FW:** adverse evidence against semantic
  attribution in that runtime/case.
- **Oracle/fresh damage differs:** a runtime-axis observation, not an automatic
  apparatus failure, provided G1/G2 passed.
- **Technical gate failure:** evidence about apparatus compatibility only; no
  semantic conclusion.

No p01 outcome licenses a claim about 4-bit KV, other quantizers, other model
families, live agents, task success, general compaction mitigation, or the cause
of the early 4-bit MLX headline.
