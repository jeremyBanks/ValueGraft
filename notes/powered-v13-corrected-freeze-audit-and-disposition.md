# Powered v13 corrected freeze audit and disposition

**Date:** 2026-07-12  
**Status:** **PASS for the corrected statistical design, its release-validation
artifact, and the compact pre-seed recipe foundation. NOT AUTHORIZED for paid
Phase A or treatment.**  
**Audited baseline:** `a2f92ee4c6370ba7d9152159aacd2bae5466a1ea`  
**Protocol SHA-256:**
`73548de9593e8556dbb30bc74db11f548ca1f80aef6d8fb6d1062d9dec05000c`

## Disposition

The corrected fixed-N statistical rule is mathematically valid under the frozen
stratified simple-random-sampling-without-replacement law. The production core,
an independent standard-library recomputation, golden cases, and the full
finite-population validation agree. The new validation artifact passes every
literal numerical release gate. This closes the statistical blocker that made
the original sequential-t design invalid.

The compact recipe/compiler also passes its deliberately narrower pre-seed
foundation audit. It defines and hashes the finite frame without inspecting any
ranked production history. That pass is not a pool/permutation/content-release
pass.

The preregistration remains `DRAFT`. Complete permutations and ranked first-ten
histories, carrier surface exclusions, content review, the production build
ladder, release-verifier corruption tests, live provider refresh, and the
dedicated Stage-A status/manifest commit remain prerequisites. Therefore this
audit does not authorize provider allocation, paid Phase A, primary treatment,
or unmasking.

## Artifact and provenance binding

The independently audited full artifact is
`results/coherent_state_powered_v13_validation/powered-v13-finite-population-simulation_local_20260712T174743Z.json`.

- Artifact SHA-256:
  `0989b850fa3fbf2bc95079e1173ef4b6186a2d705201e3e84062734cf532df75`.
- Schema/mode/seed:
  `coherent_state_powered_v13_finite_population_simulation_v3`,
  `release_validation`, `20260712`; `test_mode=false`.
- Embedded clean-tree commit:
  `a2f92ee4c6370ba7d9152159aacd2bae5466a1ea`; it equaled `HEAD` when
  independently checked.
- Embedded and independently recomputed hashes agreed for the simulation script
  (`18a9a20e...32b`), production core (`6e065084...7dc`), and `uv.lock`
  (`09d18804...e3fc`).
- Runtime was CPython 3.12.11, NumPy 2.5.1, and SciPy 1.18.0.
- A second execution of the frozen coverage, power, and golden functions at the
  same seed reproduced all three JSON subobjects exactly, including every
  finite-population array hash and Monte Carlo count.

The artifact's `release_eligible=true` means only that this Section-12
simulation artifact met its own release-validation gates. It is not the
preregistration's `STATIC_FROZEN_PHASE_A_AUTHORIZED` or `TREATMENT_RELEASED`
status.

## Mathematical audit

### Independent unit, sampling law, and alpha

The protocol fixes eight strata, six fixtures per stratum, and one final
analysis at N=48 (`COHERENT-STATE-POWERED-SUCCESSOR-V13-PREREGISTRATION.md:127-159,
463-475`). Renders remain nested and are averaged within fixture. Stable IDs are
assigned before a PCG64 without-replacement permutation; candidate seeds cannot
depend on rank. Conditional on the sixth eligible candidate appearing by rank
ten, the unordered first six eligible identities remain uniform because the
conditioning event depends on eligible-position pattern, not eligible labels.
Thus each stratum contributes an SRSWOR of six from its finite eligible
population.

The exact alpha ledger is `.02 + .02 + .01 = .05`. For either clipped cell,
`Z in [-0.5,0.5]`, so the range is exactly one and

`radius = sqrt(log(1/.02)/(2*48)) = 0.20186688594189123`.

The implementation clips before averaging and caps each UCB at `0.5`
(`src/powered_v13_stats.py:223-274`). Fixed equal stratum weights make the
ordinary 48-row mean the declared recipe mean. Hoeffding's independent bounded
form is conservative for the independent stratified WOR samples. Constant or
all-zero data retain the positive radius; observed variance never controls the
primary guarantee.

### Heterogeneous responder prevalence

For `H_i = 1{max(X_Ri,X_Vi)>0.5}` and `S=sum H_i`, the zero branch is

`U0 = 1 - .01^(1/48) = 0.09148242434831322`.

If stratum responder prevalences are `p_h`, the zero-event probability under
WOR is at most `product_h(1-p_h)^6`, which by AM--GM is at most
`(1-mean_h p_h)^48`. The inversion therefore covers the equal-stratum average
even when prevalences differ. For `S>0`, the frozen branch is
`min(1, S/48 + sqrt(log(100)/(2*48)))`, implemented at
`src/powered_v13_stats.py:205-220` and applied at lines 260-270.

The hybrid branch also has a complete tail proof. Let
`epsilon=sqrt(log(100)/96)=0.21902174040653885`. If prevalence is at most `U0`,
both branches cover. Between `U0` and `epsilon`, failure is possible only at
`S=0`, whose probability is below `.01`. Above `epsilon`, every failure,
including `S=0`, is contained in the Hoeffding lower-tail event. The tail
component therefore spends `.01`, not `.01` once per branch.

The union bound gives simultaneous coverage of both clipped means and the one
any-cell responder-prevalence endpoint at at least 95%; cell correlation is
irrelevant (`...PREREGISTRATION.md:492-503`). Raw-nat t, bootstrap, Fieller,
render, behavioral, and placebo summaries remain mandatory companions, not
substitutes for this guarantee (`...PREREGISTRATION.md:521-569`).

## Full validation results

The full matrix used 200,000 independent repeated stratified WOR samples from
fixed populations of 4,096 units per stratum for all 18 crossings of six family
shapes with clipped Pearson targets `0`, `.5`, and `.9`. All 18 population array
hashes were distinct and recorded. Every realized dependence was within the
frozen `.03` tolerance; the maximum absolute error was `0.0195898727198185`.

The exact two-sided 99% Binomial(200000, .05) upper acceptance count was `10252`.
The largest observed joint noncoverage was only `1167/200000 = 0.005835`, in the
independent two-point scenario. The other nonzero counts were `1026`, `788`, and
`1`; the remaining 14 scenarios observed zero. All counts were below the exact
acceptance limit. Rare-responder and contaminated families had nonzero,
stratum-heterogeneous responder prevalences (recorded prevalence ranges reached
`0.04150390625`) and also passed.

The literal 17-population power grid was complete:

| Frozen headline | Observed joint resolution | Required | Result |
|---|---:|---:|---|
| independent worst-variance `{-0.5,+0.5}` cells, realized mean `0.050048828125`, zero responders | `0.867955` | `>=0.80` | PASS |
| independent mean-zero cells, SD `0.25`, zero responders | `0.999985` | `>=0.95` | PASS |

All seven responder-prevalence grid points, including six nonzero points, were
also retained. At the
requested `p=.10` point the realized finite prevalence was `0.10009765625` and
tail noncoverage was `1253/200000=.006265`, explicitly exercising the nonzero
tail regime. The all-zero, one-responder, clipping, conditional-estimand, and
label golden checks all passed. The simulation is implementation and power
evidence; the finite-sample coverage claim comes from the proof above, not from
treating Monte Carlo as proof.

## Independent implementation agreement

`scripts/recompute_powered_v13_bound_independent.py` is standard-library-only
and does not import the production core. It independently enforces two C-origin
renders, ranks 1--6 in every stratum, clipping, both mean UCBs, the zero/nonzero
tail branches, and the numerical conclusion. Production and independent paths
freeze `STRATA` then rank/case order and use `math.fsum`, while `Fraction` proves
the alpha identity (`src/powered_v13_stats.py:22-40,223-231,711-727`).

Seven golden projections, including non-dyadic and catastrophic-cancellation
cases, matched byte-for-byte; malformed booleans, missing/duplicate rows, wrong
strata, and rank overshoot fail or select as specified. The durable independent
record is `notes/powered-v13-independent-bound-reimplementation-audit.md`.

The final integrated targeted run observed `141 passed` (two tokenizer-library
deprecation warnings only) across production statistics, independent
recomputation, simulation, recipe, and dynamic-token planner tests.

## Compact recipe-foundation audit

The staged foundation at commit `b61cab3` has these bindings:

- `src/powered_v13_recipe.py`:
  `19eb7e666a705eb69a3e5577ac4ab2963338c5d76b936625a0ea11285aa4035b`;
- `tests/test_powered_v13_recipe.py`:
  `830e3d70e8085c3f504ded01370b491cf8d4f6b27176c0d0101d62e4279440e3`;
- compact manifest:
  `788d01ef024febb6de7381b86248405c19719b4453289bbc4b41cdc90901740a`.

An independent exhaustive compact pass evaluated all 32,768 tuples. All 32,768
stable IDs were globally unique and agreed with an independent canonical-JSON
SHA-256 implementation; every pure C/W rule pair had opposite outcomes and a
distinct changed input. All 16 focal words and eight nonfocal words were unique,
nonprefixing, and mutually disjoint. The eight rule shapes and both orientations
are exercised at `tests/test_powered_v13_recipe.py:186-303`.

The compiler checks a ranked authorization receipt before message compilation
(`src/powered_v13_recipe.py:1060-1099`), and the negative test proves an
unreceipted call fails before `_messages_for_variant`
(`tests/test_powered_v13_recipe.py:306-318`). The receipt is intentionally not a
proof of candidate-to-permutation membership; the later release verifier must
bind it to the literal permutation.

The schema enforces the one changed user input plus its downstream repair,
identical system/retained tail, two literal distractor chains, visible independent
nonfocal evidence, choice-free probes, no focal labels in fresh-visible text,
and history hashes (`src/powered_v13_recipe.py:1306-1417`). The structured
carrier-forbidden inventory covers focal/nonfocal answers, changed values, rule
atoms/numbers/times, and contextual names. Punctuation, spelled-number, Unicode,
and tokenizer-subsequence expansions remain explicitly pending at lines
922-929 and must pass before carrier generation.

Under the pinned production tokenizer, all 16 allowed out-of-pool
family-by-explicitness sentinels had exact C/W N-plan geometry. Their noncarrier
skeletons ranged `1055-1113` tokens and complete sources with the frozen
sentinel carrier ranged `1104-1162`, inside the relevant design bands. All
curated focal and nonfocal component pairs passed the 1--4-token,
length-difference, distinct, and nonprefix checks in an out-of-pool probe
context. This supports plausibility only. Literal production-history collisions,
ranked first-ten geometry, content eligibility, and realized six-of-ten yield
cannot be assessed before the frozen seed/permutation event.

## Operational text audit

The corrected preregistration text resolves the earlier operational blockers:

- **Render origin:** both stochastic carriers originate independently under C;
  exact IDs are then forced under C/W/F, and the two renders collapse within
  fixture rather than increasing N (`...PREREGISTRATION.md:208-255`).
- **Probe placement:** focal/nonfocal forks, generation header, teacher-forced
  content-token scoring, common answer position, greedy cap/EOS, and persisted
  token bindings are literal (`...PREREGISTRATION.md:271-283`).
- **VP:** only bf16 V rows move; SHA row ordering, moved counts, directions,
  float64/`math.fsum` geometry, inclusive thresholds, unavailable path, and
  exact `44/48` completeness rule are specified
  (`...PREREGISTRATION.md:323-379`). Full-KV is not mislabeled placebo-complete.
- **Phase-A blindness:** new semantic cases cannot construct, score, persist, or
  reveal treatment-arm probe outcomes; only the two named technical exceptions
  may exercise them (`...PREREGISTRATION.md:420-441`).
- **Staged release:** Stage A and B bind immutable parent trees, restrict child
  diffs to manifest plus one status transition, require post-commit launch
  receipts and detached clean-tree verification, and reject wrong parent,
  status, receipt, or extra diff (`...PREREGISTRATION.md:727-780`).
- **Phase-A economics:** the `$12/$30/$4.50/$8` buckets are separate; unresolved
  strata forecast every remaining rank through ten at the maximum observed
  balanced-batch cost; already spent dollars are not double-counted; both the
  phase bucket and live-balance inequalities must pass after every candidate
  (`...PREREGISTRATION.md:782-835`).
- **Host/resume:** partial cases quarantine and recompute, terminal reuse requires
  identical runtime binding, results from different primary fingerprints never
  combine, and at most one complete new-batch restart may occur before scores
  are inspected with regenerated foundations and a new verified receipt
  (`...PREREGISTRATION.md:667-725`).

These are text/design passes. Their implementations, corruption probes, remote
receipt path, actual e01/long measurements, VP availability, provider ledger,
and host attestations are still unobserved for v13.

## Historical artifact classification

`powered-v13-stat-simulation_local_20260712T161932Z.json` is retained negative
evidence against the old sequential nominal-t design. It observed joint
noncoverage `0.19418` for the rare-responder case and `0.10284` for the skewed
case and cannot authorize anything.

`powered-v13-bounded-stat-simulation_local_20260712T165226Z.json` correctly
checked the replacement formula and headline powers, but remains a preliminary
partial artifact: it lacked the literal 18-way finite-population WOR matrix,
exact binomial gate, full power grid, and clean provenance/recomputation binding.
Its former authorization-pass interpretation remains revoked. The new
`...174743Z.json` artifact supersedes it only for the Section-12 numerical
validation role; neither old file is deleted.

## Remaining authorization blockers

Before `STATIC_FROZEN_PHASE_A_AUTHORIZED`, the repository still needs, at
minimum:

1. committed 128-bit seeds and literal PCG64 permutations after the component
   frame is frozen;
2. ranked first-ten materialization, collision/geometry/round-trip gates, full
   carrier surface exclusions, and blind/paired content reviews;
3. complete local production runner/control/persistence/build-ladder and
   deliberate corruption/kill/warm-order evidence;
4. Stage-A verifier acceptance and every required rejection test;
5. the dedicated release-only Stage-A manifest/status commit and its subsequent
   launch receipt; and
6. a refreshed provider balance/pod inventory immediately before any allocation.

Phase A must then independently satisfy the e01/long/VP/timing gates, exact
six-per-stratum yield, durable 96-render/bundle set, live budget inequalities,
and the separate Stage-B release. No message, audit `PASS`, or
`release_eligible=true` simulation field waives those gates
(`...PREREGISTRATION.md:891-934`).
