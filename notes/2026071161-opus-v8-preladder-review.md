# Independent cross-family pre-ladder review: Amendment 8 + v8 apparatus

**Runtime model identifier:** `claude-opus-4-8` (Claude Opus 4.8).
**Role:** independent cross-family skeptical reviewer (NOT the execution-dialogue
gate-holder). This is the Opus endorsement that the v7 NO-GO note
(`notes/2026071160-sol-v7-v8-adversarial-review.md`, "Remaining release blockers"
#3) recorded as *pending*.
**Reviewed at:** clean `trunk` HEAD `0c457ad` (working tree clean except this
note). No paid pod; no v8 30B outcome exists.
**Scope:** Amendment 8 and the exact current v8 apparatus/validator/tests as a
pre-ladder / pre-launch scientific and release review, adversarial by mandate.

---

## VERDICT

**NO-GO for any paid launch** — the mandatory, already-declared release
prerequisites are unmet (fresh v8 0.6B ladder is absent; final exact-commit
code/science reviews are pending; the clean launch commit is not yet
re-tested/pushed).

**But on the two things I was asked to adjudicate, and on the closure itself, I
give an explicit GO / ENDORSE:**

- **I endorse Amendment 8's composed actual-destination schedule fixture (§2).**
- **I endorse the strict bit-exact hash + lineage snapshot-provenance waiver
  without raw-tensor archival (§3).**
- **I found NO new code/design blocker beyond the known prerequisites.** Every
  v7 counterexample class the Sol synthesis claims to have closed is, on my
  independent read, actually closed *and regression-locked*. The residual items
  I surface (A–E below) are hardening/framing recommendations, not blockers.

This matches the v7 note's framing: v8 is a **code-and-design closure, not launch
authorization.**

---

## What I independently executed (not merely read)

- **Full unit suite:** `PYTHONPATH=src uv run pytest -q tests/` → **202 passed**,
  2 deprecation warnings, ~48 s. (Reproduces the reported 202.)
- **Monitor fault-injection:** `bash scripts/coherent_monitor_selftest.sh` →
  **`59 cases passed, 0 failed` / `COHERENT MONITOR CLEARED`.** (Reproduces 59/59.)
- **On-disk v8 ladder check:** the *only* v8 artifact under
  `results/coherent_state_ladder/` is the donor validation
  (`coherent_external_donors_gapped_v8_...20260711T113054Z.json`). There is **no
  `coherent_state_ladder_gapped_v8_*` 0.6B artifact** — the fresh v8 ladder is
  confirmed missing, exactly as stated.
- **Donor artifact ↔ committed sources:** the on-disk regression
  `test_independent_donor_reconstruction_matches_committed_sources`
  (`tests/test_validate_coherent_harvest_v8.py:1055`) recomputes all 12
  wrong-prefix constructions from `data/synthetic/*.json` via
  `_reconstruct_donor_replacements` and matches the committed v8 donor artifact;
  it is part of the passing 202.

> ⚠ Operational footgun (not a science blocker): the suite has **no
> `conftest.py` / pytest pythonpath config**. My first run (`uv run pytest`
> without `PYTHONPATH=src`) collapsed with **13 collection errors**
> (`ModuleNotFoundError: No module named 'run_coherent_state_hf'`). The job
> scripts and monitor use `PYTHONPATH=src`, so this is invocation-only — but a
> reviewer running the bare command will get a false "the suite is broken"
> reading. Add a one-line `conftest.py` (`sys.path.insert(0, "src")`) or
> `[tool.pytest.ini_options] pythonpath = ["src"]`.

---

## Endorsement 1 — composed actual-destination schedule fixture (§2)

**ENDORSED.** Producer: `measure_gapped_destination_schedule`, invoked in
`src/run_coherent_state_hf.py:941-954` **after** the summary is durably saved
(`SUMMARY_DURABLE`/`SUMMARY_RESUME`, lines 902-917) and **before** wrong-source
construction (line 994) — the exact ordering §2 mandates. Independent validator:
`_validate_gapped_destination_schedule`
(`scripts/validate_coherent_harvest.py:812-990`).

The validator genuinely *reconstructs* the composed schedule rather than trusting
it. From the saved rendered conversation + summary text it rebuilds:
- the correct full-length prefix and the compacted `fresh` (system + request)
  prefix from the production tokenizer (842-846);
- the summary IDs by **re-rendering the saved summary text** and requiring an
  exact match (`_rendered_assistant_ids`, 847-849) — the summary is not taken on
  faith;
- the gapped logical positions
  `range(system_end) + range(request_logical_start, source_summary_start)`
  (884-885), the summary forced at the **same** logical positions as the correct
  source (`source_summary_start = len(correct)`, 878/886-887), contiguous
  physical positions (897-899), the production system/request partition
  `[system_end, len-system_end]` and the ordinary ≤4096-chunk partition
  (890-891), and `summary_step_widths = [1]*len` (902) — i.e. every summary token
  stepwise in **both** branches;
- both full tokenwise log-prob traces (recomputing the abs-diff array and its max,
  939-946) and, for **all 48 layers**, per-summary-row K and V max-abs
  (948-977), with a terminal `max(...) ≤ 5e-4` gate (980-990).

This composes what v7 never composed in one gate: real render length, real
compacted destination, real logical gap, production call split, contiguous
physical storage, and every saved summary token. Seven tamper counterexamples are
regression-locked (`test_...reconstructs_gapped_destination_schedule`,
`:1183-1214`): position edit, partition edit, dropped per-layer row, aggregate
edit, trace edit, missing evidence, wrong `conversation_id` — all raise.

**Scope caveat to carry into the paper (framing, not a defect):** both branches
are the *same* gapped/compacted destination with the *same* position-correct
`position_ids`; they differ only in prefill chunking. So the fixture is a
**partition/chunking-invariance witness on the fresh gapped destination** — it
proves the production system/request split reproduces ordinary chunking to
5e-4. It does **not**, by itself, prove the logical gap is applied *correctly* (a
consistent gap bug would corrupt both branches equally → ~0 diff → PASS).
Gap-correctness is carried by the *separate* technical stages —
`position_structure` (exact gapped arrays, physical contiguity, wrong-position
failure injection; validator `:1877-1945`), `physical_causal_mask_identity`
(auto vs explicit 4D mask), `future_mutation_identity`, and the 0.6B ladder's
mask/zero-gap/position-injection tests. The destination fixture is one necessary
leg; do not let the paper over-claim it as "proves the gapped schedule is
correct." It is complementary to the snapshot-lineage leg (Endorsement 2): the
fixture witnesses that the *fresh* gapped base is schedule-robust, the lineage
witnesses that the *copied-in* correct/wrong/mixed K/V have exact provenance.

---

## Endorsement 2 — bit-exact hash + lineage waiver, no raw-tensor archival (§3)

**ENDORSED as a provenance mechanism**, conditional on the fail-closed properties
I verified end-to-end.

Producer `_strict_generated_replay_witness`
(`src/run_coherent_state_hf.py:481-495`) sets
`raw_tensor_archive_waived_by_exact_replay = (actual_row_hashes ==
replay_row_hashes)` and `passes = numerical_passes AND hashes_exact`. The waiver
is therefore **fail-closed on bit-exact byte identity**, not on the 1e-4
tolerance. The driver raises if `identity["passes"]` is false
(`:981-985`). Independent validator `_validate_snapshot_provenance`
(`scripts/validate_coherent_harvest.py:1104-1259`):

- binds `summary.actual_row_hashes == actual == replay` and requires
  `replay_hashes == actual_hashes` (1127-1131) — bit-exact, per §3.2;
- recomputes the replay numeric aggregate `max(lp_max, k_max, v_max)` from the
  raw per-token/per-layer values and requires `≤ 1e-4` **and** the producer
  booleans `summary_row_hashes_bit_exact`,
  `raw_tensor_archive_waived_by_exact_replay`, `passes` all True (1157-1182) —
  so the waiver boolean is checked *against* recomputed evidence, closing the v7
  "boolean substitutes for measurement" gap;
- enforces unambiguous materialization (`live_incremental_generation_rows` xor
  `bit_exact_stepwise_resume_reconstruction`, sorted-unique, `used ∈ history`,
  and `identity.used == used`, 1184-1198);
- reconstructs the **complete 5-arm lineage graph** by hash — G_fresh←fresh,
  G_correct←actual, G_wrong←wrong, G_Vcorrect←mixed(fresh K, actual V),
  G_Kcorrect←mixed(actual K, fresh V) via `_mixed_hash_rows` (1215-1246) — plus
  non-summary-prefix preservation, equal branch lengths, and the five lineage
  flags (1247-1259). This exactly mirrors the producer's `_declared_summary_hashes`
  (`src/run_coherent_state_hf.py:454-478`).

Five waiver counterexamples are regression-locked (`:1221-1296`), including the
decisive one whose own comment states the endorsement question: *"Numerical
evidence still claims a perfect replay; only the byte witness differs. The waiver
must fail on the hashes alone"* — and it does (raises on the hash binding).
Actual-hash tamper, inserted-hash (lineage) tamper, list-valued materialization,
and used↔identity mismatch are all rejected.

**Why the waiver is scientifically legitimate despite not archiving bytes:** the
replay is a **pure function of committed inputs** — the correct prefix comes from
the committed conversation checkpoint and the summary IDs from the committed
summary text — and generation vs. replay share the frozen production primitives.
So a future auditor holding the pinned checkpoint can regenerate the tensors and
check them against the committed SHA-256 witnesses; the un-archived bytes are
*reproducible-and-verifiable*, not lost. That is a standard, honest provenance
posture (commit inputs + a hash, not the multi-GiB output). Amendment 8 §3 states
the digests are "non-reconstructive relational witnesses" and forbids the paper
from claiming raw snapshots are archived or reusable — I concur with both, and
the 4 MiB repo limit / no-LFS arithmetic (84.375 MiB per source; 1012.5 MiB for
twelve; ~2.97 GiB across correct/wrong/fresh) makes archival genuinely
infeasible, so the waiver is the right call, not a shortcut.

**One material FEASIBILITY risk the operator must expect (fail-closed, not a
design defect):** the waiver *requires* generation↔replay to be **bit-identical**
at 30B. FINDINGS.md documents Qwen3-30B-A3B **MoE routing nondeterminism** at
bf16. Teacher-forced replay of fixed IDs in one process should be deterministic,
and the amendment's mandatory **eager** attention avoids flash-attention atomics
(which *helps* determinism) — but whether bit-exactness actually holds on the 30B
MoE is **unproven and unprovable by the 0.6B ladder**, because Qwen3-0.6B is a
*dense* model. The ladder proves the *logic* is bit-exact-capable, not that the
MoE will be. This is correctly de-risked by the phased release: the first paid
attempt is **technical-only**, so if bit-exactness fails it fails closed with
**zero semantic spend**. The operator should treat "does generation==replay hash
at 30B eager" as one of the *purposes* of the technical attempt, not an
assumption. No change required; flagging so a technical FAIL here is read as an
expected apparatus-boundary discovery, not a surprise.

---

## Adversarial sweep of the remaining surfaces — all closed

- **Semantic aggregate reconstruction (§4).** `_validate_semantic_score_aggregates`
  (`:1047-1090`) + `_validate_arm_score`/`_validate_target_score` (`:993-1044`)
  recompute every target mean from its token log-probs, every plant margin from
  correct−counterfactual, every conversation margin, every arm outcome, and both
  calibration outcome maps, and enforce identical plant identity/order across
  arms. Eight tamper cases regression-locked (`:1152-1180`). The v7 "harvest
  trusted producer means" gap is closed.
- **Static provenance (§4 / A7 §2).** `_validate_static_fingerprint` (`:253-327`)
  reconstructs the launch commit's apparatus inventory, input inventory, all
  eight amendment-blob hashes, scenario/targets/summary-request hashes, frozen
  order + donor map, the pinned runtime image, and the final↔static binding, all
  from `git` at the attested commit. Two counterexamples — self-consistent-but-
  false inventory and final-binding divergence — are locked (`:1108-1141`).
- **Donor + calibration.** Both are independently *rebuilt from source*
  (`_reconstruct_donor_replacements` `:543-624`, used strictly at `:2237-2247`;
  `_reconstruct_calibration_variant` `:436-515`, used at `:2030-2090`), not
  checked for internal consistency only. Special-token injection and non-frozen
  calibration prefix are locked (`:1023-1052`, `:1076-1092`).
- **Intervention sensitivity.** The orchestrator recomputes `downstream_sensitivity`
  and `recomputed_tail_changed` from the raw ε-attempt numerics and requires them
  true (`:1959-1982`); zero-sensitivity-with-true-booleans is rejected
  (`:1095-1105`). Closes v7 counterexample #3.
- **Terminal lifecycle.** `_validate_terminal_integrity` (`:1316-1398`) +
  `verify_terminal_envelope`/`write_terminal_envelope`
  (`src/coherent_state_integrity.py:184-283`) require exactly one unique gate,
  the two heavy-stage sidecars bound by byte-count+raw+payload hash, no unindexed
  JSON, and receipt→index→payload hash chaining. `_find_forbidden_semantic_fields`
  (`:1435-1452`, applied `:2423`) proves a technical harvest carries no semantic
  outcome fields. Sidecar-byte tamper is locked (`:1313-1318`).
- **Technical→semantic authorization (§5 / A5 §9).** This is the strongest link,
  and stronger than a first read suggests. The *producer* semantic mode calls
  `verify_prior_technical_authorization`
  (`src/coherent_state_integrity.py:430-561`) **before any model work**
  (`src/run_coherent_state_hf.py:1921-1923`) and only then records the
  `authorization_checks:{...:True}` booleans. That function actually:
  `git rev-parse`/`merge-base --is-ancestor` the result vs. trunk-HEAD launch
  (`verify_committed_directory:296-352`), `git ls-tree`+`git show` every file of
  the technical dir and compares **bytes** to disk, re-derives every current
  apparatus file from the technical *launch* commit and compares bytes, and
  **re-runs the independent harvest validator as a subprocess** on the bound
  technical directory, failing closed unless it prints technical PASS
  (`verify_harvest_attestation:355-414`, subprocess at `:392-399`). So the
  authorization is genuinely re-verified, not asserted.
- **Claim scope.** Amendment 8 §1 changes no arm/tolerance/estimand/order/donor/
  stopping-rule/claim-scope. The position-preserving-only boundary (A1 §2, A5 §11)
  and co-primary intersection (both `theta_GF` and `theta_GW` conv-clustered lower
  bounds > 0; N≤12 failure = inconclusive, not equivalence) are intact. The design
  is appropriately humble: a technical PASS is explicitly *not* a semantic effect;
  a technical FAIL is an apparatus boundary, not a null. Given the render-fragility
  history in FINDINGS, this fully-teacher-forced, byte-controlled G_correct vs
  G_fresh / G_wrong contrast is a materially cleaner estimand than the earlier
  native-render/gap-closure work it supersedes.

---

## Known prerequisites (already declared — NOT new blockers)

1. **Fresh v8 0.6B bf16 eager CPU ladder** — confirmed absent on disk. Old
   v4/v6/v7 ladders cannot authorize v8.
2. **Fresh independent code + scientific reviews of the exact clean launch
   commit** (this Opus cross-family review is now on record; code + science on
   the *final* commit remain).
3. **Re-run** full tests, monitor self-test, strict donor/ladder validation, and
   preflight on the final clean commit, then **push** it. Paid run is
   technical-only first; semantics remain separately conditional on a committed,
   independently harvest-valid technical PASS with unchanged apparatus inventory.

## Additional code/design observations (hardening/framing — none are blockers)

- **A. Semantic-run harvest accepts the authorization booleans.**
  `_validate_complete` (`:2287-2298`) checks `authorization_checks[...] is True`
  and the binding *structure/format*, but does not itself re-derive git ancestry
  or re-harvest the bound technical commit. This is acceptable because the
  *producer* §9 gate (above) does the real work and the release process requires
  a *separate* independent `validate_coherent_harvest.py <tech_dir> technical`
  run — but the semantic harvest could close the loop fully by re-running the
  technical harvest on `technical_result_commit`. Recommend either that or an
  explicit runbook step. Defense-in-depth, not a hole.
- **B. Destination-fixture scope** (Endorsement 1 caveat): frame it as
  partition-invariance, not gap-correctness, in the paper.
- **C. Snapshot lineage does not assert `wrong_hashes != actual_hashes`** (nor
  `correct_source_summary != wrong_source_summary`). If a bug made the arms'
  summary K/V bit-identical, the graph would still validate — but the science
  fails closed (theta_GW cannot clear > 0 when arms are identical), so it is not a
  false-positive path. Optional one-line hardening (assert the correct/wrong
  summary hashes differ) would make the intent explicit.
- **D. 30B bit-exact feasibility** (Endorsement 2 risk): expected to be resolved
  by the technical-only attempt; not a code change.
- **E. Missing `conftest.py`/pythonpath** (above): trivial, prevents a false
  broken-suite reading.

## Required actions before any launch (in order)

1. Implement + commit the fresh **v8 0.6B bf16 eager CPU ladder**; it must
   exercise the full A5 §5 ordering incl. gapped self-replacement, explicit-mask
   vs auto, future-mutation invariance, position-injection failure, per-arm tail
   recomputation, and — critically for Endorsement 2 — **generation↔replay
   bit-exact identity** on the dense 0.6B (proving the *logic*, with the MoE
   caveat noted for 30B).
2. Re-run tests (`PYTHONPATH=src`), monitor self-test, strict donor validation,
   preflight on the final clean commit.
3. Obtain fresh independent **code** and **scientific** reviews of that exact
   commit + ladder (this Opus cross-family endorsement stands for Amendment 8's
   destination fixture and the bit-exact hash+lineage waiver).
4. Push the reviewed commit; launch exactly one **technical-only** paid attempt;
   treat generation↔replay bit-exactness at 30B eager as a first-class thing that
   attempt exists to discover. Semantics only after a committed, independently
   harvest-valid technical PASS.

## Bottom line on the two asked questions

- **Destination fixture (§2): ENDORSED**, with the partition-invariance framing
  caveat (B).
- **Bit-exact hash + lineage waiver without raw-tensor archival (§3): ENDORSED**
  as a fail-closed, reproducible-from-committed-inputs provenance mechanism, with
  the 30B MoE bit-exact feasibility flagged as a fail-closed operational risk (D),
  and agreeing the paper must not claim raw snapshots are archived/reusable.

The v8 code-and-design closure is sound and its adversarial regressions are real.
The remaining barriers to spend are the **known prerequisites**, not additional
code defects.
