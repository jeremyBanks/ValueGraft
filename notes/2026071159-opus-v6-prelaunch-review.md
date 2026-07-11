# Opus v6 prelaunch review

**Consulted model:** `claude-opus-4-8`

This note is the durable record of the final cross-family review of the
Amendment 6 technical-only apparatus before any further paid run. The reviewer
was asked to inspect the current committed implementation and evidence without
editing apparatus files, then record a concrete GO/NO-GO verdict here. A GO
must mean only that one exact 30B bf16 eager technical attempt is justified; it
does not authorize semantic execution unless that attempt independently passes,
is harvested, committed, and re-authorized under the frozen two-process rules.

## Claude — claude-opus-4-8

Runtime model, stated exactly as required: **claude-opus-4-8** (Claude Opus 4.8).
I am not Fable and did not run as Fable; this is an independent cross-family
review by the Opus 4.8 runtime.

**One-line topline: NO-GO at the current state** — the launch candidate is not
frozen (apparatus actively edited mid-review and the working tree currently
fails 3 of its own harvest tests), no committed v6 ladder exists, and the
producer/validator schema was defective at the review-anchor commit `5e22611`.
The *design* is sound and the eager rescue is empirically confirmed; this is a
"freeze + ladder + re-review", not a "the science is wrong". Conditional GO once
the four blockers below are closed.

### Evidence base (what I read and ran)

- All requested docs: `AGENTS.md`, `INCIDENTS.md` (through #38), `DECISIONS.md`,
  `FINDINGS.md`, `STATE.md` (stale — dated 07-09, predates this line),
  Amendments 4/5/6, and `notes/2026071158-opus-30b-zero-gap-failure-review.md`
  (the prior joint PROCEED-TECHNICAL-ONLY agreement).
- The committed v6 apparatus: `run_coherent_state_hf.py`,
  `l_coherent_state_hf.py`, `coherent_state_runtime.py`, `kvlib_hf.py`,
  `coherent_state_hf.py`, `coherent_state_tokens.py`,
  `coherent_state_calibration.py`, `coherent_state_integrity.py`,
  `validate_coherent_external_donors.py`, `scripts/validate_coherent_harvest.py`,
  the two job scripts, `watch_coherent_state_pod.sh`,
  `coherent_monitor_selftest.sh`.
- Ran the coherent-state unit suite (14 files, `PYTHONPATH=src`): **106 passed**
  at ~06:20 local. Re-ran the two harvest suites at ~06:25 after a concurrent
  edit landed: **3 failed** (see Blocker 1).
- Ran the committed v6 0.6B/bf16/eager/CPU ladder (`l_coherent_state_hf.py
  --output`) to a temp path **outside the repo** (no repo file created). It
  confirms the eager rescue and lifecycle (see "Verified" below); it had not
  reached the committed-case stage when I concluded.

### Blocker 1 (most severe) — there is no frozen launch candidate; the apparatus was edited *during* this review and currently fails its own tests

A paid attempt clones an exact pinned commit (`SC_EXPECTED_COMMIT`, checked
`HEAD == expected`, exit 3 otherwise). That fixed target does not exist:

- At session start: `HEAD = 5e22611`, tree clean. During the review `HEAD`
  advanced to `24ed900` ("align v6 producer and verifier schemas") **and** the
  working tree accreted 6 uncommitted modified apparatus files
  (`validate_coherent_harvest.py` +198 lines, `l_coherent_state_hf.py` +26,
  `run_coherent_state_hf.py` +4, `job_coherent_state_bf16.sh` +7, two test
  files). `validate_coherent_harvest.py` was last written at 06:24:19, one
  second before I sampled the clock.
- The 106-pass suite at 06:20 became **3-fail at 06:25** on the same tree after
  that edit: `test_technical_harvest_validates_full_sidecar_and_terminal_contract`,
  `test_technical_harvest_rejects_sidecar_binding_tamper`, and
  `test_failure_harvest_preserves_terminal_pass_rejected_by_independent_validator`.
  The current on-disk apparatus does not pass its own harvest contract.

A "final prelaunch review" cannot certify a moving, uncommitted, red target; the
code I read is already partly stale. This alone is a NO-GO until the apparatus is
frozen, committed clean, and green.

### Blocker 2 — the producer/validator schema was mis-aligned at the review-anchor commit `5e22611`; a genuine 30B technical PASS would have failed its own independent harvest

The task's "field-by-field producer/validator" concern is not hypothetical — it
was real at the commit this review note was opened against. `git diff 5e22611
24ed900` shows two independent mismatches that each fail-close a *legitimate*
PASS at the harvest stage (wasting the paid attempt):

1. **Backend attestation scope labels.** `eager_backend_fingerprint` emitted
   `scope="model.config"` / `"model.config.text_config"`, but the independent
   validator's `_validate_backend_attestation` requires `scope in
   {"model_config","text_config"}` (`row.get("scope") != scope` → raise). At
   `5e22611` the 48-layer backend record would be rejected as "model_config
   record malformed". Fixed only in `24ed900`.
2. **Donor row fields.** At `5e22611` the validator checked
   `structural_slots_equal` / `special_ids_excluded`, which
   `validate_with_tokenizer` never emits (it emits `structural_tokens_equal`,
   `system_request_header_retained_tail_unchanged`,
   `changes_confined_to_declared_content_positions`,
   `replacement_spans_non_overlapping`, …). Every donor row would raise
   "external donor row differs". Fixed/expanded in `24ed900` + the uncommitted
   working-tree edit.

I verified the surviving alignment is correct where it has settled — most
importantly the hash functions match byte-for-byte: `sha256_ids`
(`coherent_state_hf.py`) and the validator's `_sha256_ints` both do
`int(v).to_bytes(8,"little",signed=True)`, and `WrongContentReplacement`
carries `target_ids`/`donor_pool_ids`/`replacement_ids` exactly as producer and
validator consume them, and `validate_calibration_constructions` emits
`model_forwards=0`/`semantic_scoring_performed=False`. But the alignment is
*still landing* (Blocker 1's failing tests are in this same surface), so it is
not yet trustworthy as a frozen contract.

### Blocker 3 — no committed v6 ladder artifact exists

Amendment 6 ("A fresh v6 ladder … required before paid execution"), Amendment 5
§5, and the standing house rule ("Never skip the build ladder") require a fresh
0.6B bf16 **eager** ladder that runs the exact v6 gate ordering, PASSES, and is
committed, plus independent code + science review on that exact commit and a
clean launch commit. `git grep coherent-state-gapped-v6 -- results/**` returns
**empty**; the newest committed ladder is `gapped_v3` and the newest committed
donor artifact is `gapped_v4`. The v6 apparatus (all-12 committed-case fixtures
at the 9509 ceiling, heavy-stage sidecar externalization, the new gate
lifecycle) has never been positive-controlled end-to-end on disk. This is
exactly the incident-#24/#31 "declared ready on a static code read, not a run"
class the repo has repeatedly been bitten by.

My own out-of-repo ladder run is reassuring but is not a substitute: it must be
re-run against the FINAL frozen commit and committed.

### Blocker 4 (operational precondition) — launch commit not pushed

`job_coherent_state_bf16.sh` clones `github.com/jeremyBanks/ValueGraft.git`,
`git checkout trunk`, and asserts `HEAD == SC_EXPECTED_COMMIT`. The local branch
is 26+ commits ahead of `origin/trunk`, so the intended launch commit is not on
GitHub yet; the job would abort (exit 3) until it is pushed.

### Verified strengths (do not re-litigate these — they are sound on the tree as read)

- **Fail-closed terminal verdict.** `run_loaded_gapped_gates` recomputes
  `passes = all(status == "PASS")` over the 12 declared stages from persisted
  per-stage status, never a status-only success marker; PENDING/RUNNING/ERROR/
  SKIPPED_DEPENDENCY all force FAIL. The driver additionally requires
  `production_gate.passes is True`, the backend stage's `passes` and fingerprint
  identity, a durable-vs-returned gate equality check, and "no `conv_*.json`".
- **Technical-only is the hard default with layered semantic-leak guards.**
  `_technical_main` never constructs a `Runner`, renders, or scores;
  `assert_technical_gate_has_no_semantic_scores` + the harvest validator's
  `_find_forbidden_semantic_fields` + log-marker checks (`CHECKPOINT_SCORED` /
  `PHASE RENDER` forbidden) + the `conv_*.json` guard are independent belts. The
  gate's only generations are a trivial "reply OK" replay, synthetic fixtures,
  and a `fake_conv` structural probe — no plant/target/calibration margin.
- **Backend attestation** (`eager_backend_fingerprint`) inspects
  `_attn_implementation` and `_attn_implementation_internal` at model, text_config,
  and every attention module's own config, requires exactly `num_hidden_layers`
  ordered records resolving to `eager`, and fails closed otherwise; the harvest
  validator re-checks all 48 layers independently.
- **All-twelve committed-case fixtures** run in frozen order, recompute each
  prefix length with the loaded (production) tokenizer, require exact match to
  the frozen 9509 table, and require `max_observed_logical_position == 9509`
  (Amendment 6). Position/mask/schedule identity is guarded by
  `validate_position_schedule` (contiguous physical, gapped logical) and the
  mask + future-mutation identity stages.
- **Independent harvest validator** (`scripts/validate_coherent_harvest.py`, no
  `src` imports) recomputes per-layer maxima against `5e-4`/`1e-4`, re-derives
  int-array SHA-256s, resolves the two heavy-stage sidecars, and rejects
  semantic fields — it does not trust stored `passes`.
- **Persistence-before-failure**, sealed atomic payloads, terminal index +
  receipt, and a sound cryptographically-bound semantic authorization chain
  (`verify_prior_technical_authorization`: committed-directory bytes, ancestry,
  on-trunk, apparatus-inventory equality, re-run harvest) — the latter is out of
  scope for this GO but was inspected and is not fail-open.
- **Monitor** sources the pure `classify_pod.sh`, harvests-before-terminate
  (refuses to kill a result-bearing pod unverified), double-confirms
  termination, and has cost (>$1.50/hr) and 8h wall-clock caps; the selftest does
  fault-injection + happy-path and prints "COHERENT MONITOR CLEARED".
- **Empirical (my ladder run, 0.6B/bf16/eager/CPU, production tokenizer):**
  synthetic schedule fixtures L5/L64/L900 PASS at **exactly 0.0** K/V/logit/
  margin divergence between one-shot and split schedules — the eager rescue
  (Candidate 1) behaves exactly as the prior review predicted — with correct
  durable PENDING→RUNNING→PASS lifecycle sidecars. (The 4096/4097/8193 and 12
  committed-case fixtures were still computing on CPU at conclusion.)

### Non-blocking risks to disclose (not reasons to withhold GO once 1–4 close)

- **30B eager memory headroom at 8193/9509 is unverified.** Eager materializes
  ~q×kv scores per layer; the 0.6B ladder cannot catch a 30B A100-80GB OOM at
  the 8193 synthetic and ~9509 committed-case forwards. An OOM would be a
  preserved FAIL (fail-closed), not a false PASS, but it burns the one paid
  attempt. Consider verifying peak memory in the ladder/probe before spend.
- **No single technical fixture composes (large logical gap) × (≤900-token
  summary) × (real system/request islands) at production P_i.** The full
  production gapped layout at real length is exercised only in the *semantic*
  run (guarded there by the driver's per-conversation structure gates). This is
  an acknowledged, disclosed limitation and is acceptable for a technical-only
  authorization, but it means the technical PASS certifies schedule/identity
  invariance, not the end-to-end production layout at scale.

### Conditions to convert to GO (technical-only)

1. Finish the in-flight edits; commit a **clean** working tree; the full
   coherent-state unit suite (incl. both harvest suites) is green on that exact
   commit.
2. Run the **v6 0.6B bf16 eager ladder** on that exact commit to completion; it
   PASSES (all 12 stages, all-12 committed cases reproducing the 9509 ceiling);
   commit the artifact.
3. A fresh independent **code + science review** of that exact frozen commit
   (this review predates the freeze and cannot stand in for it).
4. **Push** the launch commit to `origin/trunk` and pin it as
   `SC_EXPECTED_COMMIT`.

A GO here authorizes **only** the single exact
Qwen3-30B-A3B-Instruct-2507 / bf16 / eager / technical-only A100 attempt under
the frozen `5e-4`/`1e-4` tolerances. It authorizes **no** semantic execution:
semantics remain gated on that technical attempt independently passing, being
harvested, committed, and re-authorized under the Amendment-5/6 two-process
contract. No tolerance may be loosened in response to any observed 30B number.


