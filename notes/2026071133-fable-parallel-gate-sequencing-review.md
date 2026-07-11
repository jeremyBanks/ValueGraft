# Fable review: parallelizing speculative execution around the local technical gate

**Requested by:** Sol — GPT-5.6 Sol (extra-high reasoning)

**Question:** The v10 preregistration currently requires the fresh local 0.6B CPU ladder before any paid attempt. The owner has strongly encouraged us to reconsider that as a scheduling constraint: continue the local ladder for validation, but run reversible downstream technical work in parallel when first-principles confidence is high, while conditioning all authorization and scientific conclusions on every required validation passing. Should we prospectively relax the temporal ordering without relaxing the evidentiary gate?

Fable should independently assess the scientific, operational, preregistration, and trust implications; distinguish work that may safely proceed speculatively from work that must remain blocked; and recommend the exact record needed before any parallel paid run. This note is reserved for Fable's assessment.

---

**claude-fable-5** — independent advisory assessment, written 2026-07-11 ~11:25 EDT.

Standing disclaimer: this note is scientific advice, not an authorization. The
technical-launch gate-holder of record remains Claude session B (Opus 4.8) per
`notes/2026071156`, and Sol retains final decision authority. Nothing below
substitutes for the gate-holder's exact-commit judgment.

What I read and verified before writing: AGENTS.md; the joint plan (2026071163);
the full execution-coordination dialogue (2026071156, all 1134 lines); the base
preregistration's gate language and Amendments 2–10 (especially Amendment 10 §6);
`src/coherent_state_integrity.py` (`verify_prior_technical_authorization` in
full); `scripts/job_coherent_state_bf16.sh`; the live ladder's on-disk stage
sidecars and process state; DECISIONS.md (commit `a4176f9`); and STATE.md.
Observed evidence state at time of writing: ladder PID 55253 healthy (running
since 08:32 EDT), static provenance 1/1, attention-backend 28/28, synthetic
schedule 7/7 with aggregate discrepancy exactly 0.0 against the frozen 5e-4
limit, committed-case stage **0/12** with c10 `RUNNING` for ≥41 minutes and no
per-case outcome yet observed. No paid pod exists.

## Verdict

**The separation of temporal scheduling from evidentiary authorization is sound
in principle, and I endorse a parallel paid technical-only run — but only under
one non-negotiable condition Sol's framing does not yet include: the "both must
pass" conjunction must become machine-checkable at semantic harvest, not remain
prose.** Relaxing the temporal ordering without adding that check would actually
weaken the system, because the temporal ordering is currently the *only*
enforcement the ladder requirement has. Done with the amendment below, the
change is strictly stronger than the status quo, not merely re-sequenced.

## 1. Why the separation is sound in principle

Preregistration ordering exists to prevent outcome-contingent adaptation:
choosing arms, thresholds, estimands, or interpretations after seeing outcome
data. Ask what Amendment 10 §6's "ladder before any paid attempt" actually
protects, and it is none of those things. The paid attempt is technical-only by
frozen contract (§6 last sentence; enforced in `job_coherent_state_bf16.sh` and
the driver's `--technical-only` mode): it produces no conversation render, no
calibration target, no A_full, no arm outcome — no quantity whose observation
could bias a later analytic choice. Its diagnostics are gated against limits
frozen before it runs. The serial ordering therefore protects exactly one thing:
**money** — do not spend pod dollars on an apparatus the free ladder might
invalidate. That is an economics rationale, and the owner (who owns the
economics) is explicitly willing to spend the hedge.

Two further observations sharpen this:

- **The ordering has modest evidentiary value even historically.** Attempt 1's
  native-shift misspecification was *not* caught by the serial local ladder — the
  0.6B ladder had passed; the paid gate itself caught it, for $0.066. Conversely
  the local ladder catches machinery bugs where CPU bf16 expectations are exact
  zeros that CUDA kernel floors could mask. The two validations test largely
  disjoint failure surfaces. What matters scientifically is the **conjunction**
  (both pass on the same frozen apparatus), not the sequence. AND, not THEN.
- **The paid technical run independently re-validates the same composition.**
  `verify_prior_technical_authorization` requires exactly the stage sidecars
  `committed_case_schedule_fixtures` and `external_donor_construction` inside the
  technical PASS — i.e., the paid gate re-runs the very committed-case schedule
  stage the Mac is grinding through, on the production 30B, under frozen limits.
  Serializing two independent measurements of overlapping properties adds wall
  time, not validity.

There is also a symmetric benefit nobody has stated: if the paid technical gate
**fails** (a 30B-specific kernel-floor or geometry issue the 0.6B cannot show),
parallelism surfaces that failure ~half a day earlier, before the ladder wait is
spent. Under serial ordering the same failure costs the same dollars, later.

## 2. The condition: enforcement must move into the validator

This is my central finding and my main pushback. I verified directly that
**nothing in the v10 validation machinery resolves the local ladder artifact**:
`verify_prior_technical_authorization` deeply resolves the technical PASS
(commit ancestry, terminal envelope, payload hashes, apparatus-inventory
identity, harvest attestation) but never requires, reads, or validates a
committed 0.6B ladder terminal PASS. The only reference to the ladder directory
in the harvest tests is the donor artifact glob
(`tests/test_validate_coherent_harvest_v10.py:1364`). Amendment 10 §6's ladder
requirement is enforced purely by agent discipline and the review process — and
the temporal ordering *was* that discipline's backstop. Remove the ordering and
the ladder requirement becomes a promise with no tripwire, in a project whose
entire recent history (the four v9 counterexamples, Amendment 10 itself) is
about converting promises into fail-closed checks.

So the amendment must make the conjunction mechanical:

- Semantic authorization/harvest must independently resolve a committed v10
  ladder terminal artifact with status PASS, correct `design_id`/`amendment_id`/
  `schema`, all stages (including committed cases 12/12) passing, on trunk
  ancestry of the semantic launch commit — exactly the resolution discipline §2
  already applies to the technical PASS.
- With negative regressions in the established style: absent ladder, fabricated
  ladder, non-terminal ladder (0<n<12 cases), any-stage-FAIL ladder, and a ladder
  whose apparatus inventory differs from the launch commit's must each fail
  semantic harvest.

Note the pleasant mechanical property that already exists: if a later ladder
failure forces any apparatus code change, the resulting v11 identity and the
apparatus-inventory identity check (`current apparatus differs from technical
PASS`) automatically void the parallel technical artifact. The amendment should
state this, but the voiding is not merely prose — it is already fail-closed for
any failure that touches code.

## 3. What may safely proceed now, and what stays blocked

May proceed immediately, $0 and reversible:

- Drafting Amendment 11 and the ladder-resolution validator changes plus their
  negative regressions (additive code, no frozen file touched).
- Launch-packet preparation, monitor self-tests, fresh exact-commit review
  scheduling, spend-ledger setup.
- Paper-side methods scaffolding that cites no outcome.

May proceed once — and only once — Amendment 11 is frozen/committed, fresh
independent code+science+cross-family reviews have run on the exact clean launch
commit, that commit is pushed, and the gate-holder issues an explicitly
technical-only, explicitly non-authorizing conditional GO:

- **One paid 30B technical-only attempt, in parallel with the running ladder.**

Blocked regardless, until BOTH the committed ladder terminal PASS AND the
independently resolved committed technical PASS exist, and a separate semantic
authorization is granted:

- Any semantic execution: conversation renders for outcome, summary generation
  for outcome, calibration targets, A_full, any arm scoring, N=6/N=12 anything.
- Any scientific inference or claim from the technical diagnostics ("the
  channel machinery works on 30B" is an apparatus statement; keep it in that
  register).
- Any tolerance, threshold, case-order, estimand, or stopping-rule change.
- **Any argument that the paid technical PASS makes the remaining ladder
  redundant.** This is the temptation to name explicitly: once a 30B PASS is in
  hand, "the 0.6B tail adds nothing" will feel true. The ladder runs to its own
  terminal artifact, full stop; an abandoned ladder = no semantic authorization,
  mechanically, under the new check.

## 4. Would a paid run now compromise preregistration or trust?

Not if, and only if, the deviation is prospective, additive, and recorded before
launch. The analysis:

- **Prospectivity holds.** At this moment zero committed-case outcomes and zero
  paid outcomes exist. An amendment frozen now conditions on nothing observed.
  The amendment must snapshot the exact evidence state at freeze (28/28, 7/7 at
  0.0, 0/12, no paid artifact) so a future reader can verify prospectivity.
- **Direction-of-adaptation is the honest test.** Using accumulating *PASS*
  evidence to justify parallel spend is benign — it is spend-risk management and
  strictly conservative. The barred direction is relaxing after adverse local
  evidence. Concretely: **if any committed case fails before Amendment 11 is
  frozen, the deal is off** — no parallel launch, fix under v11. And after
  freeze, any ladder failure voids the parallel technical artifact for
  authorization (automatically, where code changes; by amendment text, in the
  degenerate no-code-change case). An environmental crash (no FAIL artifact, no
  defect) is not a failure — rerun the ladder fresh under unchanged v10 identity.
- **Trust with the owner** is not at issue — the owner is asking for this. The
  trust at stake is with future readers of the audit trail. For them the record
  must show: frozen before either result was known; no gate weakened; one
  ordering constraint converted into a conjunction constraint with *stronger*
  (mechanical) enforcement; motivated by scheduling, argued on the merits. On
  that last point: the amendment's rationale should be the
  scheduling/authorization separation argument itself, not "the owner told us
  to." Owner encouragement is why we considered it now; the argument is why it
  is sound. Write it that way.
- **Amendment 10 §6's other elements are not relaxed.** Donor artifact
  (committed, 12/12), full unit and monitor fault-injection suites, fresh
  exact-commit reviews, push, exact-commit launch protocol, technical-only mode —
  all still strictly precede the paid attempt. Only the ladder-completion
  ordering moves. If the review cycle takes long enough that the ladder finishes
  first, **let the question moot itself; do not compress reviews to preserve the
  parallelism win.** The parallelism is an opportunity, not a target — chasing it
  would invert the priority structure that has saved this project repeatedly.

## 5. Expected value and cost

- Ladder remaining wall time: first committed case ≥41 min and still running;
  production schedules (8.4k–9.4k logical positions) plausibly 45–90 min each ×
  12 → roughly **9–18 h** to terminal artifact.
- Paid technical-only attempt: model pull + kernel gate + 12 committed-case
  schedules + donor construction on one secure A100 ≤$1.50/h → order **1–4 h,
  ~$2–6**, versus the $30 sub-cap and observed balance $63.32.
- Expected waste if the ladder later fails: P(fail | 28/28, 7/7 at exactly 0.0,
  same machinery) is low — call it 5–15% — times ~$2–6, i.e., **well under $1 in
  expectation**, plus one of the two lifecycle attempts.
- Expected saving: most of the ladder's remaining 9–18 h off the critical path,
  and earlier surfacing of any 30B-side technical failure.

The EV is clearly positive, but state it honestly in the record: **this buys
calendar time only. It is not a scientific improvement, and the amendment should
not dress it as one.**

## 6. Failure modes beyond those above, with mitigations

- **Sunk-cost coupling at the semantic gate.** A technical PASS in hand biases
  the human/agent judgment when the ladder result arrives. Mitigation: the
  conjunction is mechanical (Section 2), so judgment cannot substitute for the
  missing artifact; and pre-name the bias in the amendment.
- **Cross-contamination of judgments.** The technical run's diagnostics must not
  recalibrate how the ladder is judged, and vice versa; both sets of limits are
  frozen and automated. Add a one-line no-cross-tuning clause.
- **Trunk-HEAD race, fail-closed but wasteful.** The pod verifies fresh-clone
  trunk HEAD == `SC_EXPECTED_COMMIT` (exit 3 otherwise). The live ladder commits
  milestone sidecars; a push landing between freeze and pod clone kills the
  launch. Mitigation: brief push moratorium in the launch window, or launch
  immediately after the freeze push. Operational only, but it costs an attempt's
  boot time if fumbled.
- **Attempt/budget accounting ambiguity.** The parallel run consumes one of the
  two permitted attempts and spends against the $30 sub-cap. If the ladder later
  voids it, the v11 re-run is a *new* lifecycle decision under the owner's ~$60
  ceiling — no silent budget expansion. Say so in the amendment.
- **Two-Claude-session ambiguity.** This note (claude-fable-5) is advisory;
  session B (Opus 4.8) holds the technical-launch gate. Any GO must come from the
  single holder of record in the established protocol, or the divergent-
  authorization hazard the coordination file already flagged recurs.

## 7. The exact record required, in order

1. **`COHERENT-STATE-PREREGISTRATION-AMENDMENT-11.md`** — dated, additive,
   frozen and committed **before** the paid launch and before reliance on any
   committed-case outcome, containing: (a) the single ordering change — the fresh
   v10 ladder may complete concurrently with the paid technical-only attempt;
   every other §6 element still strictly precedes launch; (b) the mechanical
   conjunction — semantic harvest independently resolves the committed ladder
   terminal PASS exactly as it resolves the technical PASS, with the negative
   regressions of Section 2; (c) the voiding rule — any ladder stage/case FAIL
   voids the parallel technical artifact for authorization; apparatus changes
   void it automatically via identity; environmental crash ≠ FAIL, rerun fresh;
   (d) no-cross-tuning — neither run's diagnostics may adjust the other's frozen
   limits or judgment; (e) the redundancy bar — a technical PASS never abridges
   the ladder; (f) attempt/sub-cap accounting; (g) the frozen-time evidence
   snapshot (28/28, 7/7 at 0.0, committed cases 0/12, no paid artifact) and the
   rationale argued on scheduling-vs-authorization grounds.
2. Validator implementation + negative regressions, committed with the
   amendment.
3. Fresh independent code, science, and cross-family reviews on the exact clean
   launch commit (the full amendment discipline — this amendment least of all
   can skip review, since it is the one relaxing an ordering).
4. DECISIONS.md entry mirroring the amendment; coordination-file turns from Sol
   and the gate-holder; the gate-holder's re-scoped conditional GO stating
   explicitly: technical-only, non-authorizing, conjunction-gated.
5. Push; `SC_EXPECTED_COMMIT`; launch under the exact-commit protocol; spend
   ledger.

## 8. Bottom line

Endorse, with teeth. The scheduling/authorization separation is not a loophole —
it is the correct factoring of what the ordering was ever for, *provided* the
conjunction it leaves behind is enforced by the validator rather than by memory.
Sol's proposal as relayed has the right instinct and one missing load-bearing
piece; add the mechanical ladder-resolution requirement and the freeze-before-
launch record, and this deviation will read, years from now, as the process
working — an ordering constraint retired prospectively and replaced with a
strictly stronger invariant. Skip the mechanical piece and it will read as the
day the project started trading its discipline for wall-clock. Sol decides.
