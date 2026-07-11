# V12 pre-freeze scientific review

**Reviewer runtime model:** `claude-fable-5` (Claude Fable 5), acting as a
fresh skeptical scientific-methods reviewer. Scope per the owner's
threat-model correction (notes/2026071169): ordinary research failure modes
only; adversarial Git/symlink/fabrication defenses out of scope.

**Files reviewed:** the v12 preregistration; `src/coherent_canary_case.py`;
the technical, Phase-A, and treatment runners; both lean validators; the
independent harvester; and the aggregate stopping-rule script. Focused tests
were not re-executed; review is of the frozen contract and code against it.

## Falsification attempts and outcomes

- **Estimand:** Per-case contrasts in the harvester (`D_focal`, `D_nonfocal`,
  `SEL`, `Hplus`, `U`, `Uplus`; full-KV = CC−WW, value-only = FC−FW, FF as
  utility baseline) match preregistration §12/§10 exactly, recomputed from
  little-endian float32 bits with runner decimals/labels ignored. No mismatch
  found.
- **Gate logic:** Technical runner's `required_pass` correctly excludes the
  natural calibration (reported, not gating, per §14.2) and includes identity,
  deterministic repeats, nine-cell self-replacement, and the bidirectional
  path control with the frozen ULP ladder and 1e-4 movement rule (§14.1). The
  validator recomputes path movement from raw margin bits and enforces
  first-passing-ULP stop and complete attempt lists on failure. Consistent.
- **Phase-A/treatment leakage:** `run_phase_a_case` computes only oracle,
  fresh, and forced-carrier evidence; the Phase-A validator rejects
  treatment-shaped fields recursively and enumerates the exact allowed field
  set; the treatment runner validates the persisted Phase-A release report,
  raw-artifact bytes, binding continuity, and identical runtime fingerprint
  before importing the treatment entry point. The exact subject is barred
  from the `ESTIMAND_INADEQUATE` exception; only the local apparatus gets it,
  matching §2. No leakage path found in ordinary workflow.
- **Schema compatibility:** Schema IDs, binding sets, arm counts (31 primary +
  3 placebo = 34), selector sets (N×3 regions×9 cells + P/R2×4 cells), and
  FF-equals-shared-fresh-baseline checks are mutually consistent across
  runner → validator → harvester → aggregator.
- **Stopping rules:** `classify_four` (PASS4 = mean D>0, ≥3/4 D+SEL, mean
  Hplus>0, yardstick; STOP4 = mean D≤0 or ≤1/4; else AMBIGUOUS4) and
  `classify_six` (threshold 4/6, else STOP6) match §15. The aggregate
  yardstick uses `mean_i(D_N) >= 3*mean_i(|D_N−D_P|)` per §8; the per-case
  harvest record is explicitly labeled a component, not a decision. Six-case
  runs require a bound four-case aggregate with at least one AMBIGUOUS4
  family, preserve prior terminal families, and re-verify the bound
  aggregate's classifications from the same inputs. Local and one-case runs
  are forced to `INCOMPLETE` with no semantic decision. Correct.
- **Model/config binding:** Both validators pin exact model IDs, revisions,
  geometry (0.6B: 28/16/8/128/1e6; 30B: 48/32/4/128/1e7), bf16, eager on
  every layer, and recompute the fingerprint commitment; Phase-A and
  treatment require fingerprint identity with the technical release. Matches
  §2/§14.

## BLOCKING findings

None established from the reviewed material. I attempted to falsify the
estimand, gate logic, blinding separation, schemas, formulas, stopping rules,
and model/config bindings and found no defect that would corrupt a decision.

## IMPORTANT, nonblocking

1. **Mitigation-branch rule not machine-applied.** §15's value-only
   mitigation condition (mean U>0 and Uplus>0 among positive-damage cases) is
   not computed by the aggregator; it only reports per-case
   `phase_a_utility_damage`. Apply that rule by hand against the persisted
   fields, or add it before any mitigation claim.
2. **Subtype reporting not enforced.** §3.1/§15 require results reported by
   explicit-vs-unstated focal subtype; the aggregator carries no subtype
   labels. Pigeonhole guarantees 4-of-6 spans both subtypes, so no decision
   error is possible, but the mandated subtype report must be produced at
   write-up time.
3. **Phase-A validator reads `fixed_text_token_evidence_v2.json` by current
   repo path** rather than through a hash binding to the technical artifact;
   an accidental in-place edit between runs would shift the reference. Low
   risk given the append-only convention.
4. **Harvester mean-consistency tolerance** (`max(1e-5, |mean|·2e-6)` between
   float32 mean bits and float64 token fmean) is a heuristic; a pathological
   long-target reduction-order gap could exceed it and abort a valid case.
   Acceptable, but note it as a known abort cause, not a science failure.

## Acceptable limitations (by design, per the corrected threat model)

- Validators/harvester do not rerun model forwards; raw bits are trusted as
  bound observations with internal-consistency checks only.
- Lossless cache tensors are not retained (§17 rationale accepted).
- The 3x schedule yardstick is an exploratory heuristic, stated as such.
- Local 0.6B semantics can never authorize or discourage the exact-model
  canary; the code enforces this everywhere it matters.

## Recommendations

- **Local 0.6B end-to-end integration (technical → Phase A → treatment →
  harvest → local descriptive aggregate): READY.**
- **Bounded exact-model gates (pod provenance, path control, natural
  calibration through the $2 step-3 authorization): READY**, conditional on
  the preregistration status being flipped to FROZEN additively before any
  forward, per its own §Status clause.

