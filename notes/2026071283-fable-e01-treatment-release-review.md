# Independent review: does the e01 pre-treatment result license the frozen e01 treatment run?

**Reviewer runtime model (as reported by my execution environment):** `claude-fable-5`
(Claude Fable 5). This is the environment's own identification, not an inferred alias.

**Date:** 2026-07-12 (review conducted 2026-07-11 local session).
**Review boundary:** e01 treatment release only — not v10/v11, not the paper, not
settled prior questions.

**Evidence actually read in full during this review:**

- `COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md` (the frozen contract).
- Phase-A validation `results/coherent_canary_validation/coherent_canary_v12_phase_a_e01_exact-subject_20260712T030622Z.json`.
- Pod receipt `results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_20260712T030622Z_receipt.json`.
- Technical validation `results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T024859Z.json`.
- `scripts/run_coherent_canary_v12_treatment.py` (the machine-enforced release path).
- `scripts/harvest_coherent_canary_v12.py` (the independent harvester).

My shell/file verification pass was cut off before it ran, so hash recomputations and
raw-artifact spot-checks I intended are listed honestly in §8 as unverified residuals.
Everything below is from the documents above only.

---

## Topline verdict

**Conditional YES.** The e01 Phase-A result (`PRETREATMENT_PASS`, exact-subject,
`semantic_release_eligible: true`, empty inadequacy/invalidity, blinding held) satisfies
every literal §13 predicate I could check, and the machine-enforced release path in
`run_coherent_canary_v12_treatment.py` will re-verify all bindings at launch. One
blocking-class item must be dispositioned in writing before launch: the §14.1
path-control **ULP stop-rule ambiguity** (§2 below), under whose strictest literal
reading the technical gate *failed* and semantic scoring is not licensed at all. The
natural calibration is ADVERSE but explicitly non-gating under the frozen text. Release,
if granted, licenses exactly one e01 treatment run + harvest — nothing more.

---

## 1. Literal condition-by-condition audit

### §13 pre-treatment adequacy (from the Phase-A validation artifact)

| Frozen condition | Observed | Verdict |
|---|---|---|
| Both full-history oracles favor their own focal target | `A_C` focal margin **+16.50018310546875**; `A_W` focal margin **−19.694555282592773** | PASS |
| Non-focal oracle correct under C and W | `A_C` non-focal **+11.792686462402344**; `A_W` non-focal **+11.667630195617676** | PASS |
| Forced margin sign **and** generated content begins with the complete exact target IDs, for all four predicates | All eight competency booleans true (`A_C/A_W × focal/nonfocal × generation/margin`). Corroborated by near-zero correct-side means (`A_C` focal −5.07e−06, `A_W` counter −3.46e−05, non-focal 0.0): the oracles are near-deterministic on their targets. | PASS (validator-asserted; raw IDs not re-opened by me, §8) |
| Fresh compaction shows positive focal damage vs `A_C` | Margin damage **+22.298429489135742** (= 16.500183 − (−5.798246), arithmetic verified); correct-target damage **+22.290991485144332** | PASS (diagnostic; never gates eligibility, and the artifact says so) |
| Utility-eligible damage: `Y_focal(A_C)>Y_focal(FF)` and `mean_lp_C(A_C)>mean_lp_C(FF)` | 16.500 > −5.798; −5.07e−06 > −22.291. Both components recorded. | PASS — e01 may support U/Uplus interpretation |
| Identity fixture ends normally, no cap/special token | Technical report: 7 content IDs, `normal_stop: true`, repeat_count 2 | PASS |
| Generated conversation summaries end normally | Not applicable — e01 is engineered; carrier is externally fixed text (finite forced support is the correct predicate, and it holds) | N/A, correctly handled |
| Target-neutral carrier blind-approved | Bound revision4 blind + paired reviews (hashes in bindings); runner re-enforces `PASS` verdicts incl. `shared_carrier_anchor_review` at launch | PASS (hash-bound) |
| Forced carrier NLLs finite for every token under C and W | C_N mean 4.922062644071827 / 42 tokens; W_N 4.920622549828598 / 42 | PASS. Near-identical C/W NLL is expected for a target-neutral carrier |
| Phase A computed only allowed quantities; treatment blinding held | `treatment_scores_present: false` in receipt and (per runner contract) in the raw artifact | PASS |
| Independent validator writes and commits one of the three statuses | `PRETREATMENT_PASS`, `inadequacy_reasons: []`, `invalidity_reasons: []`; separate validator exit recorded in receipt | PASS (code-path independence; agent-level independence not verified by me) |

### §14 exact-model gates (from the technical validation artifact)

Status `PASS`, subject `exact-subject`, `failures: []`, `semantic_release_eligible: true`.
Deterministic repeat, fresh self-replacement across all nine region/family combinations,
generated-vs-forced equivalence and identity, source bindings (prereg
`fc02e86d…`, fixtures), and runtime fingerprint all `passed: true`. Path control:
`status PASS, chosen_ulp_count 4` — but see §2. Runtime fingerprint
`b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484` is **identical** in
the technical and Phase-A reports, as the runner requires. Receipt shows
`expected_commit == observed_commit == 139d621e206518b88d7553b5166a0123ee1cdd55`,
runner exit 0, on `NVIDIA A100 80GB PCIe, GPU-0396c7e5-…, driver 580.159.04`.
The natural calibration was completed 03:01:14Z, before e01 Phase A (03:06:22Z) —
the "reported before e01" ordering condition is satisfied.

### Machine-enforced release (from the runner source)

`validate_release_inputs` re-verifies, at launch time, byte-identity (sha256) of all six
bound inputs against the Phase-A record, requires `PRETREATMENT_PASS` (an
`ESTIMAND_INADEQUATE` escape exists **only** for the local apparatus, never
exact-subject), requires Phase-A/technical fingerprint equality, then loads the pinned
subject and requires the **prepared runtime fingerprint to be dict-identical** to the
released one before the treatment module is even importable. Post-run,
`assert_treatment_payload` enforces exactly the frozen 31 primary + 3 placebo arms, and
the harvester independently re-verifies bindings, recomputes float32 bits, and ignores
runner labels. This is a genuinely fail-closed release mechanism; I found no bypass.

---

## 2. Blocking-class finding: §14.1 path-control ULP stop rule

Recorded attempts (movement sign convention inferred — see caveat):

- ULP 1: plus 0.0, minus 0.5 → fail (plus immeasurable).
- ULP 2: plus 0.25, minus **−0.25** → fail.
- ULP 4: plus 0.5, minus 0.25 → **pass**; `chosen_ulp_count: 4`.

Only one sign convention makes all three recorded `passes` flags coherent:
positive = moved in the *required* direction. Under it, at ULP 2 the negative edit moved
the margin measurably the **wrong way** (bf16 quantization noise at the readout floor —
all movements are 0.25 quanta), and at ULP 4 both directions moved correctly, each
≥ 1e−4.

The frozen sentence says: *"Stop at the first count for which both persisted directions
differ from fresh and are measurable."* At ULP 2 both directions differed from fresh
measurably. A strict literal reading therefore selects ULP 2 as the evaluation count —
where the gate **fails** — and §14.1 says failure aborts semantic scoring. Because the
stack is deterministic, a re-run reproduces the same attempts; under the strict reading
the gate is failed for this stack, full stop, and only an additive prereg revision (an
owner-level decision) could reopen it. The implementation instead used
first-*passing*-count semantics (per-attempt `passes` fields), i.e. the lenient reading:
escalate past quantization noise within the frozen [1…64] ladder, then apply the ≥1e−4
bidirectional test.

My assessment as reviewer: the **lenient reading is the better construction** — the
pass condition is a separate sentence; a frozen escalation ladder is pointless if noise-
corrupted low counts are terminal; and the gate's licensed claim ("intervention/readout
sensitivity only") is demonstrably true at ULP 4 with full transparency of the lower
attempts. But this is an interpretation choice, not a fact, and it must not be made
silently. Because e01 treatment is still blinded, a disposition committed **now** cannot
be steered by any semantic outcome — it is purely a spend-risk decision. **Required
before launch:** the decision owner (Sol) commits a dated additive disposition note
adopting one reading. Lenient → gate PASS stands, proceed. Strict → NO-GO; treatment is
not licensed.

Caveat: the sign convention above is established by internal consistency of the three
attempts (the raw-delta convention would make attempt 2 pass and attempt 4 fail,
contradicting the recorded flags); the definitive raw margins live in the bound technical
raw artifact, which I did not re-open (§8).

---

## 3. Adverse but non-gating: natural downstream-note calibration

Observed: `A_g +21.500`, `A_a −16.000`, `F +7.500`, `T_g +7.000`, `T_a +0.500`;
`rho_green = −0.0357`, `rho_amber = 0.2979` — both below the required 0.5; status
`ADVERSE` (denominators positive, oracle margins and generations all correct, so the
computation itself is valid; the validator's `passed: true` checks validity of the
report, not the thresholds).

Under the frozen text this is **reported, not a plumbing gate**: "Its failure is
scientifically adverse but does not relabel a passing technical path control as broken;
the frozen engineered decision rules remain terminal." So it does not block release.
Scientifically, it says a full-KV R2 transplant recovered ~0% (green) and ~30% (amber)
of the oracle−fresh margin gap on a minimal two-message natural fixture — a genuinely
adverse prior for e01. Two things keep this proportionate: (a) the calibration fixture
carries almost no history and its carrier never states the decision, whereas e01 has
1–2k tokens with an explicitly resolved focal decision, ±16–20-nat oracle margins, and
22.3 nats of fresh damage — a far stronger stimulus; (b) the contract deliberately made
the spend decision gate-based, not prior-based, precisely so an adverse prior would be
reported rather than silently steering. **Obligation:** the ADVERSE calibration must be
reported alongside any e01 result, whatever its sign.

---

## 4. Release ≠ claim-family success

What a granted release licenses, exactly: **one** physically separate e01 treatment run
(31 primary cells: 27 N-schedule R1/R2/R3 × 9, plus 4 P-schedule R2 cells; 3 placebo
controls) via the frozen runner, followed by independent harvest, commit, pod stop.

What it does **not** license or establish:

- No claim-family outcome. §15 terminal logic needs all four primary cases: mean
  `D_focal > 0`, ≥3/4 cases with `D>0 ∧ SEL>0`, mean `Hplus > 0`, and the aggregate 3×
  N-vs-P yardstick. The harvester itself labels per-case yardstick rows "a component,
  not an independent decision."
- No automatic e02–e04. §18 step 5 (observed full-canary cost forecast and a committed
  decision) sits between e01 and e02, and each case needs its own fit check.
- No reserve e05/e06 (only on a frozen `AMBIGUOUS` at four), no conversation stratum,
  no confirmatory anything — the canary is permanently excluded from confirmation.
- A positive e01 is one fixed engineered fixture under R2's boundary definition
  (carrier + close + anchor prompt/header), not "summary-content-only" success and not a
  live-agent claim.

## 5. New physical host: another technical run, or exact runtime equality?

The pod was stopped after Phase A (recorded in the repo history), so the treatment run
happens on a restarted or fresh pod. What the frozen machinery actually enforces:

- **Necessary on any host (machine-enforced):** the prepared runtime fingerprint at
  treatment launch must be *identical* to `b6158b98…` (model `Qwen/Qwen3-30B-A3B-
  Instruct-2507` @ `0d7cf239…`, bf16, eager on every layer, same deps/eos set), plus
  byte-identical bound files. Fields visible through the harvester's checks are
  host-invariant (model/revision/dtype/backend/subject/eos); **I could not read the
  loader to confirm whether driver/GPU identity enters the fingerprint** (§8). If it
  does, a new physical host cannot satisfy equality and the release chain forces a full
  re-gate + re-Phase-A on that host; if it does not, fingerprint equality releases on a
  new admitted host.
- **§14's letter** ("Before semantic treatment scoring on the pod, persist and
  independently validate: …") is satisfied by existing persisted, validated gates plus
  fingerprint equality — §19 froze *machine-enforced technical → eligibility →
  treatment release*, and host identity is not part of that machine check. So strictly:
  a new host does **not** literally require a new technical run; it requires pod
  admission (A100-80GB, driver ≥ 580.65.06, CUDA 13) plus exact runtime equality.
- **Scientifically**, the bit-exactness gates (deterministic repeat, generated/forced
  identity) certify kernel arithmetic of (model bytes, pinned wheels, GPU architecture).
  With pinned sm_80 binaries and the same GPU model, cross-card bit-identity is strongly
  expected but *not observed*. The contract's own principle — "Same-stack repeats must
  also be bit-exact. Any discrepancy is INVALID_TECHNICAL" — supplies a free witness:
  the treatment run recomputes FF focal/nonfocal and forced-carrier quantities that
  Phase A already persisted as float32 bits on the original host. **Recommendation:**
  pre-commit (additively, before launch) an invalidate-only check that treatment FF
  focal/nonfocal float32 bits equal Phase-A FF bits; mismatch ⇒ treat the treatment
  execution as INVALID_TECHNICAL, stop, re-gate. This closes the cross-host gap at zero
  marginal GPU cost. The harvester does not currently make this cross-artifact
  comparison — worth adding to the harvest step.
- **Preference order:** (1) resume the same stopped pod (same host likely; nothing extra
  needed beyond the machine checks); (2) new host + fingerprint equality + the
  pre-committed FF bit-witness; (3) most conservative, if budget allows after the
  treatment forecast: re-run the full technical validation on the new host (~12 min
  observed wall on the prior host, roughly $0.35–0.45 at plausible A100 pricing). A new
  **Phase A is not required in any branch** — Phase-A eligibility is case evidence bound
  by hashes, not host-bound — unless fingerprint inequality breaks the chain entirely.

## 6. Cost conditions under the frozen $2.40 ceiling

Frozen §18 facts: hard **$2.40** paid-compute ceiling including provisioning and failed
starts; proceed beyond the gates only after an observed forecast and only while a full
next case fits; record the provider hourly price; forecast each next complete case from
observed wall time with a **25% buffer**; never start a case whose forecast doesn't fit;
stop the pod before repairs/analysis/writing; the ≥$15 program reserve is not spendable
here.

Observed spend to date: **$0.8113626278** (the repo's recorded acceptance figure; I
could not independently re-derive it this session). Headroom:
**$2.40 − $0.8113626278 = $1.5886373722.**

Observed wall anchors: technical runner ≈ 11.6 min (raw stamp 024940Z → validated
03:01:14.6Z); Phase A ≈ 8.26 min including model load, two full-history oracles, FF, and
probes (03:06:22Z → 03:14:37.6Z). The treatment workload is roughly: two full-history
N-source replays (≈ the two Phase-A oracles) + compact F + P-schedule sources (chunked
prefill) + 34 short destination recomputes with probe forcings — a reasoned envelope of
**≈1.5–3× the prior GPU window, i.e. ~25–50 min**, implying a forecast around
$0.9–1.6 **after** the ×1.25 buffer at the recorded hourly price. That fits the headroom
at the low-to-mid end and fails at the top end. **The go/no-go number must be computed
per §18 from the recorded hourly price and observed wall times, not from my envelope**,
and launched only if forecast ≤ $1.5886. If a new-host full re-gate is chosen (§5 option
3), its cost must fit *in addition*, before treatment starts. After e01, headroom will
almost certainly not fit e02–e04; the frozen path is then to return the observed
full-canary forecast to the owner — that is the designed outcome, not a failure.

## 7. Exact remaining pre-launch checklist

1. Owner disposition note on the §14.1 stop-rule reading, committed additively (§2).
   Strict reading ⇒ NO-GO.
2. Confirm the bound revision4 manifest enumerates all six frozen cases (§3.1: all six
   frozen before e01 treatment unblinding). Hash-bound and pre-dated, but I did not
   count the rows (§8).
3. Compute and record the §18 forecast (observed wall × recorded hourly price × 1.25 +
   provisioning) and verify ≤ $1.5886373722; record it before provisioning.
4. Host decision per §5; if new host, pre-commit the FF float32 bit-witness (and
   optionally the technical re-gate if it fits).
5. Launch only through the frozen runner with `--subject exact-subject` and pass
   `--hourly-cost-usd` so the artifact carries cost; the runner re-verifies all six
   bindings byte-for-byte (any local drift in e.g. the prereg since Phase A will
   fail-closed at this step).
6. After the run: independent harvest, commit all artifacts under `results/`, **stop the
   pod**, then analyze.

## 8. Unverified residuals (honesty section)

My verification shell pass did not execute, so the following were **not** independently
re-checked by me and rest on the validators plus the runner's launch-time re-checks:
(a) local prereg bytes still hash to `fc02e86d…`; (b) the 1,661,785-byte Phase-A raw
artifact (`2cd7f6f1…`) — the eight competency booleans, per-token NLL finiteness, and the
path-control raw margins were accepted from the validation summaries; (c) the manifest's
six-case coverage; (d) the loader's fingerprint field inventory (decides the §5 host
question); (e) the recorded hourly price and the component-level cost/wall breakdown
behind $0.8113626278; (f) agent-level (vs code-path) independence of the Phase-A
validator. None of these is presumed false — (a) is machine-re-checked at launch — but
(b)'s validator explicitly reruns no model forward, so the raw artifact remains the
authoritative record.

---

**Verdict restated:** Phase A genuinely satisfies every literal §13/§14 release
condition as implemented, and the release machinery is fail-closed; the license is real
but **conditional** on a pre-launch owner disposition of the path-control stop-rule
ambiguity, the §18 forecast fitting $1.5886 of headroom, and exact runtime-fingerprint
equality on whatever host runs it (same stopped pod preferred; new host needs admission
gates + fingerprint equality + a pre-committed FF bit-witness, not a new Phase A).
Success, if it comes, is one exploratory case — not a claim-family result.
