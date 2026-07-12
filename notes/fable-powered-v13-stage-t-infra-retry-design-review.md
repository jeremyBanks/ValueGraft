# Fable design review — powered-v13 Stage-T infrastructure-only retry

**Date:** 2026-07-12 · **Reviewer:** Fable (independent critical review, detached worktree)
**Evidence read:** DECISIONS.md (canary sub-cap/stop-clock lines), preregistration §15,
`src/powered_v13_release.py`, `src/powered_v13_lifecycle.py`,
`scripts/run_powered_v13_stage_t_lifecycle.py`,
`notes/powered-v13-stage-t-admission-rejected-provider-schema.md`,
`results/.../stage-t-admission-rejected_20260712T224225Z/session/lifecycle.json`,
both `create-response.json` files, `git show` 766db12 and 79ab77f,
`git diff --stat 766db12..79ab77f`. No network, no secrets, no other files.

## Verified factual baseline

All claimed facts check against literal artifacts. `lifecycle.json` records exactly two
attempts, both `REJECTED_DELETED` at 29 provider seconds each, `admitted_pod_id: null`,
cumulative 58 s and `$0.02239444444444444444444444444`, final status
`ADMISSION_REJECTED` with `ATTEMPTS_EXHAUSTED`. The balance delta
(57.1287946692 − 57.1090060887 = $0.0197885805) matches the incident note; the lifecycle
figure is the conservative one (its clock includes positive-deletion verification) and is
the correct number to carry. Both persisted create responses contain
`machine.secureCloud: true`, `machine.gpuTypeId: "NVIDIA A100 80GB PCIe"`, `gpuCount: 1`,
and **no** top-level `cloudType` — confirming the failure was a local response-schema
mismatch, not a wrong host. By inspection, the 79ab77f validator
(`_validate_provider_allocation_shape`) passes those exact persisted bytes: machine
mapping present ✓, `secureCloud is True` ✓, exact SKU ✓, `gpuCount == 1` ✓, absent
`cloudType` tolerated ✓. 766db12 is a one-parent (f3d4ed0), two-path authorization commit
(prereg status line flip + one manifest), exactly the §15 shape. 766db12 is an ancestor of
79ab77f, and the diff between them touches only `src/powered_v13_lifecycle.py`, its test
file, `STATE.md`, and result/incident evidence — every other file, including
`powered_v13_release.py`, `powered_v13_watchdog.py` and its script, the import-audit
module, the provider adapter, and `scripts/run_powered_v13_stage_t.py`, is byte-identical
at the two commits.

## Assessment by dimension

### 1. Scientific honesty — sound in substance, one evidence-binding gap

The scientific payload stays immutable at 766db12; semantic N remains 0 (and is enforced
mechanically — `verify_release_binding` requires `semantic_n == 0` in the bound import
report); the retry decision is conditioned only on an infrastructure outcome (schema
rejection before any model load), so there is no unblinding or outcome-contingent
reinterpretation. The §15 "at most two allocation attempts" bound is prose enforced by the
local controller constant, not by any frozen machine gate; revising it through an
explicit, committed, pre-paid-call amendment that discloses the exhaustion and carries
58 s / $0.02239… forward is honest, and matches the incident note's own requirement
("a new, explicit, outcome-transparent design/release that binds the corrected
real-provider schema").

**Gap:** the retry session's machine-readable record will state
`authorization_commit: 766db12` and nothing else — a future auditor sees a third
allocation attempt facially contradicting the §15 text at that commit. The amendment must
be woven into the run's own evidence, not just sit beside it (repair C2).

### 2. Release acyclicity — sound

The outer amendment follows the acyclic pattern: manifest names an immutable parent
(new static root), child commit changes exactly the amendment status plus adds the
manifest, no self-hash, and the amendment references 766db12 only backward. The inner
subject never references the amendment forward — that asymmetry is precisely what keeps
766db12 immutable, and is acceptable given C2. Note the frozen release code **cannot**
mint a second Stage-T authorization by construction (`_require_parent_manifest_state`
rejects any root already containing a Stage-T manifest, and the fixed DRAFT status line no
longer exists in the prereg); the proposal correctly does not fight this — that refusal is
the design working as intended, and the outer amendment is the only clean escape. Any
alternative that reverts the prereg status or strips the committed manifest to re-mint a
"real" Stage-T release would rewrite the scientific record and is strictly worse.

### 3. Authority — conditional

DECISIONS.md authorizes the technical canary inside the `$1.50` / 3,300-provider-second
sub-cap within the `$12` bucket, with the stop clock: canary run by active goal hour
7.8758, decision boundary at 11.8758. The retry does not create new spend authority; it
consumes the same sub-cap with carry-in. The root protocol status is
`TECHNICAL_CANARY_AUTHORIZED` (not DRAFT), so the "no paid work while DRAFT" rule refers
only to the amendment's own status, which the child commit flips before launch. The
binding open condition is the stop clock (C5) — this review cannot compute the live goal
hour; if 7.8758 has passed, the written feasibility-rejection path governs and this retry
is not authorized regardless of its internal correctness.

### 4. Checkout/receipt correctness — workable, with stated preconditions

Minting a fresh receipt for 766db12 in a separate detached inner checkout is mechanically
valid: `create_stage_t_launch_receipt` re-verifies the full authorization (parent
inventory, two-path diff, status transition, manifest bindings) against the inner repo and
writes into an empty directory; `verify_release_binding(repo=inner)` then enforces
detached HEAD = 766db12, clean tree (untracked included), single fresh receipt under the
unmodified 300-second gate, and recomputes the import audit — using the outer audit
module, which is byte-identical at both commits, against inner bytes, so it matches. The
receipt asserts only true facts (stage, commit, manifest hash, creation time); it does not
assert remaining attempts. The remote flow is self-consistent: the remote clones from the
repository URL, checks out 766db12 detached, verifies with **its own** 766db12 modules
(the `_verify-remote-setup` path exists at 766db12 — the 79ab77f hunk context shows it
predates the patch), rsync `--checksum` from the inner checkout is a byte no-op against
the same commit, and the remote re-checks HEAD/clean-tree **after** rsync, so local drift
fails closed. The patched outer lifecycle is deliberately never synced — the remote
verifies its own release with its own version. Preconditions: receipt directory and
session root must live outside the inner worktree (else the clean-tree gate fails, or
worse, later evidence writes dirty the subject); 766db12 must actually be reachable at the
remote URL (the original run never reached clone — rejection preceded it — so this has
never been exercised; check `ls-remote` before paying); `SC_RUNPOD_KEY_PATH` must be set
absolute, because a relative `pod.KEY_PATH` is resolved against `--repo` — the inner
checkout — where no key exists, and the run would fail closed at
`resolve(strict=True)` (safe, but wasteful to discover live).

### 5. Retry/budget bounds — correct

With `MAX_ALLOCATION_ATTEMPTS = 1` the loop makes exactly one create call; `NoCapacity`
terminates as `NO_CAPACITY`/`ATTEMPTS_EXHAUSTED` with no fallback, admission rejection
terminates as `ADMISSION_REJECTED`, and any ambiguous create still fails closed with
retry forbidden — matching "exactly one additional allocation and no fallback." The
carry-in values pass the gates: `0.02239444444444444444444444444` is canonical
fixed-point under `_exact_money`, and 58 ≤ 3,300 with 3,242 s remaining, comfortably above
`DELETE_LEAD_SECONDS`. The cumulative cap is enforced twice: the lifecycle accumulates
from the carried values, and `build_record` seeds the OS-owned watchdog with the same
priors so the provider-clock kill honors the 3,300-second total independently of the
controller. At $1.39/h the seconds cap binds first (3,242 s ≈ $1.25 < $1.4776 remaining),
so both bounds are real and consistent.

### 6. Outer controller using the inner repo for watcher paths — safe, verified

`LaunchdWatcherBackend(repo=args.repo)` executes
`<inner>/scripts/run_powered_v13_stage_t_watchdog.py` (with inner `src` on its path),
while the watchdog record's probe/harvest commands bind to the **outer** patched helper
via `Path(__file__).resolve()`. This two-codebase supervision chain is safe here for a
verified reason, not an assumed one: the diff 766db12..79ab77f touches no watchdog,
release, audit, transport, or provider file, so the inner watchdog is byte-identical to
the outer's; the only patched code (`powered_v13_lifecycle.py`) runs solely in the outer
controller process, which is the sole allocation/admission owner. The inner watchdog's
cleanup path is additionally the one component empirically proven in the failed run —
both pods deleted with direct-404 plus inventory-absence under exactly this code. The
unpatched `validate_admission` at 766db12 exists in the inner tree but is on no execution
path of the watchdog or the remote job. No repair needed; record the byte-identity check
(`git diff 766db12 79ab77f -- scripts/run_powered_v13_stage_t_watchdog.py
src/powered_v13_watchdog.py src/pod.py` → empty) in the session evidence.

## Fatal issues

None found, provided the conditions below are met. Nothing in the design reuses the
exhausted setup receipt, touches the frozen payload, weakens a machine gate, creates
semantic N, or exceeds the sub-cap.

## Conditions (minimal repairs, in order of importance)

- **C1 — Commit the max-attempts patch and bind it.** The
  `max-allocation-attempts=1` change must be a commit contained in the amendment's static
  root, and the amendment manifest must inventory the exact patched
  `src/powered_v13_lifecycle.py` (and controller script) bytes. An uncommitted working-tree
  edit on the paid path defeats the incident note's requirement that the retry release
  *bind* the corrected controller. This is the difference between a bound release and a
  pinky promise.
- **C2 — Weave the amendment into the run's own evidence.** Embed the amendment
  authorization commit hash in `--primary-batch-id`
  (e.g. `stage-t-retry-<amend7>-<utc>`) and state in the amendment that the session's
  `authorization_commit` field will read 766db12 by design. Without this, the retry
  session facially contradicts §15's two-attempt text with no machine-readable pointer to
  its authority.
- **C3 — Order of operations must be provable.** Amendment child commit (status flip +
  manifest) exists — and is pushed — before the fresh inner receipt is minted and before
  any paid call; receipt `created_utc` after amendment commit time. Also confirm 766db12
  is reachable at the repository URL (`git ls-remote`) before allocation; the original run
  never exercised the clone.
- **C4 — Stop clock.** Confirm against the live goal-hour ledger that the retry completes
  before active goal hour 7.8758. If not, DECISIONS.md's feasibility-rejection path
  governs and this retry must not run.
- **C5 — Operational preconditions.** Fresh receipt directory and session root outside the
  inner worktree; `SC_RUNPOD_KEY_PATH` set absolute; carry-in passed exactly as
  `--prior-stage-t-spend-usd 0.02239444444444444444444444444
  --prior-stage-t-provider-seconds 58`.

## Rejected apparatus

- **The fresh outer receipt.** No frozen verifier consumes it: the remote gate and
  `verify_release_binding` consume only the inner 766db12 receipt, and the frozen release
  code cannot even construct a receipt for a non-v13-prereg amendment path. A receipt no
  gate checks is ceremony imitating a gate, which is worse than absent — it manufactures
  the appearance of machine enforcement. Drop it; the amendment authorization commit hash
  is the operative record. (If a local freshness token is genuinely wanted, keep it but
  label it operator-checked provenance, never a verified receipt.)
- **Any new verifier stage for the amendment.** Do not extend `_StageKey`/specs or write a
  one-off amendment verifier for a single bounded infra retry. The committed one-parent,
  two-path amendment plus C1's manifest binding is sufficient, auditable apparatus.

## Verdict

**CONDITIONAL** — the core design (immutable inner 766db12 subject + patched outer
controller + explicit committed amendment + carried budget + one-attempt bound) is the
minimal honest shape for this retry and is approved as designed, subject to C1–C5; drop
the outer receipt.

## Main-agent disposition after implementation

The review preceded the dedicated outer paid-entry wrapper. Its objection to an
unconsumed outer receipt therefore described the proposal at review time, not the
implemented boundary. The DRAFT implementation now consumes the outer receipt and
authorization before provider construction, and it closes the review conditions as
follows:

- C1: provider-schema repair `79ab77f` and one-attempt controller `ffc90ce` are
  committed and the outer manifest verifier byte-compares the paid controller paths to
  the latter.
- C2: the batch ID must contain the outer authorization hash prefix; a canonical
  authorization-consumption record binds the full outer and inner authorization,
  manifest, and receipt hashes to that batch and exact session path.
- C3: the wrapper verifies both detached clean checkouts, both fresh receipts, and the
  exact inner import report before atomically consuming the outer authority and before
  delegation. `origin/trunk` was independently observed to contain `766db12` in the
  ancestry of its advertised tip before authorization.
- C4: the live goal clock remained below hour `7.8758` at DRAFT integration.
- C5: the wrapper rejects a session inside either checkout and fixes its basename to the
  authority-linked batch ID. The launch procedure requires an absolute RunPod key path.

Independent review found one further blocker after this Fable review: a per-run cap was
replayable across invocations. DRAFT commit `d2e9da9` repairs that by writing one fixed
OS-account-home, full-authorization-keyed `O_EXCL`/fsynced consumption record before the
delegate. A real-flow test proved that even a failed first delegate permanently prevents
a second delegate call. The main agent retains authority over the final authorization
and paid launch.
