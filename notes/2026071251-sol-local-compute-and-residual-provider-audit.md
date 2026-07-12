# Local compute and residual-provider accounting audit

**Author:** GPT-5.6 Sol, ultra reasoning (`gpt-5.6-sol-ultra`)

**Prepared:** 2026-07-12 13:31 UTC

## Bottom line

The repository supports an auditable floor of **30,732.859043836593536
seconds (8.536905289954609 hours)** of Mac-local, inference-bearing
experimental-run wall time across 88 retained result files. That is not the
project's total local compute and not pure accelerator-busy time. It is the sum
of exact per-result `wall_seconds` records that can be retained without
guessing. The tree proves substantial additional local work, but its older
schemas omitted timing, so no complete Mac-hour or generated-token total can be
reconstructed.

RunPod is better bounded. Two settled provider snapshots already establish an
exact account-window total of **304.913444722222 billed hours and
$424.871205699397252292 in consumed Pod credits** across 76 Pods. Independent
project evidence attributes at least **283.679473055556 hours and
$394.948127557756380202** across 73 Pods. The remaining three IDs account for
21.233971666667 hours and $29.92307814164087209; they remain attribution-unknown,
not silently assigned to or excluded from the project.

The later official RunPod audit also returned exact empty responses for the two
distinct non-Pod billing categories: **$0 for serverless endpoints** and **$0
for network volumes**. Those categories are additive to Pod consumption, so the
exact all-covered-category provider total remains
**$424.871205699397252292**. It is still consumed credits, not cash paid.

These are deliberately separate quantities. No single “total project cash”
number is defensible from the available evidence.

Machine-readable detail, per-file hashes, arithmetic, exclusions, and safe
commitments to ignored launch records are in
[`experimental-compute-residual-provider-audit_gpt-5.6-sol_20260712T133100Z.json`](../results/end_to_end_accounting/experimental-compute-residual-provider-audit_gpt-5.6-sol_20260712T133100Z.json).

## Mac-local experimental work

The timed floor selects every current `results/**/*.json` object with both:

1. a top-level model identifier beginning `mlx-community/`; and
2. a numeric top-level `wall_seconds` field.

The retained components are:

| Result stratum | Files | Recorded seconds | Recorded hours |
|---|---:|---:|---:|
| `raw` | 20 | 7,188.724311113357441 | 1.996867864198155 |
| `raw_brief` | 12 | 3,203.61439847946172 | 0.889892888466517 |
| `raw_30b` | 20 | 6,795.482462644576995 | 1.887634017401271 |
| `raw_30b_brief` | 12 | 2,655.10276842117310 | 0.737528546783659 |
| `raw_brief_repro` | 24 | 10,889.93510317802428 | 3.024981973105007 |
| **Auditable floor** | **88** | **30,732.859043836593536** | **8.536905289954609** |

All 88 full-file hashes are unique. They also remain unique after removing only
`wall_seconds` and canonicalizing the JSON, so no retained timed row is a
byte-copy or an otherwise identical payload with a different clock value. This
is a conservative duplicate check, not proof that every historical render was
unique or saved.

The floor excludes work that plainly occurred but lacks additive timing:

- 108 current result artifacts explicitly identify an MLX-community model but
  omit `wall_seconds`: 12 `phase2_4b`, 12 `phase2_30b`, 48
  `longmemeval_4b`, and 36 `longmemeval_30b` artifacts.
- 80 alpha-sweep artifacts across four directories were previously audited as
  local 4-bit MLX work, but their files retain neither model identity nor wall
  time.
- Local ladder, identity, diagnostic, failed, and ad hoc development runs are
  not comprehensively timed.
- Incident 31 records three completed local 4B corpus renders at roughly 7–8
  minutes each. That estimated 21–24 minutes is reported separately and is not
  added to the exact-record floor.

Artifact counts are not execution counts. Some files contain many arms or
passes, and some later analyses reuse saved generations. I therefore did not
turn the 188 untimed artifacts into hours. Historical generated-token counts
are likewise **unknown**: schemas differ, some outputs have no token counter,
and not every render survived.

Mac electricity and hardware amortization are also **unknown**, not zero. No
wall-power meter, power trace, experiment-only duty cycle, tariff allocation,
purchase price, useful-life convention, or residual-value convention was
recorded. A modeled number could be produced only by introducing owner-chosen
assumptions; it would not be a measurement of this experiment.

## RunPod rental and workload context

The settled provider ledger remains authoritative for rental consumption:

| View | Class | Pods | Billed hours | Consumed Pod credits |
|---|---|---:|---:|---:|
| Complete provider-account window | exact source record | 76 | 304.913444722222 | $424.871205699397252292 |
| Independently project-attributed floor | lower bound | 73 | 283.679473055556 | $394.948127557756380202 |
| Attribution residual | unknown | 3 | 21.233971666667 | $29.92307814164087209 |

The provider's official non-Pod endpoints add this separate reconciliation:

| Distinct RunPod category | Class | Rows | Amount | Additive to Pod billing? |
|---|---|---:|---:|---|
| Serverless endpoints | exact source record | 0 | $0 | yes |
| Network volumes | exact source record | 0 | $0 | yes |
| **All covered RunPod categories** | **exact source record** | — | **$424.871205699397252292** | Pod total + $0 + $0 |

Forty-five ignored launch snapshots could be safely joined to their exact
provider rows without serializing their credentials, environment, networking,
machine IDs, or consumer metadata. All 45 identified one A100 80 GB PCIe GPU:

| Launch-time context | Pods | Exact provider hours | Exact provider credits |
|---|---:|---:|---:|
| Declared $1.19/h; `secureCloud` absent | 13 | 39.028294444444 | $47.92052225489169329 |
| Declared $1.39/h; `secureCloud=true` | 32 | 155.175041388889 | $220.005807745736092906 |
| **Locally retained launch subset** | **45** | **194.203335833333** | **$267.926330000627786196** |

The launch-time hourly rate is context, not an invoice. Provider charges exceed
the simple rate-times-runtime products, and the exact provider rows are the
controlling values. The 31 Pods without local launch snapshots are not assigned
a GPU type because the committed provider grouping contains no hardware field.

The repository also retains 105 agent-episode records whose `wall_seconds` sum
to **165,992 seconds (46.108888888889 episode-hours)**: 81 synthetic episodes
sum to 12.459444444444 hours and 24 SWE-bench episodes sum to
33.649444444444 hours. This is useful workload context only. Episode clocks can
overlap, include agent/tool/wait time, and sit inside Pod rental; they are not
GPU-busy hours and are never added to the 304.9134 provider hours.

The final 4-bit precision comparison is also already inside the provider total:

| Protocol | Exact billed hours | Exact consumed credits | Outcome |
|---|---:|---:|---|
| P01 | 1.954605555556 | $2.7724572848528624 | cap-interrupted; valid partial packages retained |
| P02 | 1.185347777778 | $1.6823556371964514 | PASS; four complete outcomes and 42 lossless packages |

Incident 38 separately records approximately 6.3 hours and roughly $9 for a
24-conversation render that was lost before persistence. It is a historical
estimate and a missing-artifact disclosure—not another charge to add. Its
underlying rental is already represented in the provider ledger if the Pod is
among the project-attributed IDs.

I excluded parent/child and copied timing repetitions from additive totals.
This includes champion-harvest summary timings that repeat child payload
timings, redraw/champion-validation work trees without uniform execution IDs,
and precision-probe stage/provider durations repeated through decision,
compact, timing, and run-manifest files.

## Residual providers and access costs

### OpenRouter

The later official-API snapshot is an exact source record for both the current
key and authenticated-account context. The key reports **0.0232311 OpenRouter
credits used**, all within the current monthly counter, with 49.9767689
remaining against a 50-credit limit. The account-lifetime endpoint reports
**114 total credits** and **95.541220588 credits used**. The account counter
overlaps the current-key counter, so they are not additive. Neither endpoint
exposes generation timestamps or project labels. Project records document at
least fourteen GPT-4o-mini simulation episode executions, but request tokens,
project-attributable credits, cash deposited, refunds, and cash paid remain
**unknown**. These counters are account/key envelopes, not exact project cash.

### Gemini

Gemini is credited as a contributor, but the invoking surface was not recovered.
At audit time `~/.gemini` contained 154 files and none had a filesystem
modification time at or after the 2026-07-04 20:05:15 UTC research start. This
does not prove zero Gemini use; it shows that this local history cannot account
for the credited in-window contribution. In-window tokens, requests, access
plan, and cash are **unknown**.

### Hugging Face

Repository jobs used authenticated Hugging Face Hub access to download model
checkpoints. I found no Hugging Face inference-endpoint receipt, usage export,
download-byte ledger, or subscription invoice. The models were executed on the
Mac or RunPod, so those compute costs belong in the corresponding ledgers; no
separate HF inference charge is invented. Hub-plan cash and any provider-side
egress remain **unknown**.

### Subscriptions

Claude CLI access is documented as Claude Max/subsidized usage, while Codex is
documented as ChatGPT-authenticated usage. Their reconstructed token workloads
and any counterfactual API list-price analogues belong in separate final-ledger
sections. No Claude or ChatGPT subscription invoice or project-allocation rule
was supplied, so attributable subscription cash remains **unknown**, not zero.

## Integrity and nonadditivity rules

- Do not add the 45-launch subset, P01/P02, episode hours, or incident-38
  estimate to the RunPod provider total.
- Do add the distinct RunPod serverless-endpoint and network-volume billing
  categories to Pod billing; both exact values are zero, so the covered total
  is numerically unchanged.
- Do not equate consumed Pod credits with deposits or cash paid.
- Do not add OpenRouter current-key usage to the overlapping authenticated-
  account lifetime counter, or add OpenRouter usage credits to RunPod credits
  and label the result cash.
- Do not add subscription list-price analogues to subscription invoices.
- Do not relabel the 8.5369-hour retained Mac floor as total local compute or
  accelerator-busy time.

No provider request or model invocation was made. Existing ignored Pod launch
snapshots—which contain sensitive fields—were parsed locally, but no credential,
environment, user, machine-ID, IP, or port value was selected or serialized.
The machine-readable output retains only an allowlisted aggregate plus a
cryptographic commitment.
