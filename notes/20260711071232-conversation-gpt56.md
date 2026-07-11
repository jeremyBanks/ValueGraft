_This conversation covers a rigorous audit and successive redesigns of the
ValueGraft gapped-cache experiment, from the successful Luna archive rebuild
through a failed 30B bf16 technical gate and increasingly strict v4–v8
validation contracts. The current state is pre-launch: v8 is blocked by
semantic-evidence and stopping-rule validation gaps, with no paid semantic run
authorized._

**Participants:** gpt-5.6-sol-xhigh.

**Scientific direction.** The project pivoted from packed/reset-position
transplantation to a position-preserving gapped-cache assay. The intended
estimands remain conversation-level `GF = G_correct − G_fresh` and
`GW = G_correct − G_wrong`, with twelve fixed benchmark cases, unique disjoint
external donors, no efficacy declaration at N=6, and final claims limited to
this checkpoint, benchmark, bf16 eager backend, and compaction layout.
Wrong-history effects remain potentially confounded by summary
congruence/surprisal; positive results must not be described as proof that a
planted fact alone caused the benefit.

**Corrections established.** Early audits found stale packed-arm code,
pseudo-replicated calibration variants, incorrect N=6 efficacy interpretation,
cyclic donors, weak order/fingerprint checks, incomplete failure
terminalization, and harvest acceptance of tampered checkpoint provenance. These
were repaired across gapped-v2/v3. V3 eventually passed science and code review
with 124 tests, monitor clearance, exact tokenizer donor preflight, and a valid
ladder, but the paid 30B run failed closed before any semantic outcome:
contiguous versus split bf16 SDPA execution diverged by K 1.625, V 0.4502, and
logits 1.125, exceeding the unchanged 5e-4 threshold.

Local reproduction isolated the schedule discrepancy to bf16 SDPA query-shape
sensitivity: automatic and explicit masks both diverged, while eager attention
was exact on the tested CPU/0.6B path. MPS eager also failed schedule
equivalence, so device-specific eager results cannot authorize CUDA. `G_delta`
independently failed bf16 invariants and was retired rather than relaxed. The
rescue design therefore froze eager attention, preserved all thresholds,
retained GF/GW, and required representative schedule placebos plus a single
exact-30B technical authorization attempt before semantic scoring.

**Validation evolution.** V4/v5 reviews exposed inadequate gate ordering,
persist-before-raise behavior, all-donor revalidation, independent harvest
recomputation, technical-PASS binding, and missing release artifacts. V6/v7
added durable stage records, sealed terminal envelopes, sidecars, cryptographic
technical/semantic binding, per-render source schedule checks, independent
donor/calibration/static-provenance validation, and a composed
compacted-destination schedule gate. Snapshot raw tensors were ultimately waived
only with bit-identical generation/replay K/V hashes, explicit dtype/shape
metadata requirements, and source-to-inserted-span lineage; approximate replay
is insufficient.

V8 still received a NO-GO. The validator can accept fabricated plants, probes,
targets, positions, destination tails, calibration construction, and actual
`G_wrong` evidence; it does not independently bind scored outcomes to committed
source text and frozen target definitions, reconstruct the complete scoring
continuation, validate wrong-source NLL, or recompute/bind N=6/N=12 analyses,
intervals, bootstrap, interpretation, and stopping decisions. Required follow-up
is an additive amendment and identity bump implementing those source-derived
checks, adversarial regressions, tensor dtype/shape validation, and fresh
ladder/review artifacts.

**Current handoff state.** No pod is running and no further paid spend is
authorized. The v8 donor artifact is development evidence only. A historical v4
CPU bf16-eager ladder completed PASS and was preserved, while an older paused v6
process was resumed for audit-trail preservation; neither can authorize the
corrected apparatus. Before any launch, the team must finish the next amendment,
rerun the complete suite and monitor fault-injection tests, produce a fresh
ladder and donor validation artifact on the exact clean commit, obtain
independent code/science/cross-family reviews, push the exact launch commit, and
only then provision one bounded technical attempt. Any technical failure remains
a fail-closed result with no semantic interpretation.

## Conversation sources

- `019f4ffd-2c0c-7120-aa25-a76bfdf36a3f`
- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f5005-efb7-71a1-94ed-5476ce540d9a`
- `019f5005-b3c6-75a1-9325-5294bce64e9f`
- `019f500a-ee56-7c22-bc0c-fa9d47e0dd44`
- `019f5005-dab0-7bb3-97eb-025953672c17`
- `019f5017-4efc-71a0-97a5-f106bd6168eb`
- `019f5017-6555-7d30-8b71-e3ec1bb9652b`
- `019f5058-1604-7f00-b34e-d3c3769f6c40`
- `019f5058-7003-7ae3-a350-002f48c5801c`
- `019f5058-492a-7b63-b964-76607ebad880`
- `019f5063-3339-75d1-9163-c170ac27877e`
- `019f5063-0b20-7620-a01f-723430451e18`
- `019f5063-2089-7bb2-b7f9-98d4db349fe9`
- `019f5068-38de-7450-91df-efcf46aff968`
- `019f5068-b28d-7782-ab40-489270741582`
- `019f506d-1c4b-7a23-875a-12c7f86bc530`
- `019f506d-3f15-7ff0-984a-0c3277426ffa`
- `019f5078-86c1-7343-9503-71f3e8cbcea8`
- `019f507f-1be4-7d30-ad96-38b6321f1e27`
- `019f507f-0a4a-7d20-b87a-16e7985f072e`
- `019f5085-b6a9-7511-a2b9-7523fd9c1feb`
- `019f508b-9d23-71d2-8a4e-d06e570a86e8`
- `019f508b-f523-7731-86d5-02802cc035f0`
- `019f50a9-7f5d-7fb0-8468-dd7986ae343f`
- `019f50a9-6853-7442-a1eb-b447e6d53dce`
- `019f50ca-ef44-7f32-9c2b-758080fecd8d`
- `019f50cb-0cdd-7270-86db-09ab637c400f`
- `019f50dc-d4c0-79f1-ab4a-0c37b42e3b96`
- `019f50f5-733f-7212-9e05-a290900d986b`
