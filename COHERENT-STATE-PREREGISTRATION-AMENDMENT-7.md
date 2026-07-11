# Coherent summary state: Amendment 7 — independent evidence and actual-render schedule closure

**Frozen:** 2026-07-11, after independent code, science, and Claude Opus 4.8
reviews of the unexecuted Amendment-6 apparatus, and before any 30B eager
technical or semantic outcome. This is additive. No tolerance, arm, estimand,
case order, donor map, stopping rule, or claim scope changes.

## Why this amendment is necessary

The v6 reviews found that several implementation checks were real in-memory
assertions but were not independently reconstructible from the terminal
artifact. They also identified one remaining exposure to the original
schedule confound: the technical battery tested the twelve committed source
files, while the semantic process freshly renders new conversation text. A
fresh render can have a different exact prefix length and message-block
partition. A static proxy therefore cannot alone prove schedule equivalence for
the token stream that will actually be scored.

The same reviews found concrete operational defects before paid execution:
model-load backend progress collided with a PENDING-only stage transition;
CUDA OOM could be swallowed inside row batteries; a terminal technical PASS
rejected by independent harvest could become unharvestable; and failure harvest
did not always verify the terminal envelope. Those defects are corrected and
regression-gated under this amendment.

## 1. Version identity

Artifacts governed by all seven additive amendments use exactly:

- `amendment_id = COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7`
- `design_id = coherent-state-gapped-v7`
- `schema = 2`

No v6 ladder, donor artifact, technical attempt, harvest attestation, or
semantic artifact can authorize v7.

## 2. Static provenance is an explicit first lifecycle stage

`static_provenance` is the first item in the terminal technical gate order,
before backend attestation. It is predeclared PENDING, durably enters RUNNING,
and closes PASS only after recomputing the exact launch commit, amendment hashes,
apparatus inventory, input inventory, model/revision/dtype/backend request,
tokenizer and request hashes, and technical-only mode. A failure while a stage
is RUNNING closes that stage ERROR; later PENDING stages close
SKIPPED_DEPENDENCY.

Model-load backend enumeration may durably begin `attention_backend` before the
loaded-gate function is entered. That function may resume only this already-
RUNNING stage; every other stage remains PENDING-only.

CUDA OOM, lost model residency, backend ambiguity, persistence corruption, or
input/tokenizer mismatch is re-raised out of per-row batteries after the failing
row is persisted. It prohibits all later model-backed work.

## 3. Independent harvest reconstructs raw evidence

The independent validator remains source-independent from the apparatus modules
and must reconstruct, rather than trust, at least the following:

- frozen synthetic literal, token pool, token/position hashes, partitions,
  logical-gap schedule, continuation position, rejection injection, every
  per-layer row, and all aggregates;
- every committed source file's raw and canonical hashes against both disk and
  the input inventory; production-tokenizer vocabulary/template/request
  binding; exact re-tokenized generation prefix; every token and position hash;
  exact message boundaries, conceptual widths, ordinary chunks, message-block
  chunks, continuation position, per-layer rows, and aggregates;
- replay/rebuild/mask/future-mutation layer order, raw K/V maxima, scalar
  aggregates, and thresholds;
- mathematical logical/physical position schedules and exact wrong-source token
  partitions;
- calibration prefixes, hashes, changed/structural positions, special-token
  exclusion, label coverage, and target lengths without a model forward;
- all donor source files and inventory bindings, authors, structural/content
  partition completeness, replacement hashes/bounds/cycling, special-token
  exclusion, reconstructed wrong-prefix hash, uniqueness, disjointness, and
  frozen mapping; and
- intervention source, destination, inserted-span, before-span, full-branch,
  and post-summary per-layer hashes sufficient to independently verify fresh
  self-replacement, exact selected insertion, non-summary preservation, equal
  branch lengths, and a causally changed recomputed tail.

A producer boolean, plausible-looking digest, stored aggregate, or status string
cannot substitute for these recomputations.

## 4. Exact per-render schedule equivalence before semantics

After a separately authorized semantic process has freshly rendered a
conversation, but **before** it generates that conversation's summary, creates
a wrong source, constructs an arm, scores a calibration, or computes any target
or outcome, it runs an exact schedule fixture on the actual correct generation
prefix:

1. derive the complete production-tokenizer prefix for the freshly rendered
   messages plus the frozen summary request and assistant header;
2. derive `system`, `history`, and `request_header` boundaries by exact
   `<|im_start|>` scanning and the same fresh-suffix equality used by production;
3. compare the identical full token and logical-position stream under ordinary
   consecutive chunks of at most 4096 and under separate system/history/request
   blocks using the same chunking helper;
4. require at most `5e-4` for all 48-layer prefix K/V maxima, final logits,
   frozen margin shift, and the same q=1 continuation logits/new-row K/V maxima;
5. atomically persist complete tokens, positions, hashes, partitions, raw layer
   values, and aggregates in the conversation checkpoint before applying the
   verdict; and
6. fail the conversation before any semantic outcome unless this exact fixture
   passes.

The terminal semantic validator independently re-tokenizes the saved rendered
conversation, reconstructs its exact partitions and hashes, recomputes every
stored aggregate, and requires this PASS for every scored checkpoint. This gate
computes no semantic result. It may cover a prefix longer than the static 9509
technical fixtures, provided the exact per-render comparison passes and the
frozen model context check also passes.

## 5. Terminal and environment closure

A technical terminal PASS that is subsequently rejected by the independent
validator remains harvestable as failure evidence: its PASS envelope is still
verified byte-for-byte, it authorizes no semantics, and the pod can terminate
after preservation. Every post-model FAIL harvest verifies its complete
index/receipt/sidecar envelope; setup-only failures remain separately
quarantined.

The frozen RunPod image is
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`; both paid jobs require
the observed ambient `torch==2.4.1+cu124` and `torch.version.cuda==12.4` before
model load. The apparatus inventory includes pod lifecycle and preflight code.
The standard launcher forwards the technical-result commit and directory needed
by the separate semantic process.

## 6. Release gate

Before any paid v7 attempt: run and commit a fresh production-tokenizer donor
validation artifact; run and commit a complete v7 0.6B bf16 eager CPU ladder;
run the full unit and monitor fault-injection suites; obtain fresh independent
code, science, and cross-family reviews on the exact clean launch commit; and
push that commit. A PASS authorizes only one technical-only 30B attempt. Semantic
execution remains separately conditional on a committed, independently
harvest-valid v7 technical PASS with an unchanged apparatus inventory.
