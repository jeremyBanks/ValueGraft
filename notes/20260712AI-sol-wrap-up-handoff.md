# Immediate wrap-up handoff

**Author:** OpenAI GPT-5.6 Sol (extra-high and ultra reasoning)  
**Recorded:** 2026-07-12  
**Reason:** owner ordered the paper saved, committed, and wrapped up immediately

## Published state

`PAPER.md` was promoted byte-for-byte to `README.md`, committed as `b803739`
(`Publish final evidence-bounded paper`), and pushed to `origin/trunk`. The
paper now explicitly states that the project did **not** obtain a powered upper
or equivalence bound and that most of the available terminal Pod budget remained
unused. It removes working-paper status and incorporates the latest frozen
Claude usage prefix and the missing Luna-telemetry qualification.

## Four-bit pod requirement: complete

The owner's required real four-bit model comparison ran on pods before
publication. P01 and P02 compared actual bitsandbytes NF4 eligible-linear
weights against bfloat16 weights for exact checkpoint
`Qwen/Qwen3-30B-A3B-Instruct-2507` revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` on two distinct A100 80 GB hosts.
Both NF4 technical records attest `load_in_4bit=true`, NF4 double quantization,
uint8 storage, bfloat16 compute, all 18,672 eligible linears / all
29,909,581,824 eligible logical weight elements covered, and bfloat16 K/V in
all 48 layers. The independent cross-pod comparator is `PASS` with SHA-256
`588229df5f4ca5c8613dc4f21564043214bfe889fc4dba176300ace0435442be`.
The focused local integrity suite observed 83 passing tests immediately before
publication.

Scientific boundary: this is a genuine four-bit-*weight* comparison, not
four-bit KV. It bundles weight representation with kernel implementation, uses
one fixed semantic fixture, lacks a matched placebo, and produced no correct
graft generation. Exact reproduction across hosts establishes apparatus
repeatability for that fixture, not efficacy, quantization causality, an effect
bound, or population/agent generalization. Canonical disposition:
`notes/20260712AD-sol-four-bit-pod-comparison-completion-gate.md`.

The RunPod API was queried immediately before wrap-up and returned zero active
pods.

## Accounting state

- Final committed Claude prefix:
  `results/end_to_end_accounting/claude-project-usage_claude-cli_20260712T140932Z.json`
  (SHA-256 `86a9a3735e6f5b171dc60369690c370cb1d3ea66658c029afea58122f79c9178`).
  It freezes 405 files / 207,424,274 bytes, 8,965 unique assistant message IDs,
  8,966 expanded request rows, and 3,262,758,065 tokens. The paper's
  `$3,034.01657825` is a frozen-rate, standard-speed, token-only API
  counterfactual, not observed cash.
- Exact RunPod account-window and project-lower-bound accounting, local-compute
  lower bound, and OpenRouter envelopes remain committed under
  `results/end_to_end_accounting/` and notes `20260712AE` / `20260712AF`.
- The deterministic nine-input summary combiner and nine passing focused tests
  were committed as `15a64cf`. No final combined output was emitted before the
  deadline.
- A final Codex snapshot attempt failed closed on a newly observed counter reset
  at rollout line 111 (`counter reset does not restart at last usage`). No
  partial output was committed. Codex totals therefore remain reconstructed
  working-prefix values, not a source-exact final ledger.
- The notes manifests retain 77 Luna-attributed outputs, but ephemeral Luna
  calls are absent from Codex state. Historical requests/retries/tokens/cash
  remain unknown.

## Checks that remain open

The targeted four-bit suite passed `83/83`. The broader
`tests/test_validate_semantic_release.py` run observed `32 passed, 2 failed`:

1. a test expected a `committed-case` rejection but current retained evidence
   first rejects the c10 schedule row; and
2. the historical v10 apparatus inventory test expects 35 files while the
   current repository has 51.

These appear to be stale historical-release assertions rather than failures of
the published paper or P01/P02 artifacts, but they were **not diagnosed or
fixed** before the deadline and must not be described as passing.

The transcript duration-repair dry run found zero overlong existing notes, one
large continuation, and three new notes. Those live-tail summaries were not
generated before wrap-up. The failed Codex accounting boundary and those
deferred transcript updates are the two concrete follow-up items.

## Commit sequence for this wrap-up

- `d4fbb89` — state the missing powered bound and terminal goal drift
- `a8a3dab` — freeze Claude usage after terminal review activity
- `15a64cf` — add deterministic end-to-end accounting combiner and tests
- `b803739` — publish `PAPER.md` byte-for-byte to `README.md`

All four were pushed to `origin/trunk`.
