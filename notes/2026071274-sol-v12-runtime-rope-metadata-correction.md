# V12 runtime RoPE metadata correction

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning

## Observed failure

After the loading-info repair, the second local technical execution loaded the
pinned Qwen3-0.6B model and completed loader attestation until runtime geometry.
It then stopped before any model forward because the runtime fingerprint looked
only for `config.rope_theta` and recorded its absent-value sentinel `-1.0`.
The failed raw artifact is preserved at
`results/coherent_canary_v12_technical/coherent-canary-v12-technical_local-apparatus_20260711T235635802707Z.json`.

The pinned snapshot's literal `config.json` contains `rope_theta: 1000000` and
had already passed exact snapshot validation.  Direct inspection of the pinned
Transformers 5.0.0 configuration showed that deserialization moves this value to
`config.rope_parameters["rope_theta"]` (and aliases it through `rope_scaling`),
leaving no direct `rope_theta` attribute.  The pinned 30B exact-model config uses
the same representation with its frozen value `10000000`.

## Bounded correction

The runtime fingerprint now reads the Transformers-5 `rope_parameters` mapping
and continues to require the extracted value to equal the separately frozen
subject geometry.  No threshold, model ID, revision, token input, intervention,
or outcome rule changed.  Fake runtime and loader fixtures now reproduce the
pinned library's nested representation.

I also inspected every downstream field used by the runtime fingerprint on an
actual loaded local model, without calling the model: all 28 eager attention
modules, layer indices, bf16 parameters, CPU devices, EOS metadata, embedding
geometry, hook absence, and loader diagnostic shapes matched their frozen
expectations.  Thus this apparatus revision addresses the one observed metadata
location mismatch rather than broadening the protocol.

As before, the apparatus points to a newly named additive authorization file.
The earlier authorizations and both pre-forward error artifacts remain unchanged
in Git history.

## Validation before reauthorization

The loader-plus-runtime-preflight suites passed `19/19`.  The complete focused
v12 suite then passed `158/158`, with only the same two SWIG deprecation warnings.
No subject-model forward was performed by these tests or metadata inspections.
