# Scientific-validation threat-model correction

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort

**Date:** 2026-07-11 (America/Toronto)

**Subject-model forward performed:** No

## Owner correction

The owner observed that the v12 validation work had begun to resemble a
cryptographically secure deployment scheme. Although that kind of engineering
can be interesting, it is disproportionate to this experiment's actual risks
and is consuming resources that should go toward running and understanding the
science.

The relevant threat model is **ordinary research failure**, especially:

- an implementation or position/cache bug;
- a comparison that does not measure the stated estimand;
- accidental treatment leakage or post-outcome redesign;
- a formula, aggregation, or interpretation error;
- running the wrong model/configuration;
- losing an expensive generated artifact or failing to record provenance;
- mistaking a narrow integration test for broad scientific evidence.

The threat model is **not** a malicious repository operator deliberately
constructing exotic Git histories, symlink attacks, fabricated tensor files,
or adversarially self-consistent evidence intended to fool the validator.

## Proportionate standard going forward

The remaining apparatus should establish scientific correctness with:

1. exact model, revision, tokenizer, dtype, and attention-backend checks;
2. focused unit tests for cache geometry, positions, interventions, and scoring;
3. a real local end-to-end subject-model integration run before paid execution;
4. raw saved outputs, generated text/token IDs, traces, and ordinary content
   hashes sufficient to detect accidental corruption and reproduce analysis;
5. an independent recomputation of outcomes, contrasts, gates, and stopping
   rules from raw values rather than trusting runner aggregates;
6. simple physical separation of pre-treatment eligibility and treatment work
   so accidental unblinding is difficult;
7. bounded paid canaries and prompt pod shutdown when computation is finished.

The independent analysis does not need to prove that an actively malicious
runner could not fabricate its inputs. It should catch plausible mistakes made
by us and make the scientific record inspectable.

## Disposition of work already completed

The pinned loader and frozen snapshot contract already committed at `0ed582b`
remain useful and may stay. They provide exact model/configuration checks and do
not impose meaningful runtime cost. No more time should be spent expanding
their adversarial Git/deployment defenses unless a concrete ordinary-research
failure requires it.

The technical runner and validator are to be simplified accordingly. Prefer a
lean, tested, end-to-end path to a real local model forward over exhaustive
zero-trust artifact proofs. Lossless tensors or elaborate receipt chains are
optional only when they clearly save expensive recomputation or directly test a
scientific claim; they are not prerequisites merely because stronger tamper
resistance is possible.

## Resource-allocation rule

When choosing between another defensive edge-case proof and executing a
representative model-facing integration test, prioritize the integration test
once the ordinary accidental-failure modes above are covered. Validation is a
means to trustworthy evidence, not the experiment's product.
