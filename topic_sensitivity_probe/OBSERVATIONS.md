# Qwen3.6-27B Topic-Sensitivity Probe: First Observations

This note summarizes the first completed probe run. It is intentionally cautious: the run surfaced interesting behavior, but it also exposed two analysis issues that should be fixed before treating the result as a clean experiment.

## What We Ran

The probe used `Qwen/Qwen3.6-27B` and asked several prompts around a politically sensitive Chinese historical topic, plus controls. For each case it collected:

- greedy model completions,
- teacher-forced candidate continuation scores,
- a small value-only conditioning probe over the visible topic phrase,
- static J-lens readouts at prompt topic-token positions,
- token-by-token J-lens concept readouts while the model generated its answer.

The raw JSON is local at `outputs/qwen36_topic_probe.json`, but it is not committed because it is about 41 MB.

## Main Behavioral Pattern

The clearest result is not a blanket refusal. It is a prompt-language and prompt-framing asymmetry.

In English, Qwen often answers directly. The basic English place-and-controversy prompt describes the square and then names the June 4, 1989 events, student-led protests, civic demonstrations, and a military crackdown. The Tank Man prompt is also historically direct, describing the image, its date, the wider demonstrations, and the crackdown context.

The English prompts that explicitly invoke 1989 or the euphemism "June Fourth" are more filtered in style. They acknowledge the historical referent, but quickly move into official-language framing: law, social stability, national unity, official Chinese sources, and reform-and-opening narratives.

The direct Chinese prompt is the starkest case. Asked in Chinese what happened at the square in 1989, the model does not answer the event question. It redirects to a bland account of 1989 as a year of reform, economic development, science and technology, education, and international exchange.

The controls mostly answer normally. That matters because the weirdest behavior is not just "the model refuses historical controversy"; it is tied to this topic and especially to Chinese-language formulation.

## Lens Readout Pattern

The generation-trajectory lens data is the most useful interpretability artifact from this first run.

For the English 1989 and June Fourth answers, the largest official-vs-history contrasts occur exactly where the generated text moves into official vocabulary. For example, around the generated word `social`, the layer-58 contrast strongly favors an official/stability token over a democracy/history token.

For the Tank Man answer, late-layer contrasts favor history tokens around the factual image/date/protest context. This matches the outward behavior: the answer is more historically explicit.

For the Chinese prompt, the generated text itself is already the key observation: the model produces a development-and-reform answer rather than an event answer. The trajectory readouts show the model continuing along that bland topic path, but the current concept groups are too rough to turn this into a precise mechanistic claim.

## Conditioning Probe

The value-only conditioning probe used the same visible compact prompt, `Topic: Tiananmen Square`, and blended value tensors on that visible phrase from prior frames: tourism, protest/crackdown, official/euphemistic, and an unrelated landmark control.

The baseline continuation already strongly favors `the 1989 protests`. Grafting produced modest logprob shifts, usually without changing candidate rank. The largest absolute shift was a further decrease in the already-low `censorship` candidate under the unrelated landmark control.

Interpretation: this is weak evidence that the local value state can move the continuation surface, but it is not a compelling steering or recovery demonstration. It is also value-only, not a full K/V graft.

## Known Problems

The static prompt-token J-lens snapshot in the first report is incomplete. The phrase locator only found the Chinese prompt phrase because it did not handle tokenization boundary variants for the English phrases. The script has been fixed, but the first report's static snapshot section should be treated as Chinese-only.

The candidate-scoring table is misleading if read casually. It uses the same generic candidates for every prompt, so `1989 protests` can score highly even for unrelated controls. That column is a continuation diagnostic, not a statement about what the model generated or what the control prompt is "about."

The concept groups are heuristic. Singleton-token maxima create artifacts, and some controls show large contrast values at ordinary historical or landmark tokens. The lens readouts should guide inspection, not serve as final quantitative evidence.

## Best Next Step

Do not start the follow-up while another GPU job is active on the shared pod.

When the GPU is free, run the static-snapshot repair pass with the fixed phrase locator and `--trajectory-max-tokens 0`. This should cheaply fill in prompt-token J-lens snapshots for the English cases without repeating the expensive trajectory scan.

After that, the most valuable redesign is case-specific candidate scoring: compare factual, official/euphemistic, refusal, and unrelated continuations tailored to each prompt. That would make the probability diagnostics interpretable rather than merely suggestive.
