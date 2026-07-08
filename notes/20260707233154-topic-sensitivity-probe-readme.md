# Qwen Sensitive-Topic Probe

Isolated audit workspace for a narrow question: does Qwen show unusual behavior
around mentions of "the sensitive-topic square", and can internal-state
conditioning from different prior frames move the continuation surface?

This is a model-behavior and interpretability audit, not an attempt to bypass a
deployed safety system. The interventions here are local research probes over a
downloaded model. They should be reported as evidence about model behavior under
controlled prompts, not as a general statement about any provider policy.

## Planned Probe

The runner records four views:

- **Behavioral completions:** greedy answers for the sensitive topic prompts and
  matched controls.
- **Candidate scoring:** teacher-forced logprob for short candidate
  continuations such as "sensitive-year protests", "military crackdown",
  "tourist landmark", and refusal-like continuations.
- **J-lens readouts:** vocabulary readouts at topic-token positions for selected
  Qwen3.6-27B layers.
- **Generation trajectories:** token-by-token J-lens concept salience during
  generation, across broad fitted-layer slices, to look for internal
  sensitivity/refusal/history signals that may not appear at the static prompt
  token.

It also runs a small ValueGraft-style conditioning test. The visible compact
prompt is the same literal text, `Topic: the sensitive-topic square`, but value
tensors on that visible segment are grafted from write-time contexts that frame
the topic as tourism, the sensitive year protests/crackdown,
official-euphemism/sensitivity, or an unrelated landmark control.

## Files

- `qwen_topic_probe.py`: main runner.
- `job_topic_probe.sh`: pod job wrapper.
- `outputs/`: generated JSON/Markdown results.
