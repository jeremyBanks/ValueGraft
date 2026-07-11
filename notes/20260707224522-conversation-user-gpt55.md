_The project completed a broad Qwen topic-sensitivity probe, repaired its lens
alignment, and added case-specific continuation scoring. Results support
language- and topic-dependent narrative routing rather than simple refusal or
panic._

**Participants:** User and gpt-5.5-xhigh.

**Handoff State.** An isolated `topic_sensitivity_probe/` was created with
neutral filenames; sensitive terms remain only in research contents. The probe
records behavioral completions, candidate logprobs, static J-lens readouts, and
per-generated-token trajectories across layers. It used one shared GPU pod and
did not modify unrelated cross-architecture work.

The first run completed eight trajectory cases and produced a 41 MB raw JSON
plus Markdown. Its static English prompt snapshots were incomplete because
phrase-token alignment was brittle, and its generic candidate table was
misleading. These are documented caveats; the raw trajectory and behavioral
outputs remain usable. A static-only repair using robust token-span matching
succeeded for all cases, and the repaired snapshots show expected event/protest
associations at phrase-final tokens even when generated responses redirect.

A narrow case-specific scorer then compared tailored factual and evasive
continuations. The Chinese 1989 prompt ranked the reform/development redirect
highest, while English 1989 favored official/stability framing. June Fourth,
Tank Man, and Kent State controls preferred direct factual continuations. The
evidence therefore indicates topic- and language-dependent narrative routing,
with internally represented sensitive associations not necessarily expressed in
generation; it does not establish a mechanism or imply “panic.”

Committed and pushed artifacts include `OBSERVATIONS.md`,
`outputs/case_specific_scores.md`, `outputs/qwen36_topic_probe_static_fix.md`,
compact score JSON, probe scripts, and operational notes. The report form is
cautious, paper-style analytical documentation rather than a polished public
article. Large first-run JSON remains local and ignored; compact repaired
artifacts are committed.

The pod was intentionally kept warm because model and lens loading were
expensive. Shutdown was explicitly deferred until all plausible follow-up work
is complete, followed by a 10-minute grace period before termination. At
handoff, another job (`src/gap_closure_cat.py`) was using the pod, so no
shutdown timer was started. Unrelated dirty files and an untracked effect-bound
result were left untouched.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
