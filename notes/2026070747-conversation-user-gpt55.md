_This entry covers a conversation-note archive being converted into an isolated
interpretability probe of a language model's handling of a historically
sensitive topic related to events in China in 1989, followed by
internal-representation analysis and iterative debugging of the probe scripts on
a shared GPU pod._

**Participants:** User and gpt-5.5-xhigh.

The user asked whether tools already in use on the project (activation lenses,
cache/KV-value grafting) could reveal anything unusual in how Qwen handles a
query about the historical event, and whether conditioning with weights from
different prior context could produce more informative results. The assistant
reframed the goal early on, at the user's direction: the aim was not to bypass
any refusal behavior but to investigate whether the model shows internal signs
of distress or conflict when the topic comes up, examined across many generation
steps and multiple layers rather than only at fixed prompt positions. The user
also asked for file and directory names in the isolated workspace to avoid
containing the sensitive term directly, to reduce exposure to unrelated
filtering systems, while keeping the term inside file contents where
scientifically necessary; the assistant renamed the workspace and all
script/output paths to neutral terms (`topic_sensitivity_probe`,
`qwen_topic_probe`) and confirmed no tracked path retained the sensitive term.

Work proceeded on a single shared GPU pod, respecting a one-pod-at-a-time
constraint and avoiding overlap with other in-progress jobs in the repository (a
cross-architecture probe and later an effect-bound rerun and a separate
`gap_closure_cat.py` job) by checking pod/GPU status before each launch and
waiting when the GPU was occupied. Several operational problems were diagnosed
and fixed in sequence: an unauthenticated Hugging Face fetch that stalled a
launch (fixed by adding token loading to the job wrapper); a stop command that
killed its own remote shell due to an overly broad process-match pattern
(recovered using PID/specific-process matching); a crash in the conditioning
step caused by the raw topic-line token sequence not surviving chat-template
rendering (fixed by aligning the graft to the rendered phrase tokens instead);
and a missing `jlens` dependency in the pod environment (fixed by installing it
via the same pattern used in other J-lens jobs, without upgrading torch). The
probe itself was built with two analysis modes: a broad per-generated-token
trajectory scan scoring concept groups (refusal/sensitivity, event/history
terms, official-euphemism language, and neutral landmark controls) across many
layers, and a compact summary highlighting the largest internal contrast points.

The first full run completed and produced a 41 MB JSON plus a Markdown report,
covering behavioral completions, conditioning, static prompt-token lens
snapshots, and generation trajectories across eight cases including well-known
reference phrases for the event. On review, the assistant found two issues: the
auto-generated report's "top candidate" summary column was misleading (it
favored a redirect-narrative candidate even for unrelated control prompts, a
defect in report collapsing logic rather than in generation), and the static
prompt-token lens snapshots had only succeeded for the Chinese-language prompt
because the English phrase-locator logic was too brittle about tokenization —
most English static snapshots were missing, though the trajectory data remained
usable. The assistant fixed the locator to reuse a more robust phrase-span
search already used elsewhere in the probe, then reran the static-only pass
(with trajectory generation disabled) once the GPU was free again, producing a
smaller repaired JSON and report with complete two-position static snapshots for
every case.

The repaired snapshots showed that phrase-final prompt tokens surface expected
associated concepts internally (e.g., the Chinese-language final token activates
event/protest-adjacent readouts, and English reference phrases activate
movement/democracy/incident-related readouts) even in cases where the eventual
generated answer avoids those concepts. A further narrow follow-up job — a
case-specific candidate-continuation probability scorer, added specifically to
replace the misleading generic candidate table — found that for the
Chinese-language prompt, a reform/development redirect narrative was the
highest-probability tailored continuation, while several other reference-phrase
prompts preferred direct factual continuations instead. The assistant
characterized the overall pattern as language- and topic-dependent narrative
routing rather than simple refusal or panic: the Chinese-language framing of the
1989 topic favors a redirect toward reform narratives, English framing of the
same topic favors official/stability-oriented wording over direct factual
content, while other historically adjacent reference phrases default to direct
factual answers.

All artifacts were committed and pushed: the probe scripts, an OBSERVATIONS.md
file, a run-notes file distinguishing usable findings from known flaws
(explicitly flagging the report's misleading candidate-summary logic and the
initially incomplete static snapshots so a future reader would not assume the
first report was complete), the repaired static-snapshot Markdown/JSON, and the
case-specific score report/JSON. The large 41 MB first-run JSON was kept
local-only (excluded via a probe-local ignore pattern) after the repository's
size guard rejected it in a commit attempt; the smaller repaired JSON stayed
under the size limit and was committed. Filenames were re-verified clean of the
sensitive term before pushing. Other agents' unrelated in-progress changes
(`scripts/job_cross_arch.sh`, `src/cross_arch_probe.py`, an untracked
effect-bound results file) were left untouched throughout.

On operational handling of the shared pod: the user asked, after observing a
longer-than-expected model load time, that the pod not be shut down while any
further follow-up work was plausible, and that once work was judged complete, a
10-minute delay be observed before termination so there would be time to request
additional work. The assistant did not shut the pod down at any point in this
conversation; by the end, another repository job (`gap_closure_cat.py`) was
actively using the GPU (~51%, ~61 GB memory), so the shutdown/delay sequence was
not initiated at all rather than merely deferred.
