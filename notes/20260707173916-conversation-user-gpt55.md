_This conversation advanced the ValueGraft project from a failed synthetic
referent-recovery harness toward cheaper prospecting experiments, clarified
report provenance, and established the final evidence and review workflow. The
latest state includes an active 27B effect-bound probe and an updated
paper-style report plan._

**Participants:** User and gpt-5.5-xhigh.

**Research findings.** The large-model evidence supports ValueGraft primarily
for semantic interpretation damaged by compaction: on 30B, sense and referent
improved while stance was effectively unchanged; 4B did not cleanly replicate
this. Same-model 27B checks broadly supported sense/stance but left referent
flat. Subsequent K/V experiments and free-generation checks changed the
interpretation: values appear to be the operative axis for semantic-phrase
recovery, while key-only effects remain open only for short identifier-like
targets and are based on noisy 0.6B prospecting. J-lens is secondary aggregate
corroboration, not a source of vivid per-example exhibits; the free-divergence
lens result was 0 toward-A, 17 subtle, and 26 disconfirming.

**Synthetic harness work.** The original AST/Z85 referent-recovery proposal was
preserved, then cleaned of span artifacts, attributed near the top to Google
Gemini Pro 3.1, and replaced with a clearer offline-summary/teacher-forced
extraction design. Its isolated three-case Qwen3-0.6B/MPS test produced a real
full-context-over-summary gap, but every graft policy worsened the compacted
baseline, so the large harness should not be scaled unchanged. Follow-up
searches reframed the task around lower-entropy hidden relations rather than
arbitrary exact code. The strongest prospecting shapes were:

- Private policy labels mapped to familiar action phrases: a stable, broadly
  V-sensitive lane; 30/30 cases had full-context superiority and 28/30 improved
  under some graft.
- Low-entropy identifier/timezone transforms: noisier but more promising for
  targeted K-only effects; 15/16 had a real gap and 12/15 improved.
- Exact hidden strings/code, sense labels, and bug-fix labels were deprioritized
  as weak or overly brittle.

These results are exploratory and noisy, intended to map promising regions
before scaled validation.

**Artifacts and provenance.** The top-level `REPORT.md`/`README.md` is the
integrated report, not solely GPT-5.5’s work. The closest standalone
GPT-5.5/Fable paper-style draft is
`paper-working/valuegraft-synthesis/valuegraft-focused-draft.md`; related
GPT-5.5 work includes the J-lens reports and
`referent_recovery_microtest/policy_registry_followup_notes.md`. The current
attribution names Anthropic Claude Fable 5 and OpenAI GPT-5.5, with guidance
from Jeremy Banks and assistance from Claude Opus 4.8, Claude Sonnet 5, and
Gemini Pro 3.1. A provenance correction notes that some commits labeled Fable
were actually Opus after a model handoff.

**Workflow conventions.** The best current report should be promoted by copying
it over the relevant directory `README.md`; README files are publication-style
outputs and should not be edited directly. Agent-specific guidance belongs in
`AGENTS.md`. The final paper review must include three separate no-tool Fable
perspectives: a general assessment, structure/flow/readability suggestions, and
a differently prompted critical perspective, in addition to the existing review
work. Terminology must distinguish the primary K/V grafting intervention from
J-lens diagnostic analysis.

**Repository and operations.** Stale planning documents were archived under
`docs/` with git-creation-time prefixes; `.pod2_addr` and `.pod*_addr` are
ignored while local address files remain untracked. The tree was clean at the
latest refresh, with local `trunk` ahead of `origin/trunk` by two commits
(`d160e78`, `a68fab0`) updating the master plan; origin had separately promoted
`REPORT.md` to `README.md` and added the multi-perspective Fable review
requirement. The current report and state notes say the evidence phase is
substantially concluded and final paper updating is underway.

An active RunPod A100 must be treated as live: pod `w99udryqm0szp1`,
approximately $1.19/hour, running
`effect_bound_probe.py --model Qwen/Qwen3.6-27B --alpha 0.75 --window-k 12 --n-boot 10000`.
GPU utilization was about 87% with 55 GB memory used, logs had reached case
`c09`, and no `results/effect_bound/summary.json` existed yet. Do not start
competing jobs or terminate it without checking current status.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
