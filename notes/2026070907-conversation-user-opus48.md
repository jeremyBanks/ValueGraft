_This chunk covers a corrected reproduction verdict — the referent headline
collapses further on independent verification while a packed-KV honesty effect
confirms at bf16 — followed by the user reopening the champion/per-layer-tuned
graft configuration as an untested variable, and directing a structured
bf16-plus-4bit validation of that configuration to potentially enrich the
negative/bounding paper._

**Participants:** User and claude-opus-4-8.

**Reproduction verdict deepens: the referent headline collapses under a second,
independent check.** After the b0/b1 confirmatory re-render silently failed to
launch (an unattended ssh relaunch that never actually started, discovered only
when the pods sat idle-and-billing for 45 minutes), the agent harvested all
banked renders locally rather than blindly relaunching, then ran the disciplined
test Fable's holistic assessment had flagged as still missing: reproduce the
judged sense +12pp under clean current code on a fresh render of the _same_
c01–c12 conversations. It collapsed to +1.0pp (CI spanning zero). A same-judge
disentangle (re-scoring the original answers with the new stricter judge)
isolated the cause: judge-calibration explained only a small slice of the drop
(+12.0→+8.7), while the render itself explained the rest (+8.7→+1.0) —
confirming the collapse was render-fragility, not a judge artifact, and matching
the earlier referent-logprob failure pattern exactly. This meant both of the
project's positive headlines (referent recovery and judged-sense recovery) were
now shown to be render-fragile at n=12 and non-reproducing, a result recorded as
the most important finding of the session.

**Champion-scan build and fleet validated in parallel.** Per an explicit user
override ("run at least a two-model champion scan"), the previously-unused
`SC_CHAMPION_SCAN` capability was launched for the first time, catching and
fixing a real bug (empty-string env vars crashing `int()` parsing in forwarded
launcher config) before it could waste multiple pods. Once fixed, champion scans
completed and were harvested for Qwen3-30B-A3B (MoE, reusing banked renders —
validating the render-reuse economics), Qwen3-32B (dense sibling, fresh render
banked), and Mistral-Small-24B (fresh render banked), each yielding a 10-config
per-layer/region map. A parallel claims-ledger effort (`CLAIMS.md`, later moved
from a gitignored path to a committed one) was built to recompute every headline
number from disk against its source file, recompute command, and commit — and it
caught a real overclaim: the F2 finding's "38/48 = 79% accurate" evicted-fact
claim did not reproduce (true recall was 0/24 at both 4B and 30B); the correct,
verified claim was that packed write-time KV converts fabrication into admission
rather than restoring recall.

**Fable's holistic and native-render-premise assessments (written to notes/
files, the newly adopted practice for durable Fable output) surfaced two
corrections.** First, both marquee positive results had been measured only under
a deliberately terse "brief" summary condition designed to handicap the
compacted baseline — no positive result existed under a production-faithful
summary, which Fable ranked as a bigger risk than anything then in flight.
Second, the standing claim that the graft requires natively self-rendered
replies (not just a self-generated summary) was only partly supported: the
self-generated-summary requirement was cleanly verified, but the
natively-rendered-reply requirement rested on a cherry-picked low draw from
render-to-render noise comparable in size to the effect itself, with a second
on-disk native draw (+0.116, CI excluding zero) essentially reproducing the
authored condition. The recommended fix was to present self-rendering as a
defensible design choice rather than a proven requirement. The user endorsed
pursuing a production-faithful summary arm as top priority while keeping the
brief condition as a deliberately extreme complementary probe, and asked that
cheap cross-model SWE-Gym data collection piggyback on already-warm pods given
the real cost of reloading models between runs — a point the user judged Fable
had under-weighted.

**Attempted production-faithful SWE-Gym run hit a sustained
environment/provisioning failure.** A missing `pandas` dependency on the
champion pods (undocumented, present on only one of seven job-launch scripts)
triggered an extended, ultimately unproductive debugging cycle involving stale
logs, an interpreter mismatch, and repeated ssh truncation on a flaky pod; the
pod was eventually terminated rather than continuing to fight it. The user
flagged this as a recurring, systemic pattern rather than a one-off and asked
for an independent subagent history-mining pass (with an explicit instruction to
weight its findings but not over-index on them) instead of further self-directed
debugging. That retrospective (`notes/2026070974`) identified a genuine root
cause: pod-side dependency installation is copy-pasted and drifted across seven
job scripts with no single source of truth, including a live contradiction where
two scripts pin `transformers==5.0.*` while a third scripts explicitly avoids
that version because it breaks pod weight loading. It proposed a shared
`pod_env.sh` sourced by every job plus a fail-closed dependency check in
preflight — recorded as a pending, not-yet-implemented recommendation.

**A user-mandated, unhoped-for statistical gate: the proposed honesty headline
was made to clear the same bar that had just killed the sense-recovery claim,
before any paper drafting proceeded.** Conversation-clustered bootstrap CIs
showed the packed-KV fabrication-reduction effect was robust — decoy fabrication
−66.7pp [+45.8, +87.5], evicted-fact fabrication −62.5pp [+41.7, +83.3] — with
the write-time-KV-specific contribution (beyond mere packed layout) also
independently significant. This effect was judged commensurately strong enough
to lead the paper, contingent on stating its caveats plainly (the arm is a
packed-keys+values "cousin" of the paper's named value-only method, not the
method itself; n=24/arm over 12 clusters; not independently
render-replication-tested; scope-limited to mid-task agentic compaction).

**A precision/provenance crisis followed, triggered by direct user objection
("we never reproduced our core results at 16-bit").** Investigation confirmed
both core behavioral pipelines (recovery and the proposed honesty headline) had
run only on the local MLX 4-bit backend, never at bf16 in scored form — a fact a
paper draft had briefly and incorrectly labeled as bf16, caught by an
adversarial review pass before anything shipped. The user's separate objection
that the honesty arm ("H-pack") might not even represent the paper's actual
named method (value-only ValueGraft) was also confirmed correct: H-pack retains
write-time keys in addition to values and uses a packed layout — a materially
different, coupled-KV intervention, not the value-only aligned graft the paper
is framed around. A full-authority deep audit (paper construction halted in the
interim) established the ground-truth state: no result on disk satisfied both
"bf16" and "value-only ValueGraft" as a positive; the one available bf16
value-only result (SWE-Gym, +0.0156 nats) was initially assessed as too
thin/possibly render-null to carry a paper, and the placebo-controlled bf16
value-only recovery test (`effect_bound`) was a preregistered null on two
models. The audit's initial verdict was that the paper had no valid positive
cornerstone and should be reframed as a negative/bounding result.

**User-directed reframing: an honest postmortem, explicitly empowered as an
acceptable primary deliverable.** The user explicitly authorized treating the
entire effort as a candid negative/postmortem paper if that was the honest
conclusion — describing the project as improvised, phone-directed "vibe
research" conducted without dedicated time to build real understanding, at
roughly $300 of compute spent — and asked that any postmortem be specific and
detailed (grounded in the actual daily summaries/transcripts, not motivational
or self-congratulatory framing) while paraphrasing rather than directly quoting
the user's own more informal or rambling statements. On resuming with full
context (including this explicit permission), Fable's second pass corrected the
prior audit's SWE-Gym assessment: recomputed properly, it was a genuine,
correctly-precision (bf16), correctly-armed (value-only) significant positive —
CI [+0.005, +0.027] excluding zero, changing the greedy output on 49/75 tasks —
not render-null as first assessed. Fable's recommendation, adopted and executed,
was a paper combining (1) the bf16 placebo-controlled bounding/null result for
value-only recovery, (2) the one small genuine bf16 value-only positive
(SWE-Gym) as a caveated real effect under the brief-summary condition, (3) the
packed-KV admission effect reported honestly as supporting-only (not the named
method), and (4) a dry, specific process postmortem of the failure patterns.
This draft was written to `paper/DRAFT.md`, verified against the ledger (no
stubs, no accidental overclaims, no quoted user language), and committed — not
yet promoted to README or pushed. A same-day follow-up scored the
previously-unscored bf16 honesty answers and confirmed the packed-KV admission
effect replicates at full precision (decoy +62.5pp, evicted +20.8pp, both
CI-significant) — a real, supporting-only, bf16-confirmed finding layered onto
the otherwise negative/bounding paper.

**Explicit repo-publishing boundary set and recorded.** The user authorized the
agent to promote the finished, fully-reviewed paper to README and push to the
repository autonomously once genuinely confident, without further sign-off — but
drew a hard line that nothing leaves the repository (no external posting, no HF
community blog, etc.) without the user's direct involvement. This was captured
durably in the relevant protocol doc so it survives context resets.

**New development, this chunk: the user reopened the champion/per-layer-tuned
graft configuration as a specific, unresolved variable that may still salvage a
positive result.** The user recalled that early testing of per-layer
("champion") tuning appeared to make a large difference and asked whether it
could still be tested cheaply on the pods. Investigation confirmed this is a
live gap: the bf16 placebo-controlled null (`effect_bound`) was run only under
the default global α=0.75 configuration, not the champion/per-layer-tuned
configs; a separate `tune_eval_30b_bf16` result showed tuned configs
(top16-layer α=1, position-tuned "posslots" α=1) consistently outperforming
baseline at bf16 (posslots: +0.054 mean logprob improvement, 6/6 wins, vs. the
default config's +0.033, 4/6 wins) — but these tuned-config runs were never
placebo-controlled, so it remains unknown whether the larger tuned effect is
content-specific or just an amplified version of the generic "any graft helps a
little" placebo effect already characterized in the default-config null. The
single decisive, low-cost experiment identified is to run the champion-tuned
config through the same placebo-controlled design as `effect_bound`.

The user rejected an earlier claim that no 4-bit-quantized checkpoint suitable
for pod deployment exists, noting official quantized checkpoints are readily
downloadable from Hugging Face, and directed a structured validation of the
champion-tuning question along two precision tracks: bf16 on pods (as already
scoped) and 4-bit — using an official quantized checkpoint if one exists, or the
closest available otherwise — run on pods in parallel, prioritizing 4-bit first
since it should be cheap and multiple instances may be runnable in parallel or
kept warm for fast iteration; the 4-bit pod run requires its own separate
champion/per-layer tuning pass rather than reusing the local-MLX 4-bit results.
The user asked that Fable be consulted only for minor refinements to this plan,
not a wholesale redirection, and directed the agent to proceed with executing
substantially this plan to determine whether it can enrich the existing
negative/bounding result. This is queued as the next concrete action; no pods
have yet been launched for it as of the end of this chunk.
