_This chunk covers the ValueGraft project's estimator-bug resolution, a major
redesign of the cross-architecture generalization sweep (from a flat 7-model
list to an attention-geometry-driven, dense/MoE-paired, 14–16 model,
multi-vendor design), and an extended debugging saga around a
self-generated-summary alignment bug that ultimately traced to a wrong model
checkpoint rather than a real apparatus defect._

**Participants:** User and claude-opus-4-8.

The session opened with confirmation that the earlier F1 "reproduction crisis"
was resolved: the underlying value-grafting effect is real and deterministic
within a fixed environment, and the instability that had appeared to break the
paper's headline dissociation was entirely an artifact of the mean-of-ratio
gap-closure estimator, which is Cauchy-unstable with small denominators. The
corrected standard metric — raw E−B (logprob lift) plus %-helped plus bootstrap
confidence intervals, with the ratio retained only as a conditioned secondary
check — was applied retroactively and found to also correct the earlier
K/V-grafting conclusion: "keys actively hurt" was an estimator artifact, and the
settled finding is now "keys are neutral, value is the operative axis," a
conclusion the user confirmed should not be reopened or subjected to further
key-specific experiments. Bootstrapped confidence intervals on the primary
effect showed referent as a statistically significant effect (CI excluding
zero), sense as directionally positive but underpowered at existing sample
sizes, and stance as a genuine null — establishing statistical power (more
independent conversations, since conversations rather than probes are the
correct clustering unit for the bootstrap) as the concrete rationale for the
planned corpus expansion and model sweep.

A design consultation with Fable (run as a subagent, briefed with current facts
each time per a durable process rule) reframed the cross-architecture sweep
before any wide spend: because value grafting touches only the attention block's
W_V and MoE structure lives entirely in the FFN, "MoE flips the sign" was
flagged as a reviewer-vulnerable claim, and the real hypothesis was reframed as
"attention geometry (KV-head count, head-dim, GQA ratio, RoPE/QK-norm details)
predicts the sign of the graft effect," with model selection redesigned around
within-vendor dense/MoE pairs at matched scale to de-confound generation and
vendor from mechanism. This led to two independent dense/MoE anchor pairs —
Qwen3-30B-A3B (MoE) vs Qwen3-32B (dense), and newly discovered Gemma-4-26B-A4B
(MoE) vs Gemma-4-31B (dense) — plus a frozen 14–16 model list spanning nine
vendors (Qwen, Mistral, Google, AllenAI, Zhipu/GLM, OpenAI's gpt-oss,
Microsoft's Phi-4, 01.ai's Yi, NVIDIA's Nemotron), all verified to actually
exist on Hugging Face after an earlier hallucinated "gemma-4-27b" repo ID was
caught. Llama-2 was deliberately excluded despite representing the one available
full-multi-head-attention (non-GQA) architecture, on the reasoning that its
off-band scale and different era introduce more confounds than the
geometry-extreme point is worth; the user accepted dropping it. Fable also
proposed a cheap "champion" per-layer graft scan while models are resident
(fitting within a self-imposed 25%-time-overhead budget), reframed by a second
Fable consult from a throwaway fingerprint into a potential mechanism result:
because the uniform graft's effect is a nonlinear composition over depth, a
"rescue test" (grafting only the layer-regions that scan positive on an
otherwise-negative dense model) could causally demonstrate that sign is set by
the depth-profile of value-graftability rather than being an intrinsic per-model
property. This scan is explicitly flagged everywhere in project records as a
fast, non-exhaustive, non-comparable exploratory result, to be described in the
paper as a hint for future work rather than a real optimization.

Corpus work included doubling the conversation count (adding 27 new synthetic
conversations to reach 54, ~100 plants per category, closer to Fable's power
target), adding a "strong_prior" probe category designed to test whether the
graft shifts probability mass off a famous/prior meaning toward
conversation-specific meaning, and switching corpus generation off a locally-run
MLX model (identified mid-session as needlessly slow filler-text generation)
onto a parallel mix of foundation-API subagents (Sonnet, Opus, Fable) and
Codex/GPT-5.5, confirmed explicitly to the user as involving no self-hosted or
locally-run models for authoring. A quiet company name mentioned in one scenario
was swapped out of the source without drawing attention to it in commit
messages, per the user's request. The corpus-generation process itself surfaced
two operational failures: a single sequential subagent tasked with all 27 new
scenarios was mistakenly killed based on a misleadingly small output-file size
when it was in fact nearly complete, and a subsequent merge script's
unconditional cleanup step deleted the uncommitted batch files before the merge
was verified — both fully recovered without data loss (the generator scripts
reproduced byte-identical output), but both were captured as durable lessons:
shard multi-item generation work in parallel from the outset (both for speed
and, notably, for output diversity/quality, not just latency), and never take a
destructive action based on an unverified proxy signal.

The bulk of the later session was consumed by a debugging saga that began when
the redesigned harness's positive-control run (meant to validate the apparatus
before any wide spend) failed to reproduce the known +0.136 referent effect,
instead returning near-null-to-negative results across every category.
Root-cause investigation went through several incorrect turns — a mistaken
belief that a hardened token-alignment routine was performing risky "token
matching" against arbitrary conversation text (investigated and disproven: it
performs same-summary-to-itself positional alignment, later further simplified
into a direct exact-span map, verified byte-identical to the old method on
existing data), then a diagnosis that a fixed external (Sonnet-authored) summary
was suppressing the grafting effect (confirmed as real and reframed into a
durable mechanistic finding: the value-graft effect depends on the model having
generated its own summary — the graft recovers the model's own act of
compression, not information from a foreign summary of equivalent content — so
the sweep switched entirely to self-generated summaries), and then, once
self-generation was adopted, a hard crash caused by a `<think>` reasoning block
appearing in the model's self-generated summary, whose token span diverged
between the write-time and compacted renderings enough to trip the hardened
alignment's strict-match assertion. A subagent tasked with fixing this crash
spent roughly two hours running full re-verification passes without ever
executing the cheap decisive test it had been asked to run (a null self-graft,
which by construction must return ~0 and would instantly distinguish a plumbing
bug from a real content-loss), a pattern the user explicitly called out as
failing too slowly; the subagent was stopped and the debugging taken over
directly. Tracing the actual code revealed that an earlier "alignment hardening"
change (intended to remove the fuzzy difflib matcher in favor of a strict direct
map) had only been equivalence-tested against fixed-summary inputs, not the
self-generated/thinking-model production path, and had replaced the shared
alignment function used by both the trusted original apparatus and the
redesigned cross-architecture harness — meaning both tools broke identically
once self-generated summaries with reasoning blocks were introduced. Reverting
that one function to the original tolerant difflib-based alignment fixed the
crash, but subsequent runs still returned a negative/null effect. The eventual
resolution was that the debugging had been run against the wrong Qwen3-30B-A3B
checkpoint — the plain (thinking-enabled) variant rather than the
`-Instruct-2507` non-thinking checkpoint the original +0.156 result was measured
on — which explained every downstream symptom (the reasoning block, the
token-span mismatch, the alignment crash, and the negative numbers) as
consequences of testing an unintended model rather than any genuine defect in
the apparatus or the effect. Running the correct checkpoint reproduced the known
result cleanly: original conversations gave referent +0.125 (71% helped,
matching the prior +0.136), with sense positive and stance null exactly as
expected — confirming the underlying effect and apparatus are sound. A remaining
open issue is that the 42 newly authored conversations, when tested on the same
correct checkpoint, showed a substantially weaker referent effect (+0.009 versus
+0.125) despite retaining a comparable or larger pre-graft gap between full and
compacted contexts, indicating the new conversations are not a valid drop-in
doubling of the corpus but a structurally different (and currently
graft-resistant) set that dilutes rather than strengthens the signal; this was
left as an open question for the user, with the working recommendation being to
run the wide cross-architecture sweep on the original, validated 12-conversation
corpus while treating the new-conversation quality issue as a separate problem
to diagnose.

Operational and process decisions recorded during this stretch: an idle-pod
watchdog was built to check all RunPod pods every five minutes and alert on GPU
usage under 10% with no active job, layered on top of per-job watchers;
per-conversation progress logging was added to the cross-architecture harness,
which previously gave no signal between start and finish; a rule was adopted to
prefer cheap CPU/tokenizer-only smoke tests before any GPU/pod spend, since the
alignment bug that triggered hours of debugging could have been caught in
seconds locally; a stray feature branch created by a debugging subagent was
consolidated back onto trunk via fast-forward, with a hard rule recorded that
all work stays on trunk and any subagent that branches gets folded back; and a
standing rule was set to commit and push to trunk after every unit of work
rather than let it accumulate only locally. The Codex CLI was installed and
confirmed to run under the user's existing ChatGPT authentication, defaulting to
GPT-5.5 at extra-high reasoning effort, with the process rule that
Codex/GPT-5.5-xhigh review runs at least once per major, shareable paper
revision, invoked directly by the agent rather than requested from the user. The
user also indicated Fable may be given more authority to draft the paper's final
prose directly (given the right factual briefing) while the agent continues to
iterate, review, and make changes as needed. Cost tracking showed roughly
$1.00–1.30 per model run (dominated by download/load time rather than compute),
with the full expanded sweep estimated at
$30–50, comfortably within the reported ~$74–78 account balance; pods are torn
down between phases to avoid idle burn. At the point this chunk ends, the
positive control has passed on the correct model and original corpus, the wide
sweep is understood to be blocked only on deciding how to handle the diluted
new-conversation subset, and the user's standing directive — validate first,
then go "extra wide" without further hesitation — remains the active mandate for
the next step.
