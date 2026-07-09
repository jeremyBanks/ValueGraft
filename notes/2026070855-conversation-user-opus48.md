_This chunk covers the resolution of the reproduction-crisis arc (estimator bug,
K/V reframe, positive-control failure and the resulting own-summary mechanistic
finding), Fable's redesign of the cross-architecture sweep around an
attention-geometry-predicts-sign hypothesis rather than a naive MoE-vs-dense
claim, a substantial build phase (corpus expansion, harness hardening,
champion-layer scan), the assembly and iterative widening of the final
16-model/9-vendor sweep list, and the discovery mid-launch of a new alignment
bug that the pre-flight gate correctly caught before any wide spend._

**Participants:** User and claude-opus-4-8.

**Resolution of the reproduction crisis.** A live re-run of the original F1 code
did not cleanly reproduce saved F1 numbers, and an initial hypothesis blaming
Mixture-of-Experts routing nondeterminism was wrong (the user pushed back,
correctly, that this looked like a bug rather than inherent noise). Two
same-hardware runs came back byte-identical, proving the apparatus is fully
deterministic within a fixed environment; the earlier instability came from
comparing runs across a transformers version drift (5.0.x vs 5.13.0) and from an
unstable metric. On the user's instruction to consult Fable rather than continue
generating ad hoc hypotheses, Fable diagnosed the actual defect from data
already on disk, at no GPU cost: the paper's headline used a mean-of-ratio
estimator, (E−B)/(A−B), which is Cauchy-unstable when denominators are near
zero. On robust metrics (raw E−B, %-helped, median ratio) both runs agreed and
reproduced the paper's qualitative dissociation. The correction was reframed as
a numbers/estimator fix, not a retraction: the primary effect is real, referent
significant (95% CI excludes zero), sense positive but underpowered, stance null
as claimed. This robust-metric standard (raw E−B + %-helped + bootstrap CI,
mean-ratio retired) was locked in as the standard for all subsequent work,
including a parallel finding that the same broken estimator had also produced an
overstated "keys actively hurt" claim in the earlier key/value ablation; on the
robust metric, keys are neutral rather than harmful, and "value is the operative
axis" stands as a closed, settled question requiring no further investigation.

A related scare arose when a description of the alignment mechanism ("token
matching," "spurious matches to the conversation") alarmed the user, who
believed that approach had been discarded long ago. Reading the actual code
resolved this: the mechanism is a positional 1:1 map within regions (summary
compared to itself, tail compared to itself), not fuzzy matching against
arbitrary conversation text, and it produced a 100%, zero-dropped alignment on
the tested case. Per the user's engineering instinct that a fuzzy difflib-based
matcher was unnecessary complexity for what should be a direct span lookup, it
was replaced with a direct exact-span map, verified byte-for-byte equivalent to
the old method on all 12 existing conversations, and changed to raise loudly
rather than silently degrading on any future tokenization mismatch — a design
choice that later proved decisive (see below).

**The positive-control failure and the mechanistic finding.** Running the
cross-architecture harness on Qwen3-30B-A3B (the model with a known positive
effect, ~+0.156) using a fixed, externally supplied (Sonnet-authored) summary
came back null (+0.004), failing to reproduce the known result. An isolation
test — same model and harness, but with the model's own self-generated summary
instead of the fixed one — reproduced the known positive almost exactly (+0.136
referent, matching F1's +0.156 within noise). This resolved a design flaw into a
genuine mechanistic finding: the value-graft effect depends on the model having
generated its own summary at the compaction boundary, not merely on a
semantically equivalent summary being present. The write-time state that matters
is the model's own act of summarization, not generic full-context information.
This reframed the earlier Qwen2.5-32B "architecture reversal" result as still
valid (it was measured on the trusted, self-gen apparatus) while invalidating
any sweep numbers gathered on fixed summaries; the sweep design was switched to
self-generated summaries throughout, and this became a durable project learning
(recorded to persistent memory: validate with a positive control before trusting
a pipeline, mine on-disk data before spending, isolate one variable at a time,
read code rather than narrate from a plausible-sounding story).

**Fable's redesign of the sweep.** Before any wide spend, Fable was consulted on
the overall sweep design, per the user's directive that this was major
decision-making warranting outside input, with the explicit condition that Fable
be briefed on all current facts each time (since findings had evolved
substantially through the session). Fable's key structural catch: value vectors
are computed by the attention block, and Mixture-of-Experts routing lives in the
feed-forward block, so "MoE flips the sign" was flagged as a probable
reviewer-fatal overclaim — the true driver is more likely attention geometry
(KV-head count, head dimension, GQA ratio, QK-norm, RoPE treatment), not MoE per
se. This reframed the marquee experiment from "does the effect generalize" to
"what attention property predicts the sign of the effect," a stronger and more
specific claim. Consequences: a same-generation dense Qwen3 model was added to
de-confound MoE from generation/vendor effects; the design shifted from
wide-and-shallow toward a mix of deep anchor models (with placebo grafts, an
identity/integrity check, an alpha dose-response sweep, and full
raw-trace/hyperparameter capture) plus a shallower breadth tier; and per-model
attention hyperparameters are now logged specifically to support a downstream
regression of effect sign on architecture. In a later consult, Fable proposed a
further "grand insight": because the uniform-alpha graft composes nonlinearly
over depth (later layers consume values grafted into earlier ones), a per-layer
"champion" scan — measuring which depth regions are graftable — could be
promoted from a descriptive side figure into a causal mechanism test via a
"rescue test": grafting only the positive-scanning regions on an architecture
that grafts negative overall, to see if the net effect flips sign. If it does,
that causally demonstrates the sign is a depth-composition effect rather than an
intrinsic model property. This scan also yields, essentially for free, a
per-layer write-vs-read value-alignment cosine (a candidate geometric
explanation for each region's effect) and a region="all" sanity check that must
reproduce the uniform-alpha number. Per explicit user instruction, this
champion-scan work is to be reported prominently as a quick, non-exhaustive,
weak/tentative scan — not a real optimization or a valid cross-model comparison
— with that framing constraint recorded so it cannot be inflated in the eventual
paper.

**Build phase.** With pods shut down to avoid burn during design work, the team
executed a CPU/local build phase in parallel with Fable consults: (1) corpus
augmentation, initially via the original local MLX-based conversation-generation
pipeline (`compose.py`), which the user flagged as an unnecessarily slow,
low-value use of a local model to write filler dialogue; this was corrected by
switching corpus authoring to a fast parallel mix of Sonnet, Opus, Fable, and
Codex subagents, cutting corpus generation from ~90 minutes to minutes and
improving text quality. The corpus was expanded from 12 to 27 conversations
(adding a new "strong_prior" probe category testing whether the graft shifts
probability mass off a famous prior meaning toward the true in-conversation
meaning, using invented codenames), verified 100% clean (correct plant placement
relative to the compaction boundary, verbatim plant text, balanced categories),
and frozen. Later in the session the user requested doubling the corpus again
(to ~54 conversations, ~100 plants/category) for additional statistical power,
on the reasoning — clarified to the user when asked — that conversations, not
probes, are the true unit of statistical power under cluster-bootstrap
resampling, so power is bought by adding independent conversations rather than
adding probes to existing ones; this second expansion (c28–c54) was in progress
at the end of the covered period. (2) Harness hardening: placebo graft, an
identity/integrity check, alpha dose-response, full raw-trace capture, per-model
attention-hyperparameter logging (feeding the planned sign-regression), and a
conversation-level cluster bootstrap (verified to give meaningfully wider, more
honest CIs than probe-level bootstrap on correlated data) were added; later,
corpus-dependent additions (multi-probe averaging across paraphrased probes per
plant, the strong-prior signed mass-shift readout, and the fractional-depth
champion-layer scan with the rescue-test and value-alignment-cosine upgrades)
were also completed. A separate self-gen toggle was added to the sweep job so it
can never silently regress to the fixed-summary design that was shown to
suppress the effect. It was also explicitly decided and recorded that the
reproduction/entry-point script (a planned unified, filterable, idempotent CLI
wrapping the trusted probes) must have no dependency on the local MLX
corpus-generation tooling — the frozen corpus is a committed data artifact, and
replication runs entirely on transformers on Linux/GPU.

**Model set assembly.** The final wide-sweep model list was built iteratively in
direct response to user questions probing for gaps, each time verified for real
existence on Hugging Face before being trusted (a discipline adopted after an
earlier incident where a probe silently ran the wrong model, and reinforced when
a hallucinated "gemma-4-27b-it" repo ID was caught by direct lookup). Starting
from a Qwen-heavy list of 3 anchors, the set was expanded, per user prompts, to
include cross-vendor coverage (OpenAI's gpt-oss-20b, Microsoft's phi-4, 01.ai's
Yi-1.5-34B, NVIDIA's Nemotron-49B), major-version pairs where architecturally
significant (Mistral-Small 2501 vs 2506 as a training-checkpoint control, and
critically Gemma 3 vs the newly-released Gemma 4, which ships both a dense 31B
and an MoE 26B-A4B variant — providing a second, independent within-vendor
dense/MoE de-confound alongside the existing Qwen3 one). Llama-2 was explicitly
discussed and excluded per Fable's original recommendation (scale-band and era
confounds outweigh its value as the only pure multi-head-attention/GQA-ratio-1
anchor), though the user was informed of that specific tradeoff. The final
frozen set stands at 16 models across 9 vendors, organized into deep anchor
tiers (two independent dense/MoE pairs plus a cross-generation dense point) and
shallower breadth/replication/geometry tiers; anchors get full controls
(placebo, alpha sweep, champion scan), all models get self-generated summaries
and the champion scan within a per-model compute budget the user set at roughly
25% overhead above the base run. Per user direction, architecturally
risky/high-uncertainty models (e.g., Gemma's sliding-window HybridCache, which
may be fundamentally graft-incompatible) are to be run in a second wave after
the more confident models have banked results, rather than being dropped or run
first.

**Cost and operations.** Community RunPod GPU pods were used throughout; one pod
died from a disk-full condition after a 200GB container disk filled with cached
model weights (incident logged), prompting a switch to 400GB pods, pre-download
disk-headroom checks, post-run weight eviction, and a standing 5-minute idle-pod
watchdog (checking GPU utilization and job-process state across all pods) added
at the user's request to catch idle billing. Per-model cost was estimated at
roughly $1.00–1.30 (dominated by download/load, not compute), putting the full
16-model sweep with the doubled corpus in the rough range of $30–50 total
against an account balance near $75–80; the user confirmed this was an
acceptable and even underestimated scope given the value of the results.

**Fail-faster and the alignment bug catch.** Following an explicit user
instruction to seek faster failure signals given the diligence was already
paying off, a pre-flight "gate" run (validating the full self-gen pipeline
against the known-positive Qwen3-30B result before any wide launch) failed
rather than passing, with the JSON output empty and the underlying exception
initially swallowed. Diagnosis traced this to the hardened direct-span alignment
(the same replacement introduced earlier for the difflib-based aligner) raising
on a genuine mismatch: the write-time summary span (899 tokens) did not match
the compacted-context summary region (538 tokens) for a self-generated summary —
a case that had only been equivalence-verified against fixed summaries, not
against the self-gen path that is now the production configuration. This is
treated as a real, structural bug requiring root-cause repair (not a fallback or
skip), being fixed by a dedicated agent with live, cached-model pod access,
gated on actually reproducing the known +0.14 positive-control result before
being trusted. Per the user's fail-faster request, a CPU/tokenizer-only smoke
test for this class of alignment bug is being added as a mandatory pre-flight
check (catching such mismatches in seconds, with no GPU or pod needed), and the
general principle — layer verification cheapest-and-earliest, tokenizer/CPU
checks before any pod spend, a tiny fast model before the target large model,
big-model positive control last — was recorded as standing practice. The wide
16-model launch remains held until this fix is verified and the gate passes
again; the corpus-doubling authoring work continues in parallel and is not
blocked by the pod-side fix.

**Other explicit decisions and commitments.** The user asked that a specific
placeholder codename ("Palantir") used in a strong-prior probe scenario be
replaced quietly, without drawing attention to the change in commit messages;
this was done (renamed, source-verified at zero remaining occurrences, and
confirmed absent from all rendered conversations before freezing). The user gave
a standing overnight mandate for autonomous, disciplined operation while asleep:
finish the build, run a confidence gate before any wide spend, then go wide and
fast once confidence is established (since prior stages had taken longer than
expected), consult Fable on major decisions with current facts each time, keep
tracking docs current in near-real time, and route through the full existing
review stack before publishing. On review process, the user specified that a
review from Codex (GPT-5.5, extra-high reasoning effort) must be obtained at
least once per major paper revision, not repeatedly; after confusion over a
pre-existing GUI app versus the CLI, the Codex CLI was installed and confirmed
to be authenticated via existing ChatGPT login with defaults already set to
GPT-5.5/extra-high effort, invoked as `codex exec -s read-only`, and it was
agreed the agent runs this itself without asking the user to do so. The user
also proposed, and had recorded, that Fable may be given primary responsibility
for writing the paper's final prose (still subject to the existing
iteration/review cycle and further edits), rather than only reviewing drafts.
The `jlens_boundary_probe/` legacy codebase was confirmed to be unused by any
active pipeline code (a one-way dependency the other direction) and left in
place at the user's discretion ("I'll handle it"), with a low-priority
end-of-project cleanup task recorded for eventual archival of unused code/data.
A monitor-notification debounce policy (starting at 2 minutes, backing off to a
24-minute maximum, silent on routine progress, alerting only on
completion/error/state-change) remains in effect throughout.
