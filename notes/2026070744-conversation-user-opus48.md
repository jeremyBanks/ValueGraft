_This chunk covers the completion and shipping of the paper (K/V scoping and
lens-negative results folded in, Fable-reviewed, pushed to trunk), a series of
user-directed process and scope corrections (stop pursuing the K/V-keys
question, rebalance the paper toward the primary grafting contribution,
disambiguate overloaded "lens" terminology, brief Fable with explicit focus
during editing passes), a repository cleanup moving obsolete markdown into a
dated `docs/` convention, the design and buildout of a cross-architecture
generalization sweep, and — critically — the late discovery that a newly built
"effect-bound" probe contradicts the paper's original, already-published
gap-closure results on the same cases and model, putting the cross-architecture
work on hold pending a live reproduction check._

**Participants:** User and claude-opus-4-8.

**Paper finalization.** The free-generation lens probe (N=43) completed with a
clean negative: 0 FORK_TOWARD_A, 17 SUBTLE_LEAN, 26 DISCONFIRMING — no clean
internal fork toward the correct concept, even under the sharper free-generation
method. Because the disconfirming bucket was pre-registered, this result could
not be reframed as ambiguous support; it reinforced the paper's existing
"low-resolution corroborator, no vivid example" framing. This result, together
with the K/V scoped-negative finding (value is the operative axis; keys don't
help, uniform or per-layer; short-identifier targets left as an explicitly open
edge), was folded into REPORT.md (§10 K/V, §6 lens), reviewed by Fable for
honesty-of-scoping and readability, integrity-checked (28 exhibits preserved),
and pushed to trunk (8,100 words). An operational lesson recurred here: a
process that shows "GPU freed, proc dead" can indicate either a crash or a clean
completion — check whether the run reached its last case and whether the output
file exists before declaring failure.

**User-directed scope corrections.** The user directed stopping further K/V-keys
investigation (deemed sufficiently closed) and instead prioritizing the
lens-trajectory idea. Fable reframed the literal "trajectory plot" idea
(assessed as likely to invite over-narration of a muddy signal) into a more
rigorous alternative: a teacher-forced, placebo-controlled effect-bounding
experiment — comparing the true value graft against a shuffled/random-value
placebo graft on the same 43 cases, over a fixed pre-divergence window, with
bootstrap confidence intervals, pre-registering a null (CI includes zero) as an
acceptable outcome. The user approved running this (~$3, one pod).

More significantly, the user flagged that the paper had become
disproportionately focused on the lens material (a secondary theme that
underdelivered) at the expense of the grafting technique, which is the primary
novelty. This became "Phase 4": strengthen primary-grafting evidence via both
new experiments and mining additional exhibits from already-collected data (the
user clarified this was "both," not examples-instead-of-experiments), rebalance
the paper so grafting is the center of gravity, fix the overloaded "lens"
terminology (76 uses across three distinct instruments — logit lens, tuned lens,
J-lens — with 22 ambiguous bare "the lens" references confirmed by grep), and
brief Fable explicitly with this focus/north-star context during any conceptual
editing pass rather than letting it review direction-agnostically. "Terminology
consistency" was added as a new review dimension, since none of the prior
dimension-scoped Fable passes had caught the lens-overload issue.

**Repository cleanup.** At user direction, all top-level markdown deemed
one-time/obsolete was moved into `docs/` using a `YYYY-MM-DD-HH-slug` prefix
derived from each file's git first-commit date (tracked through renames),
leaving only nine agent-essential files at top level (AGENTS, DECISIONS,
FINDINGS, INCIDENTS, MASTER-PLAN, README, REPORT, STATE, writeup-guidelines).
The `referent_recovery_microtest/` directory was deleted after its markdown was
preserved in docs/ (confirmed to be a separate lane that never touched the paper
or plan). `jlens_boundary_probe/` code was held back from cleanup while still in
active use. docs/ grew from 30 to 64 files; the change was committed and pushed.

**Effect-bound experiment execution issues.** Deployment hit several bugs: a
suppressed-stderr rsync silently failed, leaving the pod with no code; a
`SC_HF_MODEL` environment override was ignored because `effect_bound_probe.py`
only respects an explicit `--model` CLI argument (defaulting to 27B), causing a
purported "30B" run to silently re-run 27B and overwrite the 27B output at a
shared default path. This produced two new standing rules: verify not just that
work is happening but that the _correct_ model loaded, and give every experiment
run a unique, self-announcing, date-stamped output path so re-runs cannot
clobber prior results.

**27B and 30B effect-bound results.** On 27B (N=43, K=12 pre-divergence window):
E−placebo = +0.445, CI [+0.286, +0.612] (excludes zero — aligned graft
decisively beats random-value placebo, confirming content-specificity); E−B =
+0.017, CI [−0.033, +0.065] (includes zero — null over this narrow window on the
weak model, consistent with 27B's known modest effect). On 30B at the corrected
model/config: initial K=12 result showed E−B negative (−0.066, CI excludes zero)
while E−placebo remained positive (+0.24); this was hypothesized to be an
artifact of the narrow window capturing only the answer preamble rather than the
content tokens carrying the benefit. A full-window (K=48) re-run was performed
to test this, but instead of resolving positive, results got _weaker_: E−B =
−0.049, CI [−0.11, +0.01] (null, slightly negative) and E−placebo = +0.128, CI
[−0.03, +0.29] (also now null).

**Critical discrepancy and current blocking state.** Recomputing the paper's own
gap-closure metric from this 30B effect-bound run's raw per-case data gave
results contradicting the published headline: only 37% of cases showed the graft
beating plain compaction (gap-closure mean −0.034 overall; sense −0.054/27%
helped; referent −0.013/48% helped) versus the paper's published 64% (sense) and
81% (referent) helped. Ruling out case-set and alpha differences (both scripts
used the same 43 cases and α=0.75 uniform), and after re-running the original
`gap_closure_cat.py` script's saved data and confirming it reproduces the
paper's published numbers exactly on identical inputs, the working diagnosis is
that the newer effect-bound probe's teacher-forcing implementation differs from
and is likely buggy relative to the original, paper-producing gap-closure code —
rather than the primary grafting effect itself being non-robust. This is being
treated as an open, unresolved risk, not a confirmed conclusion: a live re-run
of the original `gap_closure_cat.py` on 30B was launched to confirm reproduction
before drawing any conclusion, and the user explicitly directed that no further
work (including the cross-architecture sweep) proceed until this live
confirmation lands, since a measurement apparatus giving opposite answers on
identical inputs cannot be built upon. If the effect-bound probe is confirmed
buggy, its earlier 27B "content-specific, beats placebo" finding is also flagged
as suspect and will need re-validation through a corrected/trusted apparatus
(likely an α-sweep option added to `gap_closure_cat.py` itself rather than the
separate probe).

**Cross-architecture sweep design (staged, currently on hold).** A generic
value-graft harness (`cross_arch_probe.py`) was built to port the grafting
mechanism across different HF cache types, incorporating a positive-direction
smoke gate, reporting of pre-graft gap, a concrete detector for Gemma's
sliding-window cache failure mode (flags UNSUPPORTED rather than silently
grafting misaligned rows), and exclusion of MLA architectures (DeepSeek, Kimi
K2.x) as an architecturally-blocked case documented separately rather than
silently skipped. Fable identified a confound in the original design — that
different models write different-quality self-generated summaries, conflating
"grafting works less" with "this model lost less to recover" — fixed by holding
one shared, fixed summary text per conversation across all models. At user
suggestion, these fixed summaries were generated by Sonnet rather than a
frontier model (to keep summary quality within a realistic ~30B-tier ballpark)
and rather than a model from the sweep itself (to avoid home-field advantage);
Sonnet-generated summaries for six conversations were produced and committed as
`data/fixed_summaries.json`. The model list was refreshed via web research to
current (mid-2026) SOTA in the ~24–35B band: Qwen3.6-35B-A3B, Qwen3.6-27B,
Qwen2.5-32B, Gemma-4-27B, Mistral-Small-4, GLM-4.7-Flash, OLMo-2-32B, with
DeepSeek-V3/Kimi K2.x excluded as MLA. Planned run order was a cheap 3-model
pilot (Qwen2.5-32B, Mistral, Gemma-4) before the full sweep, framed by Fable as
a mechanistic falsification test where any outcome (works, attenuates, blocked)
is scientifically informative. This entire sweep is currently paused pending
resolution of the effect-bound/gap-closure discrepancy described above.

**Architecture-limitation discussion.** In response to a user question about why
cross-model attempts (Gemma, DeepSeek) previously seemed to fail, the operative
mechanistic explanation given was that value grafting requires per-position,
per-head value vectors in a standard KV cache; DeepSeek's Multi-head Latent
Attention (MLA) compresses KV into a shared low-rank latent with no directly
graftable per-head values (a likely hard architectural block), while Gemma's
sliding-window/global-attention interleaving may only attenuate rather than
block the effect if far-back grafted values fall outside a layer's attention
window. This was flagged as reconstructed from architectural knowledge requiring
Fable verification before being asserted in the paper, and became the seed for
the cross-architecture sweep.

**Champion-tuning extension (deferred, low priority).** The user proposed that
for models where cross-architecture grafting succeeds, a follow-on
per-layer/per-head "champion tuning" search could reveal an
architecture-dependent "fingerprint" of where graftable signal concentrates
(e.g., differing by sliding-window vs. dense vs. MoE design) — framed as an
interesting but non-priority extension, to be sequenced after the cross-arch
sweep and explicitly discussed with Fable at some point (timing unspecified). It
was recorded in STATE/MASTER-PLAN but not started. A refined protocol was later
folded in during the debugging discussion: per model, run a neutral-alpha (≈0.5)
baseline pass, then champion-tune per-layer on a small set, then a champion pass
— so each architecture is measured at its tuned best rather than a
possibly-unfair uniform dose, with the resulting champion profiles doubling as
the architectural-fingerprint comparison.

**External feedback and publication framing.** Feedback relayed from another
agent (an outside/second opinion, presented for consideration) recommended,
ahead of any wider posting (r/LocalLLaMA, EleutherAI, LessWrong/Discord
mentioned as possible venues, with the caveat that Discord silence means "nobody
bit" rather than "wrong," and that posting the repro command directly in any
post text is more effective than a bare repo link): keeping the repository as-is
except for adding a short "Reproduce the core result" section (setup command,
run command, expected numbers, pinned model/quant versions) and a one-line
disclaimer that the rest of the repo is exploratory. This was adopted; the
cross-arch harness was designed with the intent of doubling as this public repro
entry point once complete. Separately, at user request, Fable drafted a short
first-person, prior-art-seeking introductory blurb (framed as if posting to a
forum/chat room, ending with an invitation for input) to be placed at the very
top of REPORT.md before the title, separated by a horizontal rule, plus a title
refresh (the prior title was judged to over-index on the now-secondary lens
framing). Both were delivered by Fable, the intro was placed and committed, and
the user selected Fable's recommended new title, "Value grafting: recovering
lost semantic continuity when a conversation is compacted," which was applied
and re-promoted to README.

**Documentation upkeep.** At user request, STATE.md was comprehensively
rewritten to reflect current in-flight work, staged plans, and all standing
rules; DECISIONS.md (found to be the most stale, ~3 hours behind) was brought
current with the full run of consequential calls made this session;
INCIDENTS.md, AGENTS.md, and MASTER-PLAN.md were also confirmed current.
FINDINGS.md was noted as current through the 27B effect-bound and pending update
once the 30B/cross-arch results are confirmed.

**Handoff state at end of chunk.** No pods are idle-billing beyond the one
active pod. A live re-run of the original, paper-producing `gap_closure_cat.py`
on 30B is in progress as a gating check (expected ~5–10 minutes from launch) to
determine whether the published gap-closure headline (81% referent, 64% sense
helped) reproduces live. Until that result lands, all downstream work — the
cross-architecture pilot, the α-sweep/champion-tuning protocol, and any further
reliance on the effect-bound probe — is explicitly on hold per user instruction.
