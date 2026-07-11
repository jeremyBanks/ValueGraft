_This conversation covers a deep scientific audit of the ValueGraft project, a
clean Luna-based archive rebuild, and the design and attempted launch of a
bounded bf16 causal experiment. The current project direction is a same-position
gapped-cache assay after the first paid run correctly halted on a faulty
numerical gate before collecting semantic results._

**Participants:** User and gpt-5.6-sol-xhigh.

**Scientific verdict.** The project supports only a narrow negative result for
the original post-prefill, value-only graft; it does not settle whether coherent
generation-time state preserves information beyond summary text. The paper’s
final README was reviewed and found not ready for external sharing because it
misstated held-out provenance, treated foreign-rendered conversations as
model-native, reconstructed three compression conditions under the wrong
request, grafted summary plus retained-tail state while often describing
summary-only intervention, and overstated placebo specificity, null/equivalence,
provenance, and SWE confirmation claims. The strongest surviving positive is a
small continuous SWE likelihood signal on out-of-fitting trajectories, not
demonstrated coding-agent success; the planned 75-trajectory confirmation
actually completed only 45.

A full Luna recursive archive rebuild was completed from fresh chronological
source extraction, including final subagent assessments but excluding prompts,
reasoning, tools, and progress logs. It generated 55 conversation notes,
daily/monthly/archive rollups, exact model-based participant records, and
unlinked opaque conversation-source lists. Filenames contain only the
established user/model identifiers, with effort levels and opaque IDs excluded;
contributor roles/task labels do not belong in filenames. Referent filtering was
strengthened from lexical avoidance to making the underlying subject
non-identifiable. The archive audit reported 55/55 Luna notes with participant
lines and source footers, 269 preserved opaque IDs, no forbidden literal
matches, and 56 passing tests.

**Collaboration and attribution.** An append-only Sol–Claude/Opus research
dialogue was created for iterative planning, with watchdog instructions to avoid
silent stalls and to prompt or take over only after checking whether the other
process is genuinely inactive. Independent reviews converged on a
mechanism-first plan. Sol owns technical analysis, implementation, execution,
and evidentiary claims; Claude/Fable/Opus is intended to lead narrative
synthesis and paper prose, with the actual runtime identity recorded rather than
guessed. Future collaborative commits should use `Co-authored-by:` trailers;
local commits may be frequent, but pushes should be occasional or required for
external cloning/watchers.

**First paid attempt.** A secure RunPod A100 80 GB instance was launched only
after independent adversarial review and an explicit conditional go-ahead. The
exact audited commit was cloned, the rate was $1.39/hour, and the run stopped
before semantic scoring because the pre-frozen whole-model position-shift
diagnostic reported K/V drift far beyond its threshold. The result was correctly
classified as apparatus evidence, not hypothesis evidence. The pod was harvested
and terminated safely; cost was approximately $0.066, with zero active pods
afterward. The failure showed that independently shifted 48-layer bf16
trajectories are a poor hard gate for K-row transplantation: small
position-dependent rounding and propagation/MoE effects can produce large
late-layer tensor maxima, and even good K cosine can cause material
selected-token log-probability differences. The failed artifact and diagnostics
must remain preserved.

**Current experimental design.** Three independent reviews recommended replacing
packed post-RoPE K rerotation with a same-position gapped assay. The additive
amendment is frozen and the original preregistration remains unchanged. The
corrected primary design preserves each summary’s natural logical RoPE positions
across compaction while using contiguous physical cache indices; correct-history
K/V are copied bit-for-bit, with no rotation or dtype conversion. Fresh and
wrong-history arms receive the identical summary tokens at identical logical
positions. Wrong histories preserve system, role, structural, request, tail,
total length, and every registered token slot, replacing only predeclared
evicted content spans with frozen same-length donor/counterfactual content.
Summary-only intervention is primary; retained-tail transplantation is not
silently mixed into the interpretation.

Primary arms are full-history competence reference, fresh same-text gapped
restart, correct-history coherent K+V, and structurally matched wrong-history
coherent K+V. Optional secondary arms retain packed V-only, K-only, and
delta-matched placebo comparisons, but those results must not be pooled with the
gapped assay. The co-primary estimands are correct coherent state minus fresh
restart and correct coherent state minus wrong-history state; both must support
any history-specific-channel claim. Six conversations are evaluated first, with
a frozen extension to twelve only if competence/headroom and technical gates
pass; futility at six is allowed only when both primary contrasts are
nonpositive and the positive calibration control does not fire. No adaptive
tuning, scenario replacement, layer/head search, or post-outcome protocol change
is allowed.

Required gates include exact generated/replayed identity, bit-identical
source-state insertion, identical logical position schedules, separate
physical-cache and logical-RoPE counters, zero-gap equivalence, explicit
causal-mask validation, future-token mutation protection, exact wrong-history
structural checks, atomic checkpoint/resume, and complete provenance. The design
must recompute downstream tail/probe state per arm when measuring causal
operational effects; freezing a common tail would estimate only a direct
summary-row perturbation.

**Current handoff state.** The gapped amendment and pure token-construction
implementation are complete, with 11 focused tests passing. Runtime
implementation is now in progress, specifically separating logical positions
from physical cache indices and forking arms before assistant-close/tail
computation. No second paid run should occur until zero-gap equivalence and the
full 0.6B/runtime gates pass, the amendment and launch packet are reviewed by
the collaborating agent, and a fresh exact-commit preflight succeeds. The final
paper should be rewritten only after the corrected data are available, with Sol
providing the factual/methodological gate and Claude/Fable/Opus leading the
narrative draft and revision cycle.

## Conversation sources

- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f4f16-8cc4-7d10-b574-9f41c2e4d817`
- `019f4f16-73cb-7be3-acff-071c4f9df64b`
- `019f4f2d-27ee-7f10-a846-f24facc12a19`
- `019f4f2d-1b08-7b61-aa7c-8d55f26f2f5b`
- `019f4f90-ed25-7b92-bad2-fa28babdc90b`
- `019f4fb0-cc98-7c23-8911-c71cd0751761`
- `019f4fef-ec58-7d82-a595-482b94ff2e34`
