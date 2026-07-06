_This chunk covers the discovery that the SWE-bench Lite real-task track was
floor-limited for the 30B subject model even under oracle-mode retrieval, the
resulting pivot to full-capability-first validation gating and a new
chained-exercise task tier, and a parallel audit of serving-configuration
defects (disabled sampling, disabled native tool-calling) that may have been
silently handicapping every episode to date._

**Participants:** User and claude-fable-5.

**Instrumentation and secondary endpoints.** Following the earlier day's
incidents, the team added post-hoc enrichment of every scored row: SDK event
count (steps), wall-clock time, recall-probe text, self-termination flag,
timeout flag, task source label, and git diff lines/files touched as a
"breadcrumb" complexity proxy (real-task-only, since synthetic workspaces aren't
git repos). Four secondary endpoints were pre-registered before any real score
existed — steps-to-success, steps-burned-in-failure, compactions-per-episode,
and wall-clock (caveated) — explicitly to preserve a weaker-but-real signal if
the primary pass-rate comparison floors or ceilings. A later addition,
"promise-keeping" (does the agent execute verification steps it committed to
earlier in the episode), was pre-declared as a direct behavioral probe of
context integrity after forensic analysis of the first real-task failure (see
below).

**First real-task episode and forensic autopsy.** The first-ever completed
real-task episode was `flask-5063` under the Compacted arm at the original
("tier-0") aggressive compaction settings (9K threshold / 2.5K tail): the agent
worked ~30 minutes, self-terminated, submitted a patch, and failed the hidden
test — a clean within-budget failure, not a timeout or thrash. A background
forensic agent autopsied the full event log and found three compaction
fingerprints: the agent forgot a self-written todo ("test the updated routes
command") after it was evicted, re-derived the same wrong file path four times,
and produced two internally inconsistent re-derivations of its own fix plan.
Verdict was split: premature self-termination (skipping verification) was
compaction-shaped; the wrong-fix content (missing Flask's `host_matching` mode)
was capability-shaped, since that requirement lived in hidden tests never
explored.

**Compaction-severity correction ("humane tier").** The user flagged that
9K/2.5K settings left agents with ~2.9K tokens of working memory across
17–27K-token episodes (51 compacted calls in one case) — an overcorrection
relative to production practice. The team ran the cycle-length arithmetic (cycle
= threshold − 1000 − tail) against the user's target of 2–6 recompactions per
episode and converged on 12,000-token threshold / 6,000-token tail, informally
dubbed the "humane tier" (to be renamed "production-calibrated compaction" for
any external write-up, per the canonical-names/translation-table convention).
Real-task rows were moved to this tier; the tier-0 rows already scored were
retained as a labeled, non-headline stratum; synthetics kept their original
calibration on a dedicated lane.

**Adversarial summarizer defect (incident #15).** Questioning the summarization
prompt's provenance revealed the shim had been using a deliberately detail-free
summary prompt (built for an earlier mechanism-isolation experiment, explicitly
instructed not to include decisions/names/numbers) for every real-task
Compacted-arm episode — a self-inflicted strawman baseline, unlike any real
production condenser. Fixed by switching to a production-faithful summary prompt
(specifics, files, state, next steps, OpenHands-condenser-style) and re-running
the humane tier. This prompted a full adversarial pipeline audit
(`PIPELINE-AUDIT.md`), which found three CRITICAL issues (Original arm's hidden
compute penalty under a uniform clock — a bias against the team's own
hypothesis; missing config provenance in score artifacts; silent test-ID drops)
plus a miscounted-compaction-events bug, all fixed and deployed. New standing
rules: re-derive inherited parameters at every purpose boundary; audit the
baseline as rigorously as the intervention; instruments must self-report their
own configuration in the data they produce.

**Summary size and graft-value provenance clarified.** The production summary
prompt targets 300–500 words (~500–650 tokens) against a 6K tail, a ratio within
normal range for tail-weighting but toward the small end for summary size, with
compression ratio worsening as episodes lengthen (11:1 early, ~40:1 late in a
27K-token episode) — logged as a pre-declared sensitivity knob, not changed
mid-run. Graft mechanics were also pinned down precisely: grafted values come
from the tail and summary tokens' _natural, original_ encoding positions in the
one full-context prefill pass that also generates the summary — no second copy
or re-encoding of the source; only the destination (compacted) layout is freshly
encoded, with fresh keys and blended-in original values at aligned positions.

**Operational incidents (#13–#17).** A KV-cache VRAM leak (icache v1, later
confirmed as arithmetic: 61GB model + stored snapshot + per-call clone +
activations exceeding an 80GB card) killed shims repeatedly; a rebuilt v2
(in-place cache extension, no clone, VRAM predictor, headroom guard) was
validated and canary-deployed on one lane with a hard admission gate, catching a
pre-existing mis-wired tunnel in the process (incident #14). A dead-tunnel
outage stalled two of four lanes for hours undetected because monitoring
measured process liveness, not throughput; Codex (a second, read-only
cross-model reviewer) surfaced this via read-only log inspection, prompting
incident #17 and a redesign toward outcome-based monitoring: a per-lane (not
just global) throughput-floor alarm (100 minutes without a scored row triggers
by lane name), with link-level checks (tunnels, ports, processes) demoted to
diagnostics that fire only after the outcome alarm trips. A subsequent full
coverage-map exercise (INCIDENTS rule 12) proactively found two further gaps: a
silently-failing repo-sync path (accepted, with analysis reading the primary
store directly as a documented fallback) and no low-balance spend alarm (added).

**SWE-bench Lite floor discovered; oracle-mode pivot.** After the humane-tier
restart, the user directed a redesign to an "A-first gating" priority queue:
each candidate instance is validated once with the full-capability (Original, no
compaction) arm before any constrained arm is run on it, with validated
instances' four constrained arms immediately promoted ahead of remaining
validation work across all lanes (a strict dynamic priority queue:
validated-instance work first, then A-validation work) — because spending five
expensive constrained episodes on tasks not proven solvable at all was the
design flaw. The first several full-capability validations across two datasets
and three repos (SWE-bench Lite and the human-rated-easy "<15-minute fix" tier
of SWE-bench Verified) all failed, including on the smallest gold-patch
instances, escalating to a total of 5 straight full-capability failures by the
end of the chunk. In response the team: (1) switched to the standard "oracle"
retrieval setting (naming the fix's files in the prompt, without patch content)
to remove repo-navigation as a confound; (2) cut the per-episode timeout from 60
to 30 minutes; (3) reordered validation queues smallest-gold-patch-first; (4)
merged in the SWE-bench Verified easy tier for a multi-source interleaved
validation pool; (5) ran a single-sample Sonnet-subagent calibration probe with
zero project context on the pool's easiest instance, which solved it in 52
seconds and 9 tool calls — confirming the tasks themselves are tractable and the
30B subject model, not task difficulty, is the constraint. The Sonnet probe also
caught a scorer-killing bug (phantom test IDs aborting collection) before it
could poison newly merged Verified instances, adding a new rule that every
instance source's gold patch must pass as a scorer self-test before use.

**Scaffold-configuration audit.** Questioning why a 30B model was
underperforming its known capability, the team found and fixed two config
defects with the user's authorization to spend resources checking: (1)
`native_tool_calling=False` had been forcing the agent through prompt-text tool
conventions instead of Qwen3's trained function-calling format; (2) decoding was
hardcoded to greedy argmax in the shim's decode loop, so the configured
temperature setting never reached generation at all — contrary to Qwen3's model
card, which recommends against greedy decoding for long generations. Both are
logged as incidents (21, 21b) under the standing pattern of bring-up
conveniences never re-derived, with a new rule that serving must follow the
model card unless a documented deviation exists, and that
accepted-but-silently-ignored parameters are forbidden. Because the handicap was
symmetric across arms, prior arm comparisons remain internally valid, but all
solve rates to date are now understood as lower bounds. A controlled A/B (native
tool-calling + recommended sampling vs. current config) was launched on the
thrice-failed `pylint-7080` control instance and was pending its shim reload at
chunk's end.

**Chained-exercise task tier.** As a third diagnostic track and a fallback
difficulty rung, a new synthetic-but-realistic task tier was built from
Aider-polyglot-style Exercism exercises, chained 3–4 per session so context
grows across exercises and compaction bites mid-chain — giving a
position-resolved measure of whether recall degrades at later exercises, plus an
embedded recall probe. Self-tested (seeded, collision-checked,
fail-pre/pass-post validated) and dispatched via the existing driver; the
first-ever chain episode (`chain:s1`, Original arm, full capability) was in
flight at chunk's end, with two follower seeds queued behind it on revived
lanes.

**Model-swap ladder (documented, not yet executed).** In response to the
capability floor, the team catalogued a swap ladder without acting on it:
Qwen3-Coder-30B-A3B (same architecture/size/template, drop-in,
~$3 recalibration) as the cheapest first move; Qwen3-235B-A22B (~470GB, 6–8×A100 or multi-H200, ~$12–25/hr)
as a genuine capability jump requiring restructured statistics around fewer
episodes; GLM-4.5/4.6 (DeepSeek-flavored MoE, GQA + standard KV layout so
cache-surgery-compatible in principle, own chat-template family requiring fresh
adapter work) with GLM-4.5-Air (~106B-A12B, ~~2×A100) as the practical entry
point, positioned behind the Qwen-coder swap due to zero template cost for the
latter. A two-layer validation approach was defined: free tokenizer-only
template dry-runs (GLM-4.5-Air and GLM-4-9B tokenizers both passed the existing
context-construction dry-run without needing a bespoke branch) versus a
weights-based serving-path smoke (GLM-4-9B attempted locally, OOM'd from a
margin-arithmetic miscalculation squeezing it beside another shim; deferred
rather than retried, since the real gate is a representative smoke of
GLM-4.5-Air itself — nothing smaller in the current family would be
representative). The user pre-authorized, contingent on chain-tier and
config-A/B both failing to rescue the current model, a hard-capped two-pod-hour
(~~$6) GLM-4.5-Air trial: single full-capability validation run first, then
immediately proceeding into arm variations on success, under tightly supervised
(not economy-mode) monitoring for its duration only.

**Serving-stack performance, discussed but deferred.** In response to user
questions, the team estimated the custom shim (plain transformers forward
passes, single-threaded WSGI, full re-prefill every turn, no prefix caching, no
flash-attention, serial requests) costs roughly 3–10× wall-clock versus an
off-the-shelf server (vLLM/SGLang), while producing near-identical output
quality (kernel-noise-level differences) — the entire custom stack exists
because production servers' paged KV-cache layouts make the required
position-aligned value-grafting surgery close to impossible without deep plugin
work, and because all experimental arms must be served through the same stack
for validity (per the earlier audit's baseline-symmetry lesson). A harder
migration path (vLLM KV-connector plugin, days-to-weeks, version-brittle) was
noted as a plausible "deployment path" discussion for eventual write-up rather
than a near-term move. A lighter-weight in-stack optimization (flash-attention
kernels, compiled decode, async serving, plus the already-validated icache-v2)
estimated to recover 2–4× in roughly a day of work was committed as the first
infrastructure investment at the next phase boundary, not immediately.

**Process/monitoring corrections during this chunk.** Several corrections were
logged as recurring failure patterns rather than one-off mistakes: confusing
process liveness with work progress (fixed via per-lane, not just global,
throughput-floor alarms); reactively closing monitoring gaps only after the user
asked rather than via the standing coverage-map rule (executed once, in full,
surfacing two previously-unflagged gaps); missing an explicit two-sentences-old
user directive (priority-queue ordering) without any compaction-related excuse,
corrected immediately. The user's standing per-verdict reporting request (report
every SWE-bench validation result immediately until the first success, then
switch to consolidated reporting) was acknowledged and wired into the gate
monitor.

**State at chunk end.** Zero real-task episodes had ever passed (one clean
tier-0 failure, five straight full-capability oracle-mode validation failures
across two datasets/three repos). Three parallel diagnostic tracks were in
flight and expected to jointly determine the path forward within roughly the
next hour or two: the chain-tier capability smoke (`chain:s1` ~14+ minutes in),
the scaffold-config A/B (pending shim reload), and continuing easy-tier
SWE-bench validations (interleaved Lite + Verified, oracle mode, 30-minute cap).
The synthetic lane continued accumulating a guaranteed paired-arm table (~17+
new rows this chunk) as the floor deliverable regardless of the real-task
outcome. All incidents (numbered through 21b), decisions, and rules were
reported as captured in DECISIONS.md, INCIDENTS.md, and STATE.md/DAY-PLAN.md at
each step per the user's repeated documentation directive.
