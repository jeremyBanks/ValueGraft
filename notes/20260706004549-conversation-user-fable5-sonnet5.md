_This conversation covers the transition from benchmark validation to
production-shaped OpenHands coding experiments, with stronger monitoring,
tuning, controls, and handoff procedures. The intended output is a rigorous,
blog-style write-up supported by paper-like experimental documentation, explicit
caveats, and reproducible provenance._

**Participants:** User, claude-fable-5, and claude-sonnet-5.

**Scientific conclusions.** The 30B bf16 slot-mask result is promising but
remains exploratory: 57 validation-selected slots improved a 10-conversation
holdout (+0.038 nats, 10/10) over global α=.75 (+.024, 7/10), pending the
wrong-conversation contamination guard. It must not replace the simple global
blend as the primary method. The 30B profile is not meaningfully rank-1 (top
component 37% variance; R²≈.31), and head means explain only 1.4%, so layer×head
factorization is mainly a portable calibration interface for heterogeneous
models such as Gemma, not a validated 30B solution. Signed coefficients are
geometrically defensible because correlated intervention directions may require
cancellation, but the 4B/4-bit holdout favored clamped factored fitting (+.0074,
10/10) over signed fitting (+.0036, 6/10); this is only a small-model
possibility-space result, not evidence against signed fits at 30B. The write-up
should explain both the geometric rationale and the statistical/ nonlinear
reasons for bounded, regularized experimentation.

The extrapolation sweep confirmed a smooth decline beyond α=.75, with no cliff
through α=1–3; α=.75 remains the peak. Corrupted caches usually remain fluent
but semantically wrong: fluent amnesia, confident denial, and donor-conversation
topic bleed; token-level degeneration was rare and mainly observed at 4B under
full shuffled grafts. Corruption is frozen, while newly generated cache entries
are coherent, so continuation may stabilize onto a wrong narrative rather than
recover the true one. The Pokémon material is an expanded, quarantined
illustration only—not benchmark evidence, not an ongoing evaluation, and never
to be placed in the results tree or evidence lists. Actual sense-level evidence
comes from controlled micro-experiments and real-trace results.

**Evidence grading and benchmark correction.** Small/local 4-bit experiments are
hypothesis generators: positive results require 30B bf16 validation, and
small-model nulls must never prune the larger-model hypothesis space. Stage 1 is
complete and judged: 350 questions/2,100 verdicts, full context 52.5% versus
compacted arms roughly 4–7%, with no meaningful fact-retrieval accuracy
separation among intervention arms and no fabrication increase from grafting.
This confirms that value grafting cannot restore evicted precise facts; its
likely benefit is behavioral/continuation-level rather than literal fact
recovery. The large six-arm run was unnecessarily expensive because local
evidence predicted this null; future cloud replications should use minimum
viable n and interim stopping gates. LoCoMo-as-QA was consequently
deprioritized/dropped unless needed only for benchmark breadth.

The “honesty” suite tests 24 decoy facts and 48 evicted facts across compacted
conditions, scoring CORRECT/FABRICATED/ADMITTED. At 4-bit, standard compaction
showed roughly 19:5 fabrication-to-admission on decoys while packed/grafted
conditions mostly admitted uncertainty; a 30B bf16 replication is running.
Subagent judge reports consistently found that baseline/full-context A answers
recalled answerable facts most often, while B/E-tuned/packed variants usually
admitted lack of history; fabrication clustered around confident wrong numbers,
ordering, preference-grounding substitutions, and answers that began with an
admission before drifting into a guess. These are useful behavioral categories
and scoring caveats.

**OpenHands coding program.** The central experiment is now production-shaped:
OpenHands tasks communicate with a pod-hosted OpenAI-compatible shim; the shim
performs the same summary generation, tail construction, alignment, and α=.75
value graft as the validated runners. Conditions include A/full context, B/plain
compaction, E/grafted compaction, α dose curves including negative and
extrapolated values, shuffled-graft controls, different compaction thresholds,
per-layer tuning, slot-mask tuning, and a dissociation probe testing whether
agents follow constraints they cannot explicitly recite. Objective success is
pytest resolution, supplemented by repeated failed commands, evicted-file
rereads, steps-to-completion, compaction events, token counts, grafted
positions, and recall-probe answers. E1 began with 12 synthetic task runs; the
first clean round was B 0/2 versus E 2/2, but n=2 is explicitly insufficient and
subsequent rounds are required. E2 is intended to expand to roughly 15–30 real
SWE-Gym-style tasks depending on plumbing and early signal, with continued
effort even if a particular approach is abandoned.

The shim is stateless per request and initially re-compacted every
over-threshold call, making B/E tasks about twice as slow as A. A verified cache
now freezes the compaction boundary, reuses byte-identical summaries and
write-time KV snapshots, and rolls forward only when the tail regrows; local
testing caught and fixed a negative hit-window bug, producing the required
MISS,HIT,HIT signature with nonzero grafts. The current E1 queue remains on old
semantics for consistency; the cache deploy is a clean transition for later
matrix/E2 runs. All changes must be logged so comparison sets never mix
semantics.

**Tuning and validation policy.** The tuning ladder is B → single α → coarse
per-layer values → slot mask, with champion/challenger promotion only after
repeated fresh-task wins. At least 25% of later runs remain baseline guard runs
after promotion. Exploration uses seeds s1–s49; sealed seeds s50–s99 are held
out. After exploration, a confirmatory side-by-side compares no compaction,
plain summary, single-α graft, and the champion on roughly 15–20 fresh tasks per
arm. The final evaluation uses additional task templates authored sight-unseen
and is run once on data untouched by tuning. Final reporting should show normal
compaction, plain-summary, single-value, per-layer, and promoted configurations,
with baseline checks retained throughout. Conventional summary settings are the
primary comparison; alternate summary sizes/thresholds are secondary exploratory
factors.

**Operations and safeguards.** The unified health loop pulls all pods every five
minutes, alerts after two consecutive pull failures, checks dead processes
without completion markers and newly appended traceback/error lines, and
performs heavier stall/traceback/milestone checks every sixth pass. Job scripts
now require syntax validation, process verification, per-item progress,
exit-code capture, and count-validated `DONE(n)`/`INCOMPLETE(n)` markers.
Watchers impose roughly 40–50 minute first-item/hard deadlines; no indefinite
waits are acceptable. Results sync directly into `results/` with a maximum loss
window of five minutes. Datasets, weights, credentials, environments, and
oversized artifacts remain uncommitted; a local pre-commit hook blocks staged
files over 4 MB. Hooks must be recreated by successors from AGENTS.md.

The cloud program is paced rather than abandoned: gates control spending and
escalation, not effort. Specific approaches may be killed when they fail, but
the overall objective—determining whether compaction-time value grafting
improves real coding agents—continues through redesign, local tests, alternate
metrics, and second opinions. If progress stalls or repeated obstacles
accumulate, invoke an available Codex agent of the strongest model/effort
through the CLI for an independent repository-wide assessment; this is
documented as a strongly encouraged stuck-protocol. Concurrent agents may create
isolated notes and commit them without touching shared indexes or staging; the
primary agent must avoid broad `git add` operations.

**Current handoff state.** Stage 1 is complete. The active work includes the 30B
bf16 honesty suite, 4B bf16 calibration/tuning, the OpenHands shim matrix,
wildcard creative experiments, and the H200 standard-protocol LongMemEval
mini-run after an A100-80GB capacity wall: 110–120K-token 30B bf16 haystacks do
not fit on one A100, so the H200 is required. Gemma exploration is parked after
six failures, including a genuine hybrid-attention scoring/shape limitation; it
has a precise resume point and must not consume more resources until the
layer-type-aware path exists. The matrix was widened toward seven pods; capacity
retries may add another lane. At the latest handoff, four shim lanes were
health-checked and actively running, with expected accumulation around 30–35
coding results/hour and first post-transition scores expected within roughly ten
minutes. The handoff contract uses alternating work shifts with approximately
30-minute sleep intervals, and every shift must verify that scores are actually
landing before yielding control.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `aa1459742100f433a`
- `a569bd67ff6a7c2e8`
- `a97ff119c367a8cf4`
- `a5a4bdd7ff0b231b2`
- `a32deccd64612a97a`
- `abb80a89bb2d0c28b`
- `ac66236cfd4b284f2`
- `ada2761cd4eb7426f`
- `acc303325eced7636`
- `a3f2a314b7b9ea536`
- `a42c0846e64a42f16`
- `adee6c6f44d67fbcc`
- `aa42c098ae8b83f21`
- `a541aaca9611cc094`
