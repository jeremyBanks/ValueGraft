_This conversation reset the project’s experimental vocabulary and controls
around a two-parameter ValueGraft framework, while separating valid historical
evidence from confounded or infrastructurally invalid live-agent results. It
also produced and repeatedly refined a paper-style synthesis draft with clearer
terminology, citations, provenance boundaries, and human-readability edits._

**Participants:** User and gpt-5.5-xhigh.

**Experimental framing.** The current conceptual family is parameterized by:

`K(αK) = (1 − αK) Kfresh + αK Kwrite-time,rerotated`\
`V(αV) = (1 − αV) Vfresh + αV Vwrite-time`

Plain compaction is `(αK, αV)=(0,0)`; V-only Graft varies `αV` with `αK=0`;
K-only Graft varies `αK` with `αV=0`; coupled KV Graft varies both. An
unsubscripted α means the same policy applies to both sides unless the
experiment explicitly defines a one-sided parameter. Scalar, per-layer,
per-head, token-specific, tuned, and extrapolating policies are all valid; the
requirement is that the intended experimental axis be isolated.

Keys and values are separate learned projections of the same contextual hidden
state. Keys determine attention matching/routing and include RoPE positional
transformation; values provide the payload retrieved after attention.
Re-rotating a write-time key changes its position while preserving its
context-derived content; it is not equivalent to freshly recomputing the key
from compact context. The main scientific comparison must therefore hold summary
text, tail, layout, positions, prompt, decoding, and values fixed while varying
key policy, or hold key policy fixed while varying value policy.

**Current experiments.** In-flight `E`, `E:a0.75`, `E:a1.0`, and `E:cfg=layers`
rows remain unchanged for provenance and are interpreted as V-only Graft
experiments: fresh keys (`αK=0`) with scalar or layer-tuned write-time value
mixing (`αV`). Existing identifiers, result directories, and code paths must not
be renamed during active runs. Future K-only/KV runs require a matched
production-shaped context; the old implementation should not be treated as a
valid isolated key-policy comparison.

The historical `H-pack`/`B-min-pack` pair is auxiliary evidence only. It used a
summary-only context and varied fresh summary KV versus write-time summary KV,
changing K and V together while also differing from the current
production-shaped summary-plus-tail design. It must remain identifiable by
historical result labels for provenance but must not appear as a current method,
taxonomy branch, or clean K-versus-KV result.

**Live coding evidence.** The clean run uses synthetic harness-authored tasks,
not a standard benchmark: `t1` is a pricing/cart task, `t2` a fetcher/API retry
task, and `t3` a harder multi-constraint billing task. `A` is full-context
oracle, `B` plain summary compaction, and the `E` arms are V-only Graft
variants. SWE-bench-Lite/standard-task integration was pre-registered
conditionally; synthetic and standard rows must be stratified, and
already-scored synthetic rows remain auditable.

The clean-run process suffered multiple infrastructure incidents: cross-arm
session leakage, incorrect endpoint modes, broken probes, duplicate/shared run
directories, unbounded incremental KV-cache VRAM growth, dead shims creating
false score files, and a `TASKMOD` transition error. These were documented,
quarantined, or fixed; trusted rows require both `score.json` and
`E1_AGENT_DONE`, while invalid episodes receive no score. Earlier contaminated
results were moved to quarantine and are not evidence.

At the latest refresh, the committed synthetic snapshot contained only 10
trusted rows (`A 2/2`, `B 2/3`, `E:a0.75 4/4`, `E:a1.0 1/1`); scratchpad
filtering yielded 14 trusted synthetic rows (`A 4/4`, `B 2/3`, `E:a0.75 4/4`,
`E:a1.0 3/3`). There were no trusted `cfg=layers` or standard-task rows. This
weakly suggests that compaction can hurt solvability and that V-only Graft may
help, but sample sizes are far too small for an arm conclusion. The live
infrastructure was also intermittently unhealthy, with most matrix ports
refusing, resetting, or timing out.

**Repository and writing state.** The nomenclature note was committed through
`37eb677`; the controlled two-alpha reframing through `3ad5015`; the separate
paper-style draft was created and pushed as `a1ec803`, then revised as `ae56911`
and `7204230`. The latest readability pass was committed and pushed as
`5dc9dfa`. The draft now uses ValueGraft/αK/αV as the active taxonomy, treats
the summary-only comparison as explicitly historical auxiliary evidence,
restores richer related-work citations, and follows the established editorial
style: direct paper prose, fewer defensive “not X but Y” constructions, no
invented method labels, precise controls, and clear separation of current,
historical, and future experiments.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
