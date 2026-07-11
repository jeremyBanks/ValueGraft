_This conversation covers the transition from an initially presumed clean null
to a fragile but independently testable positive effect from in-domain per-layer
tuning, and the decision to spend remaining compute credit on decisive
out-of-sample replication before finalizing the paper and notes._

**Participants:** User and claude-opus-4-8.

**Scientific state.** The naive scalar graft remains null: α=0.75 was −0.0017 on
the full 98, and no α-sweep value robustly helped. However, selecting layers on
41 in-domain tuning trajectories identified a 12-layer middle-region champion.
It scored +0.0117 versus baseline on 57 disjoint held-out trajectories (CI
[+0.0063,+0.0172]), with 44/57 improved. Fable’s integrated diagnostics on the
full 98 strengthened this: champion − baseline +0.0111, CI [+0.0069,+0.0155],
73/98, positive on all 200 half-splits; champion − scalar +0.0128, CI
[+0.0040,+0.0230]. The earlier characterization as a clean null was corrected
accordingly.

Caveats remain material: the initial 57-case comparison did not significantly
beat the scalar head-to-head; the evaluation pool favors graft arms; the
champion’s discrete next-action match was slightly negative; and the full-pool
result may still reflect selection or regime effects. A separate discrete metric
showed a weak +7 percentage-point signal (49%→56%, CI approximately
[+0.010,+0.143] at N=98), but it is generation-parsed, not resolve rate, and
should be treated as fragile pending more data. The intended paper framing is
therefore a small per-layer-tuned coding-task effect, not a robust general graft
benefit, pending replication.

**Handoff state.** All budget-passers were exhausted. The cleanest remaining
test is applying the fixed champion—selected without seeing the original 75
trajectories—to those original idx<213 cases as a fully out-of-sample N=75
replication. It was launched successfully on the warm pod with the champion
config active, alongside scalar, α-sweep, placebo, and discrete arms. At the
latest check, t0053/t0055 were around 24–25/75, the correct original disjoint
set was being scored, the live model was bfloat16 with brief summaries,
GPU/process health was normal, and balance was $8.86.

The cost estimate was corrected from Fable’s initial $2–3 to approximately 5
hours and $5–7 because the run has roughly ten arms plus generation at about 240
seconds per trajectory. This replication is the priority use of the remaining
credit; an additional out-of-band robustness check may not fit, though roughly
$2 could remain. A backed-off monitor is harvesting each cycle and should report
only completion, crash, or stall. After the replication, the planned sequence is
to value all data, spend any residual credit on the highest-value long-shot
robustness test if feasible, finalize the paper’s results and discussion, push,
regenerate notes, and create the separate speculative pedagogical note. Task #51
was added to the shared capture file and pushed in commit `9ea8e3e`; result data
were pushed in `6ba1a70`.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
