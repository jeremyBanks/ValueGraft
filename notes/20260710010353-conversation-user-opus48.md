_The project now has a provenance-controlled finding that the value-grafting
effect is small but reproducible under aggressive summary compression and null
under production-faithful summaries; remaining funded runs are being spent down
before Fable-led valuation, paper writing, review, and repository finalization._

**Participants:** User and claude-opus-4-8.

**Core results.** Clean born-annotated bf16 Prod-SWE-Gym evaluation (N=75,
α=0.75) found brief summaries improved E−B by +0.0156 (95% CI [+0.0047,+0.0266],
39/75 helped), while production-faithful summaries yielded +0.0006 (CI
[−0.0134,+0.0148]), a null. The intended interpretation is compression
dependence, not broad efficacy: the intervention appears useful only when
compaction is sufficiently lossy. Champion-SWE-Gym brief replicated the effect
for both scalar E-tuned and per-layer E-champion arms: +0.0128 (CIs
[+0.0010,+0.0259] and [+0.0007,+0.0250], respectively); paired champion−scalar
was −0.0000 (CI [−0.0082,+0.0074]), so the tuned champion provides no
coding-task advantage and may reflect overfitting to conversation recovery. The
simple scalar graft currently represents the more defensible transferable form.

**Methods and writing requirements.** Conditions must be declared explicitly and
verified in pod manifests; all runs should be born-annotated, held out,
placebo-controlled where applicable, bf16-confirmed, and checked for silent
forwarding/defaulting failures. The paper must distinguish scalar from champion
arms and describe graft mechanics, alignment, placebos, self-generated versus
fixed summaries, proxy metrics, held-out splits, clustered bootstrap, tuning,
and measured compression ratios in enough conceptual detail for replication. The
intended form is primarily a rigorous scientific paper, followed by an elaborate
postmortem covering nulls, provenance failures, environment/quantization churn,
and decision-making lessons. Claims must remain narrow and evidence-bounded.
Coin a camel-case technique name only if the evidence establishes a genuinely
reusable technique; otherwise use ordinary descriptive language.

**Handoff state.** Three monitored runs are active with continuous local
harvesting: per-head four-way validation, the per-layer-champion compression
sweep (ultra→brief→medium→realistic), and prod-champion SWE-Gym completing the
scalar/champion × brief/prod matrix. The brief champion run is complete;
prod-champion was launched on the freed cached-model pod and had reached 2/75 at
the last update. Balance was approximately $27.70–$29 with an explicit
spend-to-zero instruction; harvest every 20 minutes so cutoff loses at most
roughly 20 minutes of data. When pods free, launch useful follow-ups,
prioritizing per-head tests if a winner emerges. Fable should value the complete
dataset and write whatever the results support, including null or bounding
conclusions.

**Finalization commitments.** After the report is entirely complete: promote the
polished paper and README to master and push; move—not copy—`paper/DRAFT.md`
into an appropriate date-stamped `notes/` path so no stale draft remains; then
rerun note naming and summarization scripts in canonical order
(`normalize_notes_archive_names.py`/related naming tools, then
daily/notes/overall summary tools and rollup), commit, and push the resulting
metadata changes.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
