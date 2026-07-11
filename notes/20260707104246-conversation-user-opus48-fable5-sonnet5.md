_This conversation covers the tau-benchmark validation, the discovery and
corroboration of a semantic-richness grafting effect at 30B, its scale and
mechanistic limits, and the subsequent expansion into varied experiments and a
substantially revised report._

**Participants:** User, claude-opus-4-8, claude-fable-5, and claude-sonnet-5.

**Experimental state.** Tau2 integration was completed end to end, including the
shim, scoring, keyless BM25 retrieval, and a capable GPT-4o-mini customer
simulator. The initial banking pilot produced only 300–1,500 tokens; the capable
simulator increased sessions to roughly 3,857 tokens but still did not approach
the nominal 12K compaction threshold. Lowering `compact_at` to 1,500 caused
compaction but evicted little meaningful context, and all A/B/graft/champion
arms scored 0.00 without separation. Tau-banking is therefore retired as
structurally unsuitable: its dialogues are either too short to compact naturally
or too short after forced compaction to provide a meaningful test. This is a
benchmark-scope limitation, not evidence against grafting. The
standard-benchmark agent benefit remains un demonstrated; tau-telecom was
identified as a possible but low-priority alternative.

**Main finding.** Re-analysis of existing semantic probes, using meaning
judgments rather than the previously inappropriate fact/honesty labels, produced
the central result. At 30B, Original/Compacted/graft performance was
approximately stance 96/93/96 (+2pp), sense ~100/46/58 (+12pp), and referent
~100/17/26 (+10pp). Grafting therefore recovered meaning specifically where
compaction caused damage, while adding little where summaries already preserved
the information. The sense and referent ceiling estimates are fragile because
the clean-ceiling samples are tiny (n=1 and n=2), so they are indicative rather
than precise. Strict scoring that counts PARTIAL as failure preserved the
pattern (+4pp stance, +9pp sense, +8pp referent). Teacher-forced exact-token
logprob independently agreed in direction: stance was null-to-negative (mean
−0.14; 39% helped), sense positive (+0.03; 64% helped), and referent positive
(+0.04; 81% helped). The smaller token-level effect is consistent with recovery
of semantic meaning rather than verbatim wording.

The qualitative exhibits are load-bearing evidence: compaction fabricated “Lena
Cho from TaxFlow Partners” and a fake Confluence path where the write-time arm
admitted uncertainty; a stance arm avoided forbidden countdown tactics without
citing the evicted rationale; a sense case resolved “the sandbox” to the correct
shared SQL environment; and a referent case recovered the real “tight jump-cut
editing with floating query captions” instead of inventing “The 3-Second Cut.”
Other compacted outputs confidently substituted a six-wheel drivetrain for
swerve modules and an incorrect cover-art scene. Full-strength grafting also
created a separate failure mode by binding a real “Echo” arc to the wrong
referent; the per-layer champion eliminated that instability, with the chain
table showing Original, Compacted, graft·75, and champion at 16/16 while
graft·1.0 fell to 12/16. Thus tuning solves graft-induced instability, not
demonstrated compaction damage.

**Boundaries and follow-up.** The semantic dissociation did not replicate at 4B:
stance showed the strongest apparent benefit, while sense became slightly
negative and referent weakened, consistent with previously observed
scale-dependent and sometimes inverted α response. A 30B logit-lens readout
showed internal concept movement—E exceeded B on roughly 68–77% of probes and
sat between B and A—but did not reproduce the category dissociation; near-floor
single-token logprobs make it partial mechanistic support only. A broader
portfolio was recorded in `EXPERIMENTS.md`: story-continuation/contradiction
tasks, long-document QA, cross-dependent chains, style and persona persistence,
gap-closure/KL metrics, contradiction and calibration measures, K-vs-V grafting,
anchor/recency ablations, and interpretability readouts. The story task was
implemented and self-tested across six ~9K-token premises with judge-based
contradiction scoring and automatic decoy checks. J-lens was retained as an
exploratory hypothesis generator, not a validator.

**Documentation and report.** The project adopted `FINDINGS.md` as the visible
home for results, with `STATE.md`, `HANDOFF.md`, `INCIDENTS.md`, and
`DECISIONS.md` maintained separately; failures and scope limits were to be
logged rather than narrated as success. The requested report form is a detailed,
accessible hybrid of blog post and academic paper: one coherent through-line,
concrete model-output exhibits, sufficient technical detail, restrained
justification, and a related-work/citations section. `REPORT.md` was revised
from a roughly 3,000-word abstract treatment into an exhibit-led report, then
checked by adversarial critics for accuracy, accessibility, gaps,
over-justification, and narrative flow. It now includes six verbatim exhibits,
citations covering compaction, KV editing, logit lenses, activation patching,
SAEs, benchmarks, and the J-lens (without inventing missing bibliographic
metadata), and narrows the novelty claim to value-state grafting across a
summarization boundary. Fable 5 and Opus 4.8 performed fresh-eyes prose passes
after Fable quota exhaustion was temporarily encountered; Opus was explicitly
credited as available and used. The final report was integrity-checked and
pushed to `origin/trunk` at commit `8ed5847`.

**Repository state.** Ten ephemeral pod-state files were removed from tracking
and added to `.gitignore`; local history was deliberately not rewritten, per the
standing rule. Scientific `results/` artifacts remained tracked as the audit
trail. The pod was terminated after the mechanistic run, leaving approximately
$37.95, no orphan pods, and a clean working tree.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `ab2f59490d13cf71a`
- `a8f5621f5a3c71f15`
- `ab109257e326cec24`
- `ae443de2036c83b04`
- `a16666a30d0db934e`
- `ac1b42454ef3e513e`
- `aecdb160687ac480a`
- `a13f8d1b12a331e23`
- `a59d0ebb7b0dcfef6`
- `a70dd61c7a7fe78d2`
- `a2e2ae63ceda27835`
- `a222f9137cba53fba`
- `a7d4abd35f6cc4c2a`
- `aabe5a7c7031dd33a`
- `aeb375c8cedacea5d`
- `a64beea35d29566e2`
