_This conversation covers the final paper update, repo cleanup, rigorous
validation of the grafting apparatus, and a planned shift toward broader,
architecture-aware evidence for value grafting. The current critical gate is
resolving contradictory results between the original F1 evaluator and a newer
placebo-controlled probe before any cross-model work proceeds._

**Participants:** User, claude-opus-4-8, and claude-sonnet-5.

**Shipped State.** The report is finalized and pushed as “Value grafting:
recovering lost semantic continuity when a conversation is compacted,” with a
short prior-art-seeking forum introduction before the title, attribution moved
directly below the title, and README re-promoted from REPORT.md. The paper’s
primary claim is value grafting across compaction boundaries; K-only grafting is
negative for semantic-phrase targets, with identifier-shaped targets explicitly
left open but no further key investigation prioritized. The J-lens remains
secondary: free generation produced 0/43 decisive forks, 17 subtle leans, and 26
disconfirming cases, supporting a modest low-resolution-correlator framing.
Fable’s review also identified the need to disambiguate logit lens, tuned lens,
and J-lens terminology and keep lens discussion proportionate to its limited
payoff.

A proposed trajectory scan was reframed, on Fable’s advice, into a
teacher-forced placebo-controlled bound: score A/B/E on identical shared gold
tokens, use a fixed pre-divergence window, report paired bootstrap CIs, and
compare the true graft with a norm-matched shuffled-value placebo. However, the
resulting 27B and 30B runs revealed a serious implementation discrepancy. On
27B, the newer probe reported E−B ≈ +0.017 with a CI crossing zero, while
E−placebo was strongly positive. On 30B, K=12 yielded E−B ≈ −0.066 with a CI
excluding zero; full-window K=48 yielded E−B ≈ −0.049 with a CI crossing zero
and E−placebo also null. Recomputing the paper’s gap-closure metric from that
probe gave only 37% of cases helped, conflicting sharply with the original F1
results.

The original saved F1 data, using the same 43 cases and α=0.75, reproduces the
paper’s reported values exactly: referent +0.042 with 81% helped, sense +0.034
with 64% helped, and stance −0.31 with 38% helped. The likely issue is a bug or
teacher-forcing mismatch in the newer effect-bound implementation, but this is
not yet established. The live rerun of the original F1 evaluator on 30B is the
mandatory foundation gate. Until it completes and is compared carefully against
the newer probe, the effect-bound results—including the
placebo/content-specificity claim—must be treated as suspect, the newer probe
should be retired or repaired, and the cross-architecture sweep must remain
paused.

**Planned Research.** Once the trusted apparatus is confirmed, the intended
Phase 4 work is to strengthen the primary grafting contribution through both new
experiments and mining additional compelling examples from existing data.
Cross-architecture testing is framed as a bounded mechanistic falsification
probe, not a headline sweep. The design should use one shared, realistically
generated summary across models—Sonnet generated six fixed summaries,
approximately 250–295 words each, using the canonical compaction prompt—to avoid
confounding graft efficacy with model-specific summary quality. Fable’s required
controls include reporting the pre-graft gap, normalizing to fraction of
available gap closed, using a positive-direction smoke test, treating per-model
results as directional and aggregate results as inferential, and acknowledging
post-training differences as a residual confound.

The current intended pilot is three models—dense GQA, a non-Qwen GQA model, and
Gemma’s sliding-window architecture—before any broader sweep. Latest candidate
models include Qwen3.6-35B-A3B, Qwen3.6-27B, Gemma 4, Mistral Small 4,
GLM-4.7-Flash, OLMo-2-32B, and Qwen2.5-32B. MLA families such as native DeepSeek
and Kimi are a principled hard-bound for naive per-head value grafting because
their KV state is compressed into shared latent representations. Champion tuning
is a lower-priority, capacity-permitting follow-up: only successful models
should receive per-layer or potentially per-head tuning, with Fable consulted
first; resulting profiles may provide an architectural fingerprint of where
continuity information is stored.

**Repository State.** Thirty-four markdown files were moved via `git mv` into
`docs/` with `YYYY-MM-DD-HH-slug.md` prefixes based on each file’s first tracked
creation time, preserving history. Markdown from the obsolete
`referent_recovery_microtest/` was preserved there before the directory and
non-markdown contents were removed. Exactly nine operational documents remain
top-level: AGENTS.md, DECISIONS.md, FINDINGS.md, INCIDENTS.md, MASTER-PLAN.md,
README.md, REPORT.md, STATE.md, and writeup-guidelines.md. STATE, DECISIONS,
INCIDENTS, AGENTS, and MASTER-PLAN were refreshed with the current experiments,
process rules, Fable review requirements, naming conventions, and queued
champion work. Operational lessons include verifying the resolved model—not
merely that GPU work is occurring—using unique self-announcing output paths,
never suppressing deployment stderr, and distinguishing clean completion from
crashes.

The intended publication form is a polished, readable report suitable for a
technical forum or research discussion, with grafting as the center of gravity,
an honest prior-art invitation at the top, concise reproduction support to be
added later, and exploratory material clearly secondary.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `ab111f768643daa90`
- `a92d0bf59796733cc`
- `ad15b94ac85016f89`
- `aa07c06588f31f2bb`
- `ae21b449692df80dc`
- `a78435c9dfe43aec9`
- `a1552770f8153ea1a`
- `ab559852bf1111bbe`
- `a7c4e8b2d76c497f0`
