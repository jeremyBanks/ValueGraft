_This conversation covers the completed Luna-based archive rebuild and a
progressively hardened experimental program for testing history-conditioned KV
state, including the current v10 technical validation and its strict separation
from semantic conclusions._

**Participants:** User and gpt-5.6-sol-xhigh.

**Archive State.** The transcript archive was rebuilt cleanly with Luna
(`gpt-5.6-luna`) as the default provider, while retaining Claude and Codex
provider/model flags. The unified workflow covers extraction, summarization,
filename normalization, and recursive daily/monthly/yearly/archive rollups.
Codex uses ephemeral sessions; Claude uses nonpersistent sessions;
summary-worker self-ingestion is explicitly excluded. Only final subagent
assessments are included, with prompts, reasoning, tools, and progress messages
removed and duplicates deduplicated. Participant/model provenance is required at
every layer, filenames contain only compact user/model identifiers, and opaque
parent/subagent IDs appear only in unlinked `Conversation sources` footers.
Referent filtering is defined as subject-level de-identification rather than
literal word avoidance. The final Luna rebuild passed 56 tests, regenerated 55
conversation notes and all rollups, preserved 269 source IDs, and passed naming,
provenance, participant, and exclusion audits.

The repository contains separate ordered notes for the empirical audit,
methodology audit, holistic consultation, final conclusions and budget plan, and
final paper review. Collaborative commits now use `Co-authored-by:` trailers.
The append-only Sol–Claude/Opus dialogue and its watcher protocol remain the
coordination record; agents are expected to re-arm monitoring after crashes and
to leave explicit handoffs rather than waiting indefinitely. The user’s
preferred division is for Sol/GPT-5.6 to lead technical analysis and execution,
with Fable when available—or accurately identified Opus when routing
occurs—providing prose, synthesis, and independent challenge. Sol retains final
methodological authority.

**Scientific Verdict.** The original project supports a qualified negative
result about training-free, post-prefill, exact-token, value-only grafting—not a
general negative result about useful latent continuity across summarization.
MEMENTO’s primary-source result is directly relevant: on Qwen3-8B, retaining the
original summary KV state outperformed identical-text restart/re-prefill on
AIME24 (66.1% versus 50.8%). The repository’s defensible contribution is the
failure analysis of a naive old-V/fresh-K intervention and the design of
stronger causal tests.

The original synthetic evidence is constrained by major provenance and design
corrections:

- Held-out conversation bodies were foreign-rendered: c07–c12 used
  Qwen3-4B/4-bit and c13–c24 were Claude-authored; none were Qwen3-30B-native.
  Source strata reverse the observed effect, so the pooled null is heterogeneous
  rather than a homogeneous native-model test.
- Three of four compression conditions reconstructed the cache under the wrong
  summary request; claims about compression severity or flatness must be
  withdrawn.
- The intervention grafted summary and retained-tail regions, not summary values
  alone.
- Main synthetic “write-time” states were reconstructed by teacher-forced
  prefill, unlike the SWE path, which retained generation snapshots.
- Existing Gaussian and shuffled placebos were not matched to the treatment
  delta and did not establish semantic content specificity.
- SWE-Gym results are offline teacher-action matching, not coding-agent success.
  The tuned continuous likelihood signal is positive on available out-of-fitting
  rows, but behavioral action-match evidence is null out of sample, the
  selected-map placebo was absent, and the planned 75-trajectory confirmation
  completed only 45.

The final paper review found the README unusually clear and honest but not
externally shareable without correcting provenance, compression,
intervention-region, placebo, selection, reproducibility, literature, and
claim-strength issues. The intended document form is a careful paper-style
negative evaluation with a concise research-audit section, related work,
explicit limitations, reproducibility guidance, and narrowly scoped future
experiments. Fable/Opus is intended to lead eventual prose drafting, with Sol
auditing every factual and methodological claim.

**Research Program.** The agreed progression is:

- At $0, reuse existing artifacts for strict SWE action rescoring,
  source/provenance stratification, and an audit table; then implement a
  same-text, same-position causal apparatus on the local Mac.
- At roughly $50–$60, prioritize a bounded mechanism experiment rather than live
  agents.
- Larger $100 and $200 plans are conditional, using sequential capability,
  technical, and futility gates. No budget is an automatic spending target.
- A real-agent study is deferred until a model-native common-prefix evaluation
  demonstrates both full-context capability and measurable compaction damage.
  The intended design forks identical workspace/transcript snapshots into
  full-context, ordinary-compaction, and one treatment arm, scoring executed
  tests or resolved patches rather than imported teacher actions.

The technical experiment was progressively amended after adversarial review.
Earlier packed post-RoPE K rerotation was abandoned after a paid 30B bf16 run
failed before semantic scoring: same-token split versus contiguous execution
diverged substantially under SDPA, with K, V, and logits discrepancies far above
the frozen threshold. Local reproduction showed SDPA query-shape sensitivity,
while eager attention was exact on the corresponding CPU fixture; explicit masks
did not rescue SDPA. A separate failed delta-matched placebo also showed bf16
quantization/moment mismatch, so `G_delta` was retired rather than given relaxed
thresholds.

The current design is v10, an eager-attention, position-preserving, gapped-cache
assay. It preserves logical RoPE positions while using contiguous physical cache
storage, copies coherent summary K/V without rerotation, and compares:

- `A_full`: full-history reference;
- `G_fresh`: identical summary text freshly encoded after compaction;
- `G_correct`: correct-history coherent summary K/V;
- `G_wrong`: exact-position, exact-length external donor-slot history;
- `G_Vcorrect`: exploratory V-only variant;
- `G_Kcorrect`: exploratory K-only variant.

The co-primary contrasts are `G_correct−G_fresh` and `G_correct−G_wrong`,
analyzed at the conversation level. The claim requires both final N=12 intervals
to clear zero; N=6 is only a competence/headroom/futility decision and cannot
produce an efficacy declaration. A null remains inconclusive and applies only to
the fixed checkpoint, benchmark, eager backend, and position-preserving layout.
It does not address SDPA, packed/reset-position serving, other models, training,
or coding-agent performance. The wrong-history comparison is explicitly limited
by source-summary congruence and surprisal; it does not prove that a planted
fact alone caused any effect.

A proposed arm copying the original retained tail was rejected. The source
ordering is history → retained tail → summary request → summary, whereas the
compacted destination is summary request → summary → recomputed tail. Direct
tail copying would mix causal order and RoPE positions, and one correct-tail arm
would lack a matched wrong-history control. A future v11 may test downstream
post-summary note rows or a matched correct/wrong tail factorial, but v10’s null
must be interpreted only as a summary-region result.

**Current Execution State.** The v10 apparatus has passed the major local code
and integrity checks: 209 full tests, 90 targeted integrity/harvest/runtime
tests, 9 adversarial tamper tests, and 59 monitor cases. Independent validation
now reconstructs source-derived plants, targets, probes, destinations, wrong
histories, calibration, tensor geometry witnesses, semantic aggregates, N=6/N=12
decisions, technical ancestry, apparatus identity, and terminal artifacts. Raw
K/V tensors are not archived; hashes are treated as relational provenance
witnesses, with strict actual-generation/replay identity and source-to-insertion
lineage requirements.

The v10 local ladder is a technical conformance test, not a semantic experiment.
It processes identical tokens and positions under alternative
chunking/message-boundary schedules and checks K/V, logits, continuation state,
logical/physical positions, and causal access. The synthetic schedule battery
passed 7/7 with aggregate discrepancy exactly `0.0` against the frozen `5e-4`
limit. The twelve committed production-tokenizer cases then began; case `c10`
was running at the latest checkpoint, with each long case expected to take
roughly 1.5–2 hours. Intermediate running checkpoints are being committed
frequently as requested. No paid pod is currently running and the experiment has
spent $0 on the current v10 attempt.

The earlier estimate was approximately 1.5–2.5 days to a validated result and
reviewed paper, with local case checks, final reviews, a paid technical-only 30B
gate, conditional N=6/N=12 semantics, and later Fable-led paper drafting. That
estimate is uncertain and should be updated from actual case completion rates.

The user authorized using a low-cost pod if it could replace the long local
validation. The current v10 contract explicitly requires the bf16 eager CPU
ladder, so a GPU result would be a non-authorizing proxy and changing that would
require a new amendment and review. Parallel paid technical work is
scientifically low-risk because it is semantic-free, but the current frozen
Amendment 10 says the local ladder must precede any paid attempt. The latest
contract review therefore recommends retaining the serial v10 authorization
unless a transparent prospective amendment is made; any early paid run would
need to remain permanently quarantined as development evidence and could not
authorize semantic scoring.

The immediate handoff priority is to preserve the running ladder checkpoints,
complete the v10 local ladder, obtain final exact-commit
code/science/Fable-or-Opus reviews, and keep any technical failure separate from
semantic interpretation.

## Conversation sources

- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f4f16-8cc4-7d10-b574-9f41c2e4d817`
- `019f4f16-73cb-7be3-acff-071c4f9df64b`
- `019f4f2d-27ee-7f10-a846-f24facc12a19`
- `019f4f2d-1b08-7b61-aa7c-8d55f26f2f5b`
- `019f4f90-ed25-7b92-bad2-fa28babdc90b`
- `019f4fb0-cc98-7c23-8911-c71cd0751761`
- `019f4fef-ec58-7d82-a595-482b94ff2e34`
- `019f4ffd-2c0c-7120-aa25-a76bfdf36a3f`
- `019f5005-efb7-71a1-94ed-5476ce540d9a`
- `019f5005-b3c6-75a1-9325-5294bce64e9f`
- `019f500a-ee56-7c22-bc0c-fa9d47e0dd44`
- `019f5005-dab0-7bb3-97eb-025953672c17`
- `019f5017-4efc-71a0-97a5-f106bd6168eb`
- `019f5017-6555-7d30-8b71-e3ec1bb9652b`
- `019f5058-1604-7f00-b34e-d3c3769f6c40`
- `019f5058-7003-7ae3-a350-002f48c5801c`
- `019f5058-492a-7b63-b964-76607ebad880`
- `019f5063-3339-75d1-9163-c170ac27877e`
- `019f5063-0b20-7620-a01f-723430451e18`
- `019f5063-2089-7bb2-b7f9-98d4db349fe9`
- `019f5068-38de-7450-91df-efcf46aff968`
- `019f5068-b28d-7782-ab40-489270741582`
- `019f506d-1c4b-7a23-875a-12c7f86bc530`
- `019f506d-3f15-7ff0-984a-0c3277426ffa`
- `019f5078-86c1-7343-9503-71f3e8cbcea8`
- `019f507f-1be4-7d30-ad96-38b6321f1e27`
- `019f507f-0a4a-7d20-b87a-16e7985f072e`
- `019f5085-b6a9-7511-a2b9-7523fd9c1feb`
- `019f508b-9d23-71d2-8a4e-d06e570a86e8`
- `019f508b-f523-7731-86d5-02802cc035f0`
- `019f50a9-7f5d-7fb0-8468-dd7986ae343f`
- `019f50a9-6853-7442-a1eb-b447e6d53dce`
- `019f50ca-ef44-7f32-9c2b-758080fecd8d`
- `019f50cb-0cdd-7270-86db-09ab637c400f`
- `019f50dc-d4c0-79f1-ab4a-0c37b42e3b96`
- `019f50f5-733f-7212-9e05-a290900d986b`
- `019f50f5-5a05-7083-ac9c-f8f62e8a9fc0`
- `019f5110-ae49-7932-8b7b-1a24b5b50165`
- `019f5110-93b5-7a90-8307-9964c9b49787`
- `019f5124-6ba0-7502-adc9-c7738cb8ef5f`
- `019f5124-45a1-70c0-a6e6-9197c6098f50`
- `019f514f-f264-7cc1-96f6-f6071736cb32`
- `019f516d-39e2-7822-aff2-202b8fdb3c90`
- `019f516d-6a84-7f73-99bf-14b84aed4ae7`
- `019f51c1-dc68-7c23-aa37-4e5ddd449938`
