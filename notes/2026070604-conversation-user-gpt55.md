_This entry covers a period of paper-drafting work on the ValueGraft write-up
conducted with the codex/gpt-5.5 agent, followed by several scheduled "refresh"
check-ins tracking a parallel pivot of the main experiment program from
LongMemEval into live E-track coding-agent evaluation._

**Participants:** User and gpt-5.5-xhigh.

Participants: Jeremy Banks (user), codex/gpt-5.5 agent (assistant, effort=xhigh
throughout).

## Repo state at start and evolving handoff discipline

A refresh at the start of this period found roughly a dozen new commits, a
stronger configuration policy (keep the primary method as simple global
alpha-per-model; per-slot/per-head/factored variants remain exploration unless
guard checks pass), a 57-slot positive-profile mask that beat global alpha=0.75
on holdout (+0.0384 vs +0.0239, 10/10 wins) but was deliberately not promoted to
headline status, extended alpha>1 testing showing smooth decline past 1.0 with
peak near 0.75 for 30B bf16, a negative-coefficient test where clamped beat
signed but both lost to the simple mid-band rule, dtype provenance now recorded
in HF runners, `scripts/launch_pod.sh` added for cloud pod launch/watchdog
workflow, `.parquet` ignored under a 4MB pre-commit policy, and roughly 330
LongMemEval 30B bf16 result files plus 13 judge batches/verdicts landed. Several
source files (`src/arms_common.py`, `src/arms_hf.py`, `src/run_tune_hf.py`)
carried uncommitted Gemma-template and RoPE/`text_config` support at that time.

## Paper synthesis workspace

To consolidate scattered documentation, an isolated working directory
(`paper-working/valuegraft-synthesis/`) was created, explicitly marked via
README as a personal derived-analysis workspace containing no original findings,
so other agents would not treat it as a source of experimental facts. A strict
commit protocol was adopted: never pre-stage files, add and commit only the
exact tracked/new path immediately before committing, verify the staged set with
`git diff --cached --name-status`, and never use directory-wide or wildcard
`git add`.

Within that workspace, a first wide-synthesis paper draft
(`valuegraft-paper-draft.md`) was written and iterated, incorporating a newer
Stage-1 LongMemEval aggregate (n=320) showing large compaction damage with flat
arm performance in QA framing — establishing that this framing mostly bounds the
mitigation claim rather than strengthening it. A second, tighter draft
(`valuegraft-focused-draft.md`) was then created (originally preserving the wide
version untouched) with a single narrative spine: compaction damage as
baseline/denominator, two interventions, what they recover, what they don't, and
why it matters — deliberately demoting side material (provider API shape,
calibration philosophy, deployment overhead, provenance) to brief notes rather
than main narrative, while retaining a concise related-work section.

Several correction passes followed on the focused draft:

- Reframing to ensure "compaction hurts recall" reads strictly as an evaluation
  baseline/denominator, not a finding or contribution — this was flagged as a
  repeated, important correction.
- An editorial pass removing "AI-writing" tells: ghost-objection caveats,
  generic importance-scaffolding phrasing, and defensive "this is not X"
  framing, replaced with direct claims.
- A citation overhaul replacing title-only name-drops with author-year inline
  citations and a proper references section (applied to both drafts), grounding
  provider-related claims in documented surfaces rather than implied internals.
  Key sources included Li 2026, Zweiger et al. 2026, KVLink, CacheBlend,
  AutoCompressor, and Anthropic/OpenAI/Gemini compaction docs.
- A methodology expansion after the user flagged that arms were described in
  single sentences: added detail on compaction-boundary selection, summary
  generation, construction of arms A/B/B-min-pack/H-pack/ValueGraft, H-pack's
  key re-rotation step, ValueGraft's exact-token alignment and value-blending,
  tuning choices, negative controls, and scoring — applied to both drafts.
- A naming cleanup: `ValueGraft` was redefined as the umbrella name for the
  approach family, with the two serious variants renamed `ValueGraft-Pack`
  (formerly H-pack; carries forward the summary's write-time KV cache,
  repositioned/repacked with re-rotated keys) and `ValueGraft-Blend` (fresh
  keys, blended old value tensors per
  `V_final = (1-alpha)*V_fresh + alpha*V_old`), and the matched fresh-encoding
  control renamed `FreshPack`. This rename was applied only in the focused
  draft, per instruction that the paper's naming need not exactly mirror the
  code.
- A Code Availability section was added (repo URL:
  https://github.com/jeremyBanks/ValueGraft), since blind review is not expected
  for this project.

The repository was pushed to a new private GitHub remote
(`jeremyBanks/ValueGraft`, branch `trunk`) partway through this work, with a
tracked-history secret-pattern scan performed first; ignored key files stayed
out of the push. Subsequent narrow commits (citations, methodology expansion,
naming) were pushed in batches, with explicit care to avoid publishing unrelated
in-flight experiment commits mixed into the same branch.

## Publication-channel discussion

Extended discussion covered academic ML publication venues
(NeurIPS/ICML/ICLR/COLM/ACL/EMNLP/MLSys etc.) and non-peer-reviewed channels
(arXiv, blogs, GitHub, Zenodo, HuggingFace). Key points: arXiv tightened its
endorsement policy as of 2026-01-21 (institutional email alone no longer
suffices; needs prior arXiv authorship in the endorsement domain or personal
endorsement) and tightened review/position-paper moderation for cs categories as
of 2025-10-31. The user's existing CHI 2018 peer-reviewed paper (DOI
10.1145/3173574.3174182, coauthored with Denae Ford, Kristina Lustig, and Chris
Parnin) was discussed as a possible arXiv-seeding candidate; ACM's author-rights
policy permits posting accepted/peer-reviewed author versions (not the ACM
version-of-record) to arXiv with the DOI included, but arXiv's submission
agreement requires the submitter to represent that coauthors consent. The user
ultimately decided not to pursue re-uploading that paper to arXiv, given the
added coauthor-coordination overhead versus the paper's already-established
value. The user's current intent is a GitHub-only release for ValueGraft,
positioning it as a rigorous but limited-scope paper-style writeup with real
citations, rather than pursuing a formal venue.

## Repository conceptual clarification (H-pack / ValueGraft)

To ground the writing, the underlying arm distinction was restated:
`H-pack`/`ValueGraft-Pack` reuses the summary's write-time KV cache from when it
was generated under full context, repositioned into a compact prefix layout with
keys re-rotated for RoPE (values unchanged); its main observed effect is
honesty/less fabrication rather than recall recovery.
`ValueGraft`/`ValueGraft-Blend` instead starts from ordinary compacted
`summary + tail`, freshly encoded, and blends old value vectors into fresh ones
at aligned token positions; its main observed effect is on
continuation/next-action likelihood.

## Parallel experiment-program pivot (tracked via scheduled refreshes)

While paper work proceeded, the main experiment track (driven by other
concurrent agent activity, not this conversation's edits) pivoted decisively
from LongMemEval toward live E-track coding-agent evaluation. `STATE.md` now
marks LongMemEval as abandoned as a live path, retaining only its
damage-quantification table. Refresh check-ins (some scheduled via a heartbeat
automation at roughly 2-hour intervals) tracked this pivot:

- First live signal: B failed both `t1`/`t2` coding tasks (0/2), E passed both
  (2/2) — tiny n but the first direct agent-task-success signal, using
  `Qwen/Qwen3-30B-A3B-Instruct-2507` (bf16) served via an OpenAI-compatible shim
  (`src/serve_shim.py`, modes `sc-A`/`sc-B`/`sc-E`) with OpenHands as the coding
  harness.
- A larger E-matrix was planned and launched across pods (`p2`, `p4`, `e1`–`e4`,
  `w1`) covering tasks, seeds, B/E/alpha variants, anti-graft and shuffled
  controls, threshold variants, and layer/slot-mask tuning configs
  (`tune_configs.json`).
- Operational issues surfaced across refreshes and were noted for follow-up: a
  `scripts/e1_driver.sh` shell-syntax failure (later apparently fixed, `bash -n`
  passes, but a suspicious comment-embedded `LLM_BASE_URL`/`LLM_API_KEY` export
  line remained); several shim pods (`e2`, `e4`, `w1`) loaded and then
  terminated rather than staying up; a likely live bug in `serve_shim.py` where
  `sess["cfg_head_map"]` may be referenced before `sess` is assigned when
  parsing `E:cfg=...`, potentially causing 500s on the tuned-config (`layers`,
  `posslots`) matrix rows; and later, a posslots guard check reportedly returned
  a negative/contamination verdict, and two matrix lanes appeared to be running
  the same `t1... t2:s3 E` task into the same output directory, a possible
  run-contamination risk flagged for confirmation.
- An offline coding-adjacent result was also reported as complete: on 75
  SWE-Gym/OpenHands traces, tuned ValueGraft improved next-action prediction by
  +0.0156 nats, winning 45/75 paired cases (~10% closure of the A-vs-B
  compaction gap).
- By the last refresh in this period, the coding matrix had grown to 92 scored
  runs, with a committed honesty results table also in place, and commit count
  on `trunk` had grown to 49 ahead of `origin/trunk` (unpushed).

The user asked whether the agent would be comfortable supervising/driving the
full experiment program autonomously for an extended period (managing pods,
running experiments, and potentially running a separate "creative" exploration
pod), with a proposed ~30-minute handoff-file cadence. The agent affirmed
willingness given explicit authorization scoping (spend limits, pod kill/restart
permission, commit/push permission, and priority: throughput vs. caution vs.
writeup quality), but this was discussed as a future possibility, not yet
approved or activated.
