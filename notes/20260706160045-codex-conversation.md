_This shard covers the ValueGraft experiment's live monitoring routine, followed
by a pivot into a substantial side investigation using Anthropic's Jacobian-lens
(J-lens) interpretability tool, culminating in document-quality corrections and
a methodological control study, before ending on a tooling question about
shellcheck._

## Live-run monitoring and a self-diagnosed refresh failure

Through several "Refresh" cycles on 2026-07-06, the ValueGraft live run kept
pivoting rapidly: from humane-tier SWE-bench (`compact_at=12000`, audited
config) to an oracle-retrieval `swbo:` variant (after plain SWE-bench scored 0
passes across 17 trusted rows, prompting an explicit "too hard for this model"
pivot), then to `chain:*` tasks as the new interim useful evidence track, with
tau2 scouted as a possible future standard benchmark. Trusted score counts
remained thin throughout (e.g., only 4 trusted humane `c12000` rows, all
failures), while legacy synthetic/brief-summary rows stayed the only sizable
clean stratum, explicitly labeled non-headline/non-production-faithful.
Operational anomalies recurred: `H1`/`SYN`/`V1` lanes stuck on "shim down,"
failing external health probes on several ports even while other lanes' agent
logs kept updating, and stale-looking log lines (`t-source: command not found`,
`alarm 3600`) that didn't match current HEAD scripts — flagged as needing
verification rather than assumed benign.

The user then directly challenged the refresh methodology ("you're always
ignoring a ton of changed notes"), and a self-critique followed: refreshes had
been treating status files like `STATE.md` as tail-readable when they actually
contain stale head sections plus newer appended blocks elsewhere, and truncated
tool output was being summarized rather than flagged as incomplete. The
corrected protocol going forward: record the last-seen commit, diff
`previous..HEAD --name-status`, read actual changed hunks in all
status/decision/incident/handoff docs, then separately assess live processes —
treating note changes as first-class state, not secondary to live metrics.

## J-lens side investigation

A handoff note on Anthropic's J-lens/global-workspace interpretability work was
committed to repo root (`handoff-note-jlens-global-workspace.md`) and assessed
as appropriately scoped: a second, corroborative mechanistic instrument, not a
replacement for behavioral evidence, best suited to narrow concept probes rather
than compositional coding-task outcomes.

Extended technical explanation established the mechanics for the user: J-lens
reads residual-stream activations (not the KV cache directly) via a pre-fitted
per-layer Jacobian/lens matrix, producing top-k vocabulary-token readouts at a
chosen token/layer — conceptually similar in shape to next-token logits but
reading an internal diagnostic state rather than the generation frontier.
Fitting a lens exactly for a given model requires expensive backward passes (not
available prebuilt for the project's own `Qwen3-30B-A3B-Instruct-2507`), whereas
Anthropic had already released fitted lenses for **Qwen3.6-27B**, a newer,
stronger, hybrid (Gated DeltaNet + Gated Attention) dense model with strong
agentic-coding benchmarks and a built-in "preserve thinking" feature —
attractive as a future subject model but not a drop-in for the current
KV-cache-surgery experiment, since much of its architecture lacks standard
per-layer K/V tensors.

**Model acquisition and isolated prototype.** Qwen3.6-27B (~~52GB, 15 safetensor
shards) was downloaded into the local HF cache after repeated
background-downloader failures (large-shard transfers died silently in detached
processes; foreground/sequential per-shard transfers succeeded). A fully
isolated prototype directory, `jlens_boundary_probe/`, was created to keep all
exploratory work out of the live pipeline. Early work there used a RunPod A100
(~~$1.39/hr): first pod was **terminated immediately after pulling results
without inspecting them first**, which the user flagged as premature; the
assistant acknowledged the lesson (verify artifact before killing
infrastructure). A second pod was launched and deliberately left running per
explicit user instruction, with a 5-minute heartbeat check-in and a 30-minute
genuine-idle auto-terminate guardrail (only shutting down after sustained
inactivity with no live process/GPU work).

**Qualitative probe results.** Using labeled-anchor sampling (exact token
positions for names/terms with private/ambiguous meanings, at layers
`[16, 32, 48, 62]`), three demo families were run and documented with notes
files in `jlens_boundary_probe/`:

- **Pokémon demo** (canonical/lean variants, later a matched-wrapper v2 to
  remove formatting confounds): tokens like `Ghost` (write-time → "died/faint";
  fresh → generic "-type/ghost"), `Vacuum` (write-time → "Zig/nickname"; fresh →
  "Cleaner/appliance"), `Dex` (write-time → "owes/trade"; fresh →
  "Nav/completion") showed clear context-conditioned divergence, while some
  anchors (`Root Cause`, `Ultra Balls`) stayed stable across encodings.
- **Plain-conversation demo** (block-party planning): `Maple` (write-time →
  "refers/means"; fresh → "Street/Ave"), `Robin` (write-time →
  "volunteer/person"; fresh → "Hood/Oak"), `Blue` (write-time →
  "chosen/selected"; fresh → "tarp/tent") reproduced the same phenomenon in an
  ordinary, non-fandom setting.
- **Multi-demo full-scan** (`multi_demo_scan.py`): rather than only hand-picked
  anchors, every summary token position was scanned and ranked by
  write-time-vs-fresh top-k divergence, surfacing both curated and
  self-discovered high-divergence tokens (with some noise from
  punctuation/format tokens).
- **SWE-Gym next-action probe** (`swegym_next_action_probe.py`): applied the
  same technique to real coding trajectories (getmoto, Dask, MONAI),
  teacher-forcing the true next action and comparing full-context vs
  compact-summary residual readouts at action/path/tool tokens. A significant
  methodological catch: SWE-Gym system prompts bias summary generation itself
  toward emitting tool-call XML instead of prose; this was fixed by hardening
  the summary prompt (forbidding function-call XML, prefilling "Summary:") and
  adding output trimming at tool/chat markers, plus switching anchor matching
  from token-exact to character-offset-based for robustness against subword
  merges (e.g., `>` merging into `/workspace`).

All work was committed incrementally inside `jlens_boundary_probe/` (a
`blog-style` writeup, `semantic_readout_blog_draft.md`, was explicitly requested
by the user as a new document distinct from the paper, describing this
application).

**Next-token control experiment.** The user raised a substantive methodological
question: how is a J-lens readout different from ordinary next-token probability
prediction, since both are vocabulary-ranked distributions? This was treated as
a real experiment to run, not just discussion. A new comparison probe computed,
at the same anchors/contexts, both J-lens top-k and actual next-token top-k,
measuring top-k Jaccard overlap. Result (documented in
`next_token_readout_control_report.md`): J-lens is **not equivalent to
next-token prediction**, but overlap is layer-dependent — mean top-20 Jaccard
`0.256` overall, separating strongly from `0.076` at layer 48 to `0.436` at
layer 62 (later layers converge more toward next-token-like behavior). The
cleanest evidence of distinct signal was `B-410`: next-token prediction offered
only generic hyphen/code continuation, while layer-48 J-lens read out
`obsolete/outdated/deprecated/expired` — content next-token prediction had no
access to. This control was integrated as a caveat into both
`semantic_readout_blog_draft.md` and the main `valuegraft-focused-draft.md`.

## Paper terminology and quality correction

Separately, the focused paper draft
(`paper-working/valuegraft-synthesis/valuegraft-focused-draft.md`) was updated
to align with the clarified taxonomy: ValueGraft as the family name, **V-Graft**
(value-side grafting with fresh keys) and **KV-Graft** (preserving both key and
value) as the two named variants, replacing legacy internal names
(`ValueGraft-Pack`/`ValueGraft-Blend`, `H-pack`, `B-min-pack`) which now appear
only in an explicit historical-mapping note. Related-work was reorganized into
three distinct categories (academic KV/cache compression, provider
compaction/caching API surfaces, and J-lens as a readout tool), with citations
added for OpenAI/Anthropic/Gemini compaction docs, LongMemEval, and model
checkpoints. A future-work section was added covering controlled
`(alpha_K, alpha_V)` arms and span-first J-lens probing on coding traces.

This editing pass drew a substantive readability critique: the J-lens section of
the draft had been compressed to bare labels and claims with no quoted examples,
side-by-side comparisons, or context a reader could inspect, and commits had not
been made incrementally as claimed. The assistant rewrote the section to include
actual quoted summary/action text, explicit write-time-vs-fresh side-by-side
tables, and plain-language interpretation for the
`Vacuum`/`Dex`/`Maple`/`B-410`/`Crane` examples plus a coding next-action case —
committed and pushed in two visible, separately described increments (`37978b0`,
`5731202`). This is a concrete instance of the "readable, quote-backed,
comparison-based" bar the user expects for any writeup summarizing raw
experimental output.

## State at shard boundary

`trunk` was pushed to `origin` through commit `bcc9c1c` (and earlier `6421511`),
containing: the J-lens boundary-probe toolkit and all its demo/control outputs
under `jlens_boundary_probe/`, the next-token control report and its raw JSON,
and the corrected/expanded `valuegraft-focused-draft.md`. The exploratory J-lens
RunPod A100 instance was terminated. The last live topic before this shard ends
is the user asking whether `shellcheck` should be integrated into script
generation — the assistant had begun narrowing a repo search for what "generate"
refers to (source/scripts vs. archived result corpus) when the shard closes.

---
