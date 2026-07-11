# Fable review: downstream-tail channel design

**Exact runtime model identity:** `claude-fable-5` (Claude Fable 5), as exposed by
this session's runtime environment. Given this repository's history of
environment-string oscillation between Fable 5 and Opus 4.8 (see the identity
corrections in `notes/2026071156-sol-fable-execution-coordination.md`), I record
that this is the environment-reported model ID, stated per the owner's directive
to stamp the exact runtime rather than an intended alias. This review was
conducted independently: I did not coordinate with Sol, with Claude session A or
B, or with any other live session, and I inspected the code and frozen documents
directly rather than trusting any party's characterization of them.

**Evidence reviewed:** `AGENTS.md`; `COHERENT-STATE-PREREGISTRATION.md`;
Amendments 1–10; `notes/2026071154-sol-claude-research-plan-dialogue.md`;
`notes/2026071156-sol-fable-execution-coordination.md` (complete, including the
session-B G_correct_tail proposal in its final turn);
`notes/2026071163-opus-prior-art-verification.md`; `src/coherent_state_cases.py`;
`src/coherent_state_tokens.py`; `src/coherent_state_runtime.py`; and the arm/tail
construction paths of `src/run_coherent_state_hf.py`. No file was modified except
this one; no ladder process was touched; no paid work was run; no semantic
outcome exists or was inspected (none exists to inspect).

---

## 1. The proposal under review

Claude session B, citing "Models Take Notes at Prefill" (arXiv 2606.17107),
proposes adding one arm, `G_correct_tail`, to the frozen v10 design before the
launch SHA freezes: "bit-exactly preserve the summary **and** the retained-tail
write-time K/V at their logical positions (trivial and rotation-free in the
position-preserving design), recompute only the probe." The claimed contrasts:
`G_correct_tail − G_fresh` tests the full downstream-notes channel;
`G_correct_tail − G_correct` localizes it to the tail. Session B's fallback ask —
that any summary-only null be interpreted through the downstream-notes mechanism —
is already committed (commits 89b4d4c, 24dc7af).

My assessment: **the scientific concern is legitimate and the citation is
load-bearing, but the proposed arm is not constructible as described in this
exact layout, one arm would not license the claim it targets, and the correct
vehicle is a separately preregistered follow-up reusing the saved renders — not
an amendment to v10.** The reasoning follows, anchored to the code.

## 2. The decisive geometry: source and destination retained-tail tokens do NOT occupy the same logical positions

This is the single fact on which the proposal's "trivial and rotation-free"
premise stands or falls, and it is false in the frozen layout. Verified directly:

**Correct source ordering** (`coherent_state_cases.py:93-95`,
`correct_source_messages`): the source prefix is
`[system][evicted history: messages 1..tsm-1][retained tail: messages tsm..end][user: SUMMARY_REQUEST][assistant header]`,
where `tsm = sections.middle_end_msg`. The retained tail therefore occupies
logical positions strictly **before** the request/header suffix, which ends at
`P_i − 1`; the summary is generated at `P_i..P_i+n_i−1`. In the source, the tail
sits hundreds of positions before the summary.

**Frozen destination ordering** (`coherent_state_cases.py:132-144`,
`compacted_messages`; positions assigned in `coherent_state_tokens.py:206-256`,
`gapped_destination_layout`): the destination is
`[system][logical gap][request/header ending at P_i−1][summary at P_i..P_i+n_i−1][assistant close + retained tail][probe]`,
with `post_summary_position_ids = range(P_i+n_i, …)` — the retained tail
occupies logical positions strictly **after** the summary. The tail is appended
per-arm as a causal descendant of the summary intervention
(`coherent_state_runtime.py:573-599`, `append_gapped_post_summary`), which is
correct for the summary-only question v10 asks.

So the same tail token IDs sit at pre-request logical positions in the source
and at post-summary logical positions in the destination. The forward shift is
per-conversation-varying and includes the request/header length, the summary
length `n_i` (itself variable, up to 900), and the assistant-close wrapper. It
is never zero.

**Consequences for "bit-exact rotation-free copying":**

- **Reading 1 — copy source tail rows into the frozen destination's tail slots:**
  impossible without key re-rotation. The stored post-RoPE keys carry the source
  positions. Re-rotation of stored bf16 post-RoPE keys is precisely what
  Amendment 1 retired from confirmatory use, on preserved evidence
  (`results/coherent_state_diagnostics/`): even float32 rotation with row cosine
  0.999993 produced a `0.1015625`-nat target-pair margin error — at or above the
  plausible effect scale, and far above the 0.005-nat bias budget the
  coordination dialogue converged on. A rotated-tail arm would reintroduce, for
  a span typically much longer than the summary, exactly the confound the
  position-preserving pivot was designed to eliminate.
- **Reading 2 — keep the tail rows at their *source* logical positions (what
  "at their logical positions" most charitably means):** then the arm is
  rotation-free, but the destination logical-position vector now differs from
  every other arm's, violating Amendment 1's frozen invariant that "every scored
  compacted arm uses the same physical token IDs and exact logical position
  vector." `G_correct_tail − G_fresh` and `G_correct_tail − G_correct` would be
  cross-geometry contrasts — differences between experiments, not between cache
  states — and licensed by nothing in the preregistration. The visible layout
  also changes (tail logically precedes request and summary), which changes what
  the probe's context *is*.

There is also a causal-ordering asymmetry under either reading: in the source,
the tail's write-time K/V never attended to the summary (the summary did not
exist yet); in the frozen destination the tail sits downstream of the summary
and is supposed to attend to it. A transplanted source tail placed after the
summary is a cache state no serving path would ever produce in this layout. The
natural home for "tail keeps its write-time state" is a **source-ordered
eviction layout** — evict the history block's storage, keep system + tail +
request + summary at their original logical positions (this is MEMENTO-style
KV-preserving compaction, and arguably the more deployment-realistic
position-preserving pattern). That is a coherent and genuinely interesting
design. It is also, unambiguously, a *different destination geometry* requiring
its own fresh baseline, its own wrong control, its own schedule fixtures
(Amendment 8's composed destination fixture is layout-specific), its own ladder
stages, and its own validator paths (`gapped_destination_layout` hard-codes the
two-island geometry at `coherent_state_tokens.py:214-216`). "One extra arm" is
not an available description of this change at any level of the frozen
apparatus.

For completeness: the frozen gates themselves already prohibit the arm as
proposed — original preregistration §9.4 / Amendment 1 gate structure: "No
destination outside the summary content span; no tail or wrapper token may be
transplanted," and Amendment 8 §3.5's lineage graph plus the driver's lineage
validation (`run_coherent_state_hf.py:463-486`, `518-519`) enumerate the exact
frozen arm set. Adding the arm is therefore necessarily an
arms/estimands/validator amendment, the first since Amendment 4 to touch the
scientific design rather than close a fail-closed gap — every one of Amendments
4–10 states "changes no arm."

## 3. One correct-tail arm does not establish history specificity

Even in a layout where it is constructible, `G_correct_tail − G_fresh` alone
conflates (a) correct-history information memoized on the tail with (b) generic
state-versus-re-encode coherence of a long re-encoded span across a gap. The
entire logic of the frozen design — the `GF`/`GW` intersection — exists because
`C − F` alone cannot distinguish those. A tail arm inherits the same
requirement: it needs a matched `G_wrong_tail` (tail write-time K/V computed
under donor history at identical positions) and a geometry-matched fresh
baseline before "history-specific tail channel" can be claimed.

The good news, verified in code: the wrong-source construction already provides
the matched control for free. `matched_wrong_prefix_ids`
(`coherent_state_tokens.py:109-186`) replaces only evicted-content slots;
retained-tail and structural tokens are ID-identical at identical positions in
the correct and wrong sources, so wrong-history tail rows exist at exactly
matched logical positions. A properly designed follow-up gets its specificity
control from machinery that is already built and validated. That is an argument
*for* the follow-up being cheap and well-posed — not for bolting a single
under-controlled arm onto v10.

## 4. What the prior art actually predicts here — weaker than session B's framing

Session B reads 2606.17107 as making v10 "structurally biased toward an
uninformative null." I think that overstates it, in three ways:

1. **The summary rows are themselves downstream-of-history writes.** In Li et
   al.'s terms, the "field" is the evicted history, and the "<1%" claim is about
   the *field's own* K/V. v10 never transplants field K/V — the history is
   evicted in every compacted arm. The summary tokens are generated one at a
   time while attending to the full history; their K/V rows are exactly the kind
   of downstream, conclusion-bearing writes the paper says carry the decision.
   v10's `G_correct` is a downstream-notes graft, restricted to the
   summary-token subset of downstream positions.
2. **MEMENTO is direct prior evidence that the summary-block-only channel can be
   nonzero.** MEMENTO's kept state is the memento/summary block's KV versus
   re-prefilling identical memento text — the closest published analogue of
   `G_correct − G_fresh` — and it reports a positive effect (magnitude to be
   verified at source per `notes/2026071163`). The two papers jointly say: the
   channel is real, and it is distributed across downstream positions of which
   the summary is one part. They do not say the summary part is empty.
3. **What v10 genuinely cannot see** is history information that bypassed the
   summary rows: (a) the source tail's write-time state (recomputed fresh per
   arm — though note the recomputed tail *does* attend to the transplanted
   summary rows, so a summary-borne channel propagates; per-arm tail
   recomputation was designed for exactly that), and (b) the request/header
   rows, which in the correct source attended to the full history but in every
   destination arm are computed fresh across the gap
   (`build_gapped_fresh_boundary`, `coherent_state_runtime.py:388-402`). Those
   template/punctuation-heavy rows immediately downstream of the history are,
   if anything, the closest match to Li et al.'s "aggregator token" description.

So the honest statement is: **a v10 null on `GF` is consistent with the channel
living on untransplanted downstream positions, and must be reported with that
caveat — which is already committed** (89b4d4c, 24dc7af, and the caveat is the
right and sufficient minimum). A v10 positive, conversely, is clean and
history-specific by the frozen intersection. The design is not foreclosed; it
is narrower than the full downstream-notes hypothesis, and it is the narrower
question this project's mitigation (summary-region grafting) actually turns on.

## 5. Does v10 remain valuable as the narrower summary-state test? Yes.

1. It is the exact MEMENTO-analogue contrast on an ordinary instruct model at
   an ordinary compaction boundary, with a specificity control MEMENTO lacks.
2. It bounds the intervention this project has spent its budget on: ValueGraft
   operates on summary rows; whether *those rows* carry a usable, history-
   specific channel is the decision-relevant question for the paper's mechanism
   claims, independent of what else lives on the tail.
3. Either outcome remains informative under the committed caveat: positive ⇒
   history-specific summary-state channel established; null ⇒ the summary-row
   subset carries no detectable channel in this regime, sharpening the follow-up
   to downstream positions rather than invalidating the program.
4. The v10 apparatus — gapped positions, per-arm tail recomputation, bit-exact
   lineage, the independent validator stack — is the platform any tail/downstream
   assay would build on. Running it is how that platform gets its first
   production validation.
5. The run's mandatory artifacts (renders, exact summary IDs, tokenwise traces,
   bit-exact replay witnesses per Amendment 8/10) are precisely what makes the
   follow-up cheap. Running v10 first *funds* the tail experiment's inputs.

## 6. Amendment now versus separately preregistered follow-up

**Follow-up, decisively.** The cost asymmetry session B assumed ("far cheaper
now than after a null") is inverted by this repository's own authorization
contract:

- An arm-adding amendment is a v11: fresh 0.6B ladder, fresh production-tokenizer
  donor artifact, fresh independent code/science/cross-family reviews, new
  validator lineage/analysis/checkpoint schemas, new launch commit — the same
  full cycle each of Amendments 5–10 required for far smaller changes. The v10
  ladder stage artifacts are already on disk awaiting terminalization; an
  amendment now discards that position in the pipeline.
- The amendment does not even buy the intended arm, because (§2) the arm is
  ill-posed in this geometry. Doing it right means a new destination layout —
  a redesign, not an amendment.
- The follow-up's marginal cost is small **by design**: every render and exact
  summary-ID set is durably saved and committed, and Amendments 8/10 require
  bit-exact stepwise-replay identity, which proves the exact write-time source
  states are reconstructible from saved artifacts by forward passes alone — no
  regeneration, seconds-to-minutes per conversation of prefill on a rented GPU.
  The ~16-minute-per-conversation generation cost is paid once, in v10.
- Sequencing adds information: if v10's `GF` clears, the follow-up localizes an
  established channel; if it nulls, the follow-up is the decisive next test and
  its design can be sharpened by v10's diagnostics (headroom, calibration,
  observed summary NLLs) without any adaptive contamination, because its
  preregistration is separate and its outcome data are new forward passes.

**Recommended follow-up shape** (for the eventual separate preregistration, not
frozen here; both components reuse the saved renders and the frozen donor map,
targets, and probes):

- **D1 — request/header transplant in the *current* geometry** (cheapest, zero
  layout change): the request/header island occupies *identical* logical
  positions in the correct source and the destination by construction
  (`gapped_destination_layout` asserts `correct[request_start:] == suffix`,
  `coherent_state_tokens.py:226-229`), so correct-source and wrong-source
  request/header rows are bit-exact, rotation-free transplantable today. Arms
  `{G_fresh, G_reqhdr_correct, G_reqhdr_wrong, G_summary+reqhdr_correct}` test
  whether the nearest downstream aggregator rows carry the channel, with the
  same intersection logic. This is the fastest honest test of the
  Models-Take-Notes mechanism available in this apparatus.
- **D2 — source-ordered eviction layout** (the real tail test): destination
  `[system][gap where history was][tail at original positions][request/header][summary][probe]`,
  where every retained token keeps its source logical position and the entire
  retained prefix K/V is bit-exact copyable rotation-free. Arms
  `{G_fresh′, G_correct_tail, G_wrong_tail}` (wrong-tail rows supplied by the
  existing matched wrong-source construction), plus a summary-only arm at this
  layout to complete the localization factorial. This requires the three-island
  generalization of the layout code, its own composed-schedule fixture, ladder,
  and validator extensions — exactly why it is a follow-up, not a bolt-on.

## 7. Claim boundaries and what I did not verify

- I did not run any code, model, or fixture; all geometry claims are from direct
  source reading of the committed files listed above, at the working tree state
  of 2026-07-11. Line references are to that state.
- I did not re-verify the two arXiv papers beyond Opus's committed verification
  note; the MEMENTO restart-ablation magnitude remains unread at source, as that
  note itself flags.
- I take no position on the eventual v10 semantic outcome; no semantic result
  exists, and nothing here licenses a prediction of one.
- Whether D1/D2 are worth their spend is a budget decision for the owner and the
  execution lead under the existing authorization; my claim is only about
  validity, sequencing, and relative cost.

## 8. Decisive recommendation

**Do not amend v10. Freeze and run the summary-only v10 exactly as gated.**
Session B's `G_correct_tail` arm is not scientifically valid in this exact
layout: source retained-tail tokens do not occupy the same logical positions as
destination retained-tail tokens, so the promised bit-exact rotation-free copy
does not exist — the arm requires either the retired lossy key re-rotation or a
different destination geometry, and in either form a single correct-tail arm
without a matched wrong-tail control and geometry-matched fresh baseline cannot
establish history specificity. The already-committed interpretation caveat (a
summary-only null does not falsify a downstream/tail channel) is the correct and
sufficient protection for the current run, whose narrower summary-state question
remains valuable in both directions. Address the downstream-notes hypothesis as
a **separately preregistered follow-up reusing the saved renders** — request/
header-row transplantation first (bit-exact and rotation-free in the existing
geometry today), then the source-ordered tail-preserving layout with correct-
and wrong-history tail arms — drafted in parallel now if desired, launched only
after v10 terminates under its own frozen rules.
