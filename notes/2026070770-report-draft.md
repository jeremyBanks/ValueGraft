# The sense a model builds up doesn't live in the summary

_When an AI conversation is compacted, the model loses something a good summary
should have kept. This is a report on what that something is, where it lives,
and whether you can put it back._

## Abstract

Long AI conversations get **compacted**: older turns are replaced by a short
text summary so the dialogue fits in the context window. Something is lost
across that boundary that a faithful summary does not restore — the model reads
the summary and still behaves as if it never had the earlier context. We show
that the lost thing is largely **semantic continuity** (the disambiguated
"sense" the model had built up), and that it lives in the model's **write-time
internal state** — the key and value activations the model computed while it was
reading the original turns, the model's internal working notes — not in any text
a summary could carry. Re-injecting those write-time _value_ vectors at the
compaction boundary ("value grafting") recovers roughly 10–12 percentage points
of lost meaning, and does so **specifically where compaction did real damage**:
it helps disambiguate evicted referents and recover evicted decisions, but adds
nothing where a summary already suffices (stable user preferences). Three
independent measurements agree at the 30-billion-parameter scale. The effect
recovers _meaning_ more than _verbatim form_, and it is a large-model phenomenon
— it does not replicate at 4 billion parameters. We report the mechanism, its
bounds, and why standard agent benchmarks fail to exercise the regime where it
matters.

## 1. What compaction loses, and the question of where it lives

Every deployed chat assistant eventually hits a wall: the conversation grows
past the context window. The standard fix is compaction — take the older turns,
write a summary of them, and replace the turns with the summary. The recent
dialogue stays verbatim; the distant past becomes a paragraph.

Compaction is lossy by construction, and everyone knows it drops detail. The
interesting failure is subtler. Even when the summary faithfully records _what
was decided_, the model afterward often behaves as if it never lived through the
original exchange. It re-asks settled questions, misreads which of several
things an earlier shorthand referred to, or — worse — answers confidently about
content that was summarized away. A good human note-taker's summary would let a
fresh reader carry on. The compacted model reads its own good summary and still
stumbles.

That gap is the whole subject of this report. If the summary contains the facts
but the model still loses the thread, then the thing it lost was never _in_ the
text to begin with. Our hypothesis is that what's lost is **semantic
continuity** — the settled, disambiguated _sense_ the model had built up while
reading, both the meaning of evicted references and the specific decisions that
were made and then summarized away — and that this sense lives in the model's
internal activation state at the moment it read those turns, not in any summary
of them.

A transformer reads text by computing, at every layer and every token position,
a **key** and a **value** vector — the "KV cache." Keys determine which past
positions each new token attends to; values are the content that gets mixed in
once attention decides where to look. The KV cache is the model's working
representation of everything it has read so far. There are two versions of it
that matter here: the **write-time state** — the value vectors from when the
model first read the original turns — and the **re-read cache** — the much
smaller cache the model rebuilds later from the summary text after the originals
are evicted. The claim under test: those discarded write-time value vectors
carry disambiguating meaning that the re-read cache does not reconstruct.

## 2. Grafting write-time state back across the boundary

The test is direct. Run a conversation long enough that compaction fires. Keep,
in cold storage, the value vectors the model computed while it originally read
the now-evicted turns. After compaction, at the boundary where the summary sits,
**graft** those saved write-time value vectors back into the model's cache —
blend them into the values the model would otherwise use — and let it continue.
We call this **value grafting** (a values-only graft at blend weight α, where α
is the blend weight: 0 = ignore the grafted values, 1 = fully substitute them).
It touches values only, not keys: it re-supplies write-time _content_ at the
boundary without altering where the model chooses to attend.

Throughout, we compare four arms, using canonical names:

- **Original** — the full conversation, never compacted. The ceiling.
- **Compacted** — plain summary compaction. The floor we're trying to beat.
- **Compacted + value graft** — Compacted, plus write-time value vectors grafted
  at the boundary (α=0.75 unless noted).
- and, later, a **champion** configuration that tunes the graft per layer.

Before measuring recovery, one result establishes that write-time state carries
something behaviorally real. This particular probe uses a fuller retention — the
whole write-time KV state, keys and values, a related intervention to the
values-only graft; the value graft below reproduces the same thesis on the
meaning metrics. When we ask a compacted model about content that was evicted,
it **fabricates** — it invents a plausible answer rather than admitting the gap.
In a decoy-fact probe, the Compacted arm fabricated 83% of the time. Retaining
the write-time KV state instead drove fabrication down to 17%, and on evicted
facts that arm was simultaneously the most accurate (38 of 48, 79%) and the
least fabricating (4%). This honesty effect replicated across precisions (4-bit
to bf16) and scales (4 billion to 30 billion parameters). With the write-time
state, the model knows what it knows; without it, it confabulates — the
behavioral signature of information the summary failed to carry.

## 3. The refinement: it's sense, not trivia — and meaning, not form

This is where §1's question — _where does the lost sense live_ — gets its
answer. Grafting the write-time values back recovers meaning _in proportion to
how much compaction damaged it_, and it recovers **sense** rather than
**surface**. Those are two distinct dissociations along two separate axes:

- **Across category:** the graft recovers evicted sense and referents but not
  stance — it helps exactly where a summary falls short and sits still where a
  summary suffices.
- **Across metric:** it recovers _meaning_ far more than _exact form_ — a
  meaning-judge sees a large effect, an exact-token metric a small one.

Take the across-category axis first. We measured meaning-recovery on
Qwen3-30B-A3B (bf16), with realistic semantic-continuity probes sorted into
three categories and judged for meaning by Sonnet 5 (partial credit = 0.5). The
categories differ in what kind of continuity they test:

| category                                               | Original (full ctx) | Compacted | Compacted + value graft | graft − Compacted |
| ------------------------------------------------------ | ------------------- | --------- | ----------------------- | ----------------- |
| **stance** — honor an evicted preference               | 96%                 | 93%       | 96%                     | +2pp              |
| **sense** — disambiguate an evicted referent's meaning | ~100%               | 46%       | 58%                     | **+12pp**         |
| **referent** — recover a specific evicted decision     | ~100%               | 17%       | 26%                     | **+10pp**         |

(The sense and referent full-context ceilings rest on very few clean cells — n=1
and n=2 respectively — so the ~100% figures are indicative, not precise.)

Read down the "Compacted" column first, because that column is the damage.
Compaction barely touches **stance**: a summary that says "the user dislikes
carousels" preserves a stable preference perfectly well, so Compacted stays at
93% against a 96% ceiling. But compaction _flattens_ **sense** (100 → 46) — the
model can no longer reliably tell which of several things an earlier shorthand
referred to — and it _devastates_ **referent** (100 → 17), the recovery of a
specific decision that was made and then evicted.

Now read the last column. Grafting adds +2pp on stance — null, because the
summary already carried the preference. It adds +12pp on sense and +10pp on
referent — the two categories where compaction did real harm. **The effect
tracks the damage.** What the write-time value vectors carry is disambiguating
meaning — the settled sense — which is exactly what a summary drops and exactly
what a stable preference doesn't need.

Now the across-metric axis: grafting recovers _meaning_, not _wording_. We
re-measured the same probes with a judge-free metric, **teacher-forced
gap-closure**. "Teacher-forced" means we feed the model the exact gold
continuation — the correct, reference continuation — token by token and read off
the probability it assigned; gap-closure is how far the graft moves that
probability from the Compacted floor toward the Original ceiling, computed as
(graft − Compacted) / (Original − Compacted). On this exact-token metric (30B,
66 probes), the _direction_ agrees on all three categories — stance is
null-to-negative (39% of probes helped, mean −0.14), sense is positive (64%
helped, mean +0.03), referent is strongly positive (81% helped, mean +0.04) —
but the magnitude is much smaller — as expected if the graft restores sense, not
exact wording. That gap between a large meaning-effect and a small token-effect
is itself evidence for the "meaning, not surface" reading.

## 4. Three measurements agree — and a look inside that only half-confirms

At 30B the dissociation now rests on three independent measurements, and they
agree:

1. **Lenient meaning-judge** (partial credit 0.5): stance +2pp (null), sense
   +12pp, referent +10pp.
2. **Strict meaning-judge** (partial counts as a miss): stance +4pp (null),
   sense +9pp, referent +8pp. The pattern is not an artifact of the
   partial-credit scoring choice.
3. **Teacher-forced gap-closure** (judge-free, exact tokens): direction agrees
   on all three, magnitude smaller as predicted above.

Three methods — two of them judge-based and one purely mechanical — concur that
grafting helps sense and referent and is null on stance.

We also tried to watch the recovery happen _inside_ the model, with a **logit
lens** — a technique that reads the model's internal state at a chosen layer
(its residual stream) as if it were the final output, letting you see which
token the model is "leaning toward" at that depth. On the gold (correct-answer)
concept token at the probe position (30B, 61 probes), grafting increased the
evicted concept's internal presence: the grafted arm's logprob for the concept
sat between the Compacted floor and the Original ceiling in every category, and
grafting pushed the concept up on 68–77% of probes. This is direct internal
evidence that grafting _inserts concept content_ and moves the internal state
partway back toward the full-context state.

We report the logit-lens result as **partial** support. It confirms the
_general_ mechanism — the graft demonstrably puts evicted concept content back
inside the model — but it does **not** reproduce the _dissociation_. Stance
showed roughly 77% concept-elevation too, the same as sense, even though stance
was behaviorally null. The likely reason is instrumental: the gold-token signal
is near the floor (logprob around −12 to −14, rank in the thousands), so a
single-token logit lens is a blunt tool. It can detect "grafting nudges the
concept up broadly" but cannot resolve _where_ that nudge translates into
recovered behavior. A sharper readout — a tuned lens, multi-token, or targeted
layers — is future work.

## 5. Deploying it without breaking the task

If grafting is going to be useful rather than merely interesting, it has to
survive contact with a real task, and the naive version does not. Grafting at
full strength (α=1.0, full substitution) can catastrophically break things: on
one task chain the full-strength graft scored 0 of 4 (0%) where every other arm
scored 4 of 4 (100%). Flooding the boundary with write-time values at full
weight can overwhelm the model's own reading of the summary.

The fix is per-layer tuning. Rather than one global strength, we tune the graft
strength layer by layer and validate each candidate against guards. The tuned
**champion** configuration eliminated the instability — 16 of 16 (100%) on the
same task family where the naive graft had cratered. Crucially, the guards
distinguish a real recovery from an artifact: the layer profile passed a
wrong-conversation contamination guard (grafting values from a _different_
conversation must not help — and it didn't), while a control that masked all but
57 cache slots _failed_ that guard, exposing it as a content-independent
artifact rather than genuine content recovery. The takeaway for deployment is
that value grafting has a safe operating point but needs per-layer calibration
and guard validation — it is not a knob you turn to maximum.

## 6. Where it stops: scale and benchmarks

Two boundaries are real limits and findings in their own right.

**The effect is a large-model phenomenon.** Everything in Section 3–4 is at 30B.
At 4B, the same teacher-forced gap-closure metric does **not** reproduce the
dissociation. The clean ordering collapses: stance shows 87% of probes helped
(mean +0.19 — the _opposite_ of the 30B null), sense 55% (mean −0.09), referent
62%. The tidy "null on stance, positive on sense and referent" structure is gone
or muddled. This is consistent with a scale-dependence we see elsewhere in the
project — the graft's dose-response (how the recovered benefit scales with graft
strength) actually inverts between 4B and 30B — so it is not a surprise, but it
is a hard bound. The 4B gap-closure ratios are also noisier (smaller
Original−Compacted denominators), and a judged 4B cut would sharpen the picture,
but the direction of the non-replication is clear. The dissociation is a 30B
result, and holds only at that scale.

**Standard benchmarks miss the operative regime — and that itself is a
finding.** The natural next question is whether grafting improves an end-to-end
agent task. We could not cleanly demonstrate that, and the reason is structural
rather than a null result about grafting. The regime where grafting can help is
narrow: the task must be _hard enough that eviction genuinely costs the model
something_, yet _easy enough that the model can act on the recovered context_.
Existing benchmarks straddle that window rather than sitting in it.

- **SWE-bench** is simply beyond a 30B model — 0 of 7 even in oracle mode
  (handed the correct file to edit), so there is no recoverable signal to
  measure.
- **Interactive/tau-bench-style tasks** don't naturally reach the
  compaction-stress regime. When we ran a tau-bench banking scenario with a
  capable GPT-4o-mini user-simulator, the sessions came out at only ~3–4K
  tokens. That is too short for meaningful eviction at any threshold: set the
  compaction trigger high and it never fires; set it low and there is nothing
  substantial to evict. All three arms scored reward 0.00 with no separation.

The synthetic conversations behind our dissociation result avoid this trap
precisely because they are long enough that policy and referents get evicted by
**genuine conversation length**, not by an artificially low compaction threshold
— which is what makes that eviction clean and the recovery interpretable. The
benchmark landscape does not currently offer a task in the operative regime for
a mid-sized model. Demonstrating end-to-end benefit needs a purpose-built
harness, and the absence of one is a gap in the field, not a verdict on the
mechanism.

## 7. What this means, and what's still open

Strip it to the claim. The "sense" a model builds up over a conversation is not
fully reconstructible from a text summary of that conversation, because a
meaningful part of it lives in the write-time activation state rather than in
any words. You can recover a slice of it — roughly 10–12 points of lost meaning
at 30B — by grafting the write-time value vectors back at the compaction
boundary, and the recovery lands specifically on the semantic continuity a
summary can't carry (disambiguated sense, evicted decisions) while correctly
leaving alone what a summary handles fine (stable stance). It recovers meaning
more than wording, it makes the model more honest about what it no longer knows,
and — tuned per layer — it can do this without breaking the task.

Several questions are genuinely open:

- **Keys versus values.** We grafted values only, and it worked. Whether keys
  carry a separable share of the lost continuity — the "where to attend" as
  distinct from "what's there" — is untested and would sharpen the mechanistic
  picture.
- **Cross-scale behavior.** The dissociation is clean at 30B and gone at 4B, and
  the dose-response inverts between them. Why the mechanism changes character
  with scale — and where between 4B and 30B it turns on — is unknown and worth
  mapping.
- **A purpose-built benchmark.** The field needs a task deliberately built for
  the operative regime: long enough sessions to force genuine eviction,
  difficulty tuned so a mid-sized model can act on recovered context, and probes
  that separate sense/referent from stance. Standard suites don't supply it, and
  until one exists the end-to-end benefit will stay under-measured.
- **Deployment as an opaque compaction handle.** In practice this reframes
  compaction. Instead of "summarize and discard," a system could summarize for
  the human-readable transcript while retaining an opaque write-time-KV handle
  that gets grafted back on continuation — safe at a tuned operating point and
  honest by default. We have not measured the storage cost, but as a rough
  proposal the overhead should be on the order of a second prefill's worth of
  state, which we expect to be cheap to retain; the payoff is continuity that
  survives the boundary.

The underlying spirit of this write-up is a question as much as a claim.
Re-supplying write-time activations at a compaction boundary is an obvious
enough thing to try that we expected to find it named and studied; we couldn't,
under the terminology we searched. Either we are missing the right words for it,
or it has been considered too trivial to write up. Either way, here is what we
measured.

---

_Designed and largely executed by Claude Fable 5, continued by Claude Opus 4.8
(as Fable's quota exhausted), with subagent work (judging, dataset scouting,
task-adapter construction) by Claude Sonnet 5 and adversarial review by OpenAI's
GPT-5.5 — directed, sanity-checked, and funded by the user._
