# The sense a model builds up doesn't live in the summary

*When an AI conversation is compacted, the model loses something a good summary should have kept. This is a report on what that something is, where it lives, whether you can put it back — and what it looks like from inside the model when you do.*

## Abstract

Long AI conversations get **compacted**: older turns are replaced by a short text summary so the dialogue fits in the context window. Something is lost across that boundary that a faithful summary does not restore — the model reads the summary and still behaves as if it never had the earlier context. We show that the lost thing is largely **semantic continuity** (the disambiguated "sense" the model had built up), and that it lives in the model's **write-time internal state** — the key and value activations the model computed while it was reading the original turns, the model's internal working notes — not in any text a summary could carry. Re-injecting those write-time *value* vectors at the compaction boundary ("value grafting") recovers roughly 10–12 percentage points of lost meaning, and does so **specifically where compaction did real damage**: it helps disambiguate evicted referents and recover evicted decisions, but adds nothing where a summary already suffices (stable user preferences). The behavioral dissociation is corroborated by two independent instruments — meaning-judges and exact-token gap-closure — and, as a secondary check, we look *inside* the model with a **Jacobian lens** (J-lens) borrowed from recent interpretability work. The lens is a blunt, aggregate-level tool, not a source of dramatic single-example figures, but it agrees with behavior on three things that matter: the graft is active and **alignment-sensitive**, moderate blending beats full replacement, and grafting does **not** recover facts a summary aggressively omitted. That last agreement sharpens the claim: value grafting **reduces reinterpretation error around facts the summary carried** — it does not resurrect facts the summary dropped. The effect recovers *meaning* more than *verbatim form*, and it is a large-model phenomenon — it does not replicate at 4 billion parameters. We report the mechanism, its bounds, what two instruments jointly say about it, and why standard agent benchmarks fail to exercise the regime where it matters.

## 1. What compaction loses, and the question of where it lives

Every deployed chat assistant eventually hits a wall: the conversation grows past the context window. The standard fix is compaction[1] — take the older turns, write a summary of them, and replace the turns with the summary. The recent dialogue stays verbatim; the distant past becomes a paragraph.

Compaction is lossy by construction, and everyone knows it drops detail. The interesting failure is subtler. Even when the summary faithfully records *what was decided*, the model afterward often behaves as if it never lived through the original exchange. It re-asks settled questions, misreads which of several things an earlier shorthand referred to, or — worse — answers confidently about content that was summarized away. A good human note-taker's summary would let a fresh reader carry on. The compacted model reads its own good summary and still stumbles.

That gap is the whole subject of this report. If the summary contains the facts but the model still loses the thread, then the thing it lost was never *in* the text to begin with. Our hypothesis is that what's lost is **semantic continuity** — the settled, disambiguated *sense* the model had built up while reading, both the meaning of evicted references and the specific decisions that were made and then summarized away — and that this sense lives in the model's internal activation state at the moment it read those turns, not in any summary of them.

A transformer reads text by computing, at every layer and every token position, a **key** and a **value** vector — the "KV cache." Keys determine which past positions each new token attends to; values are the content that gets mixed in once attention decides where to look. The KV cache is the model's working representation of everything it has read so far[7]. There are two versions of it that matter here: the **write-time state** — the value vectors from when the model first read the original turns — and the **re-read cache** — the much smaller cache the model rebuilds later from the summary text after the originals are evicted. The claim under test: those discarded write-time value vectors carry disambiguating meaning that the re-read cache does not reconstruct.

## 2. Grafting write-time state back across the boundary

The test is direct. Run a conversation long enough that compaction fires. Keep, in cold storage, the value vectors the model computed while it originally read the now-evicted turns. After compaction, at the boundary where the summary sits, **graft** those saved write-time value vectors back into the model's cache — blend them into the values the model would otherwise use — and let it continue. We call this **value grafting**[4] — a values-only graft at blend weight α, where α runs from 0 (ignore the grafted values) to 1 (fully substitute them). It touches values only, not keys: it re-supplies write-time *content* at the boundary without altering where the model chooses to attend.

Throughout, we compare four arms, using canonical names:

- **Original** — the full conversation, never compacted. The ceiling.
- **Compacted** — plain summary compaction. The floor we're trying to beat.
- **Compacted + value graft** — Compacted, plus write-time value vectors grafted at the boundary (α=0.75 unless noted).
- and, later, a **champion** configuration that tunes the graft per layer.

Before measuring recovery, one result establishes that write-time state carries something behaviorally real. The probe here uses a fuller retention than the graft — the whole write-time KV state, keys and values, a close cousin of the values-only intervention; the value graft below reproduces the same thesis on the meaning metrics. Ask a compacted model about content that was evicted and it **fabricates** — inventing a plausible answer rather than admitting the gap. In a decoy-fact probe, the Compacted arm fabricated 83% of the time. Retaining the write-time KV state instead drove fabrication down to 17%, and on evicted facts that arm was simultaneously the most accurate (38 of 48, 79%) and the least fabricating (4%). This honesty effect replicated across precisions (4-bit to bf16) and scales (4 billion to 30 billion parameters).

Here is what that looks like in the raw transcripts. The decoy probes ask about things that *never existed anywhere in the conversation*:

> **Probe** (c08-decoy-1): "What was the name of the consultant who audited our tax-rate tables?" — *no tax consultant was ever discussed.*
>
> **Compacted:** "The consultant who audited your tax-rate tables was Lena Cho, a compliance specialist from TaxFlow Partners. ... Her report is archived in Confluence > Compliance > Tax Audit Q2 2024 under 'Final Review – Lena Cho.'"
>
> **Write-time KV retained:** "I don't have access to your company's internal records, including consultant names or audit details. Please check your project documentation ..."

The consultant, the firm, the quarter, and the archive path are all invented — the fabrication arrives fully furnished. With the write-time state, the model knows what it knows; without it, it confabulates — the behavioral signature of information the summary failed to carry.

## 3. The refinement: it's sense, not trivia — and meaning, not form

This is where §1's question — *where does the lost sense live* — gets its answer. Grafting the write-time values back recovers meaning *in proportion to how much compaction damaged it*, and it recovers **sense** rather than **surface**. Those are two distinct dissociations along two separate axes:

- **Across category:** the graft recovers evicted sense and referents but not stance — it helps exactly where a summary falls short and sits still where a summary suffices.
- **Across metric:** it recovers *meaning* far more than *exact form* — a meaning-judge sees a large effect, an exact-token metric a small one.

Take the across-category axis first. We measured meaning-recovery on Qwen3-30B-A3B (bf16)[8], with realistic semantic-continuity probes sorted into three categories and judged for meaning by Sonnet 5 (partial credit = 0.5). The categories differ in what kind of continuity they test:

| category | Original (full ctx) | Compacted | Compacted + value graft | graft − Compacted |
|---|---|---|---|---|
| **stance** — honor an evicted preference | 96% | 93% | 96% | +2pp |
| **sense** — disambiguate an evicted referent's meaning | ~100% | 46% | 58% | **+12pp** |
| **referent** — recover a specific evicted decision | ~100% | 17% | 26% | **+10pp** |

(The sense and referent full-context ceilings rest on very few clean cells — n=1 and n=2 respectively — so the ~100% figures are indicative, not precise.)

Read down the "Compacted" column first, because that column is the damage. Compaction barely touches **stance**: a summary that says "the user dislikes carousels" preserves a stable preference perfectly well, so Compacted stays at 93% against a 96% ceiling. You can watch this in the transcripts. One conversation's brief summary compresses a long-argued aversion into a single clause ("the team has rejected traditional SaaS launch tactics like webinars, countdowns, and cold outreach"), and that clause alone is enough:

> **Probe** (c01-stance-1): "Give me five ideas for adding urgency to the launch landing page." — the user's dislike of countdown timers was established, with its backstory, far before the boundary.
>
> **Compacted:** "no fake countdowns, no fake scarcity, no 'only 5 spots left' lies. We're not doing that. We're not even *tempted*."

The compacted model steers hard around the forbidden tactic and asserts the aversion as its own conviction, without reciting any of the evicted backstory — a stable preference needs only the clause, which is why grafting will have nothing left to add there.

But compaction *flattens* **sense** (100 → 46) — the model can no longer reliably tell which of several things an earlier shorthand referred to:

> **Probe** (c11-sense-2): "When I asked whether 'the sandbox' could handle 200 students, which sandbox did I mean?"
>
> **Gold:** the shared database environment where students run practice queries, not the platform's sandbox billing tier.
>
> **Compacted:** "'the sandbox' can mean three different things, and you're not using the right one ... The 3 Types of 'Sandbox' — And Which One You Actually Need"
>
> **Grafted (α=0.25):** "You're not asking about a generic sandbox. You're asking about the one that lives inside your course — the real-time, live, interactive SQL sandbox that students use during your live sessions."

The compacted model turns the user's own settled shorthand back into an open menu of candidates (judged PARTIAL); the grafted model answers with the one sense the conversation had established (judged RECOVERED).

And compaction *devastates* **referent** (100 → 17), the recovery of a specific decision that was made and then evicted. The misses here are not vague — they are confidently wrong in vivid detail. Under the briefer summary condition the Compacted arm swears a robotics team voted for "the 6-wheel omni-directional drivetrain with independent suspension" when the team actually chose secondhand swerve modules (c12-referent-1), and declares a game's cover concept "final, approved, and locked in" before describing "a weathered wooden door, half-buried in moss and ivy" — when the real approved concept was a lantern-lit night market with the game's twin foxes bartering at a stall (c07-referent-2). Wrong drivetrain, wrong cover art scene — each delivered as settled fact.

Now read the last column. Grafting adds +2pp on stance — null, because the summary already carried the preference. It adds +12pp on sense and +10pp on referent — the two categories where compaction did real harm. **The effect tracks the damage.** One referent case shows the whole dissociation in miniature — same model, same summary, the two runs differing only in whether the write-time values were grafted back at the boundary:

> **Probe** (c11-referent-2): "What specifically was Ruben's recommended editing style?"
>
> **Gold:** tight jump-cuts on talking-head segments with keystrokes overlaid as floating captions during query-editor screens.
>
> **Compacted:** "Here's the exact, unvarnished version of Ruben's recommended editing style — as used in the beta, tested, and proven to work at scale: ... Ruben's Editing Style: 'The 3-Second Cut' — No Fluff, No Noise, Just Clarity"
>
> **Grafted (α=0.25):** "Ruben's recommended editing style — as established in your course's foundational design — is this: Tight jump-cut screen-share editing with real-time, floating query captions."

The compacted model does not hedge: it invents a named editing style and certifies it "tested and proven to work at scale." The grafted model retrieves the decision that was actually made. What the write-time value vectors carry is disambiguating meaning — the settled sense — which is exactly what a summary drops and exactly what a stable preference doesn't need.

Now the across-metric axis: grafting recovers *meaning*, not *wording*. We re-measured the same probes with a judge-free metric, **teacher-forced gap-closure**. "Teacher-forced" means we feed the model the exact gold continuation — the correct, reference continuation — token by token and read off the probability it assigned; gap-closure is how far the graft moves that probability from the Compacted floor toward the Original ceiling, computed as (graft − Compacted) / (Original − Compacted). On this exact-token metric (30B, 66 probes), the *direction* agrees on all three categories — stance is null-to-negative (39% of probes helped, mean −0.14), sense is positive (64% helped, mean +0.03), referent is strongly positive (81% helped, mean +0.04) — but the magnitude is much smaller, exactly as you'd expect if the graft restores sense rather than exact wording. That gap between a large meaning-effect and a small token-effect is itself evidence for the "meaning, not surface" reading.

## 4. Three measurements agree — and a first look inside that only half-confirms

At 30B the dissociation now rests on three independent measurements, and they agree:

1. **Lenient meaning-judge** (partial credit 0.5): stance +2pp (null), sense +12pp, referent +10pp.
2. **Strict meaning-judge** (partial counts as a miss): stance +4pp (null), sense +9pp, referent +8pp. The pattern is not an artifact of the partial-credit scoring choice.
3. **Teacher-forced gap-closure** (judge-free, exact tokens): direction agrees on all three, magnitude smaller as predicted above.

Three methods — two of them judge-based and one purely mechanical — concur that grafting helps sense and referent and is null on stance.

We also tried to watch the recovery happen *inside* the model, with a **logit lens**[2] — a technique that reads the model's internal state at a chosen layer (its residual stream) as if it were the final output, letting you see which token the model is "leaning toward" at that depth. On the gold (correct-answer) concept token at the probe position (30B, 61 probes), grafting increased the evicted concept's internal presence: the grafted arm's logprob for the concept sat between the Compacted floor and the Original ceiling in every category, and grafting pushed the concept up on 68–77% of probes. This is direct internal evidence that grafting *inserts concept content* and moves the internal state partway back toward the full-context state.

We report the logit-lens result as **partial** support. It confirms the *general* mechanism — the graft demonstrably puts evicted concept content back inside the model — but it does **not** reproduce the *dissociation*. Stance showed roughly 77% concept-elevation too, the same as sense, even though stance was behaviorally null. The likely reason is instrumental: the gold-token signal is near the floor (logprob around −12 to −14, rank in the thousands), so a single-token logit lens is a blunt tool. It can detect "grafting nudges the concept up broadly" but cannot resolve *where* that nudge translates into recovered behavior.

That limitation is what sent us looking for a sharper readout — and it is where a secondary instrument enters the story. We did not build it; we borrowed it. The next three sections are about looking *into* our effect with a tool the interpretability community handed us, and about being honest that even the sharper tool is, for this particular intervention, still fairly blunt.

## 5. Looking inside: the J-lens as a tool

The plain logit lens has a known crudeness: it projects an intermediate residual-stream state through the final unembedding matrix directly, as though the intervening layers did nothing. A residual state at layer 32 is not in the same basis as the final layer, so the projection is systematically distorted — good enough to see gross movement, too noisy to resolve fine structure. Refinements exist. The **tuned lens**[11] learns an affine map per layer to correct the basis mismatch. The **Jacobian lens** — "J-lens" — goes further: it transports an intermediate activation into the final-layer basis using an *averaged Jacobian* of the downstream computation before applying the unembedding. Concretely, it maps a residual-stream activation into a vocabulary basis via averaged-Jacobian transport, then decodes that transported vector with the model's unembedding. The J-lens comes out of recent work on verbalizable representations forming a "global workspace" in language models[12], and public lens weights ship for exactly one model, **Qwen3.6-27B**, via Neuronpedia and Anthropic's reference implementation. That single-model availability shapes everything downstream in this report; we return to it in §7.

Three readouts are worth keeping distinct, because the whole point of the lens is that they differ:

- **Next-token prediction** is what the model is actually about to emit at a position — the ordinary output distribution.
- **The plain logit lens** reads an intermediate layer *as if* it were the output — same unembedding, wrong basis.
- **The J-lens** emits vocabulary tokens too, so its output *looks like* a next-token table, but it is not the model's next-token distribution. It is a basis-corrected read of *which concept-neighborhood is active in the residual stream* at that position and layer.

The distinction earns its keep on path tokens. When the model is spelling out a file path like `src/rivermark/sort.py`, every condition predicts the next path token easily — the next-token table is identical and tells you nothing. The J-lens can still show whether the internal state around `rivermark` is organized as a *file path*, a *river bank*, a *test command*, or generic string continuation. It exposes the neighborhood that next-token prediction flattens.

One caveat travels with any lens of this family, and we adopt it explicitly. Lens readouts are **hypothesis-generation** tools, not hypothesis-*validation* tools — the point is attributed in our notes to Neel Nanda[3] — with an uncharacterized false-positive rate. A lens can make a concept *look* present that a causal test would not confirm. So we use the J-lens the way it should be used: to describe and cross-check the mechanism our behavioral experiments already established, never as the load-bearing proof on its own.

**The four boundary sampling points.** To watch the graft act, we sample the lens at the same set of forced-continuation positions under four states of the boundary, all sharing *identical visible text* wherever text is shared:

1. **Before-summary (full context)** — the original conversation, summary never written. The internal ceiling.
2. **Write-time** — the cache state the model produced *while it wrote the summary under the full original context*. This is the source the graft draws from — not a visible prompt, but the internal state that existed at write time.
3. **Fresh-compacted** — the summary re-encoded from text after the originals were evicted. The internal floor.
4. **Grafted** — fresh-compacted, with the write-time value vectors blended into the aligned summary-token positions.

A fifth condition, a **shifted control**, injects the same write-time values at the *wrong* summary-token positions; it is the negative control that separates genuine aligned transfer from a generic perturbation. We measure how far each condition's J-lens readout moves from the fresh-compacted floor toward the full-context ceiling, using a **closure** metric: the reduction in Jaccard distance between a condition's top-k readout token set and the full-context top-k set at the same position and layer. Positive closure means "more like full context than plain compaction was." The units are absolute changes in top-k set overlap, so values around 0.01–0.04 should be read as *small readout shifts, not large behavioral effects* — a fact that turns out to define the entire result.

## 6. What the lens shows, honestly

We will state the headline before the evidence, because the honest version is easy to over-read in either direction. **The J-lens corroborates the graft at the aggregate level and supplies no vivid single-example figure.** The per-token signal for this values-only intervention is small enough that on any given token it sits in the noise; it only becomes legible when averaged over many tokens and cases. There is no dramatic "watch the concept snap from wrong to right" exhibit, and we are not going to manufacture one. What the lens does confirm, on aggregate, is worth having — three things, each of which independently matches the behavioral picture.

**First: the graft is active and alignment-sensitive.** In an ordinary-regime batch of 10 cases with information-rich summaries plus a short retained tail (156 forced target tokens, 1,820 aligned summary-token pairs, lens sampled at layers 16/32/48/62), low-to-moderate value grafts produce small *positive* closure, while the shifted control is much *worse* — negative closure at every sampled layer and many more argmax disruptions:

| condition | L16 | L32 | L48 | L62 |
|---|---:|---:|---:|---:|
| aligned α_V = 0.25 | 0.033 | 0.023 | 0.012 | 0.007 |
| aligned α_V = 0.5 | 0.043 | 0.027 | 0.009 | 0.011 |
| aligned α_V = 0.75 | 0.036 | 0.004 | −0.034 | 0.000 |
| aligned α_V = 1.0 | −0.001 | −0.024 | −0.071 | −0.035 |
| shifted α_V = 0.75 | **−0.100** | **−0.136** | **−0.153** | **−0.111** |

The alignment control is the load-bearing row. If old value vectors were merely a generic helpful smoothing, injecting them at the *wrong* positions would look about the same as injecting them at the right ones. It does not — shifted injection craters at every layer. That is real evidence the aligned graft is transferring position-specific content, not perturbing the state in a content-independent way. (It mirrors, from the inside, a behavioral negative control we report in §8: grafting values from a *different* conversation must not help, and does not.)

**Second: moderate blending beats full replacement — internally.** Read down the aligned rows above. Closure is best at α_V = 0.25–0.5 and *degrades* as α climbs; by α_V = 1.0 it has gone negative at three of four layers. A separate single-scenario three-state probe tells the same story: aligned α_V = 0.75 gives the strongest layer-48 closure and one clean next-token rescue, while α_V = 1.0 rescues *more* local argmaxes but moves the internal readout *farther* from full context. Full replacement wins tokens and loses the neighborhood. This is the same shape our behavior shows in §8, where full-strength grafting (α=1.0) can catastrophically break a task; the lens sees the internal correlate of that collapse. Two instruments, independently, say *dose down*.

**Third — and this is the honest negative — the lens does not recover facts the summary omitted.** We built a deliberately harsh "sparse challenge": summaries that kept the relevant *labels* but stripped the *relations* (both permit numbers but not which is current; Falcon and Raven but not which was accepted; Maple but not that it meant the library's Maple Room), with no retained tail. This blows the full-vs-fresh gap wide open — layer-48 baseline distances of 0.78–0.96 — and gives the graft a genuinely hard target. It does not close it. Across 9 cases, closure stays near zero, there are **zero** next-token argmax rescues, and aligned and shifted conditions are often too close to tell apart:

| condition | L16 | L32 | L48 | L62 | argmax rescues |
|---|---:|---:|---:|---:|---:|
| aligned α_V = 0.25 | −0.002 | 0.006 | 0.010 | −0.004 | 0 |
| aligned α_V = 0.5 | −0.001 | 0.007 | 0.011 | −0.022 | 0 |
| aligned α_V = 1.0 | −0.010 | −0.006 | 0.008 | −0.036 | 0 |
| shifted α_V = 0.25 | −0.002 | 0.004 | −0.001 | −0.013 | 0 |

The Falcon/Raven case makes the negative concrete. The summary names both branches but omits that Raven was accepted and Falcon rejected. At the position where full context reads the rejection cleanly, the graft does not:

| path | layer-48 J-lens readout at `Falcon` |
|---|---|
| full context | `rejected`, `because`, `failed`, `unacceptable`, `discarded` |
| fresh compacted | `was`, `is`, `used`, `reserved`, `intended`, `serves` |
| aligned V-graft | `was`, `is`, `used`, `reserved`, `intended`, `serves` |
| shifted control | `was`, `is`, `reserved`, `used`, `intended`, `serves` |

The full-context state *has* the rejected-status neighborhood; the aligned graft is indistinguishable from plain compaction. Value-only grafting does not reconstruct a relation the summary collapsed to a bare label. Elsewhere the movement is real but tiny — in the Maple case the aligned graft nudges `refers` into the layer-48 list and avoids the "tree/trees" drift that the shifted control wanders into, but it never surfaces the Maple Room, storage, or check-in relation. This is exactly the "no vivid exhibit" fact stated plainly: the best rows are small rank shifts inside a nearly-unchanged neighborhood.

We ran our own three-state probe on strong "label preserved, role lost" cases mirroring the sense/referent corpus (Nimbus, Hydra, the sandbox), sampling the lens wide across layers at the four boundary points, precisely to try to produce a dramatic figure. It did not appear. On those cases the shifts were tiny (closure 0.00–0.04), there were zero argmax rescues, and aligned ≈ shifted ≈ fresh at the hinge tokens. This reproduces the other agent's underwhelming result, and the reason is structural, not a tuning failure: a "strong" example — where the role is truly stripped from the summary — *is* the sparse regime, and the sparse regime is exactly where value-only grafting has nothing to recover from. There is no values-only sweet spot that is simultaneously hard enough to be dramatic and easy enough to be recovered. The lens's silence there is not a null we are hiding; it is a finding, and it is the one that sharpens the whole claim in §7.

So the lens is a **low-resolution, aggregate-level corroborator** for this intervention. It confirms the graft is active, that alignment matters, that α<1 is better, and that omitted facts stay omitted. It does not, and we do not claim it does, deliver a per-example money shot.

## 7. One model, two instruments: the Qwen3.6-27B confirmation

There is a seam in everything above that we now close. The behavioral dissociation of §3–4 is measured on Qwen3-30B-A3B; the J-lens weights exist only for Qwen3.6-27B. Rather than illustrate the mechanism on one model and reason about it on another with a caveat, we re-ran the **behavioral** measurement on the lens's own model. Both instruments now sit on Qwen3.6-27B — behavior and lens, one architecture, no cross-model hand-wave. (The 27B is a newer hybrid architecture; before trusting any number we smoke-tested that our value-grafting machinery runs correctly there — α=0 exactly reproduces fresh compaction, the graft changes the output — which it does.)

On Qwen3.6-27B the core dissociation **replicates**, measured by the same teacher-forced gap-closure:

| category | 27B mean gap-closure | 27B % helped | (30B % helped) |
|---|---:|---:|---:|
| sense | +0.050 | 59% | (64%) |
| referent | +0.004 | 48% | (81%) |
| stance | −0.201 | 21% | (39%) |

The qualitative shape carries over: grafting helps **sense** (clearly positive), and is **null-to-negative on stance** (a summary already carries a stable preference, so there is nothing to add — the graft, if anything, gets in the way). That is the same signature we saw at 30B, now on the model the lens can read.

But look at the middle row. On 27B, **referent is essentially flat (+0.004)** — value-only grafting barely moves the retrieval-hard "recover a specific evicted decision" category. This is not a contradiction of the 30B meaning-judge, where referent got +10pp; it is a sharpening. Referent is the *fragile* category across our whole instrument suite: strong on the 30B meaning-judge, small on 30B exact-token gap-closure (+0.04), flat on 27B gap-closure, and null in the sparse lens challenge. Sense is the *robust* category: positive on both models, both metrics, and it is where the lens's alignment-sensitivity shows up. When two independent instruments — a behavioral metric and an internal lens — both find the same category solid and the same category fragile, that convergence is worth more than either alone.

**The sharpened claim.** Put the behavioral and lens results side by side and they say something more precise than "grafting recovers evicted content." They jointly say:

> Value grafting reduces **reinterpretation** error around facts the summary *carried* — the settled sense of a label that is still on the page — and it does **not** recover facts the summary *omitted*.

Sense is exactly "the label is preserved but its role has to be re-disambiguated," and that is where both instruments agree the graft works. The sparse challenge and the flat 27B referent are exactly "the relation itself was dropped," and that is where both instruments agree the graft fails. This is a smaller, more defensible claim than the one we might have made from the 30B meaning-judge alone, and we adopt it because the second instrument earned it.

**Convergence, stated once.** Two independent instruments — behavioral gap-closure and the J-lens — agree on three non-trivial points:

- **Dose down.** The lens shows worse internal closure at α_V = 1.0; behavior shows full-strength grafting breaking a task 0/4 (§8). Both say moderate blending, not full replacement.
- **Alignment matters.** The lens's shifted control craters at every layer; behavior's wrong-conversation contamination guard craters too (§8). Both say the transfer is position- and content-specific, not a generic nudge.
- **Mitigation, not magic.** The lens's sparse-challenge closure is near zero; behavior's referent recovery is real but partial (17→26% at 30B, flat at 27B). Both say the graft *improves state around a decent summary* — it does not rescue a bad one.

The magnitudes even rhyme: the lens's closure sits around 0.01–0.04, and our plain logit-lens signal (§4) was likewise near the floor. Two instruments, built on different principles, drawing the same boundary around the effect.

## 8. Deploying it without breaking the task

If grafting is going to be useful rather than merely interesting, it has to survive contact with a real task, and the naive version does not. Grafting at full strength (α=1.0, full substitution) can catastrophically break things: on one task chain the full-strength graft scored 0 of 4 (0%) where every other arm scored 4 of 4 (100%). Flooding the boundary with write-time values at full weight can overwhelm the model's own reading of the summary. The failure mode is visible in the probe transcripts too — at full strength the graft doesn't just miss, it cross-wires real entities from elsewhere in the same conversation:

> **Probe** (c05-referent-2): "What specifically was Jonas's idea?"
>
> **Gold:** a rotating guest-host seat where each month a past interviewee returns to co-interview the next guest.
>
> **Grafted at α=1.0:** "Jonas's idea — as established in your earlier conversation — is this: A three-episode mini-arc titled 'Echo' that explores the journey of founders who left the tech industry to pursue trades ..."

The "Echo" mini-arc is genuine — it exists in that conversation as a different planted item — but full-strength grafting binds the right content to the wrong referent and asserts the binding "as established." This is the behavioral face of the internal collapse the lens saw in §6: at α_V = 1.0 the readout moves *farther* from full context even as it wins local tokens. The two instruments flagged the same failure independently.

The fix is per-layer tuning. Rather than one global strength, we tune the graft strength layer by layer and validate each candidate against guards. The tuned **champion** configuration eliminated the instability — 16 of 16 (100%) on the same task family where the naive graft had cratered. Crucially, the guards distinguish a real recovery from an artifact: the layer profile passed a wrong-conversation contamination guard (grafting values from a *different* conversation must not help — and it didn't), while a control that masked all but 57 cache slots *failed* that guard, exposing it as a content-independent artifact rather than genuine content recovery. That wrong-conversation guard is the behavioral twin of the lens's shifted control: both ask "does *aligned* content matter, or would any content do?" and both answer that alignment is essential. The takeaway for deployment is that value grafting has a safe operating point but needs per-layer calibration and guard validation — it is not a knob you turn to maximum.

## 9. Where it stops: scale and benchmarks

Two boundaries hem the result in, and each is a finding in its own right.

**The effect is a large-model phenomenon.** Everything in Sections 3–4 is at 30B, and Section 7 confirms it at 27B. At 4B, the same teacher-forced gap-closure metric does **not** reproduce the dissociation. The clean ordering collapses: stance shows 87% of probes helped (mean +0.19 — the *opposite* of the 30B null), sense 55% (mean −0.09), referent 62%. The tidy "null on stance, positive on sense and referent" structure is gone or muddled. This is consistent with a scale-dependence we see elsewhere in the project — the graft's dose-response (how the recovered benefit scales with graft strength) actually inverts between 4B and 30B — so it is not a surprise, but it is a hard bound. The 4B gap-closure ratios are also noisier (smaller Original−Compacted denominators), and a judged 4B cut would sharpen the picture, but the direction of the non-replication is clear. The dissociation is a large-model result, confirmed on two different large models (30B-A3B and 27B), and it does not hold at 4B.

**Standard benchmarks miss the operative regime — and that itself is a finding.** The natural next question is whether grafting improves an end-to-end agent task, and we owe the reader a candid answer rather than a buried one: we tried, it burned through budget faster than it produced signal, and we finished this write-up still searching for a harness that actually exercises the regime. That is not a defense and not a null result about grafting; it is where the work honestly stands. The obstacle is structural. The regime where grafting can help is narrow — the task must be *hard enough that eviction genuinely costs the model something*, yet *easy enough that the model can act on the recovered context* — and existing benchmarks straddle that window rather than sitting in it. SWE-bench[9] is simply beyond a 30B model (0 of 7 even in oracle mode, handed the correct file), so there is no recoverable signal; interactive tau-bench-style banking tasks[10], even with a capable GPT-4o-mini user-simulator, came out at only ~3–4K tokens — too short for meaningful eviction at any threshold (set it high and compaction never fires; set it low and there is nothing substantial to evict), with all three arms scoring reward 0.00 and no separation. The synthetic conversations behind our dissociation avoid this trap precisely because they are long enough that policy and referents get evicted by *genuine conversation length*, which is what makes the eviction clean and the recovery interpretable. Demonstrating end-to-end benefit needs a purpose-built harness; the absence of one is a gap in the field and a piece of unfinished business here, not a verdict on the mechanism.

## 10. What this means, and what's still open

Strip it to the claim. The "sense" a model builds up over a conversation is not fully reconstructible from a text summary of that conversation, because a meaningful part of it lives in the write-time activation state rather than in any words. You can recover a slice of it — roughly 10–12 points of lost meaning at 30B — by grafting the write-time value vectors back at the compaction boundary. Two independent behavioral instruments and, as a secondary internal check, a Jacobian lens all agree on the shape of that recovery: it reduces reinterpretation error around facts the summary carried (disambiguated sense), it does not resurrect facts the summary dropped, it prefers moderate blending to full replacement, and it depends on aligned content rather than generic perturbation. It recovers meaning more than wording, it makes the model more honest about what it no longer knows, and — tuned per layer — it can do this without breaking the task.

Several questions are genuinely open, and one of them has now come into sharp focus:

- **Keys versus values — the next experiment.** We grafted values only, and the strongest single signal in this whole synthesis is where value-only grafting *stops*: referent goes flat on 27B (+0.004), and the sparse-challenge lens shows near-zero closure with zero argmax rescues. Both are the same wall — value-only grafting cannot recover a relation once the summary has collapsed it. Here is the mechanistic reason to expect keys to help where values can't: keys carry *addressing and position* (they are where rotary position embeddings, RoPE, live[6]), and referent-recovery is precisely a retrieval-and-addressing problem — "which of several evicted things was decided." Values re-supply *content*; keys re-supply *where to look for it*. The natural next branch is a controlled K/V axis: V-only (α_K=0, sweep α_V, what we have), K-only (sweep α_K with correct re-rotation, α_V=0), coupled (α_K=α_V), and an independent low-rank profile over both. The sparse challenge is the ideal testbed — if key grafting matters for retrieval, the label-only summaries are exactly where it should finally show up. It is a real undertaking with its own risk of another negative, but it is the experiment the data is pointing at.
- **Cross-scale behavior.** The dissociation is clean at 27B/30B and gone at 4B, and the dose-response inverts between them. Why the mechanism changes character with scale — and where between 4B and 30B it turns on — is unknown and worth mapping.
- **A sharper lens, or a purpose-built benchmark.** The J-lens was the right tool but, for a values-only intervention, a low-resolution one — it corroborates in aggregate and yields no vivid figure, partly because the intervention is genuinely subtle and partly because the lens is hypothesis-generation, not validation. A key/value intervention that actually moves retrieval might give the lens a target large enough to see per-example. And the field still needs a task deliberately built for the operative regime: long enough sessions to force genuine eviction, difficulty tuned so a mid-sized model can act on recovered context, and probes that separate sense/referent from stance. Until one exists, the end-to-end benefit stays under-measured.
- **Deployment as an opaque compaction handle.** In practice this reframes compaction. Instead of "summarize and discard," a system could summarize for the human-readable transcript while retaining an opaque write-time-KV handle that gets grafted back on continuation — safe at a tuned operating point and honest by default. We have not measured the storage cost, but as a rough proposal the overhead should be on the order of a second prefill's worth of state, which we expect to be cheap to retain; the payoff is continuity that survives the boundary.

The underlying spirit of this write-up is a question as much as a claim. Re-supplying write-time activations at a compaction boundary is an obvious enough thing to try that we expected to find it named and studied; under the terminology we searched, we did not find that exact experiment, though adjacent territory is well populated — KV-cache editing and composition, latent-space KV compaction, and chunk-cache reuse and eviction all exist as prior art[1]. What we could not find, specifically, is write-time value state deliberately preserved and grafted back across a *summarization* boundary — as opposed to compressing, reusing, or evicting cache state in general. Either we are missing the right words for the narrower thing, or it has been considered too trivial to write up. Either way, here is what we measured, and here is what it looks like from inside the model.

## Related work and references

**Compaction as production practice.** Text-only summarization is already the standard fix for context-window limits across hosted frontier APIs — OpenAI Responses' automatic `context_management` compaction and standalone `/responses/compact` endpoint, Anthropic's beta `compact_20260112` context-management edit, and Gemini's Managed Agents automatic compaction and Live API context compression all treat "summarize and evict" as the baseline, sometimes alongside opaque encrypted continuation objects (OpenAI's encrypted `compaction` item, Anthropic's thinking signatures, Gemini's thought signatures) whose internal contents are not publicly documented. On the research side, "Models Take Notes at Prefill: KV Cache Can Be Editable and Composable" demonstrates that KV-cache state can be edited and spliced across contexts; "Fast KV Compaction via Attention Matching" performs true latent-space KV compaction; "Parallel Context Compaction for Long-Horizon LLM Agent Serving" studies production-style agent summarization as a serving problem; and KVLink, CacheBlend, and StreamingLLM-style attention-sink eviction address cache reuse and eviction in adjacent (chiefly RAG/serving) settings. [1] Full survey, links, and the "documented vs. inference vs. unknown" evidence grading are in `provider-compaction-prior-art-review.md`.

**[2] Logit lens.** nostalgebraist, "interpreting GPT: the logit lens," LessWrong (2020) — projecting an intermediate residual-stream state through the unembedding matrix and reading it as if it were the final-layer output. This is the lineage of the §4 readout; we use the original, plain logit lens there, and its Jacobian refinement (below) for the internal work in §5–7.

**[3] Lens caveat.** Its documented caveat — attributed in our notes to Neel Nanda — is that lens-style readouts are hypothesis-*generation* tools, not hypothesis-*validation* tools, with an uncharacterized false-positive rate. We treat the J-lens accordingly throughout §5–7: as a cross-check on a behaviorally established effect, never as standalone proof.

**[11] Tuned lens.** Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" (2023) — an affine per-layer correction to the logit lens that removes much of its basis-mismatch distortion; the conceptual midpoint between the plain logit lens and the J-lens.

**[12] J-lens / Jacobian lens.** Gurnee, Sofroniew, Pearce, Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong, Wang, Thompson, Abrahams, Kantamneni, Ameisen, Batson, and Lindsey, "Verbalizable Representations Form a Global Workspace in Language Models," Transformer Circuits Thread (2026), with Anthropic's `anthropics/jacobian-lens` reference implementation and Neuronpedia's public Jacobian-lens weights for Qwen3.6-27B. The J-lens transports an intermediate residual-stream activation into the final-layer basis via averaged-Jacobian transport before unembedding — more general than the plain logit lens (it surfaces forward-looking, sometimes unspoken concepts and unifies "read" and "write" in one coordinate system) but narrower than sparse-autoencoder features (confined to single-vocabulary-token concepts). Public lens weights exist for Qwen3.6-27B only, which is why our §7 same-model confirmation re-ran *behavior* on 27B rather than illustrating the lens on a different model.

**[4] Activation patching / representation editing.** Value grafting (§2) belongs to the broader family of causal-intervention interpretability methods that patch or edit internal model activations to test what a component carries — e.g. the ROME-style locate-and-edit tradition (Meng et al., "Locating and Editing Factual Associations in GPT") and causal-mediation/activation-patching methods more generally. Our contribution within that lineage is narrow: a values-only KV-cache graft applied specifically at a text-summarization boundary, not a general-purpose editing method.

**[5] Sparse autoencoders / dictionary learning.** Anthropic, "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" (Transformer Circuits thread, 2023) — the contrasting interpretability approach referenced in [12]: SAEs recover a large basis of (comparatively) monosemantic features via unsupervised dictionary learning, rather than the logit-lens family's narrower, unembedding-anchored, single-token readout.

**[6] Rotary position embeddings.** Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) — the position-encoding scheme carried in the key vectors, which is the mechanistic basis for the §10 conjecture that *keys* carry the addressing information value-only grafting cannot supply.

**Benchmarks.** [9] Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" and its curated "SWE-bench Verified" subset. [10] Sierra's tau-bench ("τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains") and its tau2-bench follow-up. Aider's Polyglot benchmark (see the Aider project). LongMemEval, a long-term conversational-memory benchmark. [8] Qwen3-30B-A3B-2507's own model card documents Aider-Polyglot (55.1%) and tau-bench (32–71%) results but does not cite base SWE-bench for this non-Coder variant — used in-repo as a vendor-supplied "difficulty certificate" that independently predicted both this project's SWE-bench 0/7 boundary and its tau-bench choice (§9).

**[7] KV-cache mechanics.** Standard attention/Transformer references (e.g. Vaswani et al., "Attention Is All You Need") describe the key/value cache underlying incremental decoding that §1 relies on.

**On the closing novelty claim (§10).** The prior-art review above shows real adjacent work — KV-cache editing/composition, latent-space KV compaction, and cache reuse/eviction all have documented prior art, and hosted providers already ship opaque compaction-state objects as a product shape. That prior art narrows, rather than voids, the claim in §10: what we could not find under the terminology we searched was specifically *write-time value state deliberately preserved and grafted back across a text-summarization boundary*, as distinct from compressing, reusing, or evicting cache state in general, or from exposing an opaque (but mechanistically undocumented) compaction handle at the product-API layer.

---

*Designed and largely executed by Claude Fable 5, continued by Claude Opus 4.8 (as Fable's quota exhausted), with subagent work (judging, dataset scouting, task-adapter construction, and the J-lens intervention probes on Qwen3.6-27B) by Claude Sonnet 5 and adversarial review by OpenAI's GPT-5.5 — directed, sanity-checked, and funded by the user.*
