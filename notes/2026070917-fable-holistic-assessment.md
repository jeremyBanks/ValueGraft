# Fable — holistic assessment of ValueGraft (2026-07-09)

*Independent, un-rubber-stamped read. I read AGENTS.md, STATE.md, CLAIMS.md,
FINDINGS.md, paper/DRAFT.md, notes/README.md, the framing-provenance note, and
spot-checked the disk (raw_brief_repro/, cross_arch_done/, git log). Numbers
below are cited to files. Where I disagree with the current plan I say so
plainly, because that is what was asked of me.*

---

## 0. Topline

The project is in **much better epistemic health than most research at this
stage** — the audit culture (CLAIMS.md, un-anchored Fable, bank-every-render,
the `[BLOCKED]` draft) is genuinely excellent and is the main reason I trust
anything here. But the *positive* story is currently **thinner and more
conditional than the momentum around it implies**, and there is one structural
risk that I think is underweighted in the plan: **the two most-cited positive
results — judged sense +12pp and SWE-Gym +0.0156 — both live under the BRIEF
summary condition, which was deliberately designed to handicap the Compacted
baseline.** That is not disqualifying, but it means the paper does not yet have
a single positive result measured under a production-faithful condition. Fixing
that is, in my view, higher-leverage than anything currently running.

My single biggest recommendation: **do not let "sense-led, gated on the held-out
test" be the whole bet.** You are holding a robust, banked, scale-replicated
result (the honesty/anti-fabrication effect) and gating the paper's headline on
a *fragile* effect measured under a *handicapped* condition whose held-out
confirmation **isn't done yet** (5/24 renders on disk). That is backwards on
risk. More below.

---

## 1. How is it going, honestly?

**The good — and it is real, not politeness:**

- **The audit discipline is the project's crown jewel.** CLAIMS.md recomputing
  every shippable number from disk at a named commit, and *catching its own
  headline* (F2's "38/48 = 79% accurate" is `❌ UNSUPPORTED` — H-pack recalls
  **0/24** evicted facts, CLAIMS.md:195-208) is exactly the behavior that makes
  a small-lab result credible. Most projects would have shipped 38/48.
- **The un-anchored-Fable → "+0.10 is FRAGILE not a regression" verdict**
  (FINDINGS.md:615-630) is the correct call and correctly reasoned: sense,
  stance, and headroom all reproduced; only the high-variance referent moved;
  +0.012 == the *lower bound* of the original CI. That is regression-to-the-mean
  from a barely-significant unbanked draw, diagnosed without spending GPU. Good.
- **The draft's `[BLOCKED]` scaffolding is the right way to write a paper whose
  headline isn't decided.** Sections 1–7 are framing-independent and genuinely
  strong; the verdict/abstract/discussion are honestly stubbed. This is how to
  avoid writing yourself into a claim.

**The structural risks — in priority order:**

1. **The BRIEF-condition monoculture (biggest).** PROV-2 (CLAIMS.md:341-353) and
   REC-6 (CLAIMS.md:123-136) establish that *both* the judged +12pp headline
   *and* the most-stable real-content result (SWE-Gym) were measured under
   `SUMMARY_REQUEST_BRIEF` — a terse summary "designed to starve the text channel
   and handicap the Compacted baseline." A skeptical reviewer's first sentence
   will be: *"You showed your method beats a baseline you deliberately
   weakened."* The mechanism-isolation rationale is legitimate, but right now
   there is **no positive result on record under the production-faithful (std)
   summary.** FINDINGS.md's own banner (line 21) calls the std number "a
   separate, arguably more decision-relevant result." I agree — and it does not
   exist yet. This is the paper's soft underbelly.

2. **The positive story rests on one model.** The graft is clearly positive
   *only* on the Qwen3-30B-A3B MoE anchor (weakly on Mistral referent), and
   null-to-strongly-negative on every dense model (DRAFT §6 table, lines 613-619).
   And the anchor's referent pole is precisely the fragile one. So the arch map
   is currently "one fragile positive vs several solid negatives." That can be an
   honest and interesting paper — but it is *not* the "architecture-specific
   recovery mechanism" headline, and the draft (correctly) refuses to call it
   that yet.

3. **The held-out decider is unfinished and could fail its own positive
   control.** `results/raw_brief_repro/` has **5 of 24** conversations rendered;
   no `judge_semantic_holdout*` exists yet. And note the trap: this run must
   *first* reproduce +12pp on c01-c12 baseline under clean current code — the
   same kind of positive control that the *referent* anchor just **failed**
   (BLK-1, +0.012 spans 0). Treat baseline reproduction of the sense +12pp as a
   live risk, not a formality. If it regresses like referent did, the sense-led
   paper loses its spine mid-write.

**Net:** on a path to a *credible, honest* contribution — yes, almost
regardless of how the decider lands, because the honesty result + the damage
characterization + the methodology traps are real and banked. On a path to a
*strong positive* "ValueGraft recovers meaning" headline — that is genuinely
uncertain and currently over-exposed to a single pending test under a
handicapped condition.

---

## 2. Reflections on the existing plan

Plan (from STATE.md:60-62): held-out decider → arch poles → Fable un-anchored →
framing → robust-metric audit → pods off → write.

**What's right:**
- Gating the framing on the decider. Correct.
- Champion scans that *bank c01-12 renders* for the four models we never banked
  (STATE.md:38-39) — good, fills a real reproducibility gap cheaply, and reuse
  of banked renders for cs30b is validated. Fine to let run.
- CLAIMS.md as the assembly source for the paper. Keep this rule hard.

**What's mis-prioritized or missing:**

- **MISSING, and I'd rank it #1: a production-faithful (std-summary) arm for the
  headline metric.** The plan mentions "std/production-faithful held-out =
  planned second arm" (STATE.md:57) almost in passing. It should be a
  first-class deliverable, not a tail item. Without it the paper's positive
  claim is brief-only, and that is the review's easiest kill. If std shows the
  effect *shrinks but survives* → far stronger paper. If it *vanishes* → you
  learn the effect is a mechanism-isolation artifact *before* you publish, which
  is the whole point of the audit culture.

- **MIS-PRIORITIZED: leading with sense-recovery over the honesty result.** See
  §1.1. The honesty/anti-fabrication result (F2) is banked, scale-replicated
  (4B→30B), precision-replicated (4-bit→bf16), and deployment-relevant. Its one
  overclaim (38/48) was caught and the *corrected* claim is arguably cleaner and
  still strong: **compaction induces confident fabrication about evicted content;
  write-time KV converts fabrication (67%) → honest admission (96%), without
  restoring recall** (CLAIMS.md:210-215). That is a robust, defensible,
  genuinely useful headline that does *not* depend on the pending decider or the
  brief condition's favorability in the same fragile way. I would seriously
  consider **leading with honesty + the damage dissociation, and demoting
  sense-recovery to a bounded supporting result and the arch map to an
  appendix.** The owner's "sense-led" instinct is the more exciting paper; it is
  also the more fragile one, gated on unfinished work. At minimum, structure the
  paper so it degrades gracefully: if the decider disappoints, the paper should
  still stand on F2 + damage + methodology without a rewrite.

- **The robust-metric audit (#29) is partly already done** by CLAIMS.md. Don't
  re-litigate; just close the two open reconciles that touch shippable prose:
  (a) 83/17 → 79/12 decoy (F2-1), (b) locate the LME n=320 source file or quote
  the n=36 recompute (DMG-2 is "provenance-thin" — CLAIMS.md:148-162). These are
  small but they are *in the paper's honesty section*, which is the one place you
  cannot afford a soft number.

---

## 3. Suggested next steps (ordered)

1. **Finish the held-out judged decider** (running, 5/24). It is the designated
   gate; let it complete and take it to Fable un-anchored per standing rule.
   **Explicitly check the c01-c12 baseline reproduces +12pp under clean code** —
   do not assume it; the referent positive control already failed once.
2. **Run the production-faithful (std-summary) judged arm on the anchor** — the
   single highest-leverage missing experiment. This is the answer to the review's
   first objection and it is cheap on renders you are already banking. Pair it
   explicitly with the brief number in the paper.
3. **Let the champion/bank scans finish** (they fill the render-repro gap), then
   **pods off** the moment data collection is done, per the standing rule. Don't
   let writing overlap live GPU.
4. **Close the two honesty-section reconciles** (79/12 decoy; LME n=320 source or
   n=36 recompute) and lock the *corrected* F2 claim (suppresses fabrication /
   induces admission, does not restore recall). The 38/48 phrasing is already
   correctly quarantined in the draft — keep it dead.
5. **Native-render verification experiment** (see §4) — moderate, worth doing
   *before* the paper implies self-rendering is a hard requirement.
6. **Write the paper**, structured to degrade gracefully (F2 + damage +
   methodology load-bearing; sense/arch conditional on step 1). Heavy-Fable
   drafting is fine given the CLAIMS ledger exists as the fact source.

Note I have **de-prioritized "SWE-Gym across other models"** relative to the
current list — see §5; I think it is the wrong direction for that warm-pod
budget.

---

## 4. The native-render premise — my honest opinion

The owner's framing is exactly right and I want to sharpen it, because I think
there are **two different "self-rendering" claims** currently bundled together,
with **very different evidential support**:

**Claim A — own-summary dependence** ("the graft needs the model's *own*
generated summary, not a foreign one"). This is **well-supported and I believe
it.** The isolation test (FINDINGS.md:465-482, MTH-3) is clean: same model, same
harness, *only* the summary source changed — fixed Sonnet summary → referent
+0.004 / sense −0.147 (suppressed); self-gen → referent +0.136 / agg +0.090
(significant positive). That is a real mechanistic finding and it *matches
deployment* (production compaction uses the model's own summary). Keep it,
present it as a finding.

**Claim B — native-replies dependence** ("the *conversation body* must be the
test model's own in-context generations, not another model's"). This is **much
weaker than it is being treated**, and here is the specific problem: the headline
evidence for it is "foreign replies +0.009 vs native +0.12, same model"
(FINDINGS.md:503, DRAFT §4.4:503). **But that +0.12 is the exact unbanked draw
that regressed to +0.012 when banked** (§4.5 / BLK-1). So the native-vs-foreign
*gap* that motivates the whole per-model-native re-render program may be
`+0.012 vs +0.009` — i.e. **near zero** — rather than `+0.12 vs +0.009`. The
nativeness confound is *plausible* on mechanism grounds (a graft has no reason
to raise the probability of text the model wouldn't produce), and Fable's own
07-08 note flagged that the dominant term is the *target/continuation* side, not
the write side (FINDINGS.md:504-507). But the clean number that "proves" it is
now known to be fragile.

**So: required, helpful, or confound?** My read:
- **Own-summary: genuinely required** for the effect as measured, and a finding.
- **Native-replies: probably helpful and more robust/deployment-faithful, but
  NOT cleanly demonstrated to be required** — and the paper currently risks
  implying it is, on the strength of a fragile number.

**How the paper should present it honestly:** frame self-rendering as a
**design choice justified by (a) a supported mechanistic dependence on the
model's own summary and (b) deployment-faithfulness** — *not* as a proven hard
requirement on native replies. State the native-replies evidence with its
current fragility caveat.

**The verification experiment I'd actually run** (and I'd scope it tightly):
hold the model fixed (the anchor), hold the summary self-generated (so Claim A
isn't confounded in), and vary *only* the reply source — native vs
foreign-rendered replies — **on the banked sense-judged metric** (the surviving
load-bearing arm), not the fragile referent-logprob. Bank both renders. If the
native-vs-foreign gap on *sense-judged* recovery is real and survives banking →
Claim B is earned and the re-render program is vindicated. If it's near zero →
you reframe: "native replies matter for deployment realism, not as a measurement
requirement," which is *also* a clean, honest, publishable statement and
actually makes the results *more* general, exactly as the owner anticipated.

---

## 5. SWE-Gym across other models — brief opinion

The owner correctly flagged this as lower priority; I'd go further and say
**don't do the breadth version now**, and redirect the impulse.

SWE-Gym +0.0156 (REC-6, VERIFIED) is your most stable *real-content* result, but
it is (a) one model, (b) self-gen, and crucially (c) **brief condition** — the
same handicap as the judged headline. Spreading it across the warm fleet buys you
*more brief-condition, cross-model breadth* — which multiplies exposure to the
exact two weaknesses the paper already has (brief favorability + architecture
ambiguity), for a real-content metric where the dense models will likely go
negative just like they did on the synthetic map. That's a lot of "the effect
reverses on other architectures" surface for little upside.

**Better use of the same warm-pod budget: run SWE-Gym on the anchor under the
*std/production* summary condition.** That converts your most-stable result from
"holds under the favorable condition" to "holds (or shrinks, honestly reported)
under the deployment condition" — directly attacking the paper's biggest
weakness on *real* content. Depth on the anchor beats breadth across models here.
If breadth is wanted later, do it *after* the std anchor number exists, so the
cross-model story isn't built entirely on the handicapped condition.

---

## 6. One closing meta-point

The thing this project should be *proud* of is not any single effect — it is that
its process reliably converts hype into honest, bounded claims (the −0.31 stance
"null" → estimator artifact; "keys hurt" → keys neutral; +0.10 referent →
fragile draw; 38/48 → 0/24 + a cleaner corrected claim). That trajectory *is* a
contribution and the paper should say so plainly. My worry is only that the
current *plan* points the headline at the one result still moving through that
converter (sense-recovery, brief, held-out-pending) instead of at the results
that have already come out the other side (honesty, damage dissociation,
methodology). Aim the headline at what has already survived the audit, and let
the pending decider *upgrade* the paper if it lands — rather than *define*
whether there's a paper at all.

— Fable
