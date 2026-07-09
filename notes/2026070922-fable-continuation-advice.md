# Fable — how do we continue (or redirect) ValueGraft? (2026-07-09)

*Author: Fable, invoked un-anchored with full authority to redirect. Method: I re-derived every
load-bearing number from the on-disk result files and the arm code myself; I did not trust the
owner's framing, Claude's summaries, STATE.md, or even the two prior audits (2026070976/77). Where I
correct a prior claim I show the disk evidence. No GPU/pod/MLX was run — read + CPU only.*

---

## TOPLINE (read this first)

**What we actually have that is good and usable (bf16 AND value-only ValueGraft AND methodologically sound):**

1. **effect_bound** — the cleanest test we own. bf16, value-only (α_K=0), placebo-controlled, CI'd,
   on the synthetic recovery plants. Verdict on the effect we care about (E−B) is **NULL on both bf16
   models** (Qwen3.6-27B: +0.017, CI [−0.033,+0.065]; Qwen3-30B-A3B: −0.049, CI [−0.113,+0.014]).
   It *does* prove value grafting is **content-specific** (E ≫ random-value placebo, CI excludes 0)
   and **non-destructive vs a random graft** — but **indistinguishable from plain compaction**. VALID
   as a **bound/negative**, not a win.
2. **SWE-Gym E-tuned** — bf16, value-only (α_K=0, verified in `blend_values`, keys untouched),
   production layout, α_V=0.75. **I recomputed it and the prior audit UNDERSOLD it** (see correction
   below): E−B = **+0.0156 nats, and the paired task-bootstrap CI is [+0.005, +0.027] — it EXCLUDES
   ZERO**, and it **changes the greedy generation on 49/75 tasks** (not "render-null"). This is the
   **only genuine bf16 + value-only POSITIVE on disk.** It is small (~9.5% of the A−B gap), it is
   teacher-forcing logprob (not solve-rate / behavioral correctness), and it ran under the **brief
   (baseline-handicapping) summary** (median summary = 112 tokens, not the 300–500-word production
   condenser). So: real, correctly-armed, correct-precision, significant — but modest and caveated.
3. **cross_arch_done** (bf16, value-only, native render, α_V=0.75) — on the SAME synthetic plants,
   E−B is **negative and significant** across architectures (phi-4 −0.064, Qwen3-32B −0.064,
   Qwen2.5-32B −0.23; Mistral −0.012 CI [−0.027,+0.002] null). Supporting/scale evidence — and it
   **reinforces the null**: on synthetic plants the value-only graft is null-to-*harmful* at α=0.75.

**The two things the paper has been calling its cornerstones are BOTH disqualified** (I independently
confirm the prior audit here): the **recovery** headline (judged sense/referent) is **4-bit MLX only**
and its bf16 placebo-controlled version (effect_bound) is null; the **honesty** headline is 4-bit MLX,
tests the **wrong arm** (H-pack = packed α_K=1,α_V=1, no tail — not value-only ValueGraft), and is
**layout-driven and content-agnostic** (H-pack-wrongS, a *different* conversation's state, suppresses
fabrication 24/24 too).

**Is there a valid paper in what we have? YES — and there are TWO honest write-ups, not one.**
(a) a **NEGATIVE/BOUNDING + INSTRUMENT-DISSOCIATION** science note (§2), and (b) a **candid
POSTMORTEM / methods-retrospective** of how an improvised, under-specified "vibe research" project
produced a string of retracted headlines (§5). **My recommendation: the postmortem is the more
valuable and more honest primary deliverable, with the bounding result folded into it as the
technical core.** The utility question is effectively answered — the approach does not robustly
demonstrate a value-only bf16 recovery effect, and we have the bound to prove it — so the highest-value
thing left to write is *what this was and how it went wrong.* Details §2, §5.

**Should we redirect? YES, in emphasis — and I have HIGH confidence about the single reframing that
matters (below). The one live positive signal is on REAL coding trajectories, not on the synthetic
probe, and the synthetic probe actively disagrees with the real-task data. Stop leading with the
synthetic probe.**

**Recommended direction (confidence):**
- **Write the honest bounding paper now** — HIGH confidence it's valid and defensible.
- **If you want a positive headline, the ONLY place it can live is the real-task (SWE-Gym) instrument
  — and it must be re-run at bf16 under the PRODUCTION summary with a behavioral metric.** MEDIUM
  confidence that signal survives production summary; if it dies, that itself completes the negative paper.
- **Do NOT invest more in the synthetic sense/referent probe as the primary instrument.** HIGH
  confidence it is the wrong instrument (it is null-to-harmful at bf16 and disagrees with real data).

**The single most important thing to change:** *You have been driving the entire project off a
made-up synthetic sense/referent probe that (a) only ever "worked" at 4-bit on 12 hand-authored
conversations and (b) is null-to-harmful at bf16 and disagrees with your one real-task result. Pick
ONE instrument that is simultaneously bf16, value-only, a real production-like task, behaviorally
scored, and with a pre-committed α — and make the whole project that one measurement. Everything else
is supporting. The split-brain (4-bit-local core evals vs bf16-pod side evals) is the mechanism that
let a 4-bit, wrong-arm result masquerade as the bf16 cornerstone; unifying the instrument kills that
whole class of error.*

---

## §1. WHAT DATA IS ACTUALLY GOOD AND USABLE (with corrections to our docs)

Precision/arm read from on-disk `model`/`dtype` fields and the arm code. I agree with almost all of
2026070976; I am **correcting it in two material places** (SWE-Gym CI and render-null), and adding the
cross-arch negative and the SWE-Gym summary-condition facts.

### (a) bf16 + value-only + sound → can legitimately speak to presence/absence of the effect

| Result | bf16? | value-only (α_K=0)? | Sound? | Speaks to the effect | My recomputed number |
|---|---|---|---|---|---|
| `effect_bound` (27B + 30B-A3B) | ✅ | ✅ (`blend_values`) | ✅ placebo-ctrl, CI'd, preregistered | **ABSENCE (null E−B)** on recovery plants | 27B +0.017 CI[−0.033,+0.065]; 30B −0.049 CI[−0.113,+0.014] — **both NULL** ✔confirmed |
| `swegym_30b_bf16` E-tuned | ✅ (`dtype=bfloat16`) | ✅ (`blend_values` keys untouched) | mostly (TF-only, brief summary, no correctness metric) | **PRESENCE (small +), the only one** | E−B **+0.0156, CI [+0.005,+0.027] EXCLUDES 0**; gen changes 49/75 |
| `cross_arch_done` (phi-4/Qwen3-32B/Qwen2.5-32B/Mistral-24B) | ✅ | ✅ | ✅ native render, gated, CI'd | **ABSENCE→HARM** on recovery plants | E−B negative & sig on 3/4 archs |
| `cross_arch_wide` (Mistral-3.2/30B-A3B) | ✅ | ✅ | ✅ | supporting scale/diversity | (consistent) |

### (b) Corrections this consult establishes (beyond 2026070976)

1. **The SWE-Gym positive is significant and NOT render-null.** 2026070976 called it "un-CI'd, TF-only,
   render-null" by inspecting only `t0001` (where A=B=E happen to be byte-identical). Over all 75
   tasks: paired bootstrap **CI [+0.005, +0.027] excludes zero**, and E-tuned's greedy generation
   **differs from B on 49/75 tasks**. So it is a *significant* bf16 value-only effect on next-token
   prediction that *does* change decoded output. (Command: paired percentile bootstrap over the 75
   `swegym_30b_bf16/t*.json` E-tuned−B `tf_mean` deltas, seed 1, 10k resamples.) **This upgrades the
   SWE-Gym result from "too thin to matter" to "the one real positive — small, but real."**
2. **But that positive is under the BRIEF summary, not production.** `run_swegym_hf.py:58` defaults to
   `SUMMARY_REQUEST_PROD`, but the on-disk `swegym_30b_bf16` set (Jul 5) predates that knob
   (commit `9985769`) and has median summary length **112 tokens** (max 272) — i.e. the terse,
   baseline-handicapping brief condition, not the 300–500-word production condenser. So even our one
   positive inherits the owner's caveat #6: **no positive exists under a production-faithful summary.**
3. **On the synthetic recovery plants, bf16 value-only graft is null-to-HARMFUL**, not merely null:
   cross_arch E−B is significantly negative on phi-4/Qwen3-32B/Qwen2.5-32B. Combined with the
   effect_bound null, the synthetic-recovery verdict at bf16 is unambiguous: **the method does not
   recover, and at α=0.75 with native render it slightly hurts.**

### (c) NOT usable as the cornerstone (confirmed)
- **Recovery judged sense/referent** (`raw_30b`, `raw_brief_repro`, etc.) — 4-bit MLX only. Supporting/
  motivation at most, always labeled 4-bit, always stated next to the effect_bound bf16 null.
- **Honesty / H-pack** (`phase2_30b`, `phase2_4b`) — 4-bit MLX, wrong arm (packed coupled-KV), layout-
  driven, content-agnostic. Supporting at best, never a ValueGraft claim. The bf16 answer set
  (`phase2_30b_bf16`, 12 convs) exists **unscored**; queue built (`phase2_30b_bf16_judge_queue.json`).
- **LongMemEval** 4-bit headline — bf16 scoring incomplete.
- **tune_*/guard_* dirs** — bf16 by dir-name convention only (no in-file model field); LOWER confidence,
  verify against launch scripts before citing precision.

---

## §2. IS THERE A VALID PAPER IN WHAT WE HAVE? — Yes, one honest paper.

**Title it as what it is: a bounding / instrument-dissociation study of value-only write-time KV
grafting at the compaction boundary.** It can honestly claim ALL of the following, each disk-backed:

1. **A preregistered, placebo-controlled bf16 NULL for value-only recovery.** On planted sense/referent
   facts, grafting write-time summary *values* (fresh keys) is **content-specific** (E ≫ norm-matched
   random-value placebo, CI excludes 0) and **non-destructive**, but **statistically indistinguishable
   from plain summary compaction** (E−B null on two bf16 models; negative on 3 more archs). *Value
   state alone does not carry the evicted meaning back through fresh keys.* This is a real, citable,
   mechanistically-interesting negative — it isolates that **keys, not values, gate recovery** (see
   prior art, §3).
2. **An instrument dissociation (the most novel, honest thing you have).** The *same* method, same
   precision, same α: **null-to-harmful on the synthetic probe, small-but-significant-positive on real
   coding trajectories** (SWE-Gym E−B +0.0156, CI excludes 0). Your synthetic instrument does not track
   the real-task effect. This is a genuinely useful methodological result for the field (people build
   these synthetic KV probes all the time) and it directly validates the owner's own suspicion that the
   made-up probe might be measuring the wrong thing.
3. **A packed-KV-retention behavioral phenomenon (supporting).** Packing write-time summary keys+values
   (H-pack) shifts a model from fabrication to admission on decoys — but this is **layout-driven and
   content-agnostic** (wrongS reproduces it), so it is a compaction-*regime/calibration* finding, not a
   value-carried-meaning finding. Score `phase2_30b_bf16` (cheap) to state whether it's precision-robust.
4. **Damage characterization + α/layer/head/champion analysis** as mechanism/robustness, honestly
   labeled 4-bit vs bf16.

**What it CANNOT claim:** that ValueGraft recovers post-summarization accuracy/behavior in general; any
positive behavioral (solve-rate) result; anything at production-summary strength; anything from the
honesty arm about the value-only method.

**Is it worth writing? Yes** — as an honest "here is a clean bound + a cautionary instrument result,"
it is publishable (workshop / short paper / strong blog) and it is *true*. It is not a flagship
"we made compaction better" paper, and you should not dress it as one.

---

## §3. SHOULD WE REDIRECT? — Yes, in emphasis. The only positive lives in the real task.

The data draws a bright line: **synthetic recovery probe → the effect is absent/harmful at bf16;
real coding trajectory → the effect is present (small) at bf16.** Two implications:

**(i) The synthetic sense/referent probe is very likely the wrong instrument.** It was, by the owner's
own account, made up on a whim; it only ever produced a positive at 4-bit on 12 hand-authored convs;
at bf16 it is null-to-harmful; and it disagrees with your one real-task signal. Continuing to pour
effort into it (native-render verification, more plants, per-head α on it) is polishing an instrument
that is telling you "no." **Stop treating it as primary.**

**(ii) If a positive paper is wanted, run the real-task instrument properly.** Proposed decisive
experiment:

- **Design:** SWE-Gym (or any real agent-trajectory) compaction, bf16, 30B, value-only (α_K=0),
  **α_V swept {0.25,0.5,0.75,1.0}**, under **BOTH the production condenser summary AND the brief
  summary** (the prod-vs-brief contrast is the whole ballgame per caveat #6).
- **Metric — add a behavioral one, don't stop at TF-logprob:** (a) keep TF-logprob of the gold turn
  with a proper task-clustered CI (cheap, already significant under brief); (b) add a **behavioral
  proxy that doesn't require solving the task**: LLM-judge agreement between the *generated* next
  action and the gold next action (edit-distance or judge "same intended action?"), on ~200–300 tasks.
  Full unit-test solve-rate is the gold standard but is prohibitively expensive on a 30B (rarely
  solves) and slow under KV-edited inference — do NOT gate the paper on it; the gen-vs-gold-action
  judge is the cheap behavioral signal.
- **Pre-commit α and the stopping rule** (α is currently 0.75 everywhere, tuned on the 4-bit synthetic
  sweep — i.e. it is *out-of-sample* for SWE-Gym, which is good; keep it fixed or sweep-then-report-all).
- **Rough cost:** one warm bf16 30B pod, ~300 tasks × 6 arms × 2 summary conditions × 4 α ≈ tractable
  in a day of pod time; the behavioral judge is cheap CPU/API. Much cheaper than any solve-rate run.
- **Confidence:** MEDIUM that the +0.0156 survives the production summary (the brief summary
  handicaps B, so prod will shrink headroom — this is the real risk); HIGH that this is the *right*
  place to look and that the result (positive or null) will be *decisive* either way.

**Prior art to stand on (verify before citing — I flag these from the compaction/KV literature, and
your own note 2026070629 already reviews provider compaction):**
- **StreamingLLM / attention sinks** (Xiao et al.) — motivates the packed-sinks layout; explains why
  H-pack's *layout* (not content) drives the honesty shift. Directly relevant to reframing honesty.
- **CacheBlend** (Yao et al., 2024) and the KV-cache-fusion line — when you splice precomputed KV from
  another context you must **recompute a small fraction of KV (the high-attention-deviation tokens),
  and keys/attention structure matter more than values.** This *predicts your null*: value-only grafting
  through fresh keys should not recover meaning, because the keys carry the positional/attention
  addressing. Your effect_bound null is a clean confirmation of that prediction and should be framed
  as such — it makes the negative result *expected and mechanistically grounded*, which is stronger,
  not weaker. **This is the single most important citation to anchor the paper.**
- **H2O / Scissorhands / FastGen** (KV eviction) and **activation/KV steering** (the "add a value
  vector" line) — position value-only grafting as a steering-style intervention and explain why it's
  content-specific but weak for *recovery*.

---

## §5. THE POSTMORTEM PAPER — assess + structure (owner's added option)

**Assessment: yes, write it, and make it the primary deliverable.** The owner is right that we have
effectively answered the utility question (value-only bf16 ValueGraft does not robustly recover
post-summarization accuracy; here is the placebo-controlled bound and the one small real-task signal
that dies outside a handicapped summary). Once that is the answer, the most useful and most honest
document is not a dressed-up null — it is a candid account of *what the experiment was, how it was run,
and every way it went wrong.* This is a real genre (postmortem / negative-results methods retrospective)
and it is worth the write. Tone must be **dry, specific, a little clinical — a postmortem, not an
inspirational "we learned so much" arc.** The failures are the content; do not redeem them.

**Honest framing of the process (part of the story, stated plainly, not self-flagellating):** this was
improvised research — directed casually and intermittently (largely dictated on the fly), with no
dedicated block of focused time spent building a rigorous measurement framework or orchestrating
execution up front, ~US$300 of compute spent, and repeated doubling-down on whatever line still showed
a number when the previous line failed. That process is not an aside; it is the *mechanism* that
produced most of the failure modes below. The lesson is about how under-specified, momentum-driven
research goes wrong, not about any one bug.

**The failure patterns to document (each is disk/history-grounded — I pulled these from the daily
meta-summaries and the audit trail; paraphrase owner intent, do not quote the casual dictation):**

1. **Publishing infrastructure before a validated result.** A public repo, paper draft, and blog draft
   were started 07-05, before any result had survived its own controls. This created standing
   sunk-cost pressure to keep *a* headline alive and to relabel rather than retract.
2. **Reframing churn driven by whatever survived.** The central claim moved mechanism → "mitigation"
   → recovery(sense/referent) → honesty-led → negative, each step forced by the prior line failing
   external review or its own controls. Direction was set by "what still shows signal," not by a
   pre-registered question. (External review as early as 07-05 correctly said the base mechanism claim
   was near-self-evident; the project kept re-spining instead of concluding.)
3. **Chasing surviving numbers across instruments.** LongMemEval came back null (full-context 52.5% vs
   2.8–6.9% for every compacted arm) → dropped, pivot to coding; coding hit a capability floor (a 30B
   rarely solves SWE tasks, and compaction never fails at the tested granularity, so the graft could
   only "do no harm") → pivot to chains; the chain recall probe was then found invalid (it referenced
   constants from an unrelated task family) → pivot to J-lens; and so on. Each pivot chased the
   instrument where a number still looked alive.
4. **A statistical artifact masqueraded as the headline.** The recovery effect was estimated with a
   mean-of-ratios (E−B)/(A−B), which is Cauchy-unstable near zero denominators; on robust metrics the
   headline shrank to a qualitative dissociation. The earlier "keys actively hurt" claim was the *same*
   artifact. A headline lived for days on an unstable estimator.
5. **Render-fragility mistaken for / entangled with signal.** The recovery effect only appears when the
   model generates *its own* summary at the boundary (native render); a semantically-equivalent
   supplied summary does not carry it. Large amounts of work (native re-rendering, banking) were
   motivated on the premise that self-render is the valid measurement — a premise never rigorously
   verified before it drove the effort.
6. **Split-brain runtime → provenance/precision confusion.** Core behavioral evals ran LOCAL at 4-bit
   MLX; the pods ran bf16 on different harnesses. The "bf16 cornerstone" the paper assumed was never
   true for the core evals — the scored recovery and honesty results are 4-bit MLX, yet were labeled
   bf16 in CLAIMS/paper. 4-bit and bf16 are different models and were treated as convertible.
7. **Arm/method conflation.** The honesty headline used H-pack (packed keys+values, α_K=1,α_V=1, no
   tail) — a different intervention from the value-only method the paper is named after. A 07-06
   arm-vocabulary reframing froze historical arm IDs and made "write-time KV" (H-pack) easy to conflate
   with "ValueGraft" (value-only).
8. **Effect size treated as validity.** A large, significant honesty number was promoted to cornerstone
   without first checking its precision or its arm identity. Both were wrong. A huge significant effect
   on the wrong arm at the wrong precision is not a cornerstone.
9. **Controls were run late, and when run, they killed the headlines.** The placebo (effect_bound) and
   the wrong-source control (H-pack-wrongS) — the things that should gate a claim — were added after the
   headlines were already in the draft, and both came back showing the effect was null vs compaction
   (recovery) or content-agnostic (honesty).
10. **Process-integrity lapses compounded the confusion:** an autonomy overreach, a retroactive
    misrepresentation of progress, and a multi-hour model-attribution error in commit trailers — small
    individually, but they degraded the trustworthiness of the running record the project depended on.

**Suggested structure (dry, ~6 short sections):**
- **1. What we set out to test.** Value-only write-time KV grafting at the compaction boundary to
  recover post-summarization accuracy; why it's plausible; the intended bf16 ~30B target.
- **2. What we actually found (the technical core = the §2 bounding result).** The placebo-controlled
  bf16 null on synthetic recovery; the content-specific-but-non-destructive finding; the one small
  significant real-task (SWE-Gym) signal and its brief-summary caveat; the instrument dissociation.
  Ground it in prior art (CacheBlend/StreamingLLM, §3) so the null reads as *expected*.
- **3. How it was run.** Improvised/vibe-directed, intermittent, ~$300, MLX-local + pod split, no
  up-front measurement framework. State it plainly as the setting, not an excuse.
- **4. Failure patterns (the heart of the piece).** The ten above, each: what happened, why the process
  allowed it, how it was caught, what it cost.
- **5. What would have caught each earlier.** Pre-registered single instrument; controls before drafts;
  precision+arm recorded per result; robust estimators from the start; no repo until a result survives
  its own controls. (Derive from the failures; do not moralize.)
- **6. Honest status.** No robust positive bf16 value-only recovery effect; a clean bound; a
  methodological caution about synthetic KV probes. What we would and would not do again.

**Combine or separate?** One document is better: the postmortem *is* the paper, with §2's bounding
result as its technical core (section 2 above). A standalone "clean null" science note is defensible
but weaker and slightly dishonest-by-omission about how the null was arrived at — the postmortem is the
more complete truth and the more useful artifact. It also needs **zero new GPU.**

---

## §4. HONEST ORDERED RECOMMENDATION (proportionate)

1. **Score `phase2_30b_bf16` now (CPU/LLM-judge, queue already built).** Cheap, strictly informative:
   confirms or kills the "honesty is a 4-bit quant artifact" objection. Report as SUPPORTING (packed-KV
   regime, layout-driven), never as ValueGraft. **Do it — highest value-per-dollar action on disk.**
2. **Add the corrected SWE-Gym number to CLAIMS.md with its real CI and caveats:** E−B +0.0156, CI
   [+0.005,+0.027], gen changes 49/75, **BRIEF summary**, TF-logprob only, α out-of-sample. This is your
   only bf16 value-only positive; record it accurately (the current docs undersell it as un-CI'd/render-null).
3. **Do the precision-provenance correction pass** on CLAIMS/FINDINGS/paper (per METHODS-PROVENANCE-
   REQUIREMENTS.md): demote both "cornerstones," add the §1 table, state the effect_bound null and the
   cross-arch negative as the *bf16 value-only recovery verdict*.
4. **Write the POSTMORTEM as the primary deliverable (§5), with the §2 bounding result as its technical
   core.** You can write ~80% of it from disk today, zero new GPU. This is the most honest and most
   useful thing left to produce; the "vibe research → cascading retracted headlines" story is real,
   specific, and worth telling dry. A standalone clean-null note is the fallback if a postmortem is
   unwanted, but it is the weaker, less-complete artifact.
5. **THEN, only if you want a positive headline, run the §3 SWE-Gym prod-vs-brief + behavioral-metric
   experiment.** This is the one experiment that can either (a) rescue a modest positive claim or
   (b) cleanly close the negative paper ("even on the one real signal, the production summary erases it").
   Either outcome is a finish line. Do NOT run more synthetic-probe or honesty experiments hoping for a
   cornerstone — the disk already says no.
6. **Do NOT** spend on: fresh synthetic-recovery runs, more native-render verification *of the synthetic
   probe*, per-head α on the synthetic probe, or breadth-for-breadth cross-arch. All are polishing the
   wrong instrument.

**Minimal path to a truthful, defensible result:** score phase2_30b_bf16 (step 1) → correct the ledger
(steps 2–3) → write the bounding/dissociation paper (step 4). That is fully honest and needs **zero new
GPU**. The §3 experiment is the *only* thing worth new GPU, and only if a positive headline is the goal.

---

## Answers to the four asks, in one line each
1. **Good/usable bf16+value-only data:** effect_bound (null bound, sound), SWE-Gym E-tuned (small
   *significant* positive — corrected), cross_arch (bf16 negative on synthetic). That's it.
2. **Valid paper?** Yes — TWO honest write-ups: a bounding + instrument-dissociation science note (§2)
   and, better, a candid POSTMORTEM/methods-retrospective (§5) with the bound as its technical core.
   Neither can claim recovery / behavioral win / production-strength / honesty-as-ValueGraft.
3. **Redirect?** Yes in emphasis: abandon the synthetic probe as primary; make the real-task (coding)
   instrument, at bf16, under production summary, with a behavioral metric, the thing you measure (§3).
4. **Next step:** score phase2_30b_bf16 (cheap, do it) → correct ledger → write the honest paper;
   spend new GPU only on the §3 SWE-Gym prod-vs-brief behavioral run if a positive headline is wanted.

*Certainty: effect_bound null, SWE-Gym CI/gen-diff, cross_arch negatives, value-only arm identity, and
the brief-summary provenance are all recomputed from disk/code this session and I am confident.
Prior-art citations are from memory and flagged to verify. The MEDIUM confidence on "does SWE-Gym
survive prod summary" is the one genuine unknown and is exactly what §3 resolves.*
