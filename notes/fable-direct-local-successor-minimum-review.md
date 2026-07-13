# Fable review — minimum direct-local powered successor

**Date:** 2026-07-12  
**Reviewer:** Claude Fable 5, invoked through the Claude CLI with a bounded,
decision-specific factual brief.  
**Scope:** Whether the direct local MLX continuation design can support the
mechanistic claim, and the minimum render/control structure required. This is
advice, not authority.

## Review supplied verbatim

Verdict below, from the stated facts plus the repo's own hard rules (which already bind you to applied-dtype controls and conversation-level N).

### 1. Does the primary answer the mechanistic question?

Yes, with one wording discipline. Paired E−B on identical visible text, held-out true-continuation teacher-forced mean logprob, is the direct operationalization of "do write-time value states improve later prediction." Value-only grafting is also the mechanistically correct minimal graft under RoPE (values are position-unrotated, so write-time values are coherent at compacted positions; keys are not). But as specified, E−B alone supports only "grafting *something* through this codepath changes prediction." Two confounds stand between that and the stated claim: (a) the graft machinery itself (dtype/geometry perturbation B never passes through), and (b) generic nonsemantic sharpening (any structured values lowering entropy raise mean logprob of plausible continuations). Wrong-history covers semantic specificity but is **not a clean null for (b)** — wrong values inject interference, so E > W is compatible with part of E−B being nonsemantic. Without closing (a) at minimum, the primary is a weaker proxy.

### 2. Minimum defensible version at N=48

- **Estimand:** mean over 48 conversations of x_i = clip(E_i − B_i, ±0.5), where E_i and B_i are each the average over the two carrier/render replicates *within* conversation i (replicates reduce variance; they are never units). One-sided 95% Hoeffding lower bound on mean(x). Claim requires LCB > 0.
- **Detectability, pre-registered:** with range 1 and N=48, the Hoeffding margin is √(ln 20 / 96) ≈ **0.177 clipped nats**. That is your minimum confirmable mean effect. Pre-register the gate: if median A−B headroom across the corpus is not comfortably above 0.177, the design cannot distinguish "effect absent" from "effect real but sub-threshold," and the honest outcome is "not confirmed at this N," not "no effect." Check this on the first few conversations before committing all 48.
- **Arms per conversation (5, all on identical visible text):** A (oracle, headroom gate), B (fresh compaction), B′ (sham graft: B's *own* freshly-encoded values pushed through the identical graft codepath), E (write-time value graft), W (wrong-history value graft, from the paired v13 fixture). At 1–3 s per continuation arm, the extra arms are noise in the budget; probe generations dominate — cut probes ruthlessly, they buy nothing for this estimand.
- **Corpus:** the frozen outcome-blind 6-per-stratum selection from the v13 32,768-candidate frame, blind review of complete histories and target minimality, replacement strictly by next frozen rank, all before any E/B measurement. c13–c54 demoted to descriptive replication at most.
- **Secondary, pre-registered:** E−W (specificity) and E−B′ ordering. Report unclipped means descriptively.

### 3. Is the missing nonsemantic placebo fatal?

Fatal to the *mechanistic* claim as currently specified, and your own AGENTS.md rule already says so ("a mathematical placebo is unavailable unless it passes at the actual dtype and state geometry"). But the fix is trivial and requires no new apparatus:

- **B′ sham graft (mandatory gate, run today):** graft the fresh-compaction values through the exact graft path. This is an identity test, not a tuned control — pre-register a tolerance (e.g., |B′−B| per conversation must be an order of magnitude below observed A−B headroom, threshold fixed before looking at E). If B′ ≠ B, every E−B number is confounded with 4-bit graft numerics and nothing else matters until that's fixed.
- **Position-permuted write-time values (cheap upgrade, recommended):** same tensors, same dtype, same norms, deterministic fixed-seed permutation of token positions within the grafted span. This is the true nonsemantic-structure placebo closing confound (b). No post hoc freedom: one seed, frozen now.

With B′ as a passing gate plus W, the primary claim "write-time value states carry history-specific information recoverable at the compacted position" is defensible. B′ alone is the floor; add the permutation arm since it costs seconds.

### 4. Shortcuts that recreate the prior failure; what launches now

Failure-recreating shortcuts, in order of danger:

1. **Counting anything but conversations as N** — renders, arms, plants, probe repeats. This is the pseudo-replication trap the c13–c54 corpus already embodies.
2. **Any contact with v13 cloud/release/lifecycle code.** That apparatus already consumed the budget once for N=1. Fixture compilation only.
3. **Trusting L0/L3 encode gates as covering the graft path.** They gate encoding identity, not applied graft numerics — that's exactly what B′ exists for.
4. **Outcome-visible selection or replacement** — the frozen rank order and blind review must complete before the first E/B logprob is computed, and replacements must be blind too.
5. **Deferring the replicate decision.** "Not yet chosen" is a post hoc doorway. Fix it now: two deterministic carrier renders per case, averaged, frozen in the prereg.
6. **Spending latency budget on 70-probe generations** that don't feed the estimand.

**Launch immediately, in parallel:** (a) run the B′ sham-graft gate on the existing 8,736-token canary — it costs seconds and a failure invalidates the whole plan, so it must go first; (b) freeze the prereg (estimand, clip, Hoeffding, arms, permutation seed, replacement rule, headroom gate at 0.177) with a new design ID; (c) compile and freeze the 48-case selection and start blind review. Nothing about (b) or (c) should wait on apparatus work, and nothing about the apparatus should grow beyond the five arms above.

## Main-agent disposition

Accepted: the direct E−B continuation endpoint, conversations as the only N,
two within-conversation renders, A/B/sham/E/wrong-history arms, and a fixed
within-region position-permutation placebo. The v13 code is limited to its
CPU-only fixture frame/compiler; its remote lifecycle/release stack is retired.

Correction: the requested confident negative result uses the one-sided 95%
**upper** Hoeffding bound. The corresponding lower bound is reported only for a
positive-effect claim. The fixed radius is the same in either direction.

