_This conversation moved the project from a fragile canary/monitoring setup
toward fault-tested, fail-closed infrastructure while beginning a focused
cross-architecture ValueGraft study. The current experiment set includes Qwen
replication, QK-norm ablation, Qwen3-32B pairing, and broader vendor-diverse
models._

**Participants:** User, claude-opus-4-8, and claude-sonnet-5.

**Scientific state.** Mistral-Small-24B completed a valid preliminary run:
referent was positive, while sense and stance were negative, a distinct
signature from Qwen. The result remains preliminary because it has only 12
independent conversations, despite 24 plants. Conversation-clustered bootstrap
CIs are now the headline inference unit; plant-level CIs were recognized as
anti-conservative and retained only as secondary diagnostics. OLMo-2 did not
have empty alignment: CPU tests showed alignment works, while the absolute task
logprob floor of −8.0 likely excluded all plants. The harness now reports
separate alignment, short-target, and competence-exclusion counters rather than
guessing the failure cause.

**Replication and mechanism plan.** Qwen3-30B-A3B-Instruct-2507’s
24-conversation run remains the primary replication gate, expected to take
roughly 4–5 hours at about 13 minutes per conversation when last checked. A
QK-norm ablation was implemented and CPU-verified: it replaces detected Q/K
normalization modules with identity layers, fails closed if none are found,
records the ablation explicitly, and is GPU-unverified. The ablation tests
whether QK-norm causally explains cross-model graft behavior; model comparisons
alone remain confounded by vendor, density/MoE structure, and training data. The
intended sequence is to interpret Qwen, Mistral, OLMo/Qwen3-32B, and ablation
results together before deciding which model categories warrant deep analysis.

**Observability incidents and corrections.** Monitoring repeatedly produced
false confidence: it missed a failed pod, treated a probe result as final,
false-alarmed on slow but active rendering, and initially failed to recognize
result-status errors. These were recorded as incidents. A shared pure classifier
now makes unknown, missing, malformed, signal-loss, result-error, stale-GPU, and
dead-process states alarm conditions; terminal completion requires the unique
`WIDE SWEEP DONE` marker and always reports sample size. Both `exp_watch.sh` and
`pod_health.sh` use it. Endpoint derivation, booting-to-timeout escalation, and
error-signature handling are centralized and fault-tested. A final review found
one remaining silent-death path—dead process plus partial result—which was
reordered and tested. The final monitor suite reported 25 cases passing and lint
clean.

**Process enforcement.** A Fable-planned prevention layer was independently
reviewed by fresh Fable assessments before being trusted, per the newly
established rule that an implementation of a Fable plan requires fresh Fable
review. The review showed that self-tests can themselves certify a bad behavior,
prompting the final dead-process fix. The durable lesson is that documentation
is insufficient when failures recur; high-risk rules must become executable
gates, fault injections, bounded launch checks, unique completion markers, exact
checkpoint checks, and source-of-truth classifiers. Process tripwires remain
partly behavioral: consult Fable after at most two failed attempts or a repeated
expensive rerun, verify the actual result and its `n`, and never infer
completion from a status label alone.

**Operational state.** The canary’s earlier three-conversation result was −0.07
with a wide CI and was correctly reclassified as an interim probe, not the
24-conversation gate. Current allocation shifted from idle same-model controls
toward model diversity: Qwen gate, QK-norm ablation, and Qwen3-32B pair-B were
running, while Wave 2 was provisioning Yi, Phi-4, and Qwen2.5-32B. The user
explicitly prioritized distinct models/vendors over multiple same-model
analyses; running jobs must not be interrupted, and freed pods should go to new
architectures first. The preflight gate correctly blocked one Wave 2 attempt
because the job argument was omitted; the invocation was corrected and
reprovisioning began. Cost was approximately $4.17/hour for three pods before
Wave 2, with six pods projected at about $8/hour.

**Handoff requirements.** Do not widen interpretation from the Mistral result
until the Qwen 24-conversation replication and corrected clustered analyses are
read. Do not loosen OLMo’s competence floor unilaterally; treat that as a
methodology decision requiring review. Preserve model diversity as the default
allocation policy, use same-model controls only as spare-capacity fill-ins, and
have Fable review any further implementation derived from its plans. The root
`REPORT.md` draft was ultimately retired as stale and preserved in `notes/`;
notes were generalized and formatted by a Sonnet subagent, with no scientific
content intended to remain from that discarded tangent.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a05e0cf097929c6a3`
- `a1fd8f794b7c5f8fc`
- `a055016f50596a218`
- `a1c9c0ddf695d02f7`
- `a1d9c63dd4a2ae068`
- `abaabd17db8266993`
- `aec338bc23be92b66`
- `a52423c0435062c45`
- `a5928d7325cddc6bd`
- `a89d3631b18db8a66`
- `a1e55427f3427927d`
- `aa191f4875debcdde`
- `aaaa4d62eaac53593`
- `af62315192265951d`
