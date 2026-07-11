_This conversation covers the completion of the experiment and paper, followed
by an independent audit that overturned key synthetic-data conclusions through a
verified provenance error and source-conditional sign reversal. The paper is now
held for a holistic correction pass; no paper changes were authorized in the
latest instruction._

**Participants:** User and claude-opus-4-8.

The experiment’s final SWE-Gym result is a small in-domain per-layer-tuned
positive: +0.0126 nats/token pooled across 143 trajectories, with
out-of-sample-only +0.0135 [0.0083, 0.0190]. The fixed-strength graft is
heterogeneous—positive on the original 75, null on the independent 98, and
pooled borderline-null (+0.0053 [−0.0028, +0.0133]); tuning beats it where the
fixed graft fails and ties it where it works. This result is bounded to a
log-probability proxy from one model under aggressive compaction; other study
components remain null. README was finalized and pushed at `1d64086`, retaining
the title “mostly-null bounding result” and cautious champion framing.

Compute spending is complete: all pods were terminated, the balance froze at
approximately $7.33, and no further pod work is planned. Notes regeneration,
renaming, and related scripts were explicitly descoped and must remain
untouched. The speculative note
`notes/2026071101-why-value-grafting-mostly-fails.md` was committed and pushed;
it presents six hypotheses, with H2’s gradient-projection check identified as
the most decisive falsification test. Task-capture state was subsequently synced
at `2360606`.

An independent audit of notes based initially on the old README raised several
obsolete criticisms, but two load-bearing findings apply to the live final
README and were independently verified. The held-out synthetic conversation
bodies are not native test-model generations: c07–c12 use
Qwen3-4B-Instruct-2507-4bit output, while c13–c24 are Claude/Fable-authored.
This contradicts final README §2.3’s claim that replies were generated
in-context by the test model without reuse of other-model replies. Recomputing
the final synthetic champion by provenance block confirmed a sign reversal:
Qwen-rendered c07–c12 +0.1197 [+.056, +.180], Claude-authored c13–c24 −0.0503
[−.075, −.028], pooled −0.0026 [−.041, +.042]. Thus the reported clean null is
an average of opposite source-conditional effects, not a homogeneous null.

The compression-sweep concern remains plausible but was not fully verified:
different summary requests were used by level, and the claim that write-time
value state was rebuilt under the realistic request requires a deeper code
trace. The test-retest dependence, champion-versus-scalar limitation, and coarse
discrete-metric caveats remain relevant. A correction note,
`notes/2026071104-corrections-from-sol-audit.md`, was committed and pushed to
record verified versus outstanding issues; the README itself was not changed
during this verification pass.

Current handoff: external publication remains on hold. Read the two remaining
audit notes—the methodology/implementation and scientific-consultation notes—and
await the auditor’s review of the actual final README. Only then perform one
authorized holistic correction pass covering provenance, source-stratified
synthetic results, compression methodology, and the paper’s conclusions.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `ac1d16212826b3eb1`
- `a3958878b33fc617d`
