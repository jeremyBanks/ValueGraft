_This conversation covers the completion of the experiment and paper, followed
by an independent audit that overturned key synthetic-data conclusions and
exposed broader provenance and methodology defects. The paper and analysis
remain held for one authorized holistic correction pass; no paper changes were
made during the audit._

**Participants:** User and claude-opus-4-8.

**Results and repository state.** The final SWE-Gym result is a small in-domain
per-layer-tuned positive: +0.0126 nats/token pooled across 143 trajectories,
with out-of-sample-only +0.0135 [0.0083, 0.0190]. The fixed-strength graft is
heterogeneous—positive on the original 75, null on the independent 98, and
pooled borderline-null (+0.0053 [−0.0028, +0.0133]); tuning beats it where the
fixed graft fails and ties it where it works. This remains a log-probability
proxy from one model under aggressive compaction; other study components are
null. README was finalized and pushed at `1d64086`, retaining the title
“mostly-null bounding result” and cautious champion framing.

Compute spending is complete: all pods were terminated, the balance froze at
approximately $7.33, and no further pod work is planned. Notes regeneration,
renaming, and related scripts were explicitly descoped and must remain
untouched. The speculative note
`notes/2026071101-why-value-grafting-mostly-fails.md` was committed and pushed;
it presents six hypotheses, with H2’s gradient-projection check identified as
the most decisive falsification test. Task-capture state was subsequently synced
at `2360606`.

**Verified audit findings.** The full set of four independent audit notes has
now been read. The held-out synthetic conversation bodies are not native
test-model generations: c07–c12 use Qwen3-4B-Instruct-2507-4bit output, while
c13–c24 are Claude/Fable-authored. This contradicts final README §2.3’s claim
that replies were generated in-context by the test model without reuse of
other-model replies. Independent recomputation with conversation-clustered
bootstrap confirmed the source-conditional sign reversal:

- Qwen-rendered c07–c12: +0.1197 [+.056, +.180].
- Claude-authored c13–c24: −0.0503 [−.075, −.028].
- Pooled 18-conversation result: −0.0026 [−.041, +.042].

The reported clean synthetic null is therefore an average of opposite
source-conditional effects, not a homogeneous null.

Three additional high-consequence implementation findings were independently
verified. The graft covers both retained-tail and summary-token regions, so the
paper’s summary-only description is false and the coding effect cannot be
attributed specifically to summary-state retention. SWE-Gym artifacts save only
summary token counts, not the generated summary text or its identifiers; the
purported exact text hash is a hash of the summary request prompt. The
SWE-Gym/champion manifests also have `git_commit: null` and `git_dirty: null`,
contradicting the README’s claim that the exact code commit was recorded.

The compression-sweep concern remains plausible but not fully verified:
different summary requests were used by level, and the claim that write-time
value state was rebuilt under the realistic request requires a deeper code
trace. Test-retest dependence, the champion-versus-scalar limitation, coarse
discrete metrics, unmatched placebo concerns, and the need to verify any
prior-art citations remain relevant. The expanded corrections note records
verified, credible, and still-to-check items; it was updated and pushed without
changing the paper.

**Handoff state.** The paper is not currently authorized for editing. Await the
auditor’s review of the actual final README, then perform one holistic
correction pass covering provenance, source-stratified synthetic results,
graft-region attribution, missing summary artifacts, commit-recording claims,
compression methodology, and the paper’s conclusions. Keep all note-regeneration
and renaming tooling untouched.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `ac1d16212826b3eb1`
- `a3958878b33fc617d`
