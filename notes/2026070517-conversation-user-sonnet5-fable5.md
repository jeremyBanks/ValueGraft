_This conversation covers the pivot from controlled KV-cache experiments to
validation on standard datasets, practical deployment overhead, and a planned
cloud-scale evaluation. The intended write-up form is a blog-style draft, with
an overhead/handle-lifecycle section added._

**Participants:** User, claude-sonnet-5, and claude-fable-5.

**Handoff State.** The primary dataset recommendation is LongMemEval, reduced to
custom ≤16K-token instances whose answer-bearing sessions fall in the evicted
region. The planned evaluation uses the established arms (full-context
reference, standard compaction, SelfGist/H-pack, B-min-pack, and tuned
ValueGraft), with QA correctness plus fabrication/admission judgments.
SWE-Gym/OpenHands-SFT-Trajectories remains a stretch follow-up for coding
traces. Dataset details should be re-verified at execution time because the
scouting used limited search; LongMemEval’s cleaned revision should be checked
before relying on it.

“Continuation gap” is defined as the full-context teacher-forced next-reply
log-probability minus the standard-compacted version’s score. An intervention’s
gap closure is the fraction of the lost score recovered toward the full-context
reference; it is a secondary fluency/coherence measure below behavioral recall
and fabrication probes.

A 4B LongMemEval batch completed with 48 questions plus two smoke tests. Judging
covered 150 items across 25 conversation groups and six variants, with aligned
keys and verified counts. The full-context/reference arm was nearly always
correct. Base and hardened compacted variants often admitted uncertainty rather
than fabricating; H-pack/H-gap variants were mixed and sometimes confidently
substituted plausible world knowledge for forgotten facts. Purely personal facts
generally elicited admission, while creative/roleplay framing increased
confident fabrication. Some variants falsely denied that prior conversations
occurred, which counts as fabrication. The complete benchmark table and
judgments were committed.

The 30B batch was launched on the same 48-question sample but slowed
substantially, with per-question time rising from roughly 191 to 380 seconds. It
was relaunched after a memory-pressure fix; four pre-fix results remained on
disk, and the process was still alive but recovering from swap. The latest
decision was to cap at approximately 24–30 questions if performance did not
stabilize, since that sample is sufficient for the fabrication contrast. No
final 30B results are recorded yet.

The practical deployment framing was added to the draft based on
`opaque-compaction-handle-notes.md`. The proposed stateless-API analogue sends
or references only a packed summary KV handle, while the provider retains the
blob with cache-style expiry; the ordinary recent context remains recomputable.
At fp16, the rough 30B KV cost is 96 KB/token, making an ~80-token SelfGist blob
about 8 MB versus roughly 1.2 GB for a 12K-token full cache; quantized blobs
could be approximately 4 MB at 8-bit or 2 MB at 4-bit, but quality is untested.
ValueGraft differs operationally because it requires the complete old cache at
compaction and is therefore server-side rather than a portable request
attachment. Exact model revision and tokenizer binding, signing/expiry, and
injected-KV trust/auditability are required caveats.

The user requested an accessible plan for testing on rented GPU hardware,
including straightforward setup/access steps and strict spend safeguards,
suitable for someone familiar with AWS but not GPU platforms. A cloud plan was
explicitly expected within roughly three to four hours of the user’s return; the
agent committed to continue autonomous research and refinement during the day.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a1849f0c5456f4128`
- `ade9696bfd520a8fb`
- `acb96e28a74edd09d`
- `a16556c2c33d05d3b`
- `a781c408b4bdd7065`
