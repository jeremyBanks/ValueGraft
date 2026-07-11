_The native per-model graft design is validated on Qwen, but the wide
cross-architecture sweep produced no usable results because infrastructure,
launcher, model-loading, and dependency failures were discovered sequentially.
The project has now switched to a secure-tier, fail-closed single-model canary
before any further fan-out._

**Participants:** User and claude-opus-4-8.

**Confirmed Science.** Native replies plus native summaries with shared gold
scoring reproduce the referent effect: CI [+0.012, +0.195], midpoint about
+0.10, with the expected referent/sense/stance dissociation and correctly
functioning validity gates. The full 54-scenario dataset remains eligible.
Batched decoding was about 2.3× faster but changed bf16 greedy outputs; referent
weakened to +0.0485 with a zero-spanning CI and `evicted_fact` shifted to
−0.305. Fable judged batched-as-is unsafe for the sign map; the validated sweep
method is per-token. A possible future off-ramp is batched decoding with fp32
argmax, gated by reproducing the native referent result and restoring
`evicted_fact`.

**Pre-registration and design.** QK-norm detection was corrected to inspect
loaded `q_norm`/`k_norm` modules rather than unreliable config flags, and the
pre-registered hypothesis was frozen before outcome data: QK-norm presence
predicts a positive graft sign. The intended sweep uses per-token native
rendering, 24 conversations for five anchors and 12 for breadth. Fable
recommended proceeding with the 11 verified text models and treating Gemma-4
text-substack loading as the only worthwhile, time-boxed attempt to recover a
second vendor-matched pair; otherwise the honest claim is one Qwen de-confound
pair plus the cross-vendor QK-norm regression.

**Reliability changes.** The launcher now has a fail-closed preflight gate
checking model IDs and causal-LM loadability, deployed paths, readable data
files, syntax, credentials, and clean repository state; it requires a fresh
fingerprinted verification token before multi-unit or material-cost launches. A
probe-first path scores three conversations before the full run.
`RELIABILITY.md` records the SRE rules: tier-1 canary before fan-out, automatic
health checks/alerting, observable failures, and no scaling on assumed
preconditions. `shellcheck` is mandatory via `scripts/lint.sh`; it caught and
enabled correction of a health-monitor stdin bug. Additional fixes covered
scenario-file placement, environment forwarding, timeout scaling,
duplicate-process guards, SSH stdin detachment, macOS Bash compatibility,
endpoint resolution, and a transformers dependency pin.

**Sweep failure and current state.** The community sweep was stopped after
detecting missing scaffold files, an incompatible multimodal Mistral checkpoint,
five multimodal wrappers initially misclassified as unavailable, launcher races,
a hung SSH detach, unreliable health reporting, and finally
`transformers 5.13.0` breaking model conversion after an unconstrained upgrade.
The dependency fix now force-reinstalls and verifies `transformers==4.57.1`,
refusing to run otherwise. All community pods were terminated; there are
currently zero cross-architecture results.

**Handoff State.** The user selected secure infrastructure for reliability, with
secure concurrency capped at three and community capacity usable only when
healthy. One secure Qwen3-30B-A3B canary (`haraf45ijgnker`, approximately
$1.39/hour) is provisioned and monitored. Required checkpoints are: transformers
verified below 5, model loads, render progresses, and a three-conversation probe
produces a sane result near +0.10. Download plus probe was estimated at about
one hour total, with dependency/load checks expected within roughly 25 minutes.
Only after all checkpoints pass should the project fan out; any failure must
stop expansion and be reported.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `aba66016710b59828`
- `aa4aae015485f75fe`
- `a79afda8a154512fa`
- `a3d09ab012bdc3d73`
- `aa660baed043f01da`
