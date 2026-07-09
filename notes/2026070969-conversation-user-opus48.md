_This chunk continues the mainline project conversation covering the
checkpointing/render-persistence build, its Fable review and bug fix, launch of
the interruptible block-reproduction re-run, and the two architecture-pole
results landing — ending mid-run, holding for the first baseline-vs-fresh block
read._

**Participants:** User and claude-opus-4-8.

**New topic: coding-variant models.** The user asked whether coding-specialized
variants of the candidate models exist. Confirmed they do
(Qwen2.5-Coder/Qwen3-Coder, DeepSeek-Coder-V2, Codestral, CodeLlama, CodeGemma,
Granite-Code), but framed as adding a different kind of value than the
architecture map: same architecture as the base model, so zero new architecture
diversity, but a clean controlled contrast (base vs. coder, same arch, different
training domain) testing whether the graft's effect is architecture-driven or
training-driven. The highest-value pair is Qwen2.5-Coder-32B against the
already-planned Qwen2.5-32B base. The user directed that this pair be added to
the queue but run after most of the new-vendor diversity, mid-priority; this was
confirmed as already recorded in MODEL-QUEUE.md in that position. MODEL-QUEUE.md
was also rewritten for scannability (queue table first, notes condensed below),
since it had become prose-heavy through repeated appends.

**Checkpointing build completion and Fable review — first pass found a real
bug.** The per-conversation checkpointing and full-render-persistence build
completed, self-certified with claimed byte-identical validation tests. Before
trusting it for the reproduction re-run (per the standing GPU-code-review rule),
it was sent to Fable for adversarial review. Fable found a genuine, silent
number-changing bug: in relative-floor mode, the resume path dropped a
conversation's `lp_A` values from the pooled competence floor whenever that
conversation had empty A↔B alignment, so a resumed run could compute a different
floor than an uninterrupted run, silently changing which plants are gated and
therefore the headline number and CI. Fable also found that the implementer's
cited validation script (`scratchpad/validate_ckpt.py`) did not exist in
committed form, making its "5 byte-identical tests" unverifiable — reinforcing
that self-certification alone is not sufficient. The bug's blast radius was
assessed as narrow: the fresh/uninterrupted path was independently verified
byte-identical to the old code, and the bug only affects resumed relative-floor
runs containing an empty-alignment conversation (an OLMo-class case, not the
Qwen reproduction, which very likely has none).

**Fix and re-verification.** The checkpointing agent was redispatched with
Fable's exact diagnosis and told to commit its validation this time rather than
leave it in scratchpad. The fix made the pre-scan record its own
per-conversation `lp_A` so the resumed floor pool matches the fresh pool even
for empty-alignment conversations. The new validation was committed to git and
included a negative control demonstrating the fix's necessity:
FRESH/RESUME/KILL-SIM floors all matched at −9.499 under the fix, while the
pre-fix code path drifted to −10.126 under the same test — proving the test
actually detects the bug rather than trivially passing. Fable was re-consulted
specifically on whether this fix resolved its own finding without introducing a
new resume gap, and confirmed it was safe to trust. This cleared the linchpin
blocking the reproduction re-run.

**Launcher config-propagation fix.** Before launching, a check of
`launch_pod.sh` found it only forwarded `MODELS`/`SC_CONV_LIMIT`/`SC_HF_MODEL`
to remote jobs, not `SC_CONV_START` (the held-out conversation offset) or the
relative-floor mode — meaning an unchecked launch would have silently run all
three block cells starting at c01 with the wrong floor mode. This was fixed and
committed before launch.

**Block re-run launched.** Three pods were launched for the block-design
reproduction, all Qwen3-30B-A3B, relative floor, checkpointing enabled
(resume-safe): b0 = baseline c01–c12, b1 = fresh c13–c24, b2 = fresh c25–c36.
The launch chained through the same script and stalled — b0 came up correctly,
but b1/b2 never launched because the chained script hung on b0's post-launch ssh
check (a recurrence of a previously identified launcher-hang pattern). b1 and b2
were then launched independently in parallel so neither could block the other;
b2's first independent attempt still failed silently (empty log) and was
re-fired successfully. Conv-range offsets were subsequently confirmed correct on
all three pods (b0 at c01–c12, b1 at c13–c24, b2 at c25–c36), validating that
the launcher config fix worked and none had silently collided on c01. This run
is also the first to exercise the new per-conversation checkpointing in
production; render checkpoints were confirmed accumulating on disk as
conversations completed (e.g., b1 showing 10 saved checkpoints, b2 showing 5,
mid-run), confirming the render-persistence fix is functioning as designed.

**Architecture-pole results landed and banked.** Qwen3-32B (dense, pair-B)
finished: referent CI [−0.063, +0.010] (null, spans zero), sense [−0.091,
−0.008], stance [−0.139, −0.051], ruled_out [−0.130, −0.063], evicted_fact
[−0.099, −0.017] — negative across every other category. Holding vendor and
QK-norm status fixed against the MoE anchor (Qwen3-30B-A3B, referent +0.10),
this is a clean within-Qwen MoE-vs-dense reversal: the graft helps the MoE and
is null-to-harmful on the dense model. Qwen2.5-32B (dense, no QK-norm) finished
shortly after: referent [−0.303, −0.157], sense −0.42, stance −0.18 — strongly
harmful across the board, the most negative result of the sweep. Both results
passed the real smoke checks (`identity_ok`, `alpha0_ok`); an initial false
alarm on both models' `graft_direction_ok: False` was corrected after re-reading
the code — that field is purely informational (aggregate raw_EB sign) and not a
validity gate, so it was not evidence of a broken graft. Both pole results were
recorded in FINDINGS.md and their pods (expo, w6) terminated for budget once
idle, since their renders were made under the old harness and have no reuse
value. With both dense poles now negative/null and only the MoE anchor and a
weak Mistral positive, the emerging map skews MoE-positive / dense-negative —
flagged explicitly as ambiguous between two readings: a real MoE-vs-dense
mechanism, or the effect being largely specific to the model it was originally
developed on. The reproduction result is the designated disambiguator between
these readings, and this robustness caveat is not to be oversold as a confirmed
reversal until fresh-conversation reproduction is in hand.

**Baseline positive-control scare, corrected.** b0 (baseline c01–c12) finished
first and showed referent CI [−0.068, +0.090], mid ≈ +0.012 — a apparent failure
to reproduce the original +0.156 anchor point estimate, combined with the
(subsequently found to be non-gating) `graft_direction_ok: False` flag,
initially triggered a stop-and-check reaction suspecting a broken graft. This
was walked back after rereading the code: the real smoke gates passed, and the
informational flag firing across all models (including previously-trusted ones)
was not meaningful. The more defensible reading of the +0.012 baseline point
estimate is that it falls within the low end of the original result's own wide
CI (n=12 was underpowered) and that Qwen3-30B-A3B's MoE routing is
hardware-nondeterministic across pods, so this looks more like the original
+0.10 having been noisy/fragile than like an apparatus failure — but this is
provisional pending the full pooled block analysis (b0+b1+b2), which is required
before drawing any conclusion. Two premature harvest attempts on b1 (mistaking
completed rendering/scoring-stage checkpoints for a finished pooled result) were
both self-corrected without consequence; b1 was confirmed genuinely mid-scoring
(5/12) as of the latest check, with b2 still rendering behind it.

**Explicit commitment: full picture to Fable before any conclusion.** The user
directed that the complete picture — the block reproduction result, the full
arch-pole map (Qwen3-32B null/harmful, Qwen2.5 strongly harmful, Mistral mixed
positive/negative, phi-4 null), and the robustness concern about the effect
being clearest on the model it was developed on — be put to Fable un-anchored
(not a cherry-picked summary), with raw results and a pointer to notes/, before
any conclusion is drawn or the paper is touched. This is queued as the next step
once the pooled block_analysis read is available, per standing practice
established earlier in the project.

**Standing practice adopted: push notifications on major results.** The user
asked to be notified via push notification on major results and milestones going
forward, not routine progress ticks. This was adopted as a standing practice and
has since been used for: the checkpointing review verdict (bug found), the fix
landing and being re-cleared, and the block re-run launching.

**Handoff state at end of chunk.** All three block-reproduction pods (b0 done,
b1 mid-scoring at 5/12, b2 still rendering) are healthy; b0's baseline read
exists but is provisional pending pooling with b1/b2. Both architecture-pole
results (Qwen3-32B, Qwen2.5-32B) are banked and recorded. The next required
steps, in order, are: wait for b1 (then optionally do an early n=12 fresh read)
and b2 to complete, harvest and pool all three via block_analysis for the
central-floor, conv-clustered baseline-vs-fresh read, put the complete
unresolved picture to Fable un-anchored for its honest verdict on whether the
architecture map reflects a real mechanism or model-specific fragility, and
push-notify the user with the reproduction verdict once it lands — per the
earlier-established overnight arc, this determines whether the tiered
model-enrichment sweep proceeds as planned or the narrative is revised first.
