_This conversation covers completion of the polished grafting paper, the
K/V-axis investigation, and a final attempt to make the J-lens evidence more
informative through free-generation and broader trajectory analysis. The central
result is that value grafting helps semantic-referent behavior in the tested
corpus, while key grafting is negative or unhelpful there; broader
identifier-shaped recovery remains open._

**Participants:** User, claude-opus-4-8, and claude-sonnet-5.

**Handoff State.** Phase 1 paper is finished and pushed to `trunk` as
`REPORT.md` (~7,700 words, 28 exhibits, 12 citations; commit range
`ed9b0a9..e3d711a`). Its intended form is approximately 70% academic paper and
30% technically literate blog post: grafting is the main subject, with the
J-lens as a secondary interpretability tool. The paper reports that sense/stance
dissociation replicates across the tested 30B-A3B and 27B models; referent
recovery is strong at 30B (+10pp; 81% helped) but not replicated at 27B
(+0.004), with the cause explicitly left open. The 4B control distinguishes the
honesty/fabrication effect, which survives at 4B, from the dissociation, which
does not.

The real J-lens runs on Qwen3.6-27B and is characterized as an aggregate-level
corroborator, not a vivid per-example visualization. It supports alignment
sensitivity, moderate blending outperforming full replacement, and the absence
of reliable omitted-fact recovery, but teacher-forced probes produced only
small/noisy shifts. A methodological hypothesis was added: teacher forcing pins
all arms to the same continuation and suppresses downstream divergence; free
generation may reveal a later commitment point. Fable’s conceptual review
required fork margins, decoding-robustness checks, an all-case distribution
rather than hero examples, and a genuinely disconfirming bucket so null results
cannot automatically be labeled “subtle.”

**K/V Results.** Key grafting with RoPE re-rotation was implemented in
`src/kv_graft.py` and validated on Qwen3-0.6B: re-rotated versus freshly encoded
keys reached cosine 0.99999982 with relative error near fp32 rounding; unrotated
controls differed, and plumbing/end-to-end tests passed. The 30B coarse sweep
passed its sanity gate because V-only reproduced the paper’s positive referent
result (+0.057). Uniform K-only actively hurt sense, referent, and stance;
coupled and independently varied K/V policies did not beat V-only. A
referent-focused per-layer probe then found no meaningful rescue: the best
sampled layer had only +0.011 lift, 10/21 positive plants, and K-only was
negative at every sampled layer.

This supports the scoped conclusion: value is the operative axis for the tested
semantic-phrase referent tasks; key grafting did not help uniformly or at
sampled individual layers. The stronger universal claim that keys never matter
was withdrawn after review of
`referent_recovery_microtest/outputs/policy_registry_followup.md`: a noisy 0.6B
prospecting screen found K-favorable results for short identifier-like targets
(`userName`, `UTC`, etc.), while its overlapping semantic-policy tasks agreed
with the main V-only result. That screen is statistically weak because of tiny
cells, 0.6B scale, and best-of-15-policy selection, but it motivates a narrowly
scoped future 30B identifier-morphology experiment using the full policy
surface.

**Current Run.** The corrected free-divergence lens runner is deployed on the
existing pod using Qwen3.6-27B. It builds A/full, B/fresh-compacted, and
E/aligned-graft states, free-generates rather than teacher-forces, finds the
first B/E fork, records margins and J-lens readouts around the fork, and tests
decoding stability. It uses 43 corpus-derived sense/referent cases rather than
hand-picked Nimbus/Hydra/sandbox heroes; results are not yet available. The pod
must remain alive only through validation, then be terminated.

**Required Next Actions.** When the lens run completes, validate the output and
classify the distribution into decisive fork-toward-full-context, subtle lean,
and disconfirming outcomes. Update `FINDINGS.md`, `STATE.md`, `MASTER-PLAN.md`,
and the task tracker; then fold the scoped K/V result and lens result into
`REPORT.md` through the full critic and readability pipeline before pushing an
updated paper. Because the main loop is currently Opus due model filtering,
Fable must perform both conceptual framing review and final readability review;
this is a hard completion gate. The morphology/short-identifier K/V experiment
follows only after the paper is shipped.

A broader lens trajectory scan is queued as a near-term follow-up if the fork
probe is muddy: use a narrow concept/readout measure scanned token-by-token over
a longer generated span to test whether the effect is diffuse rather than
localized. It should remain secondary to publishing the current paper and should
not be used to hunt selectively for a dramatic example.

**Operational Lessons.** Every remote job requires an explicit launch
validation: confirm logs have advanced into real work, model loading or download
is progressing, GPU utilization/memory is appropriate, and outputs are being
produced. A spawned process or “launched” message is insufficient. Recent
failures included a bad working-directory path, missing `jlens`, incompatible
torchvision/torchaudio, and a CUDA 13 torch build silently causing CPU fallback;
these were recorded as incidents. Never upgrade torch blindly on a pod, verify
`torch.cuda.is_available()`, and investigate apparent stalls by checking cache
growth and active processes. Monitors should remain silent during routine
progress, use 2-minute-to-24-minute backoff, and notify only on completion,
error, or meaningful state change.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `ac24c17f5283bb110`
- `a12538dc96bc3d0aa`
- `af0a02214cabdb417`
- `a7bc898009f376eb2`
- `a65318b76a056562a`
- `ad2852e6478601448`
- `ad5aaaa6eb16e249c`
- `a733c1c25db57a9cb`
- `ab65dd68eb9cacd67`
- `adba388cce2ffa4bd`
- `adc353166f98f391e`
- `a719622783e557ff7`
- `adb4e6f5f98d881a1`
- `a0acd892e52e4a335`
- `a641bde3144e775f3`
- `ac511844ddce28306`
- `a6ab635b8a6b719e3`
- `ada3bb6a89344c9b4`
- `a959458ff41951cdc`
