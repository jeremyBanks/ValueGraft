# V12 exact technical attempt 2 authorization — apparatus revision 6

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Time:** 2026-07-12T01:00:28Z

## Frozen binding

- apparatus commit: `3cb7d5d35156d43479af9c29f0b18efa3b64ef3f`
- authorization commit: `9e960fb22d52fcd264fcb1358de96be40bf5f4e2`
- sealed inventory: `ec520bd1b263269cb927f8cb1bf9cbdff8f6dfde94ab1f7d8d144c3da99eaefa`
- sealed files: `56`
- complete focused suite: `165/165` passed
- job script SHA-256: `d763f874c0735f2d28c2806107d5a003a24165c393d059d78f5943b9a6ea6b65`
- pull script SHA-256: `e33bc6076bf78a4b85c9c231b1ba96ba786519094dbfc6bbbbe4e0c3a7c74e5e`
- loader SHA-256: `570d1a4dc5ce49d6f0802b4b561e3859ea4efd41f4da6d4a65dfa1f25d422a17`
- technical runner SHA-256: `b3b1a33e59cd682782141299ace395f8d59a41efed87509b35282270071d2746`
- independent validator SHA-256: `55f1a61442eaa100b27318f5ecd0e470f9754be2cac323a04976131c6997858f`

The production frozen verifier opened this exact A6/B6 pair on a clean trunk.
Both launch scripts passed `bash -n` and ShellCheck. The loader-focused suite
passed `17/17`; the full focused suite passed `165/165`, with only the same two
SWIG deprecation warnings.

## Why a second attempt is legitimate

Attempt 1 was the first paid exact implementation defect. The exact pinned
weights loaded, but the pre-forward attestation incorrectly required the raw
per-expert checkpoint topology to equal Transformers 5's packed runtime
topology. It stopped before any subject-model forward, generated token, cache
comparison, Phase A arm, treatment, or semantic observation. Its raw failure,
independent validation, log, receipt, and cost record are committed.

Revision 6 changes only the representation attestation and artifact-selection
mechanism. It derives all `531` expected runtime rows from the complete
`18,867`-row checkpoint inventory, conserves the exact parameter count, attests
the retained Transformers conversion recipe, and bit-checks nine complete
source/runtime tensors at three deterministic layer/expert sentinels. The
pinned model, prompt, 64-token normal-EOS requirement, generated/forced
identity criteria, deterministic replay, fresh replay, replacement grid, path
control, natural calibration, independent validator, and semantic claim gates
are unchanged.

## Authorized action and stopping rule

Only one A100-80GB exact technical attempt is authorized. The job contains no
Phase A or treatment command. It publishes one current-receipt pointer; the
puller retrieves and hash-verifies only that receipt's artifacts before pod
termination. Job success still requires runner exit zero, validator exit zero,
report `PASS`, semantic-release eligibility, and exactly one durable identity
checkpoint.

The actual-pod budget remains the owner's `$2` initial tranche and `$8` hard
v12 ceiling. Attempt 1 consumed at most `$0.1289722222` conservatively and
showed a `$0.0382625315` account-balance delta. Claude/Fable CLI usage remains a
separate subsidized category and is used mindfully without being charged to
the pod ledger.

If this second exact technical attempt fails, no third paid attempt is
automatic. Preserve and commit every artifact, terminate the pod, and perform
a fresh design audit before considering any further spend. A technical PASS
authorizes review of the observed evidence; it does not by itself authorize
Phase A, treatment, or a scientific claim.
