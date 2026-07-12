# V12 exact technical launch authorization — apparatus revision 5

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Time:** 2026-07-12T00:30Z

**Paid v12 pod compute observed:** `$0`

This additive note supersedes only the repository/launch-mechanism binding in
`coherent_canary_v12_exact_technical_launch_authorization_20260712T0023Z.md`.
The local evidence, scientific gate, budget, and claim boundary are unchanged.

## Actual frozen launch binding

- apparatus commit: `89cd38977b2d8772940c572687b6db8ec3652c06`
- authorization commit: `cda62b3beb01eab23db5c9f27c6c880d50e55f12`
- sealed inventory: `c54bfd7345dd14590dd7171c411ffad3a2d8210a05ec6826298e49722e2b96fd`
- sealed files: `55`
- complete focused suite: `162/162` passed
- job script SHA-256: `90ea7a9d45797cc67c66517f7cc6aa81de702c762d9149784f7b1a7ed9c06142`
- pull script SHA-256: `6ca84a0206d4a49a754e7204f6bae7fd4585540678d9e3a4556855fe1dd192d8`
- technical runner SHA-256: `b3b1a33e59cd682782141299ace395f8d59a41efed87509b35282270071d2746`
- independent validator SHA-256: `55f1a61442eaa100b27318f5ecd0e470f9754be2cac323a04976131c6997858f`

The launch sets `SC_EXPECTED_COMMIT` to the final pushed trunk commit containing
this additive result note.  The job logs that literal SHA, requires the fresh
clone's trunk HEAD to equal it, and the receipt records both expected and
observed commits.  The frozen verifier independently binds the apparatus and
authorization ancestors while allowing this append-only result note.

## Operational boundary

Only an A100-80GB exact technical job is authorized.  A handled or unhandled
gate failure is pulled as evidence and exits the job nonzero.  Job success
requires runner exit zero, validator exit zero, report `PASS`, exact-subject
semantic eligibility, and exactly one durable post-generation identity
checkpoint.  The local pull verifies receipt hashes for raw, checkpoint,
validation, and job-log artifacts before the pod is terminated.

The job's `$2.00/hour` field is a deliberately conservative estimate, not a
claim about provider billing.  The coordinator records RunPod `costPerHr` and
elapsed wall time from the API, stops before the initial `$2` actual-pod tranche
can be exceeded, and retains the owner's `$8` total actual-pod ceiling.  This
note still authorizes no Phase A, treatment, semantic release, or claim.
