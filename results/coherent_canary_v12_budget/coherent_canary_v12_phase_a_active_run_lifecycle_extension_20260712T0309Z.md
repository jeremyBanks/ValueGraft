# V12 e01 Phase A — active-run lifecycle extension

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

At `2026-07-12T03:06:33Z`, the authorized treatment-blind e01 Phase-A
runner began on already-qualified pod `27irwplre7fmm6`. Before scientific
model work it observed the exact authorized GPU UUID and platform, verified the
e01 and technical-report hashes, recreated the locked dependencies, and passed
the frozen repository verifier at launch head
`139d621e206518b88d7553b5166a0123ee1cdd55` with sealed inventory
`86a26693fb99f0c3ce080399abbc8814f76bcf70d708ecf17b0d1f4291f4a5b1`.
The unique raw output path was announced. No treatment process or score is
present.

The earlier `+26:00` deletion target would be `03:13:51Z`. Live observation
showed that this fresh Phase-A process must repeat the frozen loader's complete
61 GB snapshot inventory/hash verification before model load. The immediately
preceding technical run measured about 447 seconds in model preparation,
dominated by the same mandatory byte verification. Therefore the prior
7--8-minute Phase-A estimate did not leave a sound margin for both fresh
verification and the actual Phase-A forward passes. Killing the healthy,
correctly bound process at the old target would discard nearly completed paid
validation and force a more expensive rerun.

Narrowly extend the entire provisioning/requalification/Phase-A unit ceiling
from `$0.80` to `$1.10`. At the observed `$1.39/hour` rate and the prior three
provisioning-window conservative bound of `$0.1783833333`, this permits the
current pod until approximately `03:27:39Z`. Begin pull and deletion
immediately upon a terminal artifact; do not retain the pod merely to use the
available ceiling. If no terminal artifact or meaningful verified progress is
present by that time, salvage and delete.

This is a maximum incremental extension of `$0.30`, not a new allocation or a
science-scope change. It authorizes neither treatment nor a retry. The active
process, inputs, thresholds, case, model, runtime fingerprint, and terminal
decision rules remain unchanged.
