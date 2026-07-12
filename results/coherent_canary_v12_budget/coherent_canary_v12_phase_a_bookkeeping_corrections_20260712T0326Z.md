# V12 e01 Phase A — additive bookkeeping corrections

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

Two prose bookkeeping notes contain transcription errors. The underlying raw
artifacts, independent validator report, receipt, hashes, gate verdict, and
decision are unchanged.

## Correct semantic execution head

In
`coherent_canary_v12_semantic_execution_head_pin_20260712T0324Z.md`, the short
commit name `cbdfa48` is correct but the manually expanded 40-character hash is
not. The authoritative last verifier-valid head is:

`cbdfa481fe08de62bf8178d09ca040716d610f21`

This value was read with `git rev-parse cbdfa48` and `git show -s --format=%H`.
All future pinned-clone checks must use the authoritative value here, not the
mistyped expansion in the earlier note.

## Correct validator-decoded margins

In
`coherent_canary_v12_phase_a_acceptance_cost_and_stop_20260712T0316Z.md`, three
manually copied decimal margins differ slightly from the independent
validator's authoritative float32-bit reconstruction. The correct values are:

- `A_W_focal`: `-19.694555282592773`;
- `A_W_nonfocal`: `+11.667630195617676`;
- `FF_focal`: `-5.798246383666992`.

The other listed values remain:

- `A_C_focal`: `+16.50018310546875`;
- `A_C_nonfocal`: `+11.792686462402344`;
- `FF_nonfocal`: `-3.699748992919922`;
- fresh margin damage: `+22.298429489135742`;
- fresh correct-target log-probability damage: `+22.290991485144332`.

All signs, thresholds, competence checks, fresh-damage checks, and the
`PRETREATMENT_PASS` verdict are identical. Scientific interpretation must cite
the raw artifact and independent validator, never the superseded prose
transcriptions.
