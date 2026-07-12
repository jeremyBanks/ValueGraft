# V12 Transformers-5 MoE loader correction

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning

**Recorded:** 2026-07-11/12 EDT/UTC

## Observed failure and scope

The first paid exact technical attempt selected the pinned
`Qwen/Qwen3-30B-A3B-Instruct-2507` revision, installed the pinned runtime,
downloaded and loaded the bf16 weights on one A100-80GB, and then stopped in the
loader attestation with:

```text
loaded parameter names/shapes/dtypes differ from weight inventory
```

The failure occurred before the exact subject executed any forward pass. No
generated token, cache comparison, semantic arm, or treatment outcome was
observed. It is an implementation defect in the loader's representation
assumption, not evidence for or against coherent-state equivalence or the
ValueGraft effect. The preserved attempt cost is at most `$0.1289722222` under
the conservative full-rental calculation; the observed account-balance delta
was `$0.0382625315`.

## Diagnosis

The checkpoint and the loaded model expose the same parameters in different
topologies. The frozen checkpoint index contains `18,867` tensor rows. Of
those, `18,432 = 48 layers × 128 experts × 3 projections` are separately named
`gate_proj`, `up_proj`, and `down_proj` expert tensors; `435` are nonexpert
rows. Transformers 5.0.0 represents the loaded experts as two packed tensors
per layer, so the runtime has `435 + 48 × 2 = 531` named parameters.

The pinned Transformers conversion is the Qwen2-style MoE mapping used by
Qwen3MoE:

```text
gate_up_proj = concatenate(
    stack(gate_proj[0..127], dim=0),
    stack(up_proj[0..127], dim=0),
    dim=1,
)
down_proj = stack(down_proj[0..127], dim=0)
```

The installed source natural-sorts numeric key segments before collection,
stacks expert tensors on dimension 0, and concatenates source patterns in the
declared gate-then-up order. The model forward splits the packed projection in
that same order. A meta-instantiated exact model independently exposed the
expected `531` rows. Both checkpoint and runtime representations contain
exactly `30,532,122,624` parameters.

Two independent Codex reviews inspected the installed Transformers source and
the correction. Both agreed this is the library's intended conversion and
that conversion-aware attestation is a representation repair, not a gate
waiver.

## Additive correction

The exact loader now:

1. pins the checkpoint's MoE layout (`48` layers, `128` experts, hidden size
   `2,048`, expert intermediate size `768`, bf16);
2. requires the complete, unique `48 × 128 × 3` source grid and exact source
   shapes;
3. derives the expected `96` packed expert rows while passing all `435`
   nonexpert rows through unchanged;
4. requires exact parameter-count conservation and exact equality between all
   `531` derived and loaded name/shape/dtype rows;
5. attests the actual retained Transformers conversion recipe; and
6. bit-compares complete gate/up/down tensors at three deterministic
   cross-checkpoint sentinels: `(layer, expert) = (0,0), (24,64), (47,127)`.

The three sentinels read and transfer about 28 MB total. They directly check
expert order, gate/up order, and down packing without exhaustively rehashing the
61 GB model or hardening against adversarial package behavior. Full frozen
shard hashes, revision binding, empty loading diagnostics, dtype/device/no-meta
checks, evaluation mode, total parameter count, and every subsequent exact
technical gate remain unchanged.

The local dense subject retains one-to-one checkpoint/runtime topology. No
local semantic evidence is being reinterpreted by this exact-only repair.

## Pull-path correction found during review

The original pull verifier globbed every local exact receipt. Because the
attempt-one receipt is committed, it could not prove that a later attempt had
written a new receipt. Each job now publishes one explicit current-receipt
pointer after writing the receipt. The puller validates that pointer, pulls
only the files declared by that receipt, and verifies their sizes and hashes.
An identity-checkpoint directory may be absent only when the current receipt
declares no checkpoint.

## Verification and next boundary

- the loader-focused suite passed `17/17`;
- the complete focused v12 suite passed `165/165`;
- both launch scripts passed `bash -n` and ShellCheck;
- only the same two SWIG deprecation warnings were emitted;
- real 30B topology and sentinel checks have **not** yet been observed; they are
  the purpose of the next bounded exact technical attempt.

The strict exact gate remains normal EOS plus full generated/forced identity
and all other preregistered technical controls. No Phase A or treatment is
authorized by this note. A second technical failure triggers a fresh design
audit before any third paid attempt.

Claude/Fable CLI review capacity is tracked separately from pod spend because
the owner reports that CLI usage is subsidized. This does not relax the
instruction to use it mindfully; the approximately `$60` research budget is
governed by actual pod/provider charges.
