_The conversation identified a precision blind spot: the corrected experiment
has only measured bf16, while the only prior positive signal occurred under
4-bit conditions. A controlled 4-bit arm is now a priority before the paper is
finalized._

**Participants:** User and claude-opus-4-8.

**Handoff State.** The owner requested that this be recorded and communicated.
The note `notes/owner-request-precision-axis-quantization-dependence.md` was
written, committed, pushed, and relayed to Sol as an owner request. The prior
exact-commit freeze was verified inactive: Sol was committing paper files and
the e01 diagnostic pod had terminated, so the commit was considered safe. The
recurring three-hour notes-archive refresh remains paused because its
file-renumbering previously disrupted active work; resume only once the paper
work settles or Sol confirms it is safe.

The paper must scope current conclusions to bf16 and explicitly identify
quantization dependence as untested. It must not claim universally that the
graft fails, because the earlier approximately +10–12 point signal appeared in
4-bit MLX, although that run was confounded by known apparatus problems
including pseudoreplication and source reversal. The first clean bf16 canary
validates the setup and shows strong history dependence, but treatment efficacy
remains unresolved because treatment scoring is blocked by the path-control
stop-rule conflict.

A corrected 4-bit-vs-bf16 comparison should use the same 30B model, pod, and
apparatus, varying only precision. The proposed implementation is a drop-in
bitsandbytes NF4 path; prior GPTQ/compressed-tensors/MLX difficulties should be
avoided where possible. This is bounded but nontrivial work: the hardened loader
deliberately rejects bitsandbytes/GPTQ/AWQ and asserts bf16 in roughly six
places. A valid 4-bit branch therefore requires revised provenance gates,
4-bit-appropriate tolerances, identity checks, and preregistered interpretation
criteria. Sol should ideally implement or review the loader change before any
paid launch.

The current unresolved fork is whether the assistant supplies Sol with an exact
specification and preregistration immediately, or implements the loader branch
itself for Sol’s review. The owner’s stated priority is to run the corrected
4-bit arm before finalizing the paper, not merely to add an “untested” caveat.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
