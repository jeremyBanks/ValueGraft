# Referent-Recovery Microtest

This directory is an isolated cheap gate for the synthetic referent-recovery
harness idea. It is intentionally small: hand-built bespoke rules, a fixed
lossy summary, and teacher-forced scoring only.

The runner tests whether the task shape produces a useful measurable gap:

- `A`: full context contains the bespoke rule payload.
- `B`: compacted context contains only a lossy summary plus the recent tail.
- `V-only`: graft summary-token values from write-time context.
- `K-only`: graft summary-token keys from write-time context, with RoPE
  re-rotation.
- `coupled`: graft both keys and values.

The test is a falsification gate, not evidence for the paper by itself. A
useful result needs `A > B` and a policy moving toward `A`. If `A` does not
beat `B`, the model/task pair is not usable. If all grafts match `B`, this
particular harness shape is not promising at the tested scale.

`explore_problem_shapes.py` is the follow-up design search over several simpler
problem families. Start with `problem_shape_notes.md` and
`outputs/problem_shape_exploration.md` for the current recommendation.

`policy_registry_followup.py` widens the recommended private-policy and
named-transform shapes. Start with `policy_registry_followup_notes.md`, then
read `outputs/policy_registry_followup.md` and the JSON output if you need the
full policy surface.
