---
name: proactively-use-idle-capacity
description: "Proactively fill idle compute with useful, preemptible work — don't leave pods idle, don't let fill-work block higher priorities"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

Whenever there's idle capacity (a free/idle pod, spare compute), PROACTIVELY think of a useful way
to use it — don't leave it idle (idle burns the same as busy, so idle = pure waste) and don't wait to
be told. Owner had to point out an idle pod twice before this clicked.

**The nuance (use judgment):**
- Pick fill-work that ADVANCES the current highest-value goal (e.g. completing the other half of the
  fresh cell so it hits target N in parallel) over speculative/breadth work.
- Keep it PREEMPTIBLE: choose work you can stop/redirect if something higher-priority arrives; never
  let low-priority fill block, delay, or starve a higher priority. Don't let it "run too long."
- Weigh useful-fill vs. tear-down-for-budget each time — if there's genuinely nothing useful and
  budget is tight, terminate it ([[fable-full-context]] pod-termination is your call for idle/surplus).

Extends [[question-the-backend]] and the repo throughput/model-diversity policies (DECISIONS.md):
keep every running pod busy, prefer diversity/highest-value, but now also — INVENT the useful work
proactively rather than parking capacity.
