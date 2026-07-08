---
name: verify-boring-before-clever
description: "Before any expensive or clever action, verify the boring preconditions — exact model/config, committed state, real progress, right branch, the actual number. A whole session of failures shared this one root."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

07-08 was a cascade of avoidable, self-inflicted failures. Every one shared a single
shape: **I did something clever/expensive while skipping a boring verification.** The
owner had to catch most of them. The transferable lesson is the pattern, not the
individual bugs.

The cascade (all one session):
- Ran a **local 4B MLX model** to generate corpus text — never questioned the backend. [[question-the-backend]]
- Ran a **27-item job in one sequential subagent** — then **killed it ~25/27 done** off an
  output-file-SIZE proxy, destroying near-complete work. [[shard-and-verify-before-killing]]
- **`rm`'d six uncommitted batch files** after a failed merge (the merge failed on a
  zero-padding bug in my own assertion, not the data).
- Let a subagent **create a git branch**; only-trunk is the rule.
- **Hardened a non-problem** (strict alignment for a misdiagnosed "token matching" alarm),
  which introduced real brittleness that broke both harnesses.
- Spent **~2 hours** letting a subagent grind 40-min runs on a narrow fix, and read
  `status=OK` passively instead of reading the actual number myself.
- The real root cause of all that: I was running the **WRONG MODEL CHECKPOINT** the entire
  time (thinking Qwen3-30B-A3B vs the non-thinking Instruct-2507 the result was measured on).
- Consulted Fable for the strategic view **hours too late**.

**The rule: before any expensive or clever action, run the boring checklist FIRST.**
- Am I using the EXACT model id + config the known-good result used? (not just the family) [[validate-before-trusting]]
- Is the work I'm about to consume/delete COMMITTED? (never rm/kill uncommitted work)
- Am I on trunk? (never branches)
- Is this subagent actually making progress? (read real signal, not a size/mtime proxy)
- Did I read the actual NUMBER myself, or trust a "status=OK"?
- If two independent things fail identically → the bug is in SHARED code / a shared input.
- Have I asked Fable for the strategic view? (early — at 1–2 failed attempts, not after hours)

Clever debugging feels like progress; it is worthless if a boring precondition is wrong.
Check the boring things first, every time.
