_The project has shifted from proxy/trace analysis to live OpenHands
coding-agent experiments, with a disciplined exploration → fresh-seed
confirmation → sealed-final evaluation plan. Early live results favor
ValueGraft, but infrastructure failures, configuration contamination, and
duplicate runs still prevent a decisive conclusion._

**Participants:** User and gpt-5.5-xhigh.

**Handoff State.** The current branch reached 49 commits ahead of
`origin/trunk`; the honesty results table is now committed. LongMemEval is
abandoned as a live path and retained only for damage quantification; Gemma
remains parked due to a hybrid-attention scoring issue. The intended writeup is
academic-paper style.

Coding uses OpenHands with `Qwen/Qwen3-30B-A3B-Instruct-2507` in bfloat16,
served through OpenAI-compatible shim variants `sc-A`, `sc-B`, and `sc-E` on
A100 pods. Offline analysis of 75 SWE-Gym/OpenHands traces showed tuned
ValueGraft improving next-action prediction by +0.0156 nats, winning 45/75
paired cases and closing roughly 10% of the A–B compaction gap.

Live coding evidence expanded from seven scored runs to a 92-run exploratory
matrix. The initial clean signal was B failing `t1/t2` while E passed both; the
earlier aggregate was A 1/1, B 0/3, E 3/3, but the larger matrix remains noisy
and exploratory. The `posslots` guard appears to have produced a
negative/contamination verdict. Two lanes may have run the same `t2:s3 E` task
into one scratchpad directory, so that row requires validation or exclusion
before aggregation.

Operational risks remain central: some shim pods loaded and terminated, tuned
`cfg=` parsing previously appeared unsafe before session initialization, and
launcher/quote-parse failures have occurred even where current shell syntax
checks pass. Any supervisor should verify pod health, run ownership, score
provenance, and logs before treating new results as evidence.

A possible supervisory mode was discussed but not approved. If authorized, the
agent could manage experiments, restart or stop pods, maintain
exploratory/confirmation/final separation, and write a handoff checkpoint
approximately every 30 minutes; permissions must explicitly cover file
mutations, pod control, spending, commits/pushes, and priority between
throughput and evidence quality. The refresh request was implemented as one
heartbeat covering three check-ins at roughly two-hour intervals.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
