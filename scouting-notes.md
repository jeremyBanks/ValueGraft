# Scouting notes: coding traces + benchmarks (tasks #12/#13)

*Sonnet research agent, 2026-07-05. Caveat: light verification (single
search round) — re-verify dataset ids/licenses at execution time.*

## Coding-agent trajectories — GO
- Primary: `SWE-Gym/OpenHands-SFT-Trajectories` (HF, MIT). Native `messages`
  field per trajectory (tool calls as turns) → directly replayable through
  the Qwen chat template with compaction at tool-call boundaries. 13–101
  messages/trajectory; filter message-count ≤~30 to find ≤16K-token samples.
- Backup: `nebius/SWE-agent-trajectories` (80K runs, but typically
  50K–300K+ tokens — heavy filtering).
- Avoid: official SWE-bench experiment logs (S3 + AWS CLI, not a download).
- Plan when activated: 10–20 short trajectories, offline next-action
  prediction (teacher-forced) under arms B / SelfGist / tuned ValueGraft.

## Benchmarks — GO via custom subset; LongMemEval is the pick
- `xiaowu0162/LongMemEval` (GitHub) / `xiaowu0162/longmemeval-cleaned` (HF,
  MIT, ungated): 500 QA over multi-session chat histories. No ≤16K variant
  ships (S≈115K tokens), but session boundaries make a subset construction
  light-touch: keep fact-bearing early session(s) + QA session, drop rest.
  Check the `-cleaned` revision (earlier revision had a broken oracle file).
- RULER (NVIDIA, Apache): synthetic, natively 16K-configurable — controlled
  non-chat control condition only.
- SCBench (microsoft): conceptually closest (KV-lifecycle aware); per-task
  lengths unverified. LongBench v2: long-dialogue category, ~8–12K low end.
