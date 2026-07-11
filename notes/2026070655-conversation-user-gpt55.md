_This conversation refreshed the ValueGraft experiment’s live state, separated
audited humane evidence from legacy synthetic evidence, and clarified that
synthetic results support a mechanism hypothesis but are not production
conclusions._

**Participants:** User and gpt-5.5-xhigh.

**Handoff State.** On 2026-07-06, the repository was clean and `trunk` advanced
from 34 to 51 commits ahead of `origin/trunk`; all checks were read-only. The
post-audit humane real-task tier remained the instrument of record, with
production-faithful settings including `compact_at=12000`, `tail_keep=6000`,
provenance fields, true recompaction counting, dropped-test recording, and
revised timeout handling. Humane SWE-bench evidence was still too sparse for arm
comparisons: four trusted `c12000` rows were all failures, mostly B, and plain
`swb:` later reached 0/17 trusted scratch passes, including A at 0/4. This
suggested a task-difficulty or solvability floor rather than evidence against a
particular intervention.

The experiment then pivoted to A-first gating and an `swbo:` oracle-retrieval
setting, with easier/Verified instances and a 30-minute episode cap. At the
latest refresh, three `swbo:*` A validation episodes were active on ports
8022/8023, but no trusted `swbo:` outcomes existed yet. Plain SWE-bench and
humane results must remain separate strata; legacy synthetic and nonhumane rows
are supporting/mechanism evidence only.

**Synthetic Result.** In the legacy synthetic coding setup, A/original context
passed consistently, plain summary compaction B was weaker at roughly 2/4 in the
clean slice, and simple graft variants were positive (`E:a0.75` 4/4; `E:a1.0`
3/3). The defensible hypothesis is that value/KV grafting can recover behavior
lost when a detail-starved summary omits specifics. However, these rows used the
old brief-summary prompt and therefore are not production evidence; they should
not headline the paper-style synthesis. `E:cfg=layers` was mixed or poor in the
latest synthetic lane, so current synthetic support favors simple alpha variants
over the layer configuration.

**Operational Follow-up.** H1/SYN, later V1, showed persistent `shim down`
behavior; port 8021 had no listener, while 8013 refused and 8022/8023 timed out
despite some active agents continuing to update. Matrix logs contained
stale-looking `alarm 5400/3600` entries even though the checked-in driver
changed over time to `alarm 1800`, and old runner output included
`t-source: command not found` plus a `line 22` syntax error near `|`; current
HEAD did not contain those faults. Verify that active processes use the current
driver and that logs are not mixing historical and live runs. Also confirm that
`sc_debug` or equivalent provenance is recoverable in final score artifacts
before trusting humane rows.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
