#!/bin/bash
# E1 driver: run one synthetic coding task through OpenHands headless against
# a shim pod, under a given mode (A|B|E). Objective: task tests pass.
# usage: e1_driver.sh <mode> <task_id> <base_url>
# Requires: scratchpad/ohenv/.venv, src/e1_tasks.py (task generator).
set -euo pipefail
MODE=$1; TASK=$2; BASE=$3
S=/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad
WS=$S/e1_runs/${TASK//:/-}_${MODE//:/-}
rm -rf "$WS" && mkdir -p "$WS"
TASKMOD=src/e1_tasks.py
case "$TASK" in swb:*|swbo:*) TASKMOD=src/swebench_tasks.py;; esac
uv run python $TASKMOD materialize "$TASK" "$WS/repo"
cd "$S/ohenv"
export LLM_MODEL="openai/sc-$MODE"  # suffixes :aX :cN ride along
export SANDBOX_TYPE=local WORKSPACE_BASE="$WS/repo"
export LOG_ALL_EVENTS=true
uv run --project /Users/jeb/experimentation python /Users/jeb/experimentation/$TASKMOD prompt "$TASK" > "$WS/task.txt"
perl -e 'alarm 1800; exec @ARGV' -- ./.venv/bin/python \
  /Users/jeb/experimentation/src/e1_agent.py "$MODE" "$BASE" "$WS/repo" "$WS/task.txt" \
  > "$WS/agent.log" 2>&1 || true
cd /Users/jeb/experimentation
# validity-at-source v2: invalid ONLY if the episode never really ran
# (connection errors, or trivially small log). A TIMEOUT after real work is
# a legitimate failure observation — score the repo state as-is.
LOGSZ=$(wc -c < "$WS/agent.log" 2>/dev/null | tr -d " "); LOGSZ=${LOGSZ:-0}
if grep -qE "Connection refused|Connection reset|APIConnectionError" "$WS/agent.log" 2>/dev/null || [ "$LOGSZ" -lt 3000 ]; then
  echo "INVALID_EPISODE (connection errors or never started, log=$LOGSZ) — no score written" | tee "$WS/invalid.marker"
  exit 0
fi
grep -q "E1_AGENT_DONE" "$WS/agent.log" 2>/dev/null || echo "TIMEOUT_EPISODE (real work, no self-termination) — scoring repo state" | tee "$WS/timeout.marker"
uv run python $TASKMOD score "$TASK" "$WS/repo" "$WS/agent.log" \
  > "$WS/score.json"
cat "$WS/score.json"
