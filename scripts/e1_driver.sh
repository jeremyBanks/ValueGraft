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
case "$TASK" in swb:*) TASKMOD=src/swebench_tasks.py;; esac
uv run python $TASKMOD materialize "$TASK" "$WS/repo"
cd "$S/ohenv"
export LLM_MODEL="openai/sc-$MODE"  # MODE may carry :aX :cN suffixes LLM_BASE_URL="$BASE" LLM_API_KEY="sc"
export SANDBOX_TYPE=local WORKSPACE_BASE="$WS/repo"
export LOG_ALL_EVENTS=true
uv run --project /Users/jeb/experimentation python /Users/jeb/experimentation/$TASKMOD prompt "$TASK" > "$WS/task.txt"
perl -e 'alarm 3600; exec @ARGV' -- ./.venv/bin/python \
  /Users/jeb/experimentation/src/e1_agent.py "$MODE" "$BASE" "$WS/repo" "$WS/task.txt" \
  > "$WS/agent.log" 2>&1 || true
cd /Users/jeb/experimentation
uv run python $TASKMOD score "$TASK" "$WS/repo" "$WS/agent.log" \
  > "$WS/score.json"
cat "$WS/score.json"
