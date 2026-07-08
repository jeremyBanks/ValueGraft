#!/usr/bin/env bash
# Tier-0 pre-flight gate. Run this (green) BEFORE any scaled spend/fan-out.
# Pass the same MODELS/ANCHORS/BREADTH + job you are about to launch, e.g.:
#   ANCHORS="Qwen/Qwen3-30B-A3B-Instruct-2507 ..." \
#   BREADTH="google/gemma-3-27b-it ..." \
#   bash scripts/preflight.sh scripts/job_sweep.sh
set -uo pipefail
cd /Users/jeb/experimentation
JOB="${1:-${SC_JOB:-}}"
python3 scripts/preflight.py --job "$JOB" --launcher scripts/launch_pod.sh
