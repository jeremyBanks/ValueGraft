#!/usr/bin/env bash
# Fault-injection and fail-closed wiring test for watch_coherent_state_pod.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/classify_pod.sh
. scripts/classify_pod.sh
# shellcheck source=scripts/coherent_lifecycle_lib.sh
. scripts/coherent_lifecycle_lib.sh

PASS=0
case_ok() {
  local want="$1"; shift
  PC_REACH=ok PC_PROC=1 PC_GPU=90 PC_DONE=0 PC_LAST="CHECKPOINT" \
    PC_RESULT="" PC_LOGAGE=10 PC_STALL_SECS=2700
  eval "$*"
  classify_pod >/dev/null 2>&1 || true
  [ "$PC_CLASS" = "$want" ] || {
    echo "FAIL classifier wanted=$want got=$PC_CLASS"; exit 1; }
  PASS=$((PASS + 1))
}
case_ok OK ':'
case_ok DONE 'PC_DONE=1'
case_ok ERROR 'PC_LAST="FATAL injected"'
case_ok DIED 'PC_PROC=0'
case_ok SIGNAL-LOSS 'PC_GPU=""'
case_ok STALLED 'PC_GPU=0; PC_LOGAGE=3000'

WATCH=scripts/watch_coherent_state_pod.sh
bash -n "$WATCH"
grep -q '^\. scripts/classify_pod.sh' "$WATCH"
grep -q 'rsync -az --checksum' "$WATCH"
! grep -Eq 'rsync .*\|\| true' "$WATCH"
grep -q 'HARVEST_UNVERIFIED.*refusing to terminate' "$WATCH"
grep -q 'TERMINATION_UNVERIFIED' "$WATCH"
grep -q 'status_json.*coherent_pod_status_after_delete' "$WATCH"
grep -q "test -d.*run_remote" "$WATCH"
grep -q "API ERROR 404" "$WATCH"
grep -q 'terminal_confirmations.*-ge 2' "$WATCH"
grep -q 'coherent_terminal_status "$desired"' "$WATCH"
grep -q '28800' "$WATCH"
grep -q '2700' "$WATCH"
grep -q 'coherent_state_gapped_v10_' "$WATCH"
grep -q 'validate_coherent_harvest.py.*failure' "$WATCH"
grep -q 'transformers==5.0.0' scripts/job_coherent_state_bf16.sh
grep -q '2.4.1+cu124' scripts/job_coherent_state_bf16.sh
grep -q '2.4.1+cu124' scripts/job_coherent_state_semantic_bf16.sh
grep -q 'SC_TECHNICAL_RESULT_COMMIT' scripts/launch_pod.sh
grep -q 'SC_TECHNICAL_RUN_DIR' scripts/launch_pod.sh
grep -q 'SC_SEMANTIC_RUN_DIR' scripts/launch_pod.sh
grep -q -- '--technical-only' scripts/job_coherent_state_bf16.sh
grep -q 'ATTENTION_BACKEND=eager' scripts/job_coherent_state_bf16.sh
grep -q 'ATTENTION_BACKEND = "eager"' scripts/validate_coherent_harvest.py
grep -q 'COHERENT_STATE_TECHNICAL_DONE' "$WATCH"
grep -q 'production_kernel_gate_\*\.json' "$WATCH"
grep -q 'LONG_GATE_ACTIVE' "$WATCH"
grep -q 'harvest_validation.json' "$WATCH"
grep -q 'validation_args=(--read-only)' "$WATCH"
grep -q 'validation_args=(--output' "$WATCH"
PASS=$((PASS + 29))

[ "$(coherent_remote_dir_class 0)" = EXISTS ]
[ "$(coherent_remote_dir_class 1)" = ABSENT ]
[ "$(coherent_remote_dir_class 255)" = UNVERIFIED ]
[ "$(coherent_run_path_class 0)" = OBSERVED ]
[ "$(coherent_run_path_class 1)" = UNVERIFIED ]
[ "$(coherent_run_path_class 255)" = UNVERIFIED ]
for terminal in EXITED TERMINATED; do
  coherent_terminal_status "$terminal"
done
for nonterminal in RUNNING CREATED "" None; do
  if coherent_terminal_status "$nonterminal"; then
    echo "FAIL nonterminal status accepted: ${nonterminal:-<empty>}"; exit 1
  fi
done
PASS=$((PASS + 12))

# The validator's exact v10 terminal, sidecar, science, semantic-authorization,
# and failure fixtures are its executable CLI/interface self-test. Keep this
# monitor gate coupled to those current fixtures instead of embedding a stale
# second schema here.
PYTHONPATH=src uv run pytest -q \
  tests/test_validate_coherent_harvest.py \
  tests/test_validate_coherent_harvest_v10.py >/dev/null
PASS=$((PASS + 12))

echo "coherent_monitor_selftest: $PASS cases passed, 0 failed"
echo "COHERENT MONITOR CLEARED"
