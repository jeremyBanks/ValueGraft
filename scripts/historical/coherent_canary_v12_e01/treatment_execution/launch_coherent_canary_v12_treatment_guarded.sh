#!/usr/bin/env bash
# Precreate one location-pinned treatment pod and own its complete provider clock.
set -euo pipefail

NAME="${1:?usage: launch_coherent_canary_v12_treatment_guarded.sh POD_NAME [DATA_CENTER]}"
DATA_CENTER="${2:-US-KS-2}"
ROOT=/Users/jeb/experimentation
JOB="$ROOT/.sol-v4/treatment_execution/job_coherent_canary_v12_treatment_e01.sh"
WATCHDOG="$ROOT/.sol-v4/treatment_execution/provider_deadline_watchdog.py"
CREATE_HELPER="$ROOT/.sol-v4/create_runpod_in_datacenter.py"
STATE="$ROOT/.pod_${NAME}_state.json"
GUARD_DIR="$ROOT/.sol-v4/treatment_execution/provider_clock"
RECORD="$GUARD_DIR/${NAME}.json"
CREATE_RESPONSE="$GUARD_DIR/${NAME}_create_response.json"
CREATE_LOG="$GUARD_DIR/${NAME}_create_stderr.log"
LAUNCH_LOG="$GUARD_DIR/${NAME}_tracked_launcher.log"
WATCH_LOG="$GUARD_DIR/${NAME}_watchdog.log"
PINNED_COMMIT="cbdfa481fe08de62bf8178d09ca040716d610f21"
MAX_RATE_USD=1.39
MAX_RENTAL_SECONDS=2300
ALLOCATED=0
WATCHDOG_STARTED=0
cd "$ROOT"

[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "unsafe pod name: $NAME" >&2
  exit 2
}
[ "$DATA_CENTER" = "US-KS-2" ] || {
  echo "treatment placement must remain US-KS-2" >&2
  exit 2
}
[ ! -e "$STATE" ] || { echo "refusing existing pod state: $STATE"; exit 2; }
[ ! -e "$RECORD" ] || { echo "refusing existing guard record: $RECORD"; exit 2; }
[ ! -e "$CREATE_RESPONSE" ] || {
  echo "refusing existing create response: $CREATE_RESPONSE"
  exit 2
}
[ -f "$JOB" ] && [ -f "$WATCHDOG" ] && [ -f "$CREATE_HELPER" ] || {
  echo "treatment guard input is absent" >&2
  exit 2
}
mkdir -p "$GUARD_DIR"

# shellcheck disable=SC2329  # invoked indirectly by the EXIT trap
cleanup_unwatched_allocation() {
  status=$?
  trap - EXIT
  if [ "$status" -ne 0 ] && [ "$ALLOCATED" -eq 1 ] && \
     [ "$WATCHDOG_STARTED" -eq 0 ]; then
    echo "OWNED CLEANUP: terminating precreated pod after pre-watchdog failure" >&2
    python3 "$WATCHDOG" terminate-state --state "$STATE" \
      --reason pre_watchdog_launch_failure || true
  fi
  exit "$status"
}
trap cleanup_unwatched_allocation EXIT

bash -n "$JOB" scripts/launch_pod.sh
shellcheck -x "$JOB"
export MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507"
export SC_EXPECTED_COMMIT="$PINNED_COMMIT"
export SC_POD_CLOUD=SECURE
export SC_POD_DISK=200
export SC_POD_ALLOWED_CUDA=13.0
export SC_EXPECTED_GPU_NAME="NVIDIA A100 80GB PCIe"
export SC_MIN_NVIDIA_DRIVER=580.65.06
export SC_MIN_GPU_MEMORY_MIB=80000
export SC_SSH_WAIT_ATTEMPTS=20
export SC_REQUIRE_HF_TOKEN_DEPLOY=1
python3 scripts/preflight.py --verify --job "$JOB" \
  --launcher scripts/launch_pod.sh

START_EPOCH="$(date +%s)"
set +e
SC_POD_NAME="$NAME" SC_POD_STATE="$STATE" \
  uv run python "$CREATE_HELPER" \
    --data-center "$DATA_CENTER" --gpu "NVIDIA A100 80GB PCIe" \
    > "$CREATE_RESPONSE" 2> "$CREATE_LOG"
CREATE_STATUS=$?
set -e
if [ "$CREATE_STATUS" -ne 0 ]; then
  if [ -f "$STATE" ]; then
    ALLOCATED=1
    echo "CREATE failed after allocation; owned cleanup is mandatory" >&2
  fi
  exit "$CREATE_STATUS"
fi
ALLOCATED=1

set +e
python3 "$WATCHDOG" init \
  --name "$NAME" --state "$STATE" --response "$CREATE_RESPONSE" \
  --job "$JOB" --record "$RECORD" --start-epoch "$START_EPOCH"
GUARD_INIT_STATUS=$?
set -e
if [ "$GUARD_INIT_STATUS" -ne 0 ]; then
  if [ -f "$RECORD" ]; then
    python3 "$WATCHDOG" terminate --record "$RECORD" \
      --reason invalid_create_response_or_rate
    ALLOCATED=0
  fi
  exit "$GUARD_INIT_STATUS"
fi

command -v caffeinate >/dev/null || {
  echo "caffeinate is required to keep the local deadline watchdog awake" >&2
  exit 2
}
nohup caffeinate -dimsu python3 "$WATCHDOG" watch --record "$RECORD" \
  > "$WATCH_LOG" 2>&1 < /dev/null &
WATCHDOG_PID=$!
sleep 1
kill -0 "$WATCHDOG_PID" 2>/dev/null || {
  echo "deadline watchdog failed to remain alive" >&2
  exit 2
}
WATCHDOG_STARTED=1
echo "WATCHDOG pid=$WATCHDOG_PID record=$RECORD hard_seconds=$MAX_RENTAL_SECONDS max_rate=$MAX_RATE_USD"

set +e
bash scripts/launch_pod.sh \
  "$NAME" "$JOB" "NVIDIA A100 80GB PCIe" 2>&1 | tee "$LAUNCH_LOG"
LAUNCH_STATUS=${PIPESTATUS[0]}
set -e
if [ "$LAUNCH_STATUS" -eq 0 ]; then
  echo "TREATMENT LAUNCHED name=$NAME guard=$RECORD watchdog_pid=$WATCHDOG_PID"
  echo "MONITOR: python3 $WATCHDOG status --record $RECORD"
  echo "SAFE MANUAL STOP AFTER PULL: python3 $WATCHDOG terminate --record $RECORD --reason artifacts_secured"
  exit 0
fi

# A failed tracked launcher is not proof that detached work failed to start.
# Only clean up before the deadline when SSH conclusively shows no process,
# no receipt pointer, and no remote job log. Otherwise the watchdog preserves
# potentially wanted work until the machine-enforced hard deadline.
STATUS_JSON="$(SC_POD_STATE="$STATE" uv run python src/pod.py status 2>/dev/null || true)"
read -r IP PORT < <(python3 -c '
import json, sys
try: d=json.load(sys.stdin)
except Exception: d={}
print(d.get("publicIp") or "", (d.get("portMappings") or {}).get("22") or "")
' <<< "$STATUS_JSON")
PROBE="AMBIGUOUS"
if [ -n "$IP" ] && [ -n "$PORT" ]; then
  set +e
  PROBE_OUTPUT="$(ssh -n -i "$HOME/.ssh/id_ed25519_runpod" -p "$PORT" \
    -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
    -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "root@$IP" '
      PROC=$(pgrep -f "run_coherent_canary_v12_treatment.py|job.sh" | grep -v $$ | wc -l | tr -d " ")
      LOG=0; [ -f /workspace/exp/job.log ] && LOG=1
      RECEIPT=0; [ -f /workspace/exp/v12_treatment_e01_current_receipt.txt ] && RECEIPT=1
      echo "PROC=${PROC:-0} LOG=$LOG RECEIPT=$RECEIPT"
    ' 2>/dev/null)"
  PROBE_STATUS=$?
  set -e
  if [ "$PROBE_STATUS" -eq 0 ] && \
     [ "$PROBE_OUTPUT" = "PROC=0 LOG=0 RECEIPT=0" ]; then
    PROBE="PROVEN_NOT_LAUNCHED"
  else
    PROBE="$PROBE_OUTPUT"
  fi
fi

if [ "$PROBE" = "PROVEN_NOT_LAUNCHED" ]; then
  echo "tracked launch failed before any remote job evidence; owned cleanup" >&2
  python3 "$WATCHDOG" terminate --record "$RECORD" \
    --reason proven_pre_job_launch_failure
  exit "$LAUNCH_STATUS"
fi

echo "tracked launcher returned $LAUNCH_STATUS but remote state is potentially wanted/ambiguous: $PROBE" >&2
echo "PRESERVED under deadline watchdog; inspect with: python3 $WATCHDOG status --record $RECORD" >&2
exit "$LAUNCH_STATUS"
