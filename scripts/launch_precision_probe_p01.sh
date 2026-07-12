#!/usr/bin/env bash
# Fail-closed local admission, provider-clock ownership, and detached launch for
# precision-probe p01.  Internal --watch mode is the proportional watchdog: it
# pulls receipt-bound artifacts before deleting the pod on completion/deadline.
set -euo pipefail

ROOT="${SC_REPO_ROOT:-/Users/jeb/experimentation}"
SELF="$ROOT/scripts/launch_precision_probe_p01.sh"
JOB="$ROOT/scripts/job_precision_probe_p01.sh"
PULL="$ROOT/scripts/pull_precision_probe_p01.sh"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_PROVIDER_SECONDS=7200
MAX_PROVIDER_USD=4.00
SSH_KEY="$HOME/.ssh/id_ed25519_runpod"
CURRENT_STATE=""
CURRENT_LAUNCH_PID=""
CURRENT_WATCHDOG_LABEL=""
WATCHDOG_CONFIRMED=0
WATCHDOG_TERMINAL=0
LAUNCHD_DOMAIN="gui/$(id -u)"

stop_launchd_watchdog() {
  local label="$1"
  [ -n "$label" ] || return 0
  launchctl bootout "$LAUNCHD_DOMAIN/$label" >/dev/null 2>&1 || \
    launchctl remove "$label" >/dev/null 2>&1 || true
}

launchd_watchdog_is_running() {
  local label="$1" status
  status="$(launchctl print "$LAUNCHD_DOMAIN/$label" 2>/dev/null)" || return 1
  grep -Eq '^[[:space:]]*state = running[[:space:]]*$' <<< "$status"
}

# shellcheck disable=SC2329  # invoked indirectly by the --watch EXIT trap
remove_own_launchd_label() {
  local label="$1" status="$2"
  trap - EXIT
  if [ "$WATCHDOG_TERMINAL" -eq 1 ]; then
    stop_launchd_watchdog "$label"
  fi
  exit "$status"
}

terminate_owned() {
  local state="$1" reason="$2" output status
  while true; do
    set +e
    output="$(cd "$ROOT" && SC_POD_STATE="$state" \
      uv run python src/pod.py terminate 2>&1)"
    status=$?
    set -e
    printf '%s\n' "$output"
    if [ "$status" -eq 0 ] || grep -q 'API ERROR 404' <<< "$output"; then
      echo "P01 TERMINATION OWNED reason=$reason"
      return 0
    fi
    echo "P01 DELETE failed status=$status reason=$reason; retrying while clock owner remains alive" >&2
    sleep 5
  done
}

# shellcheck disable=SC2329  # invoked indirectly by the EXIT trap below
cleanup_unwatched_allocation() {
  local status=$?
  trap - EXIT
  if [ "$status" -ne 0 ] && [ -n "$CURRENT_STATE" ] && \
      [ -s "$CURRENT_STATE" ] && [ "$WATCHDOG_CONFIRMED" -eq 0 ]; then
    echo "P01 OWNED CLEANUP: failure after allocation but before watchdog confirmation" >&2
    stop_launchd_watchdog "$CURRENT_WATCHDOG_LABEL"
    CURRENT_WATCHDOG_LABEL=""
    if [ -n "$CURRENT_LAUNCH_PID" ]; then
      kill "$CURRENT_LAUNCH_PID" 2>/dev/null || true
      wait "$CURRENT_LAUNCH_PID" 2>/dev/null || true
    fi
    terminate_owned "$CURRENT_STATE" pre_watchdog_failure
  fi
  exit "$status"
}
trap cleanup_unwatched_allocation EXIT

watch_pod() {
  local name="$1" state="$2" start="$3" cap="$4" rate="$5"
  local log_dir="$ROOT/.sol-v4/precision_probe_p01"
  local deadline=$((start + cap))
  mkdir -p "$log_dir"
  echo "P01 WATCH start name=$name rate=$rate start=$start cap=$cap deadline=$deadline"
  while true; do
    local now status ip port pointer pull_status
    now="$(date +%s)"
    status="$(cd "$ROOT" && SC_POD_STATE="$state" uv run python src/pod.py status 2>/dev/null || true)"
    local observed_rate
    read -r ip port observed_rate < <(python3 -c '
import json, sys
try: value=json.load(sys.stdin)
except Exception: value={}
print(value.get("publicIp") or "", (value.get("portMappings") or {}).get("22") or "", value.get("costPerHr") or "")
' <<< "$status")

    if [ -n "$observed_rate" ] && ! python3 - "$observed_rate" "$rate" <<'PY'
import math, sys
try: observed, initial = map(float, sys.argv[1:])
except ValueError: raise SystemExit(1)
raise SystemExit(0 if math.isfinite(observed) and observed <= initial + 1e-12 else 1)
PY
    then
      echo "P01 WATCH provider rate changed/invalid ($observed_rate; initial=$rate); pulling before termination" >&2
      set +e
      bash "$PULL" "$name"
      pull_status=$?
      set -e
      terminate_owned "$state" provider_rate_changed
      WATCHDOG_TERMINAL=1
      exit "$pull_status"
    fi

    pointer=""
    if [ -n "$ip" ] && [ -n "$port" ]; then
      pointer="$(ssh -n -i "$SSH_KEY" -p "$port" \
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "root@$ip" \
        'cat /workspace/exp/p01_current_receipt.txt 2>/dev/null' 2>/dev/null || true)"
    fi
    if [[ "$pointer" =~ ^results/precision_probe_p01/precision-probe-p01-receipt_Qwen3-30B-A3B-Instruct-2507_[0-9]{8}T[0-9]{6}Z\.json$ ]]; then
      set +e
      bash "$PULL" "$name"
      pull_status=$?
      set -e
      if [ "$pull_status" -eq 0 ]; then
        echo "P01 WATCH artifacts secured; terminating name=$name"
        terminate_owned "$state" artifacts_secured
        WATCHDOG_TERMINAL=1
        exit 0
      fi
      echo "P01 WATCH receipt exists but pull failed status=$pull_status; preserving until deadline" >&2
    fi

    # A setup failure can precede the outer receipt.  Once a real remote log
    # exists and both job.sh and the runner are conclusively gone, preserve a
    # full unreceipted quarantine snapshot and stop billing immediately.
    if [ -z "$pointer" ] && [ -n "$ip" ] && [ -n "$port" ]; then
      local remote_probe
      remote_probe="$(ssh -n -i "$SSH_KEY" -p "$port" \
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "root@$ip" '
          ALIVE=$(pgrep -f "[j]ob.sh|[r]un_precision_probe_p01.py" | wc -l | tr -d " ")
          LOG=0; [ -f /workspace/exp/job.log ] && LOG=1
          echo "ALIVE=${ALIVE:-0} LOG=$LOG"
        ' 2>/dev/null || true)"
      if [ "$remote_probe" = "ALIVE=0 LOG=1" ]; then
        echo "P01 WATCH job ended without receipt; quarantining before termination" >&2
        set +e
        bash "$PULL" "$name"
        pull_status=$?
        set -e
        terminate_owned "$state" unreceipted_job_exit
        WATCHDOG_TERMINAL=1
        exit "$pull_status"
      fi
    fi

    if [ "$now" -ge "$deadline" ]; then
      echo "P01 WATCH hard provider deadline reached; attempting final artifact pull before termination" >&2
      set +e
      bash "$PULL" "$name"
      pull_status=$?
      set -e
      echo "P01 WATCH final pull status=$pull_status; terminating to enforce cap" >&2
      terminate_owned "$state" hard_provider_deadline
      WATCHDOG_TERMINAL=1
      exit "$pull_status"
    fi
    sleep 15
  done
}

if [ "${1:-}" = "--watch" ]; then
  [ "$#" -eq 7 ] || {
    echo "usage: $0 --watch NAME STATE START_EPOCH CAP_SECONDS RATE LAUNCHD_LABEL" >&2
    exit 2
  }
  [[ "$7" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "unsafe launchd watchdog label: $7" >&2
    exit 2
  }
  SELF_WATCHDOG_LABEL="$7"
  trap 'remove_own_launchd_label "$SELF_WATCHDOG_LABEL" "$?"' EXIT
  watch_pod "$2" "$3" "$4" "$5" "$6"
  exit $?
fi

BASE_NAME="${1:-precision-p01}"
MAX_ATTEMPTS="${SC_ADMISSION_MAX_ATTEMPTS:-2}"
cd "$ROOT"

[[ "$BASE_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "unsafe pod base name: $BASE_NAME" >&2
  exit 2
}
[[ "$MAX_ATTEMPTS" =~ ^[12]$ ]] || {
  echo "SC_ADMISSION_MAX_ATTEMPTS must be 1 or 2" >&2
  exit 2
}
[ "$(git branch --show-current)" = "trunk" ] || {
  echo "p01 launch requires trunk" >&2
  exit 2
}
[ -z "$(git status --porcelain)" ] || {
  echo "p01 launch requires a clean worktree" >&2
  exit 2
}
HEAD_COMMIT="$(git rev-parse HEAD)"
[ "$HEAD_COMMIT" = "$(git rev-parse origin/trunk)" ] || {
  echo "p01 launch requires HEAD == origin/trunk" >&2
  exit 2
}
[ -x "$SSH_KEY" ] || [ -f "$SSH_KEY" ] || {
  echo "RunPod SSH key is absent: $SSH_KEY" >&2
  exit 2
}
[ -s .huggingface_key ] || [ -s .hf_key ] || {
  echo "required Hugging Face token file is absent/empty" >&2
  exit 2
}
bash -n "$JOB" "$PULL" scripts/launch_pod.sh
if command -v shellcheck >/dev/null; then
  shellcheck -x "$JOB" "$PULL" "$SELF"
fi

export MODELS="$MODEL"
export SC_EXPECTED_COMMIT="$HEAD_COMMIT"
export SC_POD_CLOUD=SECURE
export SC_POD_DISK=200
export SC_POD_ALLOWED_CUDA=13.0
export SC_EXPECTED_GPU_NAME="NVIDIA A100 80GB PCIe"
export SC_MIN_NVIDIA_DRIVER=580.65.06
export SC_MIN_GPU_MEMORY_MIB=80000
export SC_SSH_WAIT_ATTEMPTS=20
export SC_TERMINATE_ON_LAUNCH_FAILURE=1
export SC_REQUIRE_HF_TOKEN_DEPLOY=1

# Regenerate, then let launch_pod.sh independently verify, the normal green
# mechanism token.  No p01-specific bypass is used.
python3 scripts/preflight.py --job scripts/job_precision_probe_p01.sh \
  --launcher scripts/launch_pod.sh

LOG_DIR="$ROOT/.sol-v4/precision_probe_p01"
mkdir -p "$LOG_DIR"
command -v caffeinate >/dev/null || {
  echo "caffeinate is required for the provider-clock watchdog" >&2
  exit 2
}
command -v launchctl >/dev/null || {
  echo "launchctl is required for the provider-clock watchdog" >&2
  exit 2
}
CAFFEINATE="$(command -v caffeinate)"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  NAME="${BASE_NAME}-${attempt}"
  STATE="$ROOT/.pod_${NAME}_state.json"
  CURRENT_STATE="$STATE"
  CURRENT_LAUNCH_PID=""
  CURRENT_WATCHDOG_LABEL=""
  WATCHDOG_CONFIRMED=0
  WATCHDOG_LABEL="com.semantic-continuity.precision-p01.${NAME//_/-}"
  WATCHDOG_TARGET="$LAUNCHD_DOMAIN/$WATCHDOG_LABEL"
  LAUNCH_LOG="$LOG_DIR/${NAME}_launcher.log"
  LAUNCH_STATUS_FILE="$LOG_DIR/${NAME}_launcher.status"
  WATCH_LOG="$LOG_DIR/${NAME}_watchdog.log"
  BUDGET_LOCAL="$LOG_DIR/${NAME}_provider_budget.json"
  [ ! -e "$STATE" ] || { echo "refusing existing state: $STATE" >&2; exit 2; }
  [ ! -e "$LAUNCH_STATUS_FILE" ] || {
    echo "refusing existing launch status: $LAUNCH_STATUS_FILE" >&2
    exit 2
  }
  if launchctl print "$WATCHDOG_TARGET" >/dev/null 2>&1; then
    echo "refusing existing launchd watchdog: $WATCHDOG_TARGET" >&2
    exit 2
  fi

  echo "P01 ADMISSION attempt=$attempt/$MAX_ATTEMPTS name=$NAME commit=$HEAD_COMMIT"
  (
    set +e
    set -o pipefail
    bash scripts/launch_pod.sh "$NAME" scripts/job_precision_probe_p01.sh \
      "NVIDIA A100 80GB PCIe" 2>&1 | tee "$LAUNCH_LOG"
    status=${PIPESTATUS[0]}
    printf '%s\n' "$status" > "${LAUNCH_STATUS_FILE}.tmp"
    mv "${LAUNCH_STATUS_FILE}.tmp" "$LAUNCH_STATUS_FILE"
    exit "$status"
  ) &
  LAUNCH_PID=$!
  CURRENT_LAUNCH_PID="$LAUNCH_PID"

  # Start clock ownership as soon as create() writes the provider response,
  # not when SSH/bootstrap/model setup later finishes.
  while [ ! -s "$STATE" ] && kill -0 "$LAUNCH_PID" 2>/dev/null; do sleep 1; done
  if [ -s "$STATE" ]; then
    read -r RATE START_EPOCH CAP_SECONDS < <(python3 - "$STATE" \
      "$MAX_PROVIDER_SECONDS" "$MAX_PROVIDER_USD" <<'PY'
from datetime import datetime
import json, math, sys
value=json.load(open(sys.argv[1]))
max_seconds=int(sys.argv[2])
max_usd=float(sys.argv[3])
rate=value.get("costPerHr")
created=value.get("createdAt")
if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or
        not math.isfinite(rate) or rate <= 0):
    raise SystemExit("invalid provider rate")
try:
    start=int(datetime.strptime(created, "%Y-%m-%d %H:%M:%S.%f %z UTC").timestamp())
except Exception as exc:
    raise SystemExit(f"invalid provider createdAt: {created!r}: {exc}")
cap=min(max_seconds, int((max_usd * 3600) // float(rate)))
if cap <= 180:
    raise SystemExit(f"provider rate leaves no useful bounded runtime: {rate}")
print(float(rate), start, cap)
PY
)
    python3 - "$BUDGET_LOCAL" "$NAME" "$HEAD_COMMIT" "$RATE" \
      "$START_EPOCH" "$CAP_SECONDS" "$STATE" <<'PY'
from datetime import datetime, timezone
import json, os, sys
path, name, commit, rate, start, cap, state_path = sys.argv[1:]
state=json.load(open(state_path))
pod_id=state.get("id")
if not isinstance(pod_id,str) or not pod_id:
    raise SystemExit("provider state lacks pod ID")
value = {
    "schema": "precision_probe_p01_provider_budget_v1",
    "protocol_id": "precision-probe-p01",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "pod_name": name,
    "provider_pod_id": pod_id,
    "expected_commit": commit,
    "hourly_cost_usd": float(rate),
    "provider_clock_started_epoch": int(start),
    "provider_wall_cap_seconds": int(cap),
    "max_provider_seconds": 7200,
    "max_cost_usd": 4.0,
}
descriptor=os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    if ! launchctl submit -l "$WATCHDOG_LABEL" -o "$WATCH_LOG" -e "$WATCH_LOG" -- \
      /usr/bin/env "HOME=$HOME" "PATH=$PATH" "SC_REPO_ROOT=$ROOT" \
      "$CAFFEINATE" -dimsu /bin/bash "$SELF" --watch "$NAME" "$STATE" \
      "$START_EPOCH" "$CAP_SECONDS" "$RATE" "$WATCHDOG_LABEL"; then
      echo "provider-clock watchdog launchctl submission failed" >&2
      kill "$LAUNCH_PID" 2>/dev/null || true
      wait "$LAUNCH_PID" 2>/dev/null || true
      terminate_owned "$STATE" watchdog_submit_failure
      CURRENT_STATE=""
      exit 2
    fi
    CURRENT_WATCHDOG_LABEL="$WATCHDOG_LABEL"
    sleep 1
    launchd_watchdog_is_running "$WATCHDOG_LABEL" || {
      echo "provider-clock watchdog failed launchd running-state verification" >&2
      stop_launchd_watchdog "$WATCHDOG_LABEL"
      CURRENT_WATCHDOG_LABEL=""
      kill "$LAUNCH_PID" 2>/dev/null || true
      wait "$LAUNCH_PID" 2>/dev/null || true
      terminate_owned "$STATE" watchdog_start_failure
      CURRENT_STATE=""
      exit 2
    }
    WATCHDOG_CONFIRMED=1
    echo "P01 WATCHDOG label=$WATCHDOG_LABEL target=$WATCHDOG_TARGET rate=$RATE cap_seconds=$CAP_SECONDS log=$WATCH_LOG"
  fi

  set +e
  wait "$LAUNCH_PID"
  TRACKED_STATUS=$?
  set -e
  CURRENT_LAUNCH_PID=""
  if [ -f "$LAUNCH_STATUS_FILE" ]; then
    TRACKED_STATUS="$(tr -d '[:space:]' < "$LAUNCH_STATUS_FILE")"
  fi

  if [ "$TRACKED_STATUS" -eq 85 ] || [ "$TRACKED_STATUS" -eq 86 ]; then
    stop_launchd_watchdog "$CURRENT_WATCHDOG_LABEL"
    CURRENT_WATCHDOG_LABEL=""
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
      echo "bounded no-allocation/rejected-host status=$TRACKED_STATUS; trying the one remaining admission" >&2
      continue
    fi
    exit "$TRACKED_STATUS"
  fi
  if [ ! -s "$STATE" ]; then
    echo "p01 launcher returned status=$TRACKED_STATUS before any allocation" >&2
    [ "$TRACKED_STATUS" -ne 0 ] && exit "$TRACKED_STATUS"
    exit 2
  fi

  # If SSH became available, bind the exact rate/clock record into the remote
  # job whether the tracked launch returned success or an ambiguous failure.
  BUDGET_DEPLOYED=0
  if [ -s "$STATE" ] && [ -f "$BUDGET_LOCAL" ]; then
    STATUS_JSON="$(SC_POD_STATE="$STATE" uv run python src/pod.py status 2>/dev/null || true)"
    read -r IP PORT < <(python3 -c '
import json,sys
try: value=json.load(sys.stdin)
except Exception: value={}
print(value.get("publicIp") or "", (value.get("portMappings") or {}).get("22") or "")
' <<< "$STATUS_JSON")
    if [ -n "$IP" ] && [ -n "$PORT" ]; then
      RSYNC_SSH="ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
      for transfer_attempt in 1 2 3; do
        if rsync -az -e "$RSYNC_SSH" "$BUDGET_LOCAL" \
             "root@$IP:/workspace/exp/p01_provider_budget.json" && \
           ssh -n -i "$SSH_KEY" -p "$PORT" \
             -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
             -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
             "root@$IP" test -s /workspace/exp/p01_provider_budget.json; then
          BUDGET_DEPLOYED=1
          break
        fi
        echo "provider-budget transfer failed attempt=$transfer_attempt/3" >&2
        sleep 5
      done
    fi
  fi
  if [ "$BUDGET_DEPLOYED" -ne 1 ]; then
    echo "FATAL: provider budget was not deployed; terminating before subject work" >&2
    terminate_owned "$STATE" budget_transfer_failure
    stop_launchd_watchdog "$CURRENT_WATCHDOG_LABEL"
    CURRENT_WATCHDOG_LABEL=""
    CURRENT_STATE=""
    WATCHDOG_CONFIRMED=0
    exit 1
  fi

  if [ "$TRACKED_STATUS" -eq 0 ]; then
    echo "P01 LAUNCHED name=$NAME commit=$HEAD_COMMIT watchdog_label=$WATCHDOG_LABEL"
    echo "The watchdog will pull and verify artifacts before termination."
    exit 0
  fi
  echo "p01 launch failed status=$TRACKED_STATUS; no ambiguous post-launch retry" >&2
  exit "$TRACKED_STATUS"
done

exit 86
