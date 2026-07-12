#!/usr/bin/env bash
# Fail-closed local admission, provider-clock ownership, and detached launch for
# precision-probe p02.  Internal --watch mode is the proportional watchdog: it
# pulls receipt-bound artifacts before deleting the pod on completion/deadline.
set -euo pipefail

ROOT="${SC_REPO_ROOT:-/Users/jeb/experimentation}"
SELF="$ROOT/scripts/launch_precision_probe_p02.sh"
JOB="$ROOT/scripts/job_precision_probe_p02.sh"
PULL="$ROOT/scripts/pull_precision_probe_p02.sh"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
PREREG="$ROOT/PRECISION-PROBE-P02-PREREGISTRATION.md"
PREREG_SHA256="dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d"
MAX_PROVIDER_SECONDS=9000
SCIENTIFIC_BUDGET_USD=3.90
ABSOLUTE_ENVELOPE_USD=4.00
FINALIZE_LEAD_SECONDS=60
# The runner receives SIGINT at C-210 and its full 60-second kill grace ends at
# C-150.  This leaves 30 seconds for outer package/receipt finalization before
# the bounded final pull starts at C-120, reserves ten more seconds after its
# C-70 bound, and prioritize DELETE initiation by C-60 over waiting longer for
# an outer PASS receipt.  An unreceipted quarantine remains valid partial data.
FORCED_PULL_LEAD_SECONDS=120
SSH_KEY="$HOME/.ssh/id_ed25519_runpod"
CURRENT_STATE=""
CURRENT_LAUNCH_PID=""
CURRENT_WATCHDOG_LABEL=""
CURRENT_NAME=""
CURRENT_RATE=""
CURRENT_START_EPOCH=""
CURRENT_CAP_SECONDS=""
CURRENT_ACTIVE_GUARD=""
WATCHDOG_CONFIRMED=0
WATCHDOG_TERMINAL=0
LAUNCHD_DOMAIN="gui/$(id -u)"

stop_launchd_watchdog() {
  local label="$1"
  [ -n "$label" ] || return 0
  launchctl bootout "$LAUNCHD_DOMAIN/$label" >/dev/null 2>&1 || \
    launchctl remove "$label" >/dev/null 2>&1 || true
}

release_active_guard() {
  local guard="$1"
  [ -n "$guard" ] || return 0
  [ "$guard" = "$ROOT/.sol-v4/precision_probe_p02/active" ] || {
    echo "refusing unsafe P02 active-guard path: $guard" >&2
    return 1
  }
  rm -f "$guard/owner.json"
  rmdir "$guard" 2>/dev/null || {
    echo "P02 active guard is nonempty or absent: $guard" >&2
    return 1
  }
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
    release_active_guard "$SELF_ACTIVE_GUARD"
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
      echo "P02 TERMINATION OWNED reason=$reason"
      return 0
    fi
    echo "P02 DELETE failed status=$status reason=$reason; retrying while clock owner remains alive" >&2
    sleep 5
  done
}

bounded_pull() {
  local name="$1" seconds="$2"
  python3 - "$PULL" "$name" "$seconds" <<'PY'
import os, signal, subprocess, sys
script, name, raw_seconds = sys.argv[1:]
seconds = max(1, int(raw_seconds))
process = subprocess.Popen(
    ["/bin/bash", script, name], start_new_session=True)
try:
    raise SystemExit(process.wait(timeout=seconds))
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    raise SystemExit(124)
PY
}

write_settlement() {
  local name="$1" state="$2" start="$3" cap="$4" rate="$5"
  local reason="$6" pull_status="$7" delete_started="$8" delete_returned="$9"
  local stamp output
  stamp="$(python3 -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')"
  output="$ROOT/results/precision_probe_p02/precision-probe-p02-provider-settlement_Qwen3-30B-A3B-Instruct-2507_${stamp}.json"
  mkdir -p "$(dirname "$output")"
  python3 - "$output" "$name" "$state" "$start" "$cap" "$rate" \
    "$reason" "$pull_status" "$delete_started" "$delete_returned" <<'PY'
from datetime import datetime, timezone
import json, math, os, sys
output, name, state_path, start, cap, rate, reason, pull_status, delete_started, delete_returned = sys.argv[1:]
start, cap, pull_status, delete_started, delete_returned = map(
    int, (start, cap, pull_status, delete_started, delete_returned))
rate = float(rate)
state = json.load(open(state_path))
pod_id = state.get("id")
if not isinstance(pod_id, str) or not pod_id:
    raise SystemExit("settlement state lacks pod ID")
absolute_seconds = math.floor(4.00 * 3600 / rate)
estimated_cost = max(0, delete_returned - start) * rate / 3600
document = {
    "schema": "precision_probe_p02_provider_settlement_v1",
    "protocol_id": "precision-probe-p02",
    "preregistration_sha256": "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "pod_name": name,
    "provider_pod_id": pod_id,
    "reason": reason,
    "pull_status": pull_status,
    "hourly_cost_usd": rate,
    "provider_clock_started_epoch": start,
    "scientific_work_cap_seconds": cap,
    "scientific_budget_usd": 3.9,
    "settlement_reserve_usd": 0.1,
    "estimated_scientific_cap_cost_usd": cap * rate / 3600,
    "runner_interrupt_epoch": start + cap - 210,
    "forced_pull_sequence_epoch": start + cap - 120,
    "forced_delete_initiation_epoch": start + cap - 60,
    "delete_initiated_epoch": delete_started,
    "delete_returned_epoch": delete_returned,
    "provider_elapsed_seconds_at_delete_initiation": max(0, delete_started - start),
    "provider_elapsed_seconds_at_delete_return": max(0, delete_returned - start),
    "absolute_envelope_seconds": absolute_seconds,
    "absolute_envelope_epoch": start + absolute_seconds,
    "absolute_envelope_usd": 4.0,
    "estimated_cost_at_delete_return_usd": estimated_cost,
    "within_absolute_envelope_at_delete_return": estimated_cost <= 4.0,
    "estimated_overage_usd": max(0.0, estimated_cost - 4.0),
    "deadline_tradeoff": (
        "forced pull starts at C-120 after the runner's 60-second SIGINT grace; "
        "DELETE initiation by C-60 outranks waiting for an outer PASS receipt; "
        "an unreceipted quarantine preserves partial data"
    ),
    "settlement_basis": "creation-rate multiplied by wall time through DELETE return; not a provider invoice",
}
data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(data); handle.flush(); os.fsync(handle.fileno())
print(f"P02 SETTLEMENT {output}")
PY
}

settle_termination() {
  local name="$1" state="$2" start="$3" cap="$4" rate="$5"
  local reason="$6" pull_status="$7" delete_started delete_returned
  delete_started="$(date +%s)"
  terminate_owned "$state" "$reason"
  delete_returned="$(date +%s)"
  write_settlement "$name" "$state" "$start" "$cap" "$rate" \
    "$reason" "$pull_status" "$delete_started" "$delete_returned"
}

# shellcheck disable=SC2329  # invoked indirectly by the EXIT trap below
cleanup_unwatched_allocation() {
  local status=$?
  trap - EXIT
  if [ "$status" -ne 0 ] && [ -n "$CURRENT_STATE" ] && \
      [ -s "$CURRENT_STATE" ] && [ "$WATCHDOG_CONFIRMED" -eq 0 ]; then
    echo "P02 OWNED CLEANUP: failure after allocation but before watchdog confirmation" >&2
    stop_launchd_watchdog "$CURRENT_WATCHDOG_LABEL"
    CURRENT_WATCHDOG_LABEL=""
    if [ -n "$CURRENT_LAUNCH_PID" ]; then
      kill "$CURRENT_LAUNCH_PID" 2>/dev/null || true
      wait "$CURRENT_LAUNCH_PID" 2>/dev/null || true
    fi
    if [ -n "$CURRENT_RATE" ]; then
      settle_termination "$CURRENT_NAME" "$CURRENT_STATE" \
        "$CURRENT_START_EPOCH" "$CURRENT_CAP_SECONDS" "$CURRENT_RATE" \
        pre_watchdog_failure 8
    else
      terminate_owned "$CURRENT_STATE" pre_watchdog_failure
    fi
  fi
  if [ "$status" -ne 0 ] && [ "$WATCHDOG_CONFIRMED" -eq 0 ] && \
      [ -n "$CURRENT_ACTIVE_GUARD" ]; then
    release_active_guard "$CURRENT_ACTIVE_GUARD"
    CURRENT_ACTIVE_GUARD=""
  fi
  exit "$status"
}
trap cleanup_unwatched_allocation EXIT

watch_pod() {
  local name="$1" state="$2" start="$3" cap="$4" rate="$5"
  local log_dir="$ROOT/.sol-v4/precision_probe_p02"
  local forced_pull_epoch=$((start + cap - FORCED_PULL_LEAD_SECONDS))
  local delete_initiation_epoch=$((start + cap - FINALIZE_LEAD_SECONDS))
  local absolute_seconds
  absolute_seconds="$(python3 -c 'import math,sys; print(math.floor(4.0*3600/float(sys.argv[1])))' "$rate")"
  local absolute_epoch=$((start + absolute_seconds))
  mkdir -p "$log_dir"
  echo "P02 WATCH start name=$name rate=$rate start=$start cap=$cap forced_pull=$forced_pull_epoch delete_by=$delete_initiation_epoch absolute=$absolute_epoch"
  while true; do
    local now status ip port pointer pull_status
    now="$(date +%s)"

    # Deadline actions are evaluated before any provider/SSH call.  Once the
    # forced-pull window opens, do not spend that window on ordinary polling.
    if [ "$now" -ge "$absolute_epoch" ]; then
      echo "P02 WATCH absolute envelope reached; initiating DELETE immediately" >&2
      pull_status=124
      settle_termination "$name" "$state" "$start" "$cap" "$rate" \
        absolute_envelope_reached "$pull_status"
      WATCHDOG_TERMINAL=1
      exit "$pull_status"
    fi
    if [ "$now" -ge "$delete_initiation_epoch" ]; then
      echo "P02 WATCH forced DELETE-initiation threshold reached; deleting immediately" >&2
      pull_status=124
      settle_termination "$name" "$state" "$start" "$cap" "$rate" \
        forced_delete_threshold "$pull_status"
      WATCHDOG_TERMINAL=1
      exit "$pull_status"
    fi
    if [ "$now" -ge "$forced_pull_epoch" ]; then
      local pull_allowance=$((delete_initiation_epoch - now - 10))
      [ "$pull_allowance" -ge 1 ] || pull_allowance=1
      echo "P02 WATCH forced final pull attempt allowance=$pull_allowance" >&2
      set +e
      bounded_pull "$name" "$pull_allowance"
      pull_status=$?
      set -e
      # Refresh immediately after the blocking pull.  Status 0 is a verified
      # receipt tree; status 8 is a demonstrably nonempty unreceipted
      # quarantine.  Either is safe to delete.  Other statuses retry until the
      # C-60 deletion threshold takes ownership.
      now="$(date +%s)"
      if [ "$pull_status" -eq 0 ] || [ "$pull_status" -eq 8 ]; then
        settle_termination "$name" "$state" "$start" "$cap" "$rate" \
          forced_finalize "$pull_status"
        WATCHDOG_TERMINAL=1
        exit "$pull_status"
      fi
      echo "P02 WATCH forced pull failed status=$pull_status at epoch=$now; retrying while time remains" >&2
      continue
    fi

    status="$(cd "$ROOT" && SC_POD_STATE="$state" uv run python src/pod.py status 2>/dev/null || true)"
    # A provider call can consume most of a polling interval.  Re-enter at the
    # top so threshold handling uses a fresh clock before any SSH work.
    now="$(date +%s)"
    [ "$now" -lt "$forced_pull_epoch" ] || continue
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
      echo "P02 WATCH provider rate changed/invalid ($observed_rate; initial=$rate); pulling before termination" >&2
      local pull_allowance=$((delete_initiation_epoch - now - 10))
      [ "$pull_allowance" -ge 1 ] || pull_allowance=1
      [ "$pull_allowance" -le 30 ] || pull_allowance=30
      set +e
      bounded_pull "$name" "$pull_allowance"
      pull_status=$?
      set -e
      now="$(date +%s)"
      settle_termination "$name" "$state" "$start" "$cap" "$rate" \
        provider_rate_changed "$pull_status"
      WATCHDOG_TERMINAL=1
      exit "$pull_status"
    fi

    pointer=""
    if [ -n "$ip" ] && [ -n "$port" ]; then
      pointer="$(ssh -n -i "$SSH_KEY" -p "$port" \
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "root@$ip" \
        'cat /workspace/exp/p02_current_receipt.txt 2>/dev/null' 2>/dev/null || true)"
    fi
    now="$(date +%s)"
    [ "$now" -lt "$forced_pull_epoch" ] || continue
    if [[ "$pointer" =~ ^results/precision_probe_p02/precision-probe-p02-receipt_Qwen3-30B-A3B-Instruct-2507_[0-9]{8}T[0-9]{6}([0-9]{6})?Z\.json$ ]]; then
      local pull_allowance=$((delete_initiation_epoch - now - 10))
      [ "$pull_allowance" -ge 1 ] || pull_allowance=1
      [ "$pull_allowance" -le 60 ] || pull_allowance=60
      set +e
      bounded_pull "$name" "$pull_allowance"
      pull_status=$?
      set -e
      now="$(date +%s)"
      if [ "$pull_status" -eq 0 ] || [ "$pull_status" -eq 8 ]; then
        echo "P02 WATCH artifacts secured; terminating name=$name"
        settle_termination "$name" "$state" "$start" "$cap" "$rate" \
          artifacts_secured "$pull_status"
        WATCHDOG_TERMINAL=1
        exit "$pull_status"
      fi
      echo "P02 WATCH receipt exists but pull failed status=$pull_status; preserving until deadline" >&2
      [ "$now" -lt "$forced_pull_epoch" ] || continue
    fi

    # A setup failure can precede the outer receipt.  Once a real remote log
    # exists and both job.sh and the runner are conclusively gone, preserve a
    # full unreceipted quarantine snapshot and stop billing immediately.
    if [ -z "$pointer" ] && [ -n "$ip" ] && [ -n "$port" ]; then
      local remote_probe
      remote_probe="$(ssh -n -i "$SSH_KEY" -p "$port" \
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "root@$ip" '
          ALIVE=$(pgrep -f "[j]ob.sh|[r]un_precision_probe_p02.py" | wc -l | tr -d " ")
          LOG=0; [ -f /workspace/exp/job.log ] && LOG=1
          echo "ALIVE=${ALIVE:-0} LOG=$LOG"
        ' 2>/dev/null || true)"
      now="$(date +%s)"
      [ "$now" -lt "$forced_pull_epoch" ] || continue
      if [ "$remote_probe" = "ALIVE=0 LOG=1" ]; then
        echo "P02 WATCH job ended without receipt; quarantining before termination" >&2
        local pull_allowance=$((delete_initiation_epoch - now - 10))
        [ "$pull_allowance" -ge 1 ] || pull_allowance=1
        [ "$pull_allowance" -le 60 ] || pull_allowance=60
        set +e
        bounded_pull "$name" "$pull_allowance"
        pull_status=$?
        set -e
        now="$(date +%s)"
        if [ "$pull_status" -eq 0 ] || [ "$pull_status" -eq 8 ]; then
          settle_termination "$name" "$state" "$start" "$cap" "$rate" \
            unreceipted_job_exit "$pull_status"
          WATCHDOG_TERMINAL=1
          exit "$pull_status"
        fi
        echo "P02 WATCH unreceipted capture failed status=$pull_status at epoch=$now; retrying while time remains" >&2
        continue
      fi
    fi
    now="$(date +%s)"
    [ "$now" -lt "$forced_pull_epoch" ] || continue
    local remaining=$((forced_pull_epoch - now))
    sleep $((remaining > 15 ? 15 : (remaining > 0 ? remaining : 1)))
  done
}

if [ "${1:-}" = "--watch" ]; then
  [ "$#" -eq 8 ] || {
    echo "usage: $0 --watch NAME STATE START_EPOCH CAP_SECONDS RATE LAUNCHD_LABEL ACTIVE_GUARD" >&2
    exit 2
  }
  [[ "$7" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "unsafe launchd watchdog label: $7" >&2
    exit 2
  }
  [[ "$2" =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo "unsafe P02 watchdog pod name: $2" >&2
    exit 2
  }
  [ "$3" = "$ROOT/.pod_${2}_state.json" ] && [ -s "$3" ] || {
    echo "P02 watchdog state path differs or is absent: $3" >&2
    exit 2
  }
  SELF_WATCHDOG_LABEL="$7"
  SELF_ACTIVE_GUARD="$8"
  [ "$SELF_ACTIVE_GUARD" = "$ROOT/.sol-v4/precision_probe_p02/active" ] || {
    echo "unsafe P02 active-guard path: $SELF_ACTIVE_GUARD" >&2
    exit 2
  }
  python3 - "$SELF_ACTIVE_GUARD" "$2" "$7" "$4" "$5" "$6" <<'PY'
import json, math, re, sys, time
from pathlib import Path
guard, name, label, start, cap, rate = sys.argv[1:]
root = Path(guard)
owner = root / "owner.json"
if not root.is_dir() or not owner.is_file() or owner.is_symlink():
    raise SystemExit("P02 active guard is absent or unsafe")
if owner.stat().st_size > 4096:
    raise SystemExit("P02 active-guard owner exceeds 4096 bytes")
if {path.name for path in root.iterdir()} != {"owner.json"}:
    raise SystemExit("P02 active guard contains unexpected state")
value = json.loads(owner.read_bytes())
required = {
    "schema", "protocol_id", "pod_name", "launchd_label",
    "expected_commit", "preregistration_sha256", "claimed_at_utc",
}
if (set(value) != required or
        value.get("schema") != "precision_probe_p02_active_guard_v1" or
        value.get("protocol_id") != "precision-probe-p02" or
        value.get("pod_name") != name or
        value.get("launchd_label") != label or
        not re.fullmatch(r"[0-9a-f]{40}", value.get("expected_commit", "")) or
        value.get("preregistration_sha256") !=
            "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d"):
    raise SystemExit("P02 active-guard ownership differs")
try:
    start = int(start)
    cap = int(cap)
    rate = float(rate)
except ValueError as exc:
    raise SystemExit("P02 watchdog clock arguments are invalid") from exc
expected_cap = min(9000, math.floor(3.90 * 3600 / rate)) if rate > 0 else 0
absolute_cap = math.floor(4.00 * 3600 / rate) if rate > 0 else 0
if (start <= 0 or start > int(time.time()) + 60 or
        not math.isfinite(rate) or rate <= 0 or cap != expected_cap or
        cap <= 210 or absolute_cap - cap < 60):
    raise SystemExit("P02 watchdog provider clock/cap differs")
PY
  trap 'remove_own_launchd_label "$SELF_WATCHDOG_LABEL" "$?"' EXIT
  watch_pod "$2" "$3" "$4" "$5" "$6"
  exit $?
fi

BASE_NAME="${1:-precision-p02}"
MAX_ATTEMPTS="${SC_ADMISSION_MAX_ATTEMPTS:-1}"
cd "$ROOT"

[[ "$BASE_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "unsafe pod base name: $BASE_NAME" >&2
  exit 2
}
[[ "$MAX_ATTEMPTS" = "1" ]] || {
  echo "SC_ADMISSION_MAX_ATTEMPTS must be exactly 1 for P02" >&2
  exit 2
}
[ "$(git branch --show-current)" = "trunk" ] || {
  echo "p02 launch requires trunk" >&2
  exit 2
}
[ -z "$(git status --porcelain)" ] || {
  echo "p02 launch requires a clean worktree" >&2
  exit 2
}
HEAD_COMMIT="$(git rev-parse HEAD)"
[ "$HEAD_COMMIT" = "$(git rev-parse origin/trunk)" ] || {
  echo "p02 launch requires HEAD == origin/trunk" >&2
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
OBSERVED_PREREG_SHA256="$(shasum -a 256 "$PREREG" | awk '{print $1}')"
[ "$OBSERVED_PREREG_SHA256" = "$PREREG_SHA256" ] || {
  echo "P02 preregistration SHA-256 differs: $OBSERVED_PREREG_SHA256" >&2
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

LOG_DIR="$ROOT/.sol-v4/precision_probe_p02"
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
ACTIVE_GUARD="$LOG_DIR/active"
if ! mkdir "$ACTIVE_GUARD" 2>/dev/null; then
  echo "refusing a second P02 launch while active guard exists: $ACTIVE_GUARD" >&2
  exit 2
fi
CURRENT_ACTIVE_GUARD="$ACTIVE_GUARD"
GUARD_NAME="$BASE_NAME-1"
GUARD_LABEL="com.semantic-continuity.precision-p02.${GUARD_NAME//_/-}"
python3 - "$ACTIVE_GUARD/owner.json" "$GUARD_NAME" "$GUARD_LABEL" \
  "$HEAD_COMMIT" <<'PY'
from datetime import datetime, timezone
import json, os, sys
path, name, launchd_label, commit = sys.argv[1:]
value = {
    "schema": "precision_probe_p02_active_guard_v1",
    "protocol_id": "precision-probe-p02",
    "pod_name": name,
    "launchd_label": launchd_label,
    "expected_commit": commit,
    "preregistration_sha256": "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d",
    "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
}
data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
if len(data) > 4096:
    raise SystemExit("P02 active-guard owner record exceeds 4096 bytes")
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
PY

# Regenerate, then let launch_pod.sh independently verify, the normal green
# mechanism token.  No p02-specific bypass is used.
python3 scripts/preflight.py --job scripts/job_precision_probe_p02.sh \
  --launcher scripts/launch_pod.sh

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  NAME="${BASE_NAME}-${attempt}"
  STATE="$ROOT/.pod_${NAME}_state.json"
  CURRENT_STATE="$STATE"
  CURRENT_LAUNCH_PID=""
  CURRENT_WATCHDOG_LABEL=""
  CURRENT_NAME="$NAME"
  CURRENT_RATE=""
  CURRENT_START_EPOCH=""
  CURRENT_CAP_SECONDS=""
  WATCHDOG_CONFIRMED=0
  WATCHDOG_LABEL="com.semantic-continuity.precision-p02.${NAME//_/-}"
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

  echo "P02 ADMISSION attempt=$attempt/$MAX_ATTEMPTS name=$NAME commit=$HEAD_COMMIT"
  (
    set +e
    set -o pipefail
    bash scripts/launch_pod.sh "$NAME" scripts/job_precision_probe_p02.sh \
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
    read -r RATE START_EPOCH CAP_SECONDS ABSOLUTE_SECONDS < <(python3 - "$STATE" \
      "$MAX_PROVIDER_SECONDS" "$SCIENTIFIC_BUDGET_USD" \
      "$ABSOLUTE_ENVELOPE_USD" <<'PY'
from datetime import datetime
import json, math, sys
value=json.load(open(sys.argv[1]))
max_seconds=int(sys.argv[2])
scientific_usd=float(sys.argv[3])
absolute_usd=float(sys.argv[4])
rate=value.get("costPerHr")
created=value.get("createdAt")
if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or
        not math.isfinite(rate) or rate <= 0):
    raise SystemExit("invalid provider rate")
try:
    start=int(datetime.strptime(created, "%Y-%m-%d %H:%M:%S.%f %z UTC").timestamp())
except Exception as exc:
    raise SystemExit(f"invalid provider createdAt: {created!r}: {exc}")
cap=min(max_seconds, math.floor(scientific_usd * 3600 / float(rate)))
absolute_cap=math.floor(absolute_usd * 3600 / float(rate))
if cap <= 210:
    raise SystemExit(f"provider rate leaves no useful bounded runtime: {rate}")
if absolute_cap - cap < 60:
    raise SystemExit("provider rate leaves less than 60 seconds inside the $4 settlement envelope")
print(float(rate), start, cap, absolute_cap)
PY
)
    CURRENT_RATE="$RATE"
    CURRENT_START_EPOCH="$START_EPOCH"
    CURRENT_CAP_SECONDS="$CAP_SECONDS"
    python3 - "$BUDGET_LOCAL" "$NAME" "$HEAD_COMMIT" "$RATE" \
      "$START_EPOCH" "$CAP_SECONDS" "$ABSOLUTE_SECONDS" "$STATE" <<'PY'
from datetime import datetime, timezone
import json, os, sys
path, name, commit, rate, start, cap, absolute_seconds, state_path = sys.argv[1:]
state=json.load(open(state_path))
pod_id=state.get("id")
if not isinstance(pod_id,str) or not pod_id:
    raise SystemExit("provider state lacks pod ID")
value = {
    "schema": "precision_probe_p02_provider_budget_v1",
    "protocol_id": "precision-probe-p02",
    "preregistration_sha256": "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "pod_name": name,
    "provider_pod_id": pod_id,
    "expected_commit": commit,
    "hourly_cost_usd": float(rate),
    "provider_clock_started_epoch": int(start),
    "provider_wall_cap_seconds": int(cap),
    "scientific_budget_usd": 3.90,
    "max_provider_seconds": 9000,
    "operational_interrupt_elapsed_seconds": int(cap) - 210,
    "forced_pull_elapsed_seconds": int(cap) - 120,
    "forced_finalize_elapsed_seconds": int(cap) - 60,
    "absolute_envelope_seconds": int(absolute_seconds),
    "absolute_envelope_usd": 4.0,
}
descriptor=os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    if ! launchctl submit -l "$WATCHDOG_LABEL" -o "$WATCH_LOG" -e "$WATCH_LOG" -- \
      /usr/bin/env "HOME=$HOME" "PATH=$PATH" "SC_REPO_ROOT=$ROOT" \
      "$CAFFEINATE" -dimsu /bin/bash "$SELF" --watch "$NAME" "$STATE" \
      "$START_EPOCH" "$CAP_SECONDS" "$RATE" "$WATCHDOG_LABEL" \
      "$ACTIVE_GUARD"; then
      echo "provider-clock watchdog launchctl submission failed" >&2
      kill "$LAUNCH_PID" 2>/dev/null || true
      wait "$LAUNCH_PID" 2>/dev/null || true
      settle_termination "$NAME" "$STATE" "$START_EPOCH" "$CAP_SECONDS" \
        "$RATE" watchdog_submit_failure 8
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
      settle_termination "$NAME" "$STATE" "$START_EPOCH" "$CAP_SECONDS" \
        "$RATE" watchdog_start_failure 8
      CURRENT_STATE=""
      exit 2
    }
    WATCHDOG_CONFIRMED=1
    echo "P02 WATCHDOG label=$WATCHDOG_LABEL target=$WATCHDOG_TARGET rate=$RATE scientific_cap_seconds=$CAP_SECONDS absolute_seconds=$ABSOLUTE_SECONDS log=$WATCH_LOG"
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
    if [ -n "$CURRENT_RATE" ]; then
      SETTLED_NOW="$(date +%s)"
      write_settlement "$NAME" "$STATE" "$START_EPOCH" "$CAP_SECONDS" \
        "$RATE" admission_or_allocation_failure "$TRACKED_STATUS" \
        "$SETTLED_NOW" "$SETTLED_NOW"
    fi
    release_active_guard "$ACTIVE_GUARD"
    CURRENT_ACTIVE_GUARD=""
    exit "$TRACKED_STATUS"
  fi
  if [ ! -s "$STATE" ]; then
    echo "p02 launcher returned status=$TRACKED_STATUS before any allocation" >&2
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
             "root@$IP:/workspace/exp/p02_provider_budget.json" && \
           ssh -n -i "$SSH_KEY" -p "$PORT" \
             -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
             -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
             "root@$IP" test -s /workspace/exp/p02_provider_budget.json; then
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
    settle_termination "$NAME" "$STATE" "$START_EPOCH" "$CAP_SECONDS" \
      "$RATE" budget_transfer_failure 8
    stop_launchd_watchdog "$CURRENT_WATCHDOG_LABEL"
    CURRENT_WATCHDOG_LABEL=""
    CURRENT_STATE=""
    WATCHDOG_CONFIRMED=0
    exit 1
  fi

  if [ "$TRACKED_STATUS" -eq 0 ]; then
    echo "P02 LAUNCHED name=$NAME commit=$HEAD_COMMIT watchdog_label=$WATCHDOG_LABEL"
    echo "The watchdog will pull and verify artifacts before termination."
    exit 0
  fi
  echo "p02 launch failed status=$TRACKED_STATUS; no ambiguous post-launch retry" >&2
  exit "$TRACKED_STATUS"
done

exit 86
