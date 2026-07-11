#!/usr/bin/env bash
# Local lifecycle watcher: observe, harvest, and terminate one coherent-state pod.
set -u

NAME="${1:?usage: watch_coherent_state_pod.sh POD_NAME}"
ROOT=/Users/jeb/experimentation
STATE="$ROOT/.pod_${NAME}_state.json"
KEY="$HOME/.ssh/id_ed25519_runpod"
START="$(date +%s)"
LAST_PROGRESS="$START"
LAST_SIG=""
FAIL_REASON=""

cd "$ROOT" || exit 1
[ -f "$STATE" ] || { echo "FATAL: state absent $STATE"; exit 2; }

status_json() {
  SC_POD_STATE="$STATE" uv run python src/pod.py status 2>/dev/null
}

harvest_and_terminate() {
  local ipp ip port ssh run_remote local_dir stamp
  ipp="$(status_json | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("publicIp") or "")+":"+str((d.get("portMappings") or {}).get("22", "")))' 2>/dev/null)"
  ip="${ipp%%:*}"; port="${ipp##*:}"
  ssh="ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 root@$ip"
  run_remote="$($ssh "grep -o '/workspace/repo/results/coherent_state/coherent_state_[^ ]*' /workspace/exp/job.log 2>/dev/null | tail -1" 2>/dev/null)"
  if [ -n "$run_remote" ]; then
    local_dir="$ROOT/${run_remote#/workspace/repo/}"
    mkdir -p "$(dirname "$local_dir")" "$local_dir"
    rsync -az -e "ssh -i $KEY -p $port" "root@$ip:$run_remote/" "$local_dir/" || true
    rsync -az -e "ssh -i $KEY -p $port" "root@$ip:/workspace/exp/job.log" "$local_dir/job.log" || true
    echo "HARVESTED $local_dir"
  else
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    local_dir="$ROOT/results/_QUARANTINE_coherent_state_${stamp}"
    mkdir -p "$local_dir"
    rsync -az -e "ssh -i $KEY -p $port" "root@$ip:/workspace/exp/job.log" "$local_dir/job.log" || true
    status_json > "$local_dir/pod_status.json" 2>/dev/null || true
    echo "HARVESTED_PARTIAL $local_dir"
  fi
  SC_POD_STATE="$STATE" uv run python src/pod.py terminate || true
}

while true; do
  NOW="$(date +%s)"
  ELAPSED=$((NOW - START))
  STATUS="$(status_json)" || { FAIL_REASON="status API failed"; break; }
  RATE="$(echo "$STATUS" | python3 -c 'import json,sys; print(float(json.load(sys.stdin).get("costPerHr") or 999))' 2>/dev/null)"
  if python3 - "$RATE" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 1.50 else 1)
PY
  then FAIL_REASON="hourly rate $RATE exceeds 1.50"; break; fi
  if [ "$ELAPSED" -gt 28800 ]; then FAIL_REASON="8-hour wall-clock cap"; break; fi

  IPP="$(echo "$STATUS" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("publicIp") or "")+":"+str((d.get("portMappings") or {}).get("22", "")))' 2>/dev/null)"
  IP="${IPP%%:*}"; PORT="${IPP##*:}"
  SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 root@$IP"
  OBS="$($SSH 'LOG=/workspace/exp/job.log
    ALIVE=$(pgrep -f "job.sh|run_coherent_state_hf.py" | grep -v $$ | wc -l | tr -d " ")
    DONE=$(grep -c "COHERENT_STATE_JOB_DONE" "$LOG" 2>/dev/null || true)
    CRASH=$(grep -cE "FATAL|Traceback|CUDA error|OutOfMemoryError" "$LOG" 2>/dev/null || true)
    READY=$(grep -c "MODEL_READY" "$LOG" 2>/dev/null || true)
    RUN=$(grep -o "/workspace/repo/results/coherent_state/coherent_state_[^ ]*" "$LOG" 2>/dev/null | tail -1)
    if [ -n "$RUN" ] && [ -d "$RUN" ]; then
      CK=$(find "$RUN" -maxdepth 1 -name "conv_*.json" | wc -l | tr -d " ")
      MT=$(find "$RUN" -maxdepth 1 -name "conv_*.json" -exec stat -c %Y {} + 2>/dev/null | sort -n | tail -1)
    else CK=0; MT=0; fi
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    DISK=$(df -BG /workspace | tail -1 | awk "{gsub(/G/,\"\",\$4);print \$4}")
    echo "ALIVE=$ALIVE DONE=$DONE CRASH=$CRASH READY=$READY CK=$CK MT=${MT:-0} GPU=$GPU DISK=$DISK"' 2>/dev/null)" || {
      echo "$(date -Is) transient SSH observation failure"
      sleep 60
      continue
    }
  echo "$(date -Is) elapsed=$ELAPSED rate=$RATE $OBS"
  eval "$(echo "$OBS" | sed -n 's/.*ALIVE=\([0-9]*\) DONE=\([0-9]*\) CRASH=\([0-9]*\) READY=\([0-9]*\) CK=\([0-9]*\) MT=\([0-9]*\).*/ALIVE=\1;DONE=\2;CRASH=\3;READY=\4;CK=\5;MT=\6/p')"
  SIG="${CK:-0}:${MT:-0}"
  if [ "$SIG" != "$LAST_SIG" ]; then LAST_SIG="$SIG"; LAST_PROGRESS="$NOW"; fi
  if [ "${DONE:-0}" -gt 0 ]; then
    harvest_and_terminate
    echo "WATCH_COMPLETE"
    exit 0
  fi
  if [ "${CRASH:-0}" -gt 0 ]; then FAIL_REASON="fatal marker in job.log"; break; fi
  if [ "${ALIVE:-0}" = 0 ]; then FAIL_REASON="no scientific/job process"; break; fi
  if [ "${READY:-0}" = 0 ] && [ "$ELAPSED" -gt 2700 ]; then
    FAIL_REASON="MODEL_READY deadline exceeded"
    break
  fi
  if [ "${READY:-0}" -gt 0 ] && [ $((NOW - LAST_PROGRESS)) -gt 2700 ]; then
    FAIL_REASON="45 minutes without atomic checkpoint progress"
    break
  fi
  sleep 300
done

echo "WATCH_FAILURE reason=$FAIL_REASON"
harvest_and_terminate
exit 1
