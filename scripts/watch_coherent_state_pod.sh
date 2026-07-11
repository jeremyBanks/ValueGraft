#!/usr/bin/env bash
# Local lifecycle watcher: observe, harvest, and terminate one coherent-state pod.
# shellcheck disable=SC2034  # PC_* variables are consumed by sourced classifier.
set -u

NAME="${1:?usage: watch_coherent_state_pod.sh POD_NAME}"
ROOT=/Users/jeb/experimentation
STATE="$ROOT/.pod_${NAME}_state.json"
KEY="$HOME/.ssh/id_ed25519_runpod"
START="$(date +%s)"
LAST_PROGRESS="$START"
LAST_SIG=""
FAIL_REASON=""
PROBLEM_STREAK=0

cd "$ROOT" || exit 1
# shellcheck source=scripts/classify_pod.sh
. scripts/classify_pod.sh
# shellcheck source=scripts/coherent_lifecycle_lib.sh
. scripts/coherent_lifecycle_lib.sh
[ -f "$STATE" ] || { echo "FATAL: state absent $STATE"; exit 2; }

status_json() {
  SC_POD_STATE="$STATE" uv run python src/pod.py status 2>/dev/null
}

harvest_and_terminate() {
  local mode="${1:-failure}" attempt ipp ip port ssh run_remote local_dir stamp
  local remote_dir_rc remote_dir_class
  local harvested=0
  for attempt in 1 2 3 4 5; do
    ipp="$(status_json | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("publicIp") or "")+":"+str((d.get("portMappings") or {}).get("22", "")))' 2>/dev/null)" || true
    ip="${ipp%%:*}"; port="${ipp##*:}"
    if [ -z "$ip" ] || [ -z "$port" ] || [ "$ipp" = ":" ]; then
      echo "HARVEST_RETRY $attempt endpoint unavailable"; sleep 60; continue
    fi
    ssh="ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 root@$ip"
    run_remote="$($ssh "grep -o '/workspace/repo/results/coherent_state/coherent_state_[^ ]*' /workspace/exp/job.log 2>/dev/null | tail -1" 2>/dev/null)" || true
    if [ -n "$run_remote" ]; then
      $ssh "test -d '$run_remote'" >/dev/null 2>&1
      remote_dir_rc=$?
      remote_dir_class="$(coherent_remote_dir_class "$remote_dir_rc")"
      if [ "$remote_dir_class" = ABSENT ]; then
        # The job announces its intended absolute path before clone/model setup.
        # A confirmed absent path is a setup failure, not a result tree.
        run_remote=""
      elif [ "$remote_dir_class" = UNVERIFIED ]; then
        echo "HARVEST_RETRY $attempt result-directory check transport failure rc=$remote_dir_rc"
        sleep 60
        continue
      fi
    fi
    if [ -n "$run_remote" ]; then
      local_dir="$ROOT/${run_remote#/workspace/repo/}"
      mkdir -p "$(dirname "$local_dir")" "$local_dir"
      if rsync -az --checksum -e "ssh -i $KEY -p $port" \
          "root@$ip:$run_remote/" "$local_dir/" && \
          rsync -az --checksum -e "ssh -i $KEY -p $port" \
          "root@$ip:/workspace/exp/job.log" "$local_dir/job.log"; then
        if python3 scripts/validate_coherent_harvest.py "$local_dir" "$mode"
        then harvested=1; echo "HARVEST_VERIFIED $local_dir"; break; fi
      fi
    else
      stamp="$(date -u +%Y%m%dT%H%M%SZ)"
      local_dir="$ROOT/results/_QUARANTINE_coherent_state_${stamp}"
      mkdir -p "$local_dir"
      if rsync -az --checksum -e "ssh -i $KEY -p $port" \
          "root@$ip:/workspace/exp/job.log" "$local_dir/job.log" && \
          [ -s "$local_dir/job.log" ]; then
        status_json > "$local_dir/pod_status.json" 2>/dev/null || true
        harvested=1; echo "HARVEST_VERIFIED_SETUP_FAILURE $local_dir"; break
      fi
    fi
    echo "HARVEST_RETRY $attempt transfer/validation failed"
    sleep 60
  done
  if [ "$harvested" != 1 ]; then
    echo "HARVEST_UNVERIFIED — refusing to terminate result-bearing pod"
    return 2
  fi

  local terminal=0 terminal_confirmations=0 desired
  for attempt in 1 2 3 4 5; do
    SC_POD_STATE="$STATE" uv run python src/pod.py terminate >/dev/null 2>&1 || true
    sleep 10
    if status_json >/tmp/coherent_pod_status_after_delete.json 2>/dev/null; then
      desired="$(python3 -c 'import json; v=json.load(open("/tmp/coherent_pod_status_after_delete.json")).get("desiredStatus"); print(v if isinstance(v,str) else "")' 2>/dev/null)"
      if coherent_terminal_status "$desired"; then
        terminal_confirmations=$((terminal_confirmations + 1))
      else
        terminal_confirmations=0
      fi
    elif grep -q 'API ERROR 404' /tmp/coherent_pod_status_after_delete.json; then
      terminal_confirmations=$((terminal_confirmations + 1))
      desired="HTTP404"
    else
      terminal_confirmations=0
      desired="UNVERIFIED_API_FAILURE"
    fi
    if [ "$terminal_confirmations" -ge 2 ]; then terminal=1; break; fi
    echo "TERMINATE_RETRY $attempt desiredStatus=$desired confirmations=$terminal_confirmations"
  done
  if [ "$terminal" != 1 ]; then
    echo "TERMINATION_UNVERIFIED — pod may still be billing"
    return 3
  fi
  echo "TERMINATION_VERIFIED"
  return 0
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
  if ! OBS="$($SSH "PC_ERROR_SIGNATURES='$PC_ERROR_SIGNATURES' bash -s" <<'REMOTE'
LOG=/workspace/exp/job.log
ALIVE=$(pgrep -f "job.sh|run_coherent_state_hf.py" | grep -v $$ | wc -l | tr -d " ")
DONE=$(grep -c "COHERENT_STATE_JOB_DONE" "$LOG" 2>/dev/null || true)
CRASH=$(grep -cE "$PC_ERROR_SIGNATURES" "$LOG" 2>/dev/null || true)
READY=$(grep -c "MODEL_READY" "$LOG" 2>/dev/null || true)
RUN=$(grep -o "/workspace/repo/results/coherent_state/coherent_state_[^ ]*" "$LOG" 2>/dev/null | tail -1)
if [ -n "$RUN" ] && [ -d "$RUN" ]; then
  CK=$(find "$RUN" -maxdepth 1 -name "conv_*.json" | wc -l | tr -d " ")
  MT=$(find "$RUN" -maxdepth 1 -name "conv_*.json" -exec stat -c %Y {} + 2>/dev/null | sort -n | tail -1)
else CK=0; MT=0; fi
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
AGE=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))
DISK=$(df -BG /workspace | tail -1 | awk '{gsub(/G/,"",$4);print $4}')
echo "ALIVE=$ALIVE DONE=$DONE CRASH=$CRASH READY=$READY CK=$CK MT=${MT:-0} AGE=$AGE GPU=$GPU DISK=$DISK"
REMOTE
)"; then
      echo "$(date -Is) transient SSH observation failure"
      sleep 60
      continue
  fi
  echo "$(date -Is) elapsed=$ELAPSED rate=$RATE $OBS"
  eval "$(echo "$OBS" | sed -n 's/.*ALIVE=\([0-9]*\) DONE=\([0-9]*\) CRASH=\([0-9]*\) READY=\([0-9]*\) CK=\([0-9]*\) MT=\([0-9]*\) AGE=\([0-9]*\) GPU=\([0-9]*\).*/ALIVE=\1;DONE=\2;CRASH=\3;READY=\4;CK=\5;MT=\6;AGE=\7;GPU_UTIL=\8/p')"
  SIG="${CK:-0}:${MT:-0}"
  if [ "$SIG" != "$LAST_SIG" ]; then LAST_SIG="$SIG"; LAST_PROGRESS="$NOW"; fi
  if [ "${READY:-0}" = 0 ] && [ "$ELAPSED" -gt 2700 ]; then
    FAIL_REASON="MODEL_READY deadline exceeded"
    break
  fi
  if [ "${READY:-0}" -gt 0 ] && [ $((NOW - LAST_PROGRESS)) -gt 2700 ]; then
    FAIL_REASON="45 minutes without atomic checkpoint progress"
    break
  fi
  if [ "${READY:-0}" -gt 0 ]; then
    PC_REACH=ok
    PC_PROC="${ALIVE:-}"
    PC_GPU="${GPU_UTIL:-}"
    PC_DONE="${DONE:-0}"
    PC_LAST="CHECKPOINT signature=$SIG"
    [ "${CRASH:-0}" -gt 0 ] && PC_LAST="FATAL signature-count=$CRASH"
    PC_RESULT=""
    PC_LOGAGE="${AGE:-}"
    PC_STALL_SECS=2700
    if classify_pod; then
      PROBLEM_STREAK=0
    else
      PROBLEM_STREAK=$((PROBLEM_STREAK + 1))
      echo "CLASSIFIER_PROBLEM streak=$PROBLEM_STREAK class=$PC_CLASS msg=$PC_MSG"
      if [ "$PROBLEM_STREAK" -ge 2 ] || [ "$PC_CLASS" = "ERROR" ] || \
          [ "$PC_CLASS" = "DIED" ]; then
        FAIL_REASON="classifier $PC_CLASS: $PC_MSG"
        break
      fi
    fi
    if [ "$PC_CLASS" = "DONE" ]; then
      harvest_and_terminate complete || exit $?
      echo "WATCH_COMPLETE"
      exit 0
    fi
    unset PC_REACH PC_PROC PC_GPU PC_DONE PC_LAST PC_RESULT PC_LOGAGE PC_STALL_SECS
  fi
  sleep 300
done

echo "WATCH_FAILURE reason=$FAIL_REASON"
harvest_and_terminate failure || exit $?
exit 1
