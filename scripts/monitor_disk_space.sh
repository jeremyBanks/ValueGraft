#!/bin/zsh
set -eu

REPO=${DISK_MONITOR_REPO:-/Users/jeb/experimentation}
WATCH_PATH=${DISK_MONITOR_PATH:-$REPO}
THRESHOLD_GIB=${DISK_MONITOR_THRESHOLD_GIB:-30}
RESET_GIB=${DISK_MONITOR_RESET_GIB:-35}
STATE_DIR=${DISK_MONITOR_STATE_DIR:-$HOME/.local/state/valuegraft-disk-monitor}
MARKER=${DISK_MONITOR_MARKER:-$REPO/.disk-space-alert.json}
NOTIFY=${DISK_MONITOR_NOTIFY:-1}

mkdir -p "$STATE_DIR"
FREE_KIB=$(df -Pk "$WATCH_PATH" | awk 'NR == 2 {print $4}')
FREE_GIB=$((FREE_KIB / 1024 / 1024))
THRESHOLD_KIB=$((THRESHOLD_GIB * 1024 * 1024))
RESET_KIB=$((RESET_GIB * 1024 * 1024))
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FLAG="$STATE_DIR/alert-active"
LOG="$STATE_DIR/transitions.log"

if (( FREE_KIB < THRESHOLD_KIB )); then
  TMP="$MARKER.tmp.$$"
  printf '{\n  "status": "ALERT",\n  "observed_utc": "%s",\n  "free_gib_floor": %d,\n  "threshold_gib": %d,\n  "watched_path": "%s"\n}\n' \
    "$NOW" "$FREE_GIB" "$THRESHOLD_GIB" "$WATCH_PATH" > "$TMP"
  mv "$TMP" "$MARKER"
  if [[ ! -e "$FLAG" ]]; then
    : > "$FLAG"
    printf '%s ALERT free=%dGiB threshold=%dGiB path=%s\n' \
      "$NOW" "$FREE_GIB" "$THRESHOLD_GIB" "$WATCH_PATH" >> "$LOG"
    if [[ "$NOTIFY" == "1" ]]; then
      /usr/bin/osascript -e \
        "display notification \"Only ${FREE_GIB} GiB remains. Free disk before continuing model work.\" with title \"ValueGraft disk-space alert\"" \
        >/dev/null 2>&1 || true
    fi
  fi
elif (( FREE_KIB >= RESET_KIB )); then
  rm -f "$MARKER"
  if [[ -e "$FLAG" ]]; then
    rm -f "$FLAG"
    printf '%s RECOVERED free=%dGiB reset=%dGiB path=%s\n' \
      "$NOW" "$FREE_GIB" "$RESET_GIB" "$WATCH_PATH" >> "$LOG"
  fi
fi
