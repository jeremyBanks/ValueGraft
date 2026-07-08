#!/usr/bin/env bash
# OBSERVABILITY: self-discovering health check across ALL sweep pods. No registry to
# drift out of sync — it queries the RunPod API for live pods, resolves each SSH
# endpoint, and reads each pod's OWN job state. Classifies every pod into one health
# state and prints one line per pod. Bad states (ERROR/DIED/STALLED/IDLE) are the
# alerting signals; the health monitor greps for them and fires notifications.
#
# States:
#   DONE:<status>     job wrote a result (OK / ERROR / UNSUPPORTED) + the reason
#   ERROR:<reason>    result status ERROR/UNSUPPORTED, or job crashed (Traceback/FATAL)
#   RENDERING:c X/Y   job alive + log advancing (healthy)
#   STALLED:<age>     job process alive but log hasn't advanced in > STALL_MIN min
#   IDLE-GPU          job supposedly running but GPU ~0% and no progress (anomaly)
#   DIED              pod reachable but NO job process and no result (crashed silently)
#   INIT              pod has no SSH endpoint yet (still booting)
#   UNREACHABLE       pod in API but SSH refused (degrading — watch)
set -uo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || cd /Users/jeb/experimentation || exit 1
K=$HOME/.ssh/id_ed25519_runpod
STALL_MIN="${STALL_MIN:-8}"

# 1. pods + endpoints from the LAUNCH LOGS (the RunPod API returns blank ports for these
# pods; the launch logs record the real 'LAUNCHED w<N> at <ip>:<port>' endpoint — reliable).
SCR=/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad
PODS=$(for lg in "$SCR"/launch_w*.log; do
  [ -f "$lg" ] || continue
  ep=$(grep -aoE "LAUNCHED w[0-9]+ at [0-9.]+:[0-9]+" "$lg" 2>/dev/null | tail -1 | awk '{print $4}')
  w=$(grep -aoE "LAUNCHED w[0-9]+" "$lg" 2>/dev/null | tail -1 | sed 's/LAUNCHED //')
  [ -n "$ep" ] && echo "$w $ep"
done | sort -u)

[ -z "$PODS" ] && { echo "HEALTH: no pods (or API error)"; exit 0; }

echo "$PODS" | while read -r pid ep; do
  [ "$pid" = "APIERR" ] && { echo "  API-ERROR: $ep"; continue; }
  if [ "$ep" = "no-ssh" ]; then echo "  $pid  INIT (booting, no ssh yet)"; continue; fi
  ip=${ep%:*}; port=${ep#*:}
  OUT=$(ssh -n -i "$K" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=2 root@"$ip" '
    cd /workspace/exp 2>/dev/null || exit 7
    MODEL=$(grep -aoE "PLAN: [^ ]+" job.log 2>/dev/null | head -1 | cut -d" " -f2)
    PROC=$(pgrep -f cross_arch_probe | grep -v $$ | wc -l | tr -d " ")
    GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    # newest result status+reason
    RES=$(for f in results/cross_arch/*.json; do [ -f "$f" ] && [ "$(basename $f|cut -c1)" != "_" ] && python3 -c "import json;d=json.load(open(\"$f\"));print(d.get(\"status\"),str(d.get(\"reason\"))[:70])" 2>/dev/null; done | tail -1)
    # last progress line + its recency: mtime of job.log
    LASTLOG=$(tr "\r" "\n" < job.log 2>/dev/null | grep -aiE "native] c[0-9]|PROBE|conv [0-9]+/|Traceback|FATAL|Error|OOM|MODELS env" | tail -1)
    LOGAGE=$(( $(date +%s) - $(stat -c %Y job.log 2>/dev/null || echo 0) ))
    echo "MODEL=$MODEL|PROC=$PROC|GPU=$GPU|RES=$RES|LOGAGE=$LOGAGE|LAST=$LASTLOG"
  ' 2>/dev/null)
  rc=$?
  if [ -z "$OUT" ] || [ $rc -ne 0 ]; then echo "  $pid ($ep)  UNREACHABLE (ssh refused/dropped)"; continue; fi
  MODEL=$(echo "$OUT" | grep -oE "MODEL=[^|]*" | cut -d= -f2)
  PROC=$(echo "$OUT" | grep -oE "PROC=[0-9]*" | cut -d= -f2)
  GPU=$(echo "$OUT" | grep -oE "GPU=[0-9]*" | cut -d= -f2)
  RES=$(echo "$OUT" | sed -n 's/.*RES=\([^|]*\).*/\1/p')
  LOGAGE=$(echo "$OUT" | grep -oE "LOGAGE=[0-9]*" | cut -d= -f2)
  LAST=$(echo "$OUT" | sed -n 's/.*LAST=//p')
  tag="$pid ${MODEL:-?}"
  # classify (bad states first — these are the alert signals)
  if echo "$LAST" | grep -qiE "Traceback|FATAL|OOM|MODELS env|FileNotFound"; then
    echo "  $tag  ERROR:crash  $LAST"
  elif echo "$RES" | grep -qiE "ERROR|UNSUPPORTED"; then
    echo "  $tag  ERROR:result  $RES"
  elif [ "${PROC:-0}" = "0" ] && echo "$RES" | grep -qi "OK"; then
    echo "  $tag  DONE:OK  $RES"
  elif [ "${PROC:-0}" = "0" ]; then
    echo "  $tag  DIED (no proc, no OK result)  last:$LAST"
  elif [ "${LOGAGE:-0}" -gt $((STALL_MIN*60)) ]; then
    echo "  $tag  STALLED (${LOGAGE}s no log advance)  last:$LAST"
  elif [ "${GPU:-0}" -lt 3 ]; then
    echo "  $tag  IDLE-GPU (gpu=${GPU}% but proc alive)  last:$LAST"
  else
    echo "  $tag  RENDERING (gpu=${GPU}%)  $LAST"
  fi
done
