#!/usr/bin/env bash
# shellcheck disable=SC2034  # PC_* are the input contract consumed by the sourced classify_pod
# OBSERVABILITY: self-discovering health check across sweep pods (endpoints discovered from
# the launch logs, because the RunPod API returns blank ports for these pods). Prints one
# line per pod; the alerting layer greps for a PROBLEM class.
#
# CLASSIFICATION IS DELEGATED to the shared, fault-tested scripts/classify_pod.sh — this
# script decides REACHABILITY only. It used to classify INLINE and lied twice: it declared
# DONE on PROC=0 + a result merely containing "OK" (not the UNIQUE "WIDE SWEEP DONE" marker
# — the #36 false-DONE), and it declared STALLED off log-age ALONE, before/independent of
# the GPU check (the false-stall class). Both are now impossible: DONE requires the unique
# marker and STALL is GPU-aware, both enforced by classify_pod and proven by monitor_selftest.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null || cd /Users/jeb/experimentation || exit 1
# shellcheck source=scripts/classify_pod.sh
. scripts/classify_pod.sh
K=$HOME/.ssh/id_ed25519_runpod
# feed the remote error grep from the classifier's OWN pattern so no crash signature the
# classifier keys on is dropped by a narrower monitor grep (gap 2).
LOG_PATTERN="native] c[0-9]|PROBE|conv [0-9]+/|WROTE .*status=|${PC_ERROR_SIGNATURES}"

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
  if [ "$ep" = "no-ssh" ]; then
    PC_REACH=booting; classify_pod || true; echo "  $pid: $PC_CLASS ($PC_MSG)"; continue
  fi
  ip=${ep%:*}; port=${ep#*:}
  OUT=$(ssh -n -i "$K" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=2 root@"$ip" 'cd /workspace/exp 2>/dev/null || exit 7
    P=$(pgrep -f cross_arch_probe | grep -v pgrep | wc -l | tr -d " ")
    G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    D=$(grep -c "WIDE SWEEP DONE" job.log 2>/dev/null || echo 0)
    A=$(( $(date +%s) - $(stat -c %Y job.log 2>/dev/null || echo 0) ))
    R=""
    for f in results/cross_arch/*.json; do [ -f "$f" ] && [ "$(basename "$f"|cut -c1)" != "_" ] && R=$(python3 -c "import json;d=json.load(open(\"$f\"));b=d.get(\"by_category_robust\",{}) or {};r=(b.get(\"referent\",{}) or {});print(d.get(\"status\"),\"ref_ci\",r.get(\"raw_EB_ci\"),\"n\",r.get(\"n\"))" 2>/dev/null); done
    L=$(tr "\r" "\n" < job.log 2>/dev/null | grep -aiE "'"$LOG_PATTERN"'" | tail -1)
    printf "P=%s|G=%s|D=%s|A=%s|R=%s|L=%s\n" "$P" "$G" "$D" "$A" "$R" "$L"' 2>/dev/null)
  if [ -z "$OUT" ]; then
    PC_REACH=unreachable; classify_pod || true; echo "  $pid ($ep): $PC_CLASS ($PC_MSG)"; continue
  fi
  PC_REACH=ok
  PC_PROC=$(echo "$OUT" | sed -n 's/.*P=\([0-9]*\).*/\1/p')
  PC_GPU=$(echo "$OUT" | sed -n 's/.*G=\([0-9]*\).*/\1/p')
  PC_DONE=$(echo "$OUT" | sed -n 's/.*D=\([0-9]*\).*/\1/p')
  PC_LOGAGE=$(echo "$OUT" | sed -n 's/.*A=\([0-9]*\).*/\1/p')
  PC_RESULT=$(echo "$OUT" | sed -n 's/.*R=\([^|]*\).*/\1/p')
  PC_LAST=$(echo "$OUT" | sed -n 's/.*L=//p')
  classify_pod || true
  echo "  $pid: $PC_CLASS ($PC_MSG)"
  unset PC_REACH PC_PROC PC_GPU PC_DONE PC_LOGAGE PC_RESULT PC_LAST
done
