#!/usr/bin/env bash
# shellcheck disable=SC2034  # PC_* are the input contract consumed by the sourced classify_pod
# Watch the EXPECTED set of exploration pods. Endpoints via STATE FILES + pod.py status
# (reliable). ABSENCE IS AN ALARM: an expected model with no state file / no endpoint / no
# process is a problem state, not a benign line (incident #35). Emits per-pod lines + a SUMMARY
# the monitor alarms on. Reads the referent CI when a result is written.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# Classification is delegated to the SHARED, FAULT-TESTED classifier so the monitor's
# live decisions are exactly the code scripts/monitor_selftest.sh validates. Do NOT
# re-implement state logic inline here (that is how every monitor bug was born).
# shellcheck source=scripts/classify_pod.sh
. scripts/classify_pod.sh
K=$HOME/.ssh/id_ed25519_runpod
EXPECTED="${EXP_MODELS:-canary expm expo}"
n_ok=0; n_done=0; n_boot=0; n_problem=0
for n in $EXPECTED; do
  sf=".pod_${n}_state.json"
  # decide REACHABILITY only here; the classifier decides everything else.
  if [ ! -f "$sf" ]; then
    PC_REACH=missing; classify_pod || true; echo "$n: $PC_CLASS ($PC_MSG)"; n_problem=$((n_problem+1)); continue
  fi
  ep=$(SC_POD_STATE="$sf" uv run python src/pod.py status 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('publicIp') or '')+':'+str((d.get('portMappings') or {}).get('22','')))" 2>/dev/null)
  if ! echo "$ep" | grep -qE "^[0-9].*:[0-9]"; then
    PC_REACH=booting; classify_pod || true; echo "$n: $PC_CLASS ($PC_MSG)"; n_boot=$((n_boot+1)); continue
  fi
  ip=${ep%:*}; port=${ep#*:}
  OUT=$(ssh -n -i "$K" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=8 -o ServerAliveCountMax=3 root@"$ip" 'cd /workspace/exp 2>/dev/null || exit 7
    P=$(pgrep -f cross_arch_probe | grep -v pgrep | wc -l | tr -d " ")
    G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    T=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
    L=$(tr "\r" "\n" < job.log 2>/dev/null | grep -aiE "native] c[0-9]|PROBE|render.graft took|FATAL|WROTE .*status=|RuntimeError|model load failed" | tail -1)
    D=$(grep -c "WIDE SWEEP DONE" job.log 2>/dev/null || echo 0)
    R=""
    for f in results/cross_arch/*.json; do [ -f "$f" ] && [ "$(basename "$f"|cut -c1)" != "_" ] && R=$(python3 -c "import json;d=json.load(open(\"$f\"));b=d.get(\"by_category_robust\",{}) or {};r=(b.get(\"referent\",{}) or {});print(d.get(\"status\"),\"ref_ci\",r.get(\"raw_EB_ci\"),\"n\",r.get(\"n\"))" 2>/dev/null); done
    printf "P=%s|G=%s|T=%s|D=%s|L=%s|R=%s\n" "$P" "$G" "$T" "$D" "$L" "$R"' 2>/dev/null)
  if [ -z "$OUT" ]; then
    PC_REACH=unreachable; classify_pod || true; echo "$n ($ep): $PC_CLASS ($PC_MSG)"; n_problem=$((n_problem+1)); continue
  fi
  # parse the remote signal into the classifier's contract, then classify.
  PC_REACH=ok
  PC_PROC=$(echo "$OUT" | sed -n 's/.*P=\([0-9]*\).*/\1/p')
  PC_GPU=$(echo "$OUT" | sed -n 's/.*G=\([0-9]*\).*/\1/p')
  PC_DONE=$(echo "$OUT" | sed -n 's/.*D=\([0-9]*\).*/\1/p')
  PC_LAST=$(echo "$OUT" | sed -n 's/.*L=\([^|]*\).*/\1/p')
  PC_RESULT=$(echo "$OUT" | sed -n 's/.*R=//p')
  t=$(echo "$OUT" | sed -n 's/.*T=\([^|]*\).*/\1/p')
  classify_pod; rc=$?
  echo "$n: $PC_CLASS tv=${t} — $PC_MSG"
  if [ "$rc" -ne 0 ]; then n_problem=$((n_problem+1))
  elif [ "$PC_CLASS" = "DONE" ]; then n_done=$((n_done+1))
  else n_ok=$((n_ok+1)); fi
  unset PC_REACH PC_PROC PC_GPU PC_DONE PC_LAST PC_RESULT
done
n_exp=$(echo "$EXPECTED" | wc -w | tr -d " ")
echo "SUMMARY: expected=$n_exp healthy=$n_ok done=$n_done booting=$n_boot problem=$n_problem"
[ "$n_problem" -gt 0 ] && echo "ALERT: $n_problem expected pod(s) in a PROBLEM state (see lines above)."
exit 0
