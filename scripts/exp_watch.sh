#!/usr/bin/env bash
# Watch the EXPECTED set of exploration pods. Endpoints via STATE FILES + pod.py status
# (reliable). ABSENCE IS AN ALARM: an expected model with no state file / no endpoint / no
# process is a problem state, not a benign line (incident #35). Emits per-pod lines + a SUMMARY
# the monitor alarms on. Reads the referent CI when a result is written.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
K=$HOME/.ssh/id_ed25519_runpod
EXPECTED="${EXP_MODELS:-canary expm expo}"
n_ok=0; n_done=0; n_boot=0; n_problem=0
for n in $EXPECTED; do
  sf=".pod_${n}_state.json"
  if [ ! -f "$sf" ]; then echo "$n: MISSING (no state file — expected model not launched)"; n_problem=$((n_problem+1)); continue; fi
  ep=$(SC_POD_STATE="$sf" uv run python src/pod.py status 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('publicIp') or '')+':'+str((d.get('portMappings') or {}).get('22','')))" 2>/dev/null)
  if ! echo "$ep" | grep -qE "^[0-9].*:[0-9]"; then echo "$n: BOOTING (no ssh endpoint yet)"; n_boot=$((n_boot+1)); continue; fi
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
  if [ -z "$OUT" ]; then echo "$n ($ep): UNREACHABLE (ssh refused)"; n_problem=$((n_problem+1)); continue; fi
  p=$(echo "$OUT" | sed -n 's/.*P=\([0-9]*\).*/\1/p'); g=$(echo "$OUT" | sed -n 's/.*G=\([0-9]*\).*/\1/p')
  t=$(echo "$OUT" | sed -n 's/.*T=\([^|]*\).*/\1/p'); dd=$(echo "$OUT" | sed -n 's/.*D=\([0-9]*\).*/\1/p'); l=$(echo "$OUT" | sed -n 's/.*L=\([^|]*\).*/\1/p'); r=$(echo "$OUT" | sed -n 's/.*R=//p')
  if echo "$l" | grep -qiE "FATAL|RuntimeError|model load failed"; then echo "$n: ERROR — $l"; n_problem=$((n_problem+1))
  elif [ "${dd:-0}" -gt 0 ]; then echo "$n: DONE(full) — $r"; n_done=$((n_done+1))
  elif echo "$r" | grep -qiE "^ERROR |^UNSUPPORTED |ERROR ref_ci|UNSUPPORTED ref_ci"; then echo "$n: ERROR(result) — $r"; n_problem=$((n_problem+1))
  elif echo "$r" | grep -q "ref_ci"; then echo "$n: SCORED-INTERIM (probe/partial, still running) — $r"; n_ok=$((n_ok+1))
  elif [ "${p:-0}" = "0" ]; then echo "$n: NO-PROC (died, no WIDE SWEEP DONE) — R=$r L=$l"; n_problem=$((n_problem+1))
  else echo "$n: rendering gpu=${g}% tv=${t} — $l"; n_ok=$((n_ok+1)); fi
done
n_exp=$(echo "$EXPECTED" | wc -w | tr -d " ")
echo "SUMMARY: expected=$n_exp rendering=$n_ok done=$n_done booting=$n_boot problem=$n_problem"
