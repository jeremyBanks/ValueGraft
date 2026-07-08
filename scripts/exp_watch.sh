#!/usr/bin/env bash
# Watch the 3 exploration pods (canary/Qwen, expm/Mistral, expo/OLMo). Endpoints resolved via
# STATE FILES + pod.py status (reliable; the RunPod list API returns blank ports). Classifies
# each pod and reads its referent CI when a result is written. One line per pod.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
K=$HOME/.ssh/id_ed25519_runpod
for n in canary expm expo; do
  sf=".pod_${n}_state.json"
  [ -f "$sf" ] || { echo "$n: no-state-file"; continue; }
  ep=$(SC_POD_STATE="$sf" uv run python src/pod.py status 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('publicIp') or '')+':'+str((d.get('portMappings') or {}).get('22','')))" 2>/dev/null)
  if ! echo "$ep" | grep -qE "^[0-9].*:[0-9]"; then echo "$n: booting/no-endpoint"; continue; fi
  ip=${ep%:*}; port=${ep#*:}
  OUT=$(ssh -n -i "$K" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=8 -o ServerAliveCountMax=3 root@"$ip" 'cd /workspace/exp 2>/dev/null || exit 7
    P=$(pgrep -f cross_arch_probe | grep -v pgrep | wc -l | tr -d " ")
    G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    T=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
    L=$(tr "\r" "\n" < job.log 2>/dev/null | grep -aiE "native] c[0-9]|PROBE|render.graft took|FATAL|WROTE .*status=|RuntimeError|model load failed" | tail -1)
    R=""
    for f in results/cross_arch/*.json; do [ -f "$f" ] && [ "$(basename "$f"|cut -c1)" != "_" ] && R=$(python3 -c "import json;d=json.load(open(\"$f\"));b=d.get(\"by_category_robust\",{}) or {};r=(b.get(\"referent\",{}) or {});print(d.get(\"status\"),\"ref_ci\",r.get(\"raw_EB_ci\"))" 2>/dev/null); done
    printf "P=%s|G=%s|T=%s|L=%s|R=%s\n" "$P" "$G" "$T" "$L" "$R"' 2>/dev/null)
  [ -z "$OUT" ] && { echo "$n ($ep): UNREACHABLE"; continue; }
  p=$(echo "$OUT" | sed -n 's/.*P=\([0-9]*\).*/\1/p'); g=$(echo "$OUT" | sed -n 's/.*G=\([0-9]*\).*/\1/p')
  t=$(echo "$OUT" | sed -n 's/.*T=\([^|]*\).*/\1/p'); l=$(echo "$OUT" | sed -n 's/.*L=\([^|]*\).*/\1/p'); r=$(echo "$OUT" | sed -n 's/.*R=//p')
  if echo "$l" | grep -qiE "FATAL|RuntimeError|model load failed"; then echo "$n: ERROR — $l"
  elif echo "$l" | grep -qiE "WROTE .*status="; then echo "$n: DONE — $r"
  elif [ "${p:-0}" = "0" ]; then echo "$n: no-proc (check) — R=$r L=$l"
  else echo "$n: rendering gpu=${g}% tv=${t} — $l"
  fi
done
