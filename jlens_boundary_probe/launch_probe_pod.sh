#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/jlens_boundary_probe"
STATE="$DIR/.pod_jlens_probe_state.json"
LOG="$DIR/launch_probe_pod.log"
GPU="${1:-NVIDIA A100 80GB PCIe}"
K="$HOME/.ssh/id_ed25519_runpod"

cd "$ROOT"

if [ ! -f "$STATE" ]; then
  SC_POD_STATE="$STATE" uv run python src/pod.py create "$GPU" | tee -a "$LOG"
fi

for _ in $(seq 1 40); do
  STATUS="$(SC_POD_STATE="$STATE" uv run python src/pod.py status 2>/dev/null || true)"
  IP="$(printf '%s' "$STATUS" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("publicIp") or "")' 2>/dev/null || true)"
  PORT="$(printf '%s' "$STATUS" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("portMappings") or {}).get("22") or "")' 2>/dev/null || true)"
  if [ -n "$IP" ] && [ -n "$PORT" ]; then
    break
  fi
  sleep 15
done

if [ -z "${IP:-}" ] || [ -z "${PORT:-}" ]; then
  echo "FAIL: pod did not expose ssh endpoint" | tee -a "$LOG"
  exit 1
fi

SSH=(ssh -i "$K" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "root@$IP")
RSYNC=(rsync -az -e "ssh -i $K -p $PORT -o StrictHostKeyChecking=accept-new")

"${SSH[@]}" 'apt-get update -q >/dev/null 2>&1; apt-get install -y -q rsync git >/dev/null 2>&1; mkdir -p /workspace/jlens_boundary_probe; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' | tee -a "$LOG"

"${RSYNC[@]}" "$DIR/" "root@$IP:/workspace/jlens_boundary_probe/"
if [ -f "$ROOT/.huggingface_key" ]; then
  "${RSYNC[@]}" "$ROOT/.huggingface_key" "root@$IP:/workspace/.huggingface_key"
  "${SSH[@]}" 'chmod 600 /workspace/.huggingface_key'
fi

"${SSH[@]}" 'cd /workspace/jlens_boundary_probe && chmod +x job_qwen36_probe.sh && nohup bash job_qwen36_probe.sh > job.log 2>&1 < /dev/null & echo "job pid $!"' | tee -a "$LOG"

cat <<EOF | tee -a "$LOG"
LAUNCHED jlens probe pod
ssh: ssh -i $K -p $PORT -o StrictHostKeyChecking=accept-new root@$IP
remote log: /workspace/jlens_boundary_probe/job.log
pull results:
  rsync -az -e "ssh -i $K -p $PORT" root@$IP:/workspace/jlens_boundary_probe/outputs/ "$DIR/outputs/"
EOF
