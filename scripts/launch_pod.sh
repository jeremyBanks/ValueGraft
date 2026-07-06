#!/bin/bash
# Hardened idempotent pod launcher.
# usage: launch_pod.sh <name> <job-script-path> [gpu-type]
# - provisions a secure-tier pod (state in .pod_<name>_state.json)
# - waits for SSH, installs rsync, syncs src+data+key
# - verifies GPU present; job script itself must verify VRAM residency
#   after model load (all runners now print nvidia-smi after load)
# - uploads and launches the given job script detached as job.sh
# - registers pod in scratchpad/pods.list for the watchdog + auto-pull
set -euo pipefail
NAME=$1; JOB=$2; GPU="${3:-NVIDIA A100 80GB PCIe}"
K=$HOME/.ssh/id_ed25519_runpod
S=/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad
STATE=".pod_${NAME}_state.json"

if [ ! -f "$STATE" ]; then
  SC_POD_STATE=$STATE uv run python src/pod.py create "$GPU"
fi
# wait for ssh endpoint
for i in $(seq 1 40); do
  IPP=$(SC_POD_STATE=$STATE uv run python src/pod.py status 2>/dev/null | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('publicIp') or '')+':'+str((d.get('portMappings') or {}).get('22','')))")
  IP=${IPP%%:*}; PORT=${IPP##*:}
  [ -n "$IP" ] && [ -n "$PORT" ] && [ "$IPP" != ":" ] && break
  sleep 15
done
[ -z "$IP" ] && { echo "FAIL: no ssh endpoint for $NAME"; exit 1; }
SSH="ssh -i $K -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$IP"

$SSH "apt-get update -q >/dev/null 2>&1; apt-get install -y -q rsync >/dev/null 2>&1; mkdir -p /workspace/exp/data; nvidia-smi --query-gpu=name --format=csv,noheader" || { echo "FAIL: bootstrap $NAME"; exit 1; }
rsync -azL -e "ssh -i $K -p $PORT" src tune_configs.json data/synthetic data/natural data/decoy_probes.json swegym.parquet .huggingface_key root@$IP:/workspace/exp/ 2>/dev/null || true
LME=/Users/jeb/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval-cleaned/snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json
rsync -azL -e "ssh -i $K -p $PORT" "$LME" root@$IP:/workspace/exp/longmemeval_s_cleaned.json
$SSH "cd /workspace/exp && mv -f .huggingface_key .hf_key 2>/dev/null; mkdir -p data && mv -f synthetic natural data/ 2>/dev/null; true"
bash -n "$JOB" || { echo "FAIL: job script syntax"; exit 1; }
for f in src/*.py; do python3 -c "import ast,sys; ast.parse(open('$f').read())" || { echo "FAIL: $f syntax"; exit 1; }; done
rsync -az -e "ssh -i $K -p $PORT" "$JOB" root@$IP:/workspace/exp/job.sh
echo "$NAME $PORT $IP" >> $S/pods.list
$SSH 'cd /workspace/exp && chmod +x job.sh && nohup bash job.sh > job.log 2>&1 & echo "job pid $!"' 
echo "LAUNCHED $NAME at $IP:$PORT"
