#!/usr/bin/env bash
# Pull every exact-v12 technical artifact off one named pod before termination.
set -euo pipefail

NAME="${1:?usage: pull_coherent_canary_v12_technical.sh POD_NAME}"
ROOT=/Users/jeb/experimentation
STATE="$ROOT/.pod_${NAME}_state.json"
KEY="$HOME/.ssh/id_ed25519_runpod"
cd "$ROOT"
[ -f "$STATE" ] || { echo "missing pod state: $STATE"; exit 2; }

STATUS="$(SC_POD_STATE="$STATE" uv run python src/pod.py status)"
read -r IP PORT < <(python3 -c '
import json, sys
d=json.load(sys.stdin)
print(d.get("publicIp") or "", (d.get("portMappings") or {}).get("22") or "")
' <<< "$STATUS")
[ -n "$IP" ] && [ -n "$PORT" ] || { echo "pod has no SSH endpoint"; exit 3; }

SSH_OPTS=(-i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new
          -o ConnectTimeout=20 -o ServerAliveInterval=15
          -o ServerAliveCountMax=4)
RSYNC_SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p .sol-v4/pod-logs
rsync -az --ignore-existing -e "$RSYNC_SSH" \
  "root@$IP:/workspace/exp/job.log" \
  ".sol-v4/pod-logs/v12-technical_${NAME}_${STAMP}.log"

if ! ssh "${SSH_OPTS[@]}" "root@$IP" test -d /workspace/repo/results; then
  echo "remote result tree is absent; job log was preserved locally"
  exit 4
fi

for directory in coherent_canary_v12_technical \
                 coherent_canary_v12_technical_checkpoints \
                 coherent_canary_validation; do
  mkdir -p "results/$directory"
  rsync -az --ignore-existing -e "$RSYNC_SSH" \
    --include='*/' --include='*exact-subject*' --exclude='*' \
    "root@$IP:/workspace/repo/results/$directory/" "results/$directory/"
done

python3 - <<'PY'
import hashlib
import json
from pathlib import Path

receipts = sorted(Path("results/coherent_canary_v12_technical").glob(
    "coherent-canary-v12-technical_exact-subject_*_receipt.json"))
if not receipts:
    raise SystemExit("no exact technical receipt was pulled")
for receipt in receipts:
    document = json.loads(receipt.read_text())
    artifacts = document["artifacts"]
    rows = [artifacts["raw"], artifacts["validation"], artifacts["job_log"],
            *artifacts["identity_checkpoints"]]
    for row in rows:
        path = Path(row["path"])
        data = path.read_bytes()
        if len(data) != row["size_bytes"] or \
                hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SystemExit(f"pulled artifact differs from receipt: {path}")
print(f"PULL VERIFIED receipts={len(receipts)}")
PY

git status --short -- \
  results/coherent_canary_v12_technical \
  results/coherent_canary_v12_technical_checkpoints \
  results/coherent_canary_validation
