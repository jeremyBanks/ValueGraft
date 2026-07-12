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

if ! REMOTE_RECEIPT="$(ssh "${SSH_OPTS[@]}" "root@$IP" \
    'cat /workspace/exp/v12_exact_technical_current_receipt.txt')"; then
  echo "current exact technical receipt pointer is absent; job log was preserved"
  exit 5
fi
if [[ ! "$REMOTE_RECEIPT" =~ ^results/coherent_canary_v12_technical/coherent-canary-v12-technical_exact-subject_[0-9]{8}T[0-9]{6}Z_receipt\.json$ ]]; then
  echo "current exact technical receipt pointer is invalid: $REMOTE_RECEIPT"
  exit 5
fi
mkdir -p "$(dirname "$REMOTE_RECEIPT")"
rsync -az --ignore-existing -e "$RSYNC_SSH" \
  "root@$IP:/workspace/repo/$REMOTE_RECEIPT" "$REMOTE_RECEIPT"

ARTIFACT_PATHS="$(python3 - "$REMOTE_RECEIPT" <<'PY'
import json
from pathlib import PurePosixPath
import sys

receipt_path = sys.argv[1]
document = json.load(open(receipt_path))
artifacts = document.get("artifacts")
if not isinstance(artifacts, dict):
    raise SystemExit("current receipt has no artifact object")
rows = [artifacts.get("raw"), artifacts.get("validation"),
        artifacts.get("job_log")]
checkpoints = artifacts.get("identity_checkpoints")
if not isinstance(checkpoints, list):
    raise SystemExit("current receipt checkpoint list is invalid")
rows.extend(checkpoints)
allowed = (
    "results/coherent_canary_v12_technical/",
    "results/coherent_canary_v12_technical_checkpoints/",
    "results/coherent_canary_validation/",
)
paths = []
for row in rows:
    if not isinstance(row, dict) or not isinstance(row.get("path"), str):
        raise SystemExit("current receipt artifact row is invalid")
    value = row["path"]
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or
            not value.startswith(allowed) or "exact-subject" not in path.name):
        raise SystemExit(f"current receipt artifact path is unsafe: {value}")
    paths.append(value)
if len(paths) != len(set(paths)):
    raise SystemExit("current receipt artifact path is duplicated")
print("\n".join(paths))
PY
)"
ARTIFACT_COUNT="$(grep -c . <<< "$ARTIFACT_PATHS")"
[ "$ARTIFACT_COUNT" -ge 3 ] || {
  echo "current exact technical receipt names fewer than three base artifacts"
  exit 6
}
while IFS= read -r relative; do
  mkdir -p "$(dirname "$relative")"
  if ! ssh "${SSH_OPTS[@]}" "root@$IP" test -f "/workspace/repo/$relative"; then
    echo "current receipt artifact is absent on pod: $relative"
    exit 6
  fi
  rsync -az --ignore-existing -e "$RSYNC_SSH" \
    "root@$IP:/workspace/repo/$relative" "$relative"
done <<< "$ARTIFACT_PATHS"

python3 - "$REMOTE_RECEIPT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

receipt = Path(sys.argv[1])
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
print(f"PULL VERIFIED receipt={receipt}")
PY

git status --short -- \
  results/coherent_canary_v12_technical \
  results/coherent_canary_v12_technical_checkpoints \
  results/coherent_canary_validation
