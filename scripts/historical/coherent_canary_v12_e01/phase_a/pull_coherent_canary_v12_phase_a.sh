#!/usr/bin/env bash
# Receipt-driven pull for the ignored e01 Phase-A orchestration job.
set -euo pipefail

NAME="${1:?usage: pull_coherent_canary_v12_phase_a.sh POD_NAME}"
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
  ".sol-v4/pod-logs/v12-phase-a_${NAME}_${STAMP}.log"

REMOTE_RECEIPT="$(ssh -n "${SSH_OPTS[@]}" "root@$IP" \
  'cat /workspace/exp/v12_phase_a_current_receipt.txt')"
if [[ ! "$REMOTE_RECEIPT" =~ ^results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_[0-9]{8}T[0-9]{6}Z_receipt\.json$ ]]; then
  echo "invalid receipt pointer: $REMOTE_RECEIPT"
  exit 4
fi
mkdir -p "$(dirname "$REMOTE_RECEIPT")"
rsync -az --ignore-existing -e "$RSYNC_SSH" \
  "root@$IP:/workspace/repo_phasea/$REMOTE_RECEIPT" "$REMOTE_RECEIPT"

ARTIFACT_PATHS="$(python3 - "$REMOTE_RECEIPT" <<'PY'
import json
from pathlib import PurePosixPath
import sys

document=json.load(open(sys.argv[1]))
artifacts=document.get("artifacts")
if not isinstance(artifacts, dict) or set(artifacts) != {"raw", "validation", "job_log"}:
    raise SystemExit("receipt artifact set differs")
paths=[]
for key in ("raw", "validation", "job_log"):
    row=artifacts[key]
    value=row.get("path") if isinstance(row, dict) else None
    if not isinstance(value, str):
        raise SystemExit(f"invalid receipt artifact row: {key}")
    path=PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "exact-subject" not in path.name:
        raise SystemExit(f"unsafe artifact path: {value}")
    if key in ("raw", "job_log") and not value.startswith(
            "results/coherent_canary_v12_phase_a/"):
        raise SystemExit(f"unexpected Phase-A path: {value}")
    if key == "validation" and not value.startswith(
            "results/coherent_canary_validation/"):
        raise SystemExit(f"unexpected validation path: {value}")
    paths.append(value)
print("\n".join(paths))
PY
)"

while IFS= read -r relative; do
  mkdir -p "$(dirname "$relative")"
  ssh -n "${SSH_OPTS[@]}" "root@$IP" test -f "/workspace/repo_phasea/$relative"
  rsync -az --ignore-existing -e "$RSYNC_SSH" \
    "root@$IP:/workspace/repo_phasea/$relative" "$relative"
done <<< "$ARTIFACT_PATHS"

python3 - "$REMOTE_RECEIPT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

receipt=Path(sys.argv[1]); document=json.loads(receipt.read_text())
for key,row in document["artifacts"].items():
    path=Path(row["path"]); data=path.read_bytes()
    if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise SystemExit(f"pulled {key} differs from receipt: {path}")
print(f"PULL VERIFIED receipt={receipt}")
PY

git status --short -- \
  results/coherent_canary_v12_phase_a \
  results/coherent_canary_validation
