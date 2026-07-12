#!/usr/bin/env bash
# Receipt-driven pull for the ignored exact-v12 e01 treatment job.
set -euo pipefail

NAME="${1:?usage: pull_coherent_canary_v12_treatment_e01.sh POD_NAME}"
ROOT=/Users/jeb/experimentation
TOOLS="$ROOT/.sol-v4/treatment_execution/receipt_tools.py"
STATE="$ROOT/.pod_${NAME}_state.json"
KEY="$HOME/.ssh/id_ed25519_runpod"
INTACT_RAW="$ROOT/.sol-v4/treatment_execution/intact_raw"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PULL_ROOT="$ROOT/.sol-v4/treatment_execution/pull_tmp/$NAME"
POD_LOGS="$ROOT/.sol-v4/treatment_execution/pod_logs"
cd "$ROOT"

[ -f "$STATE" ] || { echo "missing pod state: $STATE"; exit 2; }
[ -f "$TOOLS" ] || { echo "missing receipt helper: $TOOLS"; exit 2; }
mkdir -p "$PULL_ROOT" "$POD_LOGS" "$INTACT_RAW"

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

rsync -az --partial -e "$RSYNC_SSH" \
  "root@$IP:/workspace/exp/job.log" \
  "$POD_LOGS/v12-treatment_${NAME}_${STAMP}.log"

REMOTE_RECEIPT="$(ssh -n "${SSH_OPTS[@]}" "root@$IP" \
  'cat /workspace/exp/v12_treatment_e01_current_receipt.txt')"
if [[ ! "$REMOTE_RECEIPT" =~ ^results/coherent_canary_v12_treatment/coherent-canary-v12-treatment-e01-exact-subject-[0-9]{8}T[0-9]{6}Z_receipt\.json$ ]]; then
  echo "invalid receipt pointer: $REMOTE_RECEIPT"
  exit 4
fi
PULL_TMP="$PULL_ROOT/${REMOTE_RECEIPT##*/}"
mkdir -p "$PULL_TMP"

ssh -n "${SSH_OPTS[@]}" "root@$IP" \
  "test -f /workspace/repo_treatment/$REMOTE_RECEIPT"
rsync -az --partial -e "$RSYNC_SSH" \
  "root@$IP:/workspace/repo_treatment/$REMOTE_RECEIPT" \
  "$PULL_TMP/receipt"
python3 "$TOOLS" validate "$PULL_TMP/receipt" \
  --receipt-relative "$REMOTE_RECEIPT"

while IFS=$'\t' read -r key relative; do
  [[ "$key" =~ ^(raw|harvest|job_log)$ ]] || {
    echo "invalid artifact key: $key"
    exit 5
  }
  [[ "$relative" =~ ^[A-Za-z0-9_./-]+$ ]] || {
    echo "unsafe artifact path: $relative"
    exit 5
  }
  ssh -n "${SSH_OPTS[@]}" "root@$IP" \
    "test -f /workspace/repo_treatment/$relative"
  rsync -az --partial -e "$RSYNC_SSH" \
    "root@$IP:/workspace/repo_treatment/$relative" \
    "$PULL_TMP/$key"
done < <(python3 "$TOOLS" list "$PULL_TMP/receipt")

python3 "$TOOLS" install "$PULL_TMP/receipt" \
  --receipt-relative "$REMOTE_RECEIPT" \
  --staging "$PULL_TMP" \
  --repo "$ROOT" \
  --intact-raw "$INTACT_RAW"

TERMINAL_STATUS="$(python3 - "$PULL_TMP/receipt" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["terminal_status"])
PY
)"

echo "PULL VERIFIED receipt=$REMOTE_RECEIPT raw_archive=$INTACT_RAW terminal_status=$TERMINAL_STATUS"
git status --short -- \
  results/coherent_canary_v12_treatment \
  results/coherent_canary_v12_harvest

[ "$TERMINAL_STATUS" = "PASS" ] || {
  echo "pulled and verified terminal ERROR artifacts"
  exit 7
}
