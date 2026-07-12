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
cd /Users/jeb/experimentation

# ── FAIL-CLOSED PRE-FLIGHT GATE ────────────────────────────────────────────
# No pod launches without a fresh GREEN token (scripts/preflight.py) whose
# mechanism fingerprint matches THIS job+launcher and that verified the models
# in $MODELS/$ANCHORS/$BREADTH. This interlock does NOT depend on anyone
# remembering to check. Override only for a deliberate one-off (logged):
#   SC_SKIP_PREFLIGHT="reason" launch_pod.sh ...
if [ -n "${SC_SKIP_PREFLIGHT:-}" ]; then
  echo "PRE-FLIGHT BYPASSED for $NAME — reason: $SC_SKIP_PREFLIGHT" >&2
else
  MODELS="${MODELS:-}" ANCHORS="${ANCHORS:-}" BREADTH="${BREADTH:-}" \
    python3 scripts/preflight.py --verify --job "$JOB" --launcher scripts/launch_pod.sh \
    || { echo "REFUSING to launch $NAME: pre-flight not green (see above)."; exit 5; }
fi
# ───────────────────────────────────────────────────────────────────────────

K=$HOME/.ssh/id_ed25519_runpod
S=/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad
STATE=".pod_${NAME}_state.json"

admission_fail() {
  echo "POD ADMISSION FAILED for $NAME: $1" >&2
  if [ "${SC_TERMINATE_ON_ADMISSION_FAILURE:-0}" = "1" ] && [ -f "$STATE" ]; then
    SC_POD_STATE="$STATE" uv run python src/pod.py terminate || true
  fi
  exit 86
}

if [ ! -f "$STATE" ]; then
  SC_POD_STATE=$STATE uv run python src/pod.py create "$GPU"
fi
# wait for ssh endpoint
for _ in $(seq 1 40); do
  IPP=$(SC_POD_STATE=$STATE uv run python src/pod.py status 2>/dev/null | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('publicIp') or '')+':'+str((d.get('portMappings') or {}).get('22','')))")
  IP=${IPP%%:*}; PORT=${IPP##*:}
  [ -n "$IP" ] && [ -n "$PORT" ] && [ "$IPP" != ":" ] && break
  sleep 15
done
[ -z "$IP" ] && admission_fail "no SSH endpoint"
# BOUNDED ssh (macOS has no `timeout`): ServerAliveInterval/CountMax make a STALLED
# connection die in ~60s and return nonzero instead of HANGING FOREVER. This converts
# the recurring launcher-hang class (incidents #3, #35 — a hang after launch that
# blocked/starved later pods) into a bounded, LOUD, nonzero-exit failure the caller sees.
SSH="ssh -i $K -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 root@$IP"

GPU_INFO="$($SSH "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits")" \
  || admission_fail "nvidia-smi admission query failed"
if ! SC_OBSERVED_GPU="$GPU_INFO" uv run python src/pod_admission.py; then
  admission_fail "GPU/driver/memory requirements differ"
fi
echo "POD ADMISSION observed: $GPU_INFO requested_cuda=${SC_POD_ALLOWED_CUDA:-ANY}"

$SSH "apt-get update -q >/dev/null 2>&1; apt-get install -y -q rsync >/dev/null 2>&1; mkdir -p /workspace/exp/data" \
  || admission_fail "bootstrap failed"
rsync -azL -e "ssh -i $K -p $PORT" src tune_configs.json tune_rules.json data/scenarios.json data/model_geometry.json data/synthetic data/natural data/decoy_probes.json data/champion_configs swegym.parquet .huggingface_key "root@$IP:/workspace/exp/" 2>/dev/null || true
LME=/Users/jeb/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval-cleaned/snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json
rsync -azL -e "ssh -i $K -p $PORT" "$LME" "root@$IP:/workspace/exp/longmemeval_s_cleaned.json"
$SSH "cd /workspace/exp && mv -f .huggingface_key .hf_key 2>/dev/null; mkdir -p data && mv -f synthetic natural scenarios.json model_geometry.json champion_configs data/ 2>/dev/null; true"
bash -n "$JOB" || { echo "FAIL: job script syntax"; exit 1; }
for f in src/*.py; do python3 -c "import ast,sys; ast.parse(open('$f').read())" || { echo "FAIL: $f syntax"; exit 1; }; done
rsync -az -e "ssh -i $K -p $PORT" "$JOB" "root@$IP:/workspace/exp/job.sh"
echo "$NAME $PORT $IP" >> $S/pods.list
# forward per-pod launch env (MODELS + conv limit) into the remote job execution.
# VERIFIED-DETACH pattern (incident #3): nohup + all fds redirected + </dev/null +
# disown so the job survives the ssh close. The ssh RETURNS immediately.
$SSH "cd /workspace/exp && chmod +x job.sh && MODELS='${MODELS:-}' SC_EXPECTED_COMMIT='${SC_EXPECTED_COMMIT:-}' SC_TECHNICAL_RESULT_COMMIT='${SC_TECHNICAL_RESULT_COMMIT:-}' SC_TECHNICAL_RUN_DIR='${SC_TECHNICAL_RUN_DIR:-}' SC_SEMANTIC_RUN_DIR='${SC_SEMANTIC_RUN_DIR:-}' SC_CONV_LIMIT='${SC_CONV_LIMIT:-}' SC_CONV_START='${SC_CONV_START:-}' SC_HF_MODEL='${SC_HF_MODEL:-}' SC_TASK_COMPETENCE_MODE='${SC_TASK_COMPETENCE_MODE:-}' SC_PROBE_CONVS='${SC_PROBE_CONVS:-}' SC_ABLATE_QK_NORM='${SC_ABLATE_QK_NORM:-}' SC_CHAMPION_SCAN='${SC_CHAMPION_SCAN:-}' SC_CHAMPION_REGIONS='${SC_CHAMPION_REGIONS:-}' SC_CHAMPION_CONFIG='${SC_CHAMPION_CONFIG:-}' SC_LOAD_DTYPE='${SC_LOAD_DTYPE:-}' SC_TUNE_TAG='${SC_TUNE_TAG:-}' SC_QUANT_PKG='${SC_QUANT_PKG:-}' SC_GC_ALPHA='${SC_GC_ALPHA:-}' SC_FULL_DEPTH='${SC_FULL_DEPTH:-}' SC_SUMMARY='${SC_SUMMARY:-}' SC_SWE_N='${SC_SWE_N:-}' SC_E_ALPHA='${SC_E_ALPHA:-}' SC_SWE_TAG='${SC_SWE_TAG:-}' SC_SHARD='${SC_SHARD:-}' SC_LEVELS='${SC_LEVELS:-}' SC_SUMMARY_LEVEL='${SC_SUMMARY_LEVEL:-}' SC_SELFGEN='${SC_SELFGEN:-}' SC_STAGE='${SC_STAGE:-}' SC_E_ALPHAS='${SC_E_ALPHAS:-}' SC_SWE_PLACEBO='${SC_SWE_PLACEBO:-}' SC_PROFILE_REGIONS='${SC_PROFILE_REGIONS:-}' SC_PROFILE_ALPHA='${SC_PROFILE_ALPHA:-}' SC_SWE_MIN_IDX='${SC_SWE_MIN_IDX:-}' nohup bash job.sh</dev/null > job.log 2>&1 & disown; echo job-launched" \
  || { echo "FAIL: launch ssh for $NAME did not return cleanly (hang/drop) — NOT trusting it"; exit 1; }

# ── POST-LAUNCH REAL-WORK CHECK (incident #28: a 'launched' echo is NOT proof) ──
# A job can abort on line 4 (bad cd / missing pkg) and bill the pod for nothing while
# a grep of "launched" matches itself. Verify the detach actually took AND the job
# advanced past bare setup. This is fast (a few s) and fails LOUD + nonzero so a
# never-started pod can NOT be silently swallowed. (Full GPU-residency proof is the
# job's own responsibility + the health monitor — it happens minutes later after load.)
sleep 8
CHK=$($SSH "cd /workspace/exp 2>/dev/null || exit 7
  ALIVE=\$(pgrep -f 'job.sh|cross_arch_probe' | grep -v \$\$ | wc -l | tr -d ' ')
  LINES=\$(wc -l < job.log 2>/dev/null | tr -d ' ')
  CRASH=\$(grep -ciE 'FATAL|Traceback|No such file|command not found|cannot access' job.log 2>/dev/null || echo 0)
  DONE=\$(grep -c 'WIDE SWEEP DONE' job.log 2>/dev/null || echo 0)
  echo \"ALIVE=\$ALIVE LINES=\${LINES:-0} CRASH=\$CRASH DONE=\$DONE\"" 2>/dev/null) \
  || { echo "FAIL: post-launch check ssh for $NAME hung/dropped — cannot confirm the job started"; exit 1; }
ALIVE=$(echo "$CHK" | sed -n 's/.*ALIVE=\([0-9]*\).*/\1/p')
CRASH=$(echo "$CHK" | sed -n 's/.*CRASH=\([0-9]*\).*/\1/p')
LINES=$(echo "$CHK" | sed -n 's/.*LINES=\([0-9]*\).*/\1/p')
DONE=$(echo "$CHK" | sed -n 's/.*DONE=\([0-9]*\).*/\1/p')
if [ "${CRASH:-0}" -gt 0 ]; then
  echo "FAIL: $NAME job CRASHED at launch (setup error in job.log) — pod would bill doing nothing."; exit 1
fi
if [ "${ALIVE:-0}" = "0" ] && [ "${DONE:-0}" = "0" ]; then
  echo "FAIL: $NAME job is NOT running and not done ($CHK) — detach failed / aborted. Do NOT trust it."; exit 1
fi
echo "LAUNCHED $NAME at $IP:$PORT — verified: alive=$ALIVE log_lines=${LINES:-0} (real-work check passed)"
