#!/usr/bin/env bash
# Bounded, CUDA-compatible admission path for the exact v12 technical gate.
set -euo pipefail

ROOT="${SC_REPO_ROOT:-/Users/jeb/experimentation}"
BASE_NAME="${1:-v12tech}"
MAX_ADMISSION_ATTEMPTS="${SC_ADMISSION_MAX_ATTEMPTS:-3}"
cd "$ROOT"

[[ "$MAX_ADMISSION_ATTEMPTS" =~ ^[1-3]$ ]] || {
  echo "SC_ADMISSION_MAX_ATTEMPTS must be 1, 2, or 3" >&2
  exit 2
}
[ "$(git branch --show-current)" = "trunk" ] || {
  echo "exact v12 launch requires trunk" >&2
  exit 2
}
[ -z "$(git status --porcelain)" ] || {
  echo "exact v12 launch requires a clean worktree" >&2
  exit 2
}
HEAD_COMMIT="$(git rev-parse HEAD)"
[ "$HEAD_COMMIT" = "$(git rev-parse origin/trunk)" ] || {
  echo "exact v12 launch requires HEAD == origin/trunk" >&2
  exit 2
}

export MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507"
export SC_EXPECTED_COMMIT="$HEAD_COMMIT"
export SC_POD_CLOUD=SECURE
export SC_POD_DISK=200
export SC_POD_ALLOWED_CUDA=13.0
export SC_EXPECTED_GPU_NAME="NVIDIA A100 80GB PCIe"
export SC_MIN_NVIDIA_DRIVER=580.65.06
export SC_MIN_GPU_MEMORY_MIB=80000
export SC_SSH_WAIT_ATTEMPTS=20
export SC_TERMINATE_ON_LAUNCH_FAILURE=1
export SC_REQUIRE_HF_TOKEN_DEPLOY=1

echo "V12 EXACT ADMISSION commit=$HEAD_COMMIT cuda=$SC_POD_ALLOWED_CUDA min_driver=$SC_MIN_NVIDIA_DRIVER attempts=$MAX_ADMISSION_ATTEMPTS"
for attempt in $(seq 1 "$MAX_ADMISSION_ATTEMPTS"); do
  NAME="${BASE_NAME}-${attempt}"
  STATE=".pod_${NAME}_state.json"
  [ ! -e "$STATE" ] || {
    echo "refusing to reuse pod state: $STATE" >&2
    exit 2
  }
  set +e
  bash scripts/launch_pod.sh \
    "$NAME" scripts/job_coherent_canary_v12_technical.sh \
    "NVIDIA A100 80GB PCIe"
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    echo "V12 EXACT ADMITTED name=$NAME commit=$HEAD_COMMIT"
    exit 0
  fi
  if [ "$status" -eq 85 ]; then
    if [ "$attempt" -lt "$MAX_ADMISSION_ATTEMPTS" ]; then
      echo "provider allocated no pod for $NAME; retrying bounded allocation" >&2
      continue
    fi
    echo "provider allocated no pod after $MAX_ADMISSION_ATTEMPTS attempts" >&2
    exit 85
  fi
  if [ "$status" -eq 86 ]; then
    echo "v12 exact host $NAME rejected and termination succeeded (attempt $attempt/$MAX_ADMISSION_ATTEMPTS)" >&2
    continue
  fi
  echo "v12 exact launch failed outside bounded allocation/admission (status=$status); not retrying" >&2
  exit "$status"
done

echo "no compatible exact-v12 host admitted after $MAX_ADMISSION_ATTEMPTS attempts" >&2
exit 86
