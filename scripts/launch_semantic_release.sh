#!/bin/bash
# Amendment-11 semantic launch wrapper. This is the only designated semantic
# launch entry point after concurrent technical execution.
set -euo pipefail

NAME="${1:?usage: launch_semantic_release.sh <name> [gpu-type]}"
GPU="${2:-NVIDIA A100 80GB PCIe}"
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?SC_EXPECTED_COMMIT is required}"
ATTESTATION="${SC_RELEASE_ATTESTATION:?SC_RELEASE_ATTESTATION is required}"

[ -z "${SC_SKIP_PREFLIGHT:-}" ] || {
  echo "FATAL: Amendment-11 semantic release forbids SC_SKIP_PREFLIGHT" >&2
  exit 5
}

cd /Users/jeb/experimentation

[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "FATAL: pod name is not a safe slug" >&2
  exit 5
}
shopt -s nullglob
RELEASE_PACKETS=(results/v10_release/preflight_*.json)
[ "${#RELEASE_PACKETS[@]}" -eq 1 ] || {
  echo "FATAL: expected exactly one release preflight, found ${#RELEASE_PACKETS[@]}" >&2
  exit 5
}
[ "$ATTESTATION" = "${RELEASE_PACKETS[0]}" ] || {
  echo "FATAL: SC_RELEASE_ATTESTATION does not name the sole frozen preflight" >&2
  exit 5
}

VERIFIED_RELEASE="$(python3 scripts/validate_semantic_release.py verify \
  --repo . \
  --attestation-path "$ATTESTATION" \
  --expected-launch-commit "$EXPECTED_COMMIT")"
echo "$VERIFIED_RELEASE"
export SC_TECHNICAL_RESULT_COMMIT
export SC_TECHNICAL_RUN_DIR
SC_TECHNICAL_RESULT_COMMIT="$(python3 -c \
  'import json,sys; print(json.load(sys.stdin)["technical_result_commit"])' \
  <<<"$VERIFIED_RELEASE")"
SC_TECHNICAL_RUN_DIR="$(python3 -c \
  'import json,sys; print(json.load(sys.stdin)["technical_run_dir"])' \
  <<<"$VERIFIED_RELEASE")"

MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" \
  exec bash scripts/launch_pod.sh \
  "$NAME" scripts/job_semantic_release.sh "$GPU"
