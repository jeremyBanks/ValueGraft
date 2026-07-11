#!/bin/bash
# Amendment-11 semantic launch wrapper. This is the only designated semantic
# launch entry point after concurrent technical execution.
set -euo pipefail

NAME="${1:?usage: launch_semantic_release.sh <name> [gpu-type]}"
GPU="${2:-NVIDIA A100 80GB PCIe}"
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?SC_EXPECTED_COMMIT is required}"
ATTESTATION="${SC_RELEASE_ATTESTATION:?SC_RELEASE_ATTESTATION is required}"

cd /Users/jeb/experimentation

python3 scripts/validate_semantic_release.py verify \
  --repo . \
  --attestation-path "$ATTESTATION" \
  --expected-launch-commit "$EXPECTED_COMMIT"

MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" \
  exec bash scripts/launch_pod.sh \
  "$NAME" scripts/job_coherent_state_semantic_bf16.sh "$GPU"
