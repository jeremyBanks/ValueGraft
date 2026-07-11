#!/usr/bin/env bash
# Remote Amendment-11 outer gate. It verifies the committed L AND T release
# packet in the fresh exact launch clone before invoking the frozen v10 semantic
# job. This file is intentionally outside the v10 apparatus glob.
set -euo pipefail

BOOT=/workspace/exp
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?SC_EXPECTED_COMMIT is required}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CHECKOUT="/workspace/release_check_${STAMP}"

echo "START SEMANTIC_RELEASE_OUTER $(date -Is) EXPECTED_COMMIT=$EXPECTED_COMMIT"
[ ! -e "$CHECKOUT" ] || { echo "FATAL: release-check path exists"; exit 3; }
git clone --quiet https://github.com/jeremyBanks/ValueGraft.git "$CHECKOUT"
cd "$CHECKOUT"
git checkout --quiet trunk
[ "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" ] || {
  echo "FATAL: origin/trunk $(git rev-parse HEAD) != expected $EXPECTED_COMMIT"
  exit 3
}

shopt -s nullglob
RELEASE_PACKETS=(results/v10_release/preflight_*.json)
[ "${#RELEASE_PACKETS[@]}" -eq 1 ] || {
  echo "FATAL: expected exactly one committed release preflight, found ${#RELEASE_PACKETS[@]}"
  exit 3
}

python3 -m pip install -q "transformers==5.0.0" "huggingface_hub==1.22.0" \
  "sentencepiece==0.2.1" "safetensors==0.8.0"
for f in "$BOOT/.hf_key" "$BOOT/.huggingface_key" /workspace/.huggingface_key; do
  if [ -f "$f" ]; then
    export HF_TOKEN
    HF_TOKEN="$(tr -d '[:space:]' < "$f")"
    break
  fi
done
[ -n "${HF_TOKEN:-}" ] || { echo "FATAL: HF token absent"; exit 2; }

# Seed only the exact production config/tokenizer. The release resolver and
# independent harvest deliberately use local_files_only=True so an unpinned
# network lookup can never occur during validation.
python3 - <<'PY'
from transformers import AutoConfig, AutoTokenizer
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
AutoConfig.from_pretrained(MODEL, revision=REVISION)
AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
print("SEMANTIC_RELEASE_CACHE_SEEDED", MODEL, REVISION, flush=True)
PY

VERIFIED_RELEASE="$(python3 scripts/validate_semantic_release.py verify \
  --repo . \
  --attestation-path "${RELEASE_PACKETS[0]}" \
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

RELEASE_RAW="$(python3 -c \
  'import json,sys; print(json.load(sys.stdin)["attestation_raw_sha256"])' \
  <<<"$VERIFIED_RELEASE")"
RELEASE_PAYLOAD="$(python3 -c \
  'import json,sys; print(json.load(sys.stdin)["attestation_payload_sha256"])' \
  <<<"$VERIFIED_RELEASE")"

cp scripts/job_coherent_state_semantic_bf16.sh "$BOOT/inner_semantic_job.sh"
chmod +x "$BOOT/inner_semantic_job.sh"
echo "SEMANTIC_RELEASE_OUTER_PASS launch=$EXPECTED_COMMIT packet=${RELEASE_PACKETS[0]} raw=$RELEASE_RAW payload=$RELEASE_PAYLOAD technical_commit=$SC_TECHNICAL_RESULT_COMMIT technical_run_dir=$SC_TECHNICAL_RUN_DIR"
exec bash "$BOOT/inner_semantic_job.sh"
