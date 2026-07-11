#!/usr/bin/env bash
# Fault-injection and fail-closed wiring test for watch_coherent_state_pod.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/classify_pod.sh
. scripts/classify_pod.sh
# shellcheck source=scripts/coherent_lifecycle_lib.sh
. scripts/coherent_lifecycle_lib.sh

PASS=0
case_ok() {
  local want="$1"; shift
  PC_REACH=ok PC_PROC=1 PC_GPU=90 PC_DONE=0 PC_LAST="CHECKPOINT" \
    PC_RESULT="" PC_LOGAGE=10 PC_STALL_SECS=2700
  eval "$*"
  classify_pod >/dev/null 2>&1 || true
  [ "$PC_CLASS" = "$want" ] || {
    echo "FAIL classifier wanted=$want got=$PC_CLASS"; exit 1; }
  PASS=$((PASS + 1))
}
case_ok OK ':'
case_ok DONE 'PC_DONE=1'
case_ok ERROR 'PC_LAST="FATAL injected"'
case_ok DIED 'PC_PROC=0'
case_ok SIGNAL-LOSS 'PC_GPU=""'
case_ok STALLED 'PC_GPU=0; PC_LOGAGE=3000'

WATCH=scripts/watch_coherent_state_pod.sh
bash -n "$WATCH"
grep -q '^\. scripts/classify_pod.sh' "$WATCH"
grep -q 'rsync -az --checksum' "$WATCH"
! grep -Eq 'rsync .*\|\| true' "$WATCH"
grep -q 'HARVEST_UNVERIFIED.*refusing to terminate' "$WATCH"
grep -q 'TERMINATION_UNVERIFIED' "$WATCH"
grep -q 'status_json.*coherent_pod_status_after_delete' "$WATCH"
grep -q "test -d.*run_remote" "$WATCH"
grep -q "API ERROR 404" "$WATCH"
grep -q 'terminal_confirmations.*-ge 2' "$WATCH"
grep -q 'coherent_terminal_status "$desired"' "$WATCH"
grep -q '28800' "$WATCH"
grep -q '2700' "$WATCH"
grep -q 'coherent_state_gapped_v2_' "$WATCH"
grep -q 'validate_coherent_harvest.py.*failure' "$WATCH"
grep -q 'transformers==5.0.0' scripts/job_coherent_state_bf16.sh
PASS=$((PASS + 15))

[ "$(coherent_remote_dir_class 0)" = EXISTS ]
[ "$(coherent_remote_dir_class 1)" = ABSENT ]
[ "$(coherent_remote_dir_class 255)" = UNVERIFIED ]
[ "$(coherent_run_path_class 0)" = OBSERVED ]
[ "$(coherent_run_path_class 1)" = UNVERIFIED ]
[ "$(coherent_run_path_class 255)" = UNVERIFIED ]
for terminal in EXITED TERMINATED; do
  coherent_terminal_status "$terminal"
done
for nonterminal in RUNNING CREATED "" None; do
  if coherent_terminal_status "$nonterminal"; then
    echo "FAIL nonterminal status accepted: ${nonterminal:-<empty>}"; exit 1
  fi
done
PASS=$((PASS + 12))

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python3 - "$TMP" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
identity = {"schema": 2,
            "amendment_id": "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2",
            "design_id": "coherent-state-gapped-v2"}
arms = ["A_full", "G_fresh", "G_correct", "G_wrong", "G_Vcorrect",
        "G_Kcorrect", "G_delta"]
def write(name, doc):
    (root / name).write_text(json.dumps(doc) + "\n")
(root / "job.log").write_text("MODEL_READY\nCOHERENT_STATE_JOB_DONE\n")
write("manifest.json", {**identity, "status": "COMPLETE",
      "resume_probe_verified": True})
write("resume_probe.json", {**identity, "status": "VERIFIED",
      "resume_probe_verified": True})
write("production_kernel_gate.json", {**identity, "status": "PASS",
      "completed_at": "2026-07-11T00:00:00Z", "gates": {"passes": True}})
for i in range(1, 7):
    write(f"conv_{i:02d}_c{i:02d}.json", {
        **identity, "stage": "scored", "status": "scored", "order_position": i,
        "fingerprint": identity, "conversation": {}, "summary": {}, "sources": {},
        "destination": {}, "arm_scores": {x: {} for x in arms},
        "conversation_outcomes": {x: 0.0 for x in arms},
        "gates": {"technical_pass": True}, "runtime": {}})
PY
python3 scripts/validate_coherent_harvest.py "$TMP" complete >/dev/null
printf '{bad\n' > "$TMP/conv_01.json"
if python3 scripts/validate_coherent_harvest.py "$TMP" complete >/dev/null 2>&1; then
  echo "FAIL malformed harvest was accepted"; exit 1
fi
PASS=$((PASS + 2))

echo "coherent_monitor_selftest: $PASS cases passed, 0 failed"
echo "COHERENT MONITOR CLEARED"
