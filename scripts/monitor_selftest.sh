#!/usr/bin/env bash
# MONITOR FAULT-INJECTION SELF-TEST — a monitor may NOT be trusted until this passes.
#
# THE DISEASE (incidents #35, #36 + the endpoint-blind/stall/done/error quartet): monitors
# whose classification logic was never validated against how the system ACTUALLY emits signals.
# THE CURE (RELIABILITY.md row 0): drive the pure classifier (scripts/classify_pod.sh) with a
# FIXTURE for every state — including the HAPPY PATH, which must produce NO false alert — and
# assert the exact class. If any case is wrong, exit nonzero: the monitor is NOT cleared for use.
#
# Run before relying on exp_watch.sh / pod_health.sh:  bash scripts/monitor_selftest.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=scripts/classify_pod.sh
. scripts/classify_pod.sh

pass=0; fail=0
# case: <desc> <expect-class> <expect-rc 0|1> ; env vars set inline before calling
check() {
  local desc="$1" want_class="$2" want_rc="$3"
  classify_pod; local rc=$?
  if [ "$PC_CLASS" = "$want_class" ] && [ "$rc" = "$want_rc" ]; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    printf 'FAIL: %-46s got class=%s rc=%s  want class=%s rc=%s  [%s]\n' \
      "$desc" "$PC_CLASS" "$rc" "$want_class" "$want_rc" "$PC_MSG"
  fi
  # clear all inputs so cases never leak into each other
  unset PC_REACH PC_PROC PC_GPU PC_DONE PC_LAST PC_RESULT
}

# ── HAPPY PATH — a correct run MUST produce NO alert (RELIABILITY row 0) ──────
PC_REACH=ok PC_PROC=1 PC_GPU=98 PC_DONE=0 PC_LAST="native] c5 render.graft took 41s" \
  check "happy: rendering, gpu busy, log quiet-ish" OK 0
PC_REACH=ok PC_PROC=1 PC_GPU=95 PC_DONE=0 PC_RESULT="OK ref_ci [0.01,0.19] n 48" \
  check "happy: full-size interim result present" INTERIM 0
PC_REACH=ok PC_PROC=0 PC_GPU=0 PC_DONE=1 PC_RESULT="OK ref_ci [0.01,0.19] n 48" \
  check "happy: full run truly done (unique marker)" DONE 0
PC_REACH=booting check "happy: still booting within grace" BOOTING 0

# ── FAILURE MODES — each MUST fire (rc=1) with the right class ────────────────
# incident #35: expected pod never launched -> absence is an alarm, not silence
PC_REACH=missing check "#35 expected model never launched" MISSING 1
# endpoint-blind class: reachability could not be established -> fail closed
PC_REACH=garbage check "endpoint/reach unknown -> fail closed" UNKNOWN 1
PC_REACH=unreachable check "ssh refused" UNREACHABLE 1
# reachable but no signal at all -> must NOT be assumed healthy
PC_REACH=ok check "reachable but zero signal -> fail closed" UNKNOWN 1
# crash in log
PC_REACH=ok PC_PROC=1 PC_GPU=50 PC_LAST="RuntimeError: CUDA out of memory" \
  check "crash: RuntimeError in log" ERROR 1
PC_REACH=ok PC_PROC=0 PC_GPU=0 PC_LAST="FATAL: model load failed" \
  check "crash: FATAL model load failed" ERROR 1
# error-as-interim bug: result STATUS is authoritative even if log looks clean
PC_REACH=ok PC_PROC=1 PC_GPU=80 PC_LAST="native] c3 ok" PC_RESULT="ERROR ref_ci None n 0" \
  check "result status=ERROR beats clean log" ERROR 1
PC_REACH=ok PC_PROC=1 PC_GPU=80 PC_RESULT="UNSUPPORTED ref_ci None n 0" \
  check "result status=UNSUPPORTED is an error" ERROR 1
# #36: the probe's small n=6 result must be INTERIM (still running), NEVER DONE.
# (DONE requires the unique WIDE SWEEP DONE marker, which the probe never emits.)
PC_REACH=ok PC_PROC=1 PC_GPU=90 PC_DONE=0 PC_RESULT="OK ref_ci [-0.20,0.06] n 6" \
  check "#36 probe n=6 is INTERIM (still running), NOT DONE" INTERIM 0
# died: process gone, no terminal marker, no OK result
PC_REACH=ok PC_PROC=0 PC_GPU=0 PC_DONE=0 PC_LAST="native] c7 ..." \
  check "died: no proc, no marker, no result" DIED 1
# stall-false-positive bug: quiet log but GPU BUSY is a slow render, NOT a stall
PC_REACH=ok PC_PROC=1 PC_GPU=99 PC_DONE=0 PC_LAST="native] c2 (rendering, slow)" \
  check "quiet log + gpu busy = RENDERING not stall" OK 0
# idle-gpu: process alive but GPU idle -> genuine anomaly, must alarm
PC_REACH=ok PC_PROC=1 PC_GPU=0 PC_DONE=0 PC_LAST="native] c2" \
  check "process alive but gpu idle -> anomaly" IDLE-GPU 1

echo "monitor_selftest: $pass passed, $fail failed"
[ "$fail" -eq 0 ] || { echo "MONITOR NOT CLEARED FOR USE — classifier is wrong above."; exit 1; }
echo "MONITOR CLEARED: classifier fires on every failure mode and stays silent on the happy path."
