#!/usr/bin/env bash
# shellcheck disable=SC2034  # PC_* are the classify_pod input contract, read indirectly via the sourced function
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

# ── GAP 1a (pod_health.sh:65 false-DONE): PROC=0 + a result merely containing "OK" but with
# NO unique WIDE SWEEP DONE marker MUST be INTERIM, never DONE. If pod_health's old inline
# `PROC=0 && grep OK => DONE` logic were live, this would (wrongly) be DONE.
PC_REACH=ok PC_PROC=0 PC_GPU=0 PC_DONE=0 PC_RESULT="OK ref_ci [0.01,0.19] n 6" \
  check "gap1a: proc=0 + OK result, no marker -> INTERIM (NOT DONE)" INTERIM 0
# ── GAP 1b (pod_health.sh:69 false-stall): stall is GPU-AWARE. A stale log with the GPU BUSY
# is a slow render (healthy); a stale log with the GPU IDLE is a genuine STALL (alarm).
PC_REACH=ok PC_PROC=1 PC_GPU=97 PC_DONE=0 PC_LOGAGE=1800 PC_LAST="native] c2 (slow render)" \
  check "gap1b: stale log + GPU busy -> RENDERING not stall" OK 0
PC_REACH=ok PC_PROC=1 PC_GPU=0 PC_DONE=0 PC_LOGAGE=1800 PC_LAST="native] c2" \
  check "gap1b: stale log + GPU idle -> STALLED (alarm)" STALLED 1

# ── GAP 2 (exp_watch under-fed the classifier): the classifier keys Traceback/OOM/CUDA-error
# as ERROR. These crash proc-alive; if the monitor grep dropped them (as exp_watch did) they
# never reached the classifier. These cases prove the classifier fires on each signature; the
# single-source assertions below prove the monitors actually feed them.
PC_REACH=ok PC_PROC=1 PC_GPU=70 PC_LAST="Traceback (most recent call last):" \
  check "gap2: Traceback in log -> ERROR" ERROR 1
PC_REACH=ok PC_PROC=1 PC_GPU=70 PC_LAST="torch.cuda.OutOfMemoryError: OOM" \
  check "gap2: OOM in log -> ERROR" ERROR 1
PC_REACH=ok PC_PROC=1 PC_GPU=70 PC_LAST="CUDA error: device-side assert triggered" \
  check "gap2: CUDA error in log -> ERROR" ERROR 1

# ── GAP 4 (unknown==alarm only covered TOTAL blackout): losing just ONE key signal while the
# pod is reachable (e.g. nvidia-smi failed -> GPU blank, proc still alive) MUST be a PROBLEM,
# not a healthy "rendering gpu=%". Before the fix this fell through to OK.
PC_REACH=ok PC_PROC=1 PC_GPU="" PC_DONE=0 PC_LAST="native] c2" \
  check "gap4: reachable, proc alive but GPU signal lost -> SIGNAL-LOSS" SIGNAL-LOSS 1
PC_REACH=ok PC_PROC="" PC_GPU=90 PC_DONE=0 PC_LAST="native] c2" \
  check "gap4: reachable, GPU up but proc signal lost -> SIGNAL-LOSS" SIGNAL-LOSS 1

echo "classifier-fixture cases: $pass passed, $fail failed"
[ "$fail" -eq 0 ] || { echo "MONITOR NOT CLEARED FOR USE — classifier is wrong above."; exit 1; }

# ══ ENDPOINT-DERIVATION + BOOT-ESCALATION (gaps 3 & 5) ════════════════════════════════════
# The code that PRODUCES the reach token used to be inline+untested in exp_watch (where the
# endpoint-blind bug hid). Exercise it directly: source exp_watch (its main loop is guarded so
# sourcing does NOT run it) to get classify_endpoint + resolve_reach, and drive them.
EXP_WATCH_STATE="$(mktemp -t exp_watch_boot.XXXXXX)"; export EXP_WATCH_STATE
BOOT_GRACE_CHECKS=2; export BOOT_GRACE_CHECKS
# shellcheck source=scripts/exp_watch.sh
. scripts/exp_watch.sh

epass=0; efail=0
expect_eq() {  # desc got want
  if [ "$2" = "$3" ]; then epass=$((epass+1)); else efail=$((efail+1));
    printf 'FAIL: %-52s got=%s want=%s\n' "$1" "$2" "$3"; fi
}

# GAP 5: pure endpoint derivation — valid -> ok, blank -> booting, malformed -> problem.
expect_eq "gap5: valid ip:port -> ok"      "$(classify_endpoint '203.0.113.5:22001')" ok
expect_eq "gap5: blank endpoint -> booting" "$(classify_endpoint '')"                  booting
expect_eq "gap5: ':' only (malformed) -> problem" "$(classify_endpoint ':')"           problem
expect_eq "gap5: malformed 'no-ssh' -> problem" "$(classify_endpoint 'no-ssh')"        problem
expect_eq "gap5: malformed 'pending...' -> problem" "$(classify_endpoint 'pending...')" problem

# GAP 5: a malformed endpoint routes (via resolve_reach) to a classifier PROBLEM, not a boot state.
rm -f "$EXP_WATCH_STATE"
r=$(resolve_reach "garbage-endpoint" "podM"); expect_eq "gap5: malformed reach token" "$r" malformed
PC_REACH="$r"; classify_pod; rc=$?; unset PC_REACH
expect_eq "gap5: malformed -> class"  "$PC_CLASS" MALFORMED-EP
expect_eq "gap5: malformed -> alarm"  "$rc"       1

# GAP 3: a blank endpoint is BOOTING within grace (no false alarm) but ESCALATES to a PROBLEM
# once it stays blind past the grace window — a never-resolving blind endpoint MUST alarm.
rm -f "$EXP_WATCH_STATE"
r1=$(resolve_reach "" "podB")   # check 1 (<= grace) -> booting
r2=$(resolve_reach "" "podB")   # check 2 (== grace) -> booting
r3=$(resolve_reach "" "podB")   # check 3 (>  grace) -> escalate
expect_eq "gap3: booting check1 (happy, no alarm)" "$r1" booting
expect_eq "gap3: booting check2 (happy, no alarm)" "$r2" booting
expect_eq "gap3: booting past grace escalates"     "$r3" boot-timeout
PC_REACH="$r1"; classify_pod; rc=$?; unset PC_REACH
expect_eq "gap3: within-grace booting is silent (rc0)" "$rc" 0
PC_REACH="$r3"; classify_pod; rc=$?; unset PC_REACH
expect_eq "gap3: escalated boot -> class"  "$PC_CLASS" BOOT-TIMEOUT
expect_eq "gap3: escalated boot -> alarm"  "$rc"       1

# GAP 3/5: a valid endpoint resolves to ok AND clears any accumulated boot count (so a pod
# that finally comes up does not carry stale boot state into an escalation).
r=$(resolve_reach "203.0.113.5:22001" "podB"); expect_eq "gap5: valid endpoint reach -> ok" "$r" ok
expect_eq "gap3: valid endpoint clears boot count" "$(_boot_get podB)" 0
rm -f "$EXP_WATCH_STATE"

# GAP 2: the monitors must FEED the classifier from its single-source error pattern, not a
# hand-copied subset. Assert both live monitors reference PC_ERROR_SIGNATURES in their remote
# grep (a reverted narrow grep would drop Traceback/OOM/CUDA-error and fail here).
# NB: match the INTERPOLATION form ${PC_ERROR_SIGNATURES} in the grep-pattern line, not the
# bare word (which also appears in comments) — a narrowed grep would drop the interpolation.
if grep -qF '${PC_ERROR_SIGNATURES}' scripts/exp_watch.sh; then g=ok; else g=narrowed; fi
expect_eq "gap2: exp_watch grep interpolates PC_ERROR_SIGNATURES" "$g" ok
if grep -qF '${PC_ERROR_SIGNATURES}' scripts/pod_health.sh; then g=ok; else g=narrowed; fi
expect_eq "gap2: pod_health grep interpolates PC_ERROR_SIGNATURES" "$g" ok

echo "endpoint/escalation cases: $epass passed, $efail failed"

echo "monitor_selftest: $((pass)) classifier + $epass endpoint cases passed, $((fail+efail)) failed"
[ "$((fail+efail))" -eq 0 ] || { echo "MONITOR NOT CLEARED FOR USE — see failures above."; exit 1; }
echo "MONITOR CLEARED: classifier fires on every failure mode and stays silent on the happy path."
