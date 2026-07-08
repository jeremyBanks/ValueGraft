#!/usr/bin/env bash
# PURE POD-STATE CLASSIFIER — the single source of truth for "what state is this pod in?".
#
# WHY THIS EXISTS: every monitor bug on this project (incidents #35, #36, and the
# endpoint-blind / stall-false-positive / done-on-probe / error-as-interim quartet)
# was a classifier built on an UNVERIFIED assumption, embedded inline in a monitor
# where it could never be fault-tested. This extracts the classification into ONE
# pure function with NO I/O (no ssh, no network) so it can be driven with fixtures by
# scripts/monitor_selftest.sh. A monitor is only trustworthy once this classifier has
# passed that self-test (happy-path = NO false alert, AND every failure mode fires).
#
# CONTRACT (fail-closed): call classify_pod, reading these variables:
#   PC_REACH  = missing | booting | unreachable | ok   (reachability, decided by caller)
#   PC_PROC   = number of live render processes (integer; "" => unknown)
#   PC_GPU    = gpu utilisation % (integer; "" => unknown)
#   PC_DONE   = count of the UNIQUE terminal marker "WIDE SWEEP DONE" (integer)
#   PC_LAST   = last interesting log line
#   PC_RESULT = newest result summary, e.g. "OK ref_ci [..] n 48" or "ERROR ref_ci None n 0"
# It sets two globals:
#   PC_CLASS  = one token: DONE OK INTERIM BOOTING MISSING UNREACHABLE ERROR DIED IDLE-GPU UNKNOWN
#   PC_MSG    = human-readable detail
# and returns:  0 = healthy/expected (DONE/OK/INTERIM/BOOTING), 1 = PROBLEM (alarm).
#
# The five INVARIANTS this encodes (each maps to a real incident):
#   1. UNKNOWN == ALARM. An expected pod we cannot account for (missing/unreachable/
#      unparseable signal) is a PROBLEM, never a benign line. (#35: absence was silent.)
#   2. DONE requires the UNIQUE terminal marker, never a signal the probe also emits.
#      (#36: "WROTE status=" fired on the 3-conv probe and was mislabelled the full run.)
#   3. RESULT STATUS is authoritative for error, not just log grep. A result whose
#      status is ERROR/UNSUPPORTED is ERROR even if the log looks clean. (error-as-interim bug)
#   4. STALL is GPU-AWARE. A quiet log with the GPU BUSY is RENDERING (slow render),
#      not a stall; a quiet log with the GPU IDLE is the anomaly. (stall-false-positive bug)
#   5. n IS ALWAYS SURFACED so probe(~6) vs full(~48) is unmistakable in every line. (#36)

# error signatures in a log line (crash / load failure)
_pc_log_is_error() { echo "$1" | grep -qiE "FATAL|RuntimeError|Traceback|OOM|model load failed|CUDA error"; }
# result-status is an error verdict (authoritative)
_pc_result_is_error() { echo "$1" | grep -qiE "^ERROR |^UNSUPPORTED |ERROR ref_ci|UNSUPPORTED ref_ci"; }
# result carries a scored number
_pc_result_scored() { echo "$1" | grep -q "ref_ci"; }

# PC_CLASS / PC_MSG are the function's OUT parameters, read by callers after the call.
# shellcheck disable=SC2034
classify_pod() {
  local reach="${PC_REACH:-}" proc="${PC_PROC:-}" gpu="${PC_GPU:-}"
  local done_n="${PC_DONE:-0}" last="${PC_LAST:-}" result="${PC_RESULT:-}"
  PC_CLASS="UNKNOWN"; PC_MSG=""

  # ---- reachability layer (invariant 1: unknown == alarm) ----
  case "$reach" in
    missing)     PC_CLASS="MISSING";     PC_MSG="expected model has no state file — never launched"; return 1;;
    booting)     PC_CLASS="BOOTING";     PC_MSG="no ssh endpoint yet (boot grace)"; return 0;;
    unreachable) PC_CLASS="UNREACHABLE"; PC_MSG="ssh refused/dropped"; return 1;;
    ok) : ;;
    *)           PC_CLASS="UNKNOWN";     PC_MSG="reachability not established (fail-closed)"; return 1;;
  esac

  # reachable but no parseable signal at all => fail closed, do NOT assume healthy
  if [ -z "$last" ] && [ -z "$result" ] && [ -z "$proc" ] && [ -z "$gpu" ]; then
    PC_CLASS="UNKNOWN"; PC_MSG="reachable but emitted no signal (fail-closed)"; return 1
  fi

  # ---- error layer (crash OR authoritative result status; invariant 3) ----
  if _pc_log_is_error "$last"; then
    PC_CLASS="ERROR"; PC_MSG="crash in log — $last"; return 1
  fi
  if _pc_result_is_error "$result"; then
    PC_CLASS="ERROR"; PC_MSG="result status is an error verdict — $result"; return 1
  fi

  # ---- terminal layer (invariant 2: unique marker only) ----
  if [ "${done_n:-0}" -gt 0 ] 2>/dev/null; then
    PC_CLASS="DONE"; PC_MSG="full run complete (WIDE SWEEP DONE) — $result"; return 0
  fi

  # ---- scored-but-still-running (invariant 5: always show n) ----
  if _pc_result_scored "$result"; then
    PC_CLASS="INTERIM"; PC_MSG="scored, still running (probe/partial) — $result"; return 0
  fi

  # ---- liveness layer ----
  if [ "${proc:-0}" = "0" ]; then
    PC_CLASS="DIED"; PC_MSG="no render process and no terminal marker — last:$last"; return 1
  fi

  # ---- progress layer (invariant 4: stall is GPU-aware) ----
  if [ -n "$gpu" ] && [ "${gpu:-0}" -lt 3 ] 2>/dev/null; then
    PC_CLASS="IDLE-GPU"; PC_MSG="process alive but gpu=${gpu}% (anomaly) — last:$last"; return 1
  fi
  PC_CLASS="OK"; PC_MSG="rendering gpu=${gpu}% — $last"; return 0
}
