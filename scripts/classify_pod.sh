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
#   PC_REACH   = missing | booting | boot-timeout | malformed | unreachable | ok
#                (reachability, decided by caller — see classify_endpoint below)
#   PC_PROC    = number of live render processes (integer; "" => unknown / signal lost)
#   PC_GPU     = gpu utilisation % (integer; "" => unknown / signal lost)
#   PC_DONE    = count of the UNIQUE terminal marker "WIDE SWEEP DONE" (integer)
#   PC_LAST    = last interesting log line
#   PC_RESULT  = newest result summary, e.g. "OK ref_ci [..] n 48" or "ERROR ref_ci None n 0"
#   PC_LOGAGE  = seconds since job.log last advanced (integer; "" => unknown). Used ONLY
#                with an idle GPU to distinguish a genuine STALL from a live slow render.
#   PC_STALL_SECS = stall threshold in seconds (default 480 = 8min).
# It sets two globals:
#   PC_CLASS  = one token: DONE OK INTERIM BOOTING MISSING BOOT-TIMEOUT MALFORMED-EP
#               UNREACHABLE ERROR DIED IDLE-GPU STALLED SIGNAL-LOSS UNKNOWN
#   PC_MSG    = human-readable detail
# and returns:  0 = healthy/expected (DONE/OK/INTERIM/BOOTING), 1 = PROBLEM (alarm).
#
# classify_endpoint (below) is the PURE endpoint->reach derivation the monitors use, extracted
# here so it too is fault-testable (it is where the endpoint-blind bug hid).
#
# The INVARIANTS this encodes (each maps to a real incident/gap):
#   1. UNKNOWN == ALARM. An expected pod we cannot account for (missing/unreachable/
#      unparseable signal) is a PROBLEM, never a benign line. (#35: absence was silent.)
#   2. DONE requires the UNIQUE terminal marker, never a signal the probe also emits.
#      (#36: "WROTE status=" fired on the 3-conv probe and was mislabelled the full run.)
#   3. RESULT STATUS is authoritative for error, not just log grep. A result whose
#      status is ERROR/UNSUPPORTED is ERROR even if the log looks clean. (error-as-interim bug)
#   4. STALL is GPU-AWARE. A quiet log with the GPU BUSY is RENDERING (slow render),
#      not a stall; a quiet log with the GPU IDLE is the anomaly. (stall-false-positive bug)
#   5. n IS ALWAYS SURFACED so probe(~6) vs full(~48) is unmistakable in every line. (#36)
#   6. PARTIAL SIGNAL-LOSS == ALARM. Reachable but a KEY signal (proc or gpu) is unreadable is
#      a PROBLEM, never a healthy "rendering gpu=%" fall-through. (gap 4: losing just the GPU
#      signal was read as healthy.) A never-resolving BOOTING endpoint likewise escalates to a
#      PROBLEM rather than staying silent (gap 3, via the caller's boot-persistence).

# SINGLE SOURCE OF TRUTH for crash/load-failure log signatures. Every monitor that greps
# a remote log for errors MUST feed the classifier from THIS pattern (not a hand-copied
# subset) — a monitor grep narrower than this is how Traceback/OOM/CUDA-error crashes never
# reached the classifier (gap 2). exp_watch.sh / pod_health.sh reference PC_ERROR_SIGNATURES.
# shellcheck disable=SC2034
PC_ERROR_SIGNATURES="FATAL|RuntimeError|Traceback|OOM|model load failed|CUDA error"
# error signatures in a log line (crash / load failure)
_pc_log_is_error() { echo "$1" | grep -qiE "$PC_ERROR_SIGNATURES"; }

# PURE endpoint -> reachability-token derivation (gap 5). The code that PRODUCES the reach
# token used to live inline+untested in exp_watch (where the endpoint-blind bug hid). It is
# a pure string function so monitor_selftest can exercise it:
#   valid "ip:port"          -> "ok"       (a real target was resolved)
#   "" / all-whitespace      -> "booting"  (no endpoint yet; caller persists+escalates)
#   anything else (garbage)  -> "problem"  (an endpoint present but malformed = a fault NOW)
classify_endpoint() {
  local ep="$1"
  local stripped="${ep//[[:space:]]/}"
  if [ -z "$stripped" ]; then echo booting; return 0; fi
  if echo "$ep" | grep -qE "^[0-9]+(\.[0-9]+){3}:[0-9]+$"; then echo ok; return 0; fi
  echo problem; return 0
}
# result-status is an error verdict (authoritative)
_pc_result_is_error() { echo "$1" | grep -qiE "^ERROR |^UNSUPPORTED |ERROR ref_ci|UNSUPPORTED ref_ci"; }
# result carries a scored number
_pc_result_scored() { echo "$1" | grep -q "ref_ci"; }

# PC_CLASS / PC_MSG are the function's OUT parameters, read by callers after the call.
# shellcheck disable=SC2034
classify_pod() {
  local reach="${PC_REACH:-}" proc="${PC_PROC:-}" gpu="${PC_GPU:-}"
  local done_n="${PC_DONE:-0}" last="${PC_LAST:-}" result="${PC_RESULT:-}"
  local logage="${PC_LOGAGE:-}" stall_secs="${PC_STALL_SECS:-480}"
  PC_CLASS="UNKNOWN"; PC_MSG=""

  # ---- reachability layer (invariant 1: unknown == alarm) ----
  case "$reach" in
    missing)      PC_CLASS="MISSING";      PC_MSG="expected model has no state file — never launched"; return 1;;
    booting)      PC_CLASS="BOOTING";      PC_MSG="no ssh endpoint yet (boot grace)"; return 0;;
    # gap 3: a pod that stays BOOTING past the grace window is no longer benign — a blind /
    # never-resolving endpoint MUST alarm rather than sit SILENT forever.
    boot-timeout) PC_CLASS="BOOT-TIMEOUT"; PC_MSG="stuck BOOTING past grace window — endpoint never resolved (escalated)"; return 1;;
    # gap 5: an endpoint that is present but malformed is a fault now, not a boot state.
    malformed)    PC_CLASS="MALFORMED-EP"; PC_MSG="endpoint present but malformed — cannot reach (fail-closed)"; return 1;;
    unreachable)  PC_CLASS="UNREACHABLE";  PC_MSG="ssh refused/dropped"; return 1;;
    ok) : ;;
    *)            PC_CLASS="UNKNOWN";      PC_MSG="reachability not established (fail-closed)"; return 1;;
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

  # ---- signal-loss layer (gap 4: a MISSING KEY signal while reachable is a PROBLEM) ----
  # proc and gpu are the KEY liveness/progress signals. If the pod is reachable but we could
  # not read one of them, we CANNOT certify health — losing just the GPU signal (proc alive)
  # must NOT fall through to a healthy "rendering gpu=%" line. Fail closed. (Total blackout —
  # all four signals empty — is caught above as UNKNOWN; this catches a PARTIAL loss.)
  if [ -z "$proc" ] || [ -z "$gpu" ]; then
    PC_CLASS="SIGNAL-LOSS"; PC_MSG="reachable but missing a key signal (proc='${proc}' gpu='${gpu}') — fail-closed"; return 1
  fi

  # ---- liveness layer ----
  if [ "${proc:-0}" = "0" ]; then
    PC_CLASS="DIED"; PC_MSG="no render process and no terminal marker — last:$last"; return 1
  fi

  # ---- progress layer (invariant 4: stall is GPU-aware) ----
  # BUSY gpu => rendering, regardless of how quiet the log is (a slow render is healthy).
  # IDLE gpu => anomaly; if the log is ALSO stale beyond the threshold it is a genuine
  # STALL, otherwise it is the idle-GPU anomaly. (pod_health.sh:69 declared STALLED off
  # log-age ALONE, before/independent of the GPU check — the false-stall class.)
  if [ "${gpu:-0}" -lt 3 ] 2>/dev/null; then
    if [ -n "$logage" ] && [ "${logage:-0}" -gt "${stall_secs:-480}" ] 2>/dev/null; then
      PC_CLASS="STALLED"; PC_MSG="gpu idle ${gpu}% AND log stale ${logage}s (>${stall_secs}s) — last:$last"; return 1
    fi
    PC_CLASS="IDLE-GPU"; PC_MSG="process alive but gpu=${gpu}% (anomaly) — last:$last"; return 1
  fi
  PC_CLASS="OK"; PC_MSG="rendering gpu=${gpu}% — $last"; return 0
}
