#!/usr/bin/env bash
# shellcheck disable=SC2034  # PC_* are the input contract consumed by the sourced classify_pod
# Watch the EXPECTED set of exploration pods. Endpoints via STATE FILES + pod.py status
# (reliable). ABSENCE IS AN ALARM: an expected model with no state file / no endpoint / no
# process is a problem state, not a benign line (incident #35). Emits per-pod lines + a SUMMARY
# the monitor alarms on. Reads the referent CI when a result is written.
#
# THREE things this monitor must NOT do (each a past bug):
#  - classify inline: it SOURCEs the fault-tested scripts/classify_pod.sh. (Do NOT re-implement
#    state logic here — that is how every monitor bug was born.)
#  - under-feed the classifier: the remote error grep is built from PC_ERROR_SIGNATURES (the
#    classifier's own single-source pattern), so every crash signature the classifier keys on
#    (Traceback / OOM / CUDA error / …) actually reaches it (gap 2).
#  - let a blind endpoint sit SILENT forever: BOOTING is PERSISTED across checks and a pod
#    stuck BOOTING past the grace window ESCALATES to a PROBLEM/alarm (gap 3).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/classify_pod.sh
. scripts/classify_pod.sh
K=$HOME/.ssh/id_ed25519_runpod
EXPECTED="${EXP_MODELS:-canary expm expo}"
# gap 3: a pod BOOTING for more than this many consecutive checks escalates to a PROBLEM.
BOOT_GRACE_CHECKS="${BOOT_GRACE_CHECKS:-2}"
# small state file: one "<pod> <consecutive-boot-count>" line per still-booting pod.
EXP_WATCH_STATE="${EXP_WATCH_STATE:-$HOME/.exp_watch_boot_state}"

# ── boot-persistence helpers (gap 3): the ONLY stateful part; small, bounded (≤ one line
# per expected pod), self-evicting (cleared the moment a pod stops booting). ─────────────
_boot_get() {  # $1=pod -> current consecutive-boot count (0 if none recorded)
  [ -f "$EXP_WATCH_STATE" ] || { echo 0; return; }
  awk -v k="$1" '$1==k{print $2; f=1} END{if(!f) print 0}' "$EXP_WATCH_STATE"
}
_boot_clear() {  # $1=pod -> drop its line (called when a pod resolves or faults)
  [ -f "$EXP_WATCH_STATE" ] || return 0
  local tmp="${EXP_WATCH_STATE}.tmp.$$"
  grep -vE "^$1 " "$EXP_WATCH_STATE" > "$tmp" 2>/dev/null || true
  mv "$tmp" "$EXP_WATCH_STATE"
}
_boot_incr() {  # $1=pod -> increment + persist + echo new count
  local new; new=$(( $(_boot_get "$1") + 1 ))
  _boot_clear "$1"
  echo "$1 $new" >> "$EXP_WATCH_STATE"
  echo "$new"
}

# resolve_reach: given the resolved endpoint string + pod name, produce the reachability
# token the classifier consumes — applying the boot-persistence escalation. Pure derivation
# (classify_endpoint) + the small state I/O; exercised end-to-end by monitor_selftest.
resolve_reach() {  # $1=endpoint  $2=pod -> echoes: ok | booting | boot-timeout | malformed
  case "$(classify_endpoint "$1")" in
    ok)      _boot_clear "$2"; echo ok;;
    problem) _boot_clear "$2"; echo malformed;;
    booting)
      local c; c=$(_boot_incr "$2")
      if [ "$c" -gt "$BOOT_GRACE_CHECKS" ]; then echo boot-timeout; else echo booting; fi;;
  esac
}

main() {
  local n_ok=0 n_done=0 n_boot=0 n_problem=0 n_exp
  # feed the remote log grep from the classifier's OWN error pattern (gap 2) + progress markers.
  local LOG_PATTERN="native] c[0-9]|PROBE|render.graft took|WROTE .*status=|${PC_ERROR_SIGNATURES}"
  local n sf ep reach rc t OUT
  for n in $EXPECTED; do
    sf=".pod_${n}_state.json"
    # decide REACHABILITY only here; the classifier decides everything else.
    if [ ! -f "$sf" ]; then
      PC_REACH=missing; classify_pod || true; echo "$n: $PC_CLASS ($PC_MSG)"; n_problem=$((n_problem+1)); continue
    fi
    ep=$(SC_POD_STATE="$sf" uv run python src/pod.py status 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('publicIp') or '')+':'+str((d.get('portMappings') or {}).get('22','')))" 2>/dev/null)
    reach=$(resolve_reach "$ep" "$n")
    if [ "$reach" != "ok" ]; then
      PC_REACH="$reach"; classify_pod; rc=$?
      echo "$n ($ep): $PC_CLASS ($PC_MSG)"
      if [ "$rc" -ne 0 ]; then n_problem=$((n_problem+1)); else n_boot=$((n_boot+1)); fi
      unset PC_REACH; continue
    fi
    ip=${ep%:*}; port=${ep#*:}
    OUT=$(ssh -n -i "$K" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=8 -o ServerAliveCountMax=3 root@"$ip" 'cd /workspace/exp 2>/dev/null || exit 7
      P=$(pgrep -f cross_arch_probe | grep -v pgrep | wc -l | tr -d " ")
      G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
      T=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
      A=$(( $(date +%s) - $(stat -c %Y job.log 2>/dev/null || echo 0) ))
      L=$(tr "\r" "\n" < job.log 2>/dev/null | grep -aiE "'"$LOG_PATTERN"'" | tail -1)
      D=$(grep -c "WIDE SWEEP DONE" job.log 2>/dev/null || echo 0)
      R=""
      for f in results/cross_arch/*.json; do [ -f "$f" ] && [ "$(basename "$f"|cut -c1)" != "_" ] && R=$(python3 -c "import json;d=json.load(open(\"$f\"));b=d.get(\"by_category_robust\",{}) or {};r=(b.get(\"referent\",{}) or {});print(d.get(\"status\"),\"ref_ci\",r.get(\"raw_EB_ci\"),\"n\",r.get(\"n\"))" 2>/dev/null); done
      printf "P=%s|G=%s|T=%s|A=%s|D=%s|L=%s|R=%s\n" "$P" "$G" "$T" "$A" "$D" "$L" "$R"' 2>/dev/null)
    if [ -z "$OUT" ]; then
      PC_REACH=unreachable; classify_pod || true; echo "$n ($ep): $PC_CLASS ($PC_MSG)"; n_problem=$((n_problem+1)); continue
    fi
    # parse the remote signal into the classifier's contract, then classify.
    PC_REACH=ok
    PC_PROC=$(echo "$OUT" | sed -n 's/.*P=\([0-9]*\).*/\1/p')
    PC_GPU=$(echo "$OUT" | sed -n 's/.*G=\([0-9]*\).*/\1/p')
    PC_LOGAGE=$(echo "$OUT" | sed -n 's/.*A=\([0-9]*\).*/\1/p')
    PC_DONE=$(echo "$OUT" | sed -n 's/.*D=\([0-9]*\).*/\1/p')
    PC_LAST=$(echo "$OUT" | sed -n 's/.*L=\([^|]*\).*/\1/p')
    PC_RESULT=$(echo "$OUT" | sed -n 's/.*R=//p')
    t=$(echo "$OUT" | sed -n 's/.*T=\([^|]*\).*/\1/p')
    classify_pod; rc=$?
    echo "$n: $PC_CLASS tv=${t} — $PC_MSG"
    if [ "$rc" -ne 0 ]; then n_problem=$((n_problem+1))
    elif [ "$PC_CLASS" = "DONE" ]; then n_done=$((n_done+1))
    else n_ok=$((n_ok+1)); fi
    unset PC_REACH PC_PROC PC_GPU PC_LOGAGE PC_DONE PC_LAST PC_RESULT
  done
  n_exp=$(echo "$EXPECTED" | wc -w | tr -d " ")
  echo "SUMMARY: expected=$n_exp healthy=$n_ok done=$n_done booting=$n_boot problem=$n_problem"
  [ "$n_problem" -gt 0 ] && echo "ALERT: $n_problem expected pod(s) in a PROBLEM state (see lines above)."
  return 0
}

# Run the watch loop only when executed directly; when SOURCED (monitor_selftest) expose the
# reach-derivation + boot-persistence functions for fault-injection without running the loop.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main
  exit 0
fi
