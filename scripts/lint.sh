#!/usr/bin/env bash
# MANDATORY: shellcheck every shell script. Run before committing any *.sh.
# Exits nonzero if any script has a warning-or-worse finding.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
fail=0
for f in scripts/*.sh; do
  out=$(shellcheck -s bash --severity=warning "$f" 2>&1)
  if [ -n "$out" ]; then echo "⚠ $f"; echo "$out" | grep -E "SC[0-9]+" | head -6; fail=1; fi
done
[ $fail -eq 0 ] && echo "shellcheck: all scripts clean" || echo "shellcheck: FIX THE ABOVE before committing"
exit $fail
