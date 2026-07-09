---
name: verify-kills-by-pid
description: "After killing background processes, verify by listing PIDs — pkill -f patterns silently miss argv variations (e.g. \"-u\" flags)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

`pkill -f "python3 src/script"` does NOT match `/path/python3 -u src/script.py` — the `-u` breaks the substring. A missed kill left two 30B model instances thrashing 37GB of swap for 3 hours.

**Why:** pkill -f is substring matching on the full argv; any flag between tokens defeats it.

**How to apply:** after any kill/pkill, immediately `pgrep -fl <loose-pattern>` with the loosest possible pattern (just the script name) and assert empty before relaunching. Prefer killing exact PIDs collected via pgrep first.
