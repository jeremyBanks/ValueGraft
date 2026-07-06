"""Post-hoc episode enrichment: joins score.json + agent.log + timestamps
into results/agent_clean_run/<name>.json (adds fields, never alters
tests_pass). Idempotent; run at analysis time.
Fields added: agent_events, wall_seconds, recall_probe (raw text),
log_bytes. Token/compaction traces stay in shim logs (joinable by time)."""
import json
import re
import sys
from pathlib import Path

S = Path("/private/tmp/claude-501/-Users-jeb-experimentation/"
         "bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad/e1_runs")
OUT = Path("results/agent_clean_run")

def enrich():
    n = 0
    for d in S.iterdir():
        sc = d / "score.json"
        if not (d.is_dir() and sc.exists()):
            continue
        data = json.load(open(sc))
        log = d / "agent.log"
        txt = log.read_text(errors="replace") if log.exists() else ""
        m = re.search(r"E1_AGENT_DONE events=(\d+)", txt)
        data["agent_events"] = int(m.group(1)) if m else None
        p = re.search(r"E1_RECALL_PROBE: (.*)", txt)
        data["recall_probe"] = p.group(1)[:300] if p else None
        tt = d / "task.txt"
        if tt.exists():
            data["wall_seconds"] = round(sc.stat().st_mtime - tt.stat().st_mtime)
        data["log_bytes"] = len(txt)
        data["source"] = "swebench" if d.name.startswith("swb") else "synthetic"
        json.dump(data, open(OUT / f"{d.name}.json", "w"), indent=1)
        n += 1
    print(f"enriched {n} rows")

if __name__ == "__main__":
    enrich()
