#!/usr/bin/env python3
"""Fail-closed local validation before a coherent-state pod may terminate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(root: Path, mode: str) -> dict:
    log = root / "job.log"
    if not log.exists() or not log.stat().st_size:
        raise ValueError("job log missing or empty")
    for path in root.glob("*.json"):
        json.loads(path.read_text())
    out = {"mode": mode, "json_files": len(list(root.glob("*.json")))}
    if mode == "complete":
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("status") != "COMPLETE":
            raise ValueError("complete harvest lacks COMPLETE manifest")
        scored = [json.loads(p.read_text()) for p in root.glob("conv_*.json")]
        scored = [x for x in scored if x.get("status") == "scored"]
        if len(scored) not in (6, 12):
            raise ValueError(f"complete harvest has invalid N={len(scored)}")
        out["n_scored"] = len(scored)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("mode", choices=("complete", "failure"))
    args = ap.parse_args()
    print(json.dumps(validate(args.root, args.mode), sort_keys=True))


if __name__ == "__main__":
    main()
