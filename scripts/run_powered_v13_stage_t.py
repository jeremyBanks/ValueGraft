#!/usr/bin/env python3
"""Run the closed powered-v13 Stage-T technical worker/supervisor."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from powered_v13_stage_t import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
