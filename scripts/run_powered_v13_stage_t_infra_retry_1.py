#!/usr/bin/env python3
"""Sole provider-allocation entry point for powered-v13 Stage-T retry 1."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from powered_v13_infra_retry_entrypoint import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
