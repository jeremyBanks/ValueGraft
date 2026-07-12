#!/usr/bin/env python3
"""Create one frozen-style pod in an explicit RunPod data center.

External orchestration only. This imports the tracked provider client so image,
disk, cloud, CUDA-filter, SSH-key, timeout, and response validation stay the
same, then adds the provider-supported dataCenterIds placement constraint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pod  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="NVIDIA A100 80GB PCIe")
    parser.add_argument("--data-center", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    body = pod.create_body(args.gpu)
    body["dataCenterIds"] = [args.data_center]
    body["dataCenterPriority"] = "custom"
    if args.dry_run:
        print(json.dumps(body, indent=2, sort_keys=True))
        return

    if pod.STATE.exists():
        raise SystemExit(f"refusing to reuse state path: {pod.STATE}")
    pod.ensure_ssh_key()
    try:
        created = pod.api("POST", "/pods", body)
    except urllib.error.HTTPError as exc:
        if exc.code == 500:
            raise SystemExit(85) from exc
        raise
    if not isinstance(created, dict) or not isinstance(created.get("id"), str):
        raise SystemExit("RunPod create response lacks a pod ID")
    # Persist every allocated pod before any secondary attestation can fail, so
    # an unexpected provider response remains terminable by the tracked client.
    pod.STATE.write_text(json.dumps(created, indent=1) + "\n")
    machine = created.get("machine")
    if not isinstance(machine, dict) or machine.get("dataCenterId") != args.data_center:
        raise SystemExit(
            "RunPod response does not attest the requested data center; "
            f"allocated pod {created['id']} is preserved in {pod.STATE} and "
            "must be terminated before retry")
    print(json.dumps({
        "id": created["id"],
        "name": created.get("name"),
        "machineId": created.get("machineId"),
        "dataCenterId": machine.get("dataCenterId"),
        "gpuTypeId": machine.get("gpuTypeId"),
        "secureCloud": machine.get("secureCloud"),
        "costPerHr": created.get("costPerHr"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
