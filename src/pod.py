"""RunPod lifecycle tool: create / status / ssh-cmd / terminate / balance.

usage:
  pod.py balance
  pod.py create [gpu_type_id]     (default: NVIDIA A100 80GB PCIe, secure)
  pod.py status
  pod.py ssh-cmd                  (prints ssh command line)
  pod.py terminate

State in .pod_state.json (podId). Reads .runpod_key. Registers
~/.ssh/id_ed25519_runpod.pub via the account settings if needed (we pass the
public key through the pod env instead — RunPod injects account SSH keys
automatically for pods with SSH enabled; we also set it explicitly).

Set SC_POD_ALLOWED_CUDA to a comma-separated RunPod CUDA capability filter
(for example ``13.0``).  The provider applies this before choosing a host;
launch_pod.sh separately verifies the actual driver after allocation.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import urllib.error
import urllib.request

KEY_PATH = Path(".runpod_key")
REST = "https://rest.runpod.io/v1"
import os
STATE = Path(os.environ.get("SC_POD_STATE", ".pod_state.json"))
DEFAULT_GPU = "NVIDIA A100 80GB PCIe"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519_runpod"
RUNPOD_CUDA_VERSIONS = {
    "13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4",
    "12.3", "12.2", "12.1", "12.0", "11.8",
}


def api_timeout_seconds() -> float:
    raw = os.environ.get("SC_POD_API_TIMEOUT_S", "30")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid SC_POD_API_TIMEOUT_S: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid SC_POD_API_TIMEOUT_S: {raw!r}")
    return value


def api_key():
    return KEY_PATH.read_text().strip()


def requested_cuda_versions():
    raw = os.environ.get("SC_POD_ALLOWED_CUDA", "").strip()
    if not raw:
        return []
    versions = list(dict.fromkeys(part.strip() for part in raw.split(",")
                                  if part.strip()))
    invalid = [version for version in versions
               if version not in RUNPOD_CUDA_VERSIONS]
    if not versions or invalid:
        raise ValueError(
            f"invalid SC_POD_ALLOWED_CUDA values: {invalid or raw!r}")
    return versions


def create_body(gpu):
    body = {
        "name": os.environ.get("SC_POD_NAME", "semantic-continuity"),
        "imageName": IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "cloudType": os.environ.get("SC_POD_CLOUD", "SECURE"),
        "containerDiskInGb": int(os.environ.get("SC_POD_DISK", "200")),
        "volumeInGb": 0,
        "supportPublicIp": True,
        "ports": ["22/tcp"],
    }
    cuda_versions = requested_cuda_versions()
    if cuda_versions:
        body["allowedCudaVersions"] = cuda_versions
    if os.environ.get("SC_POD_SPOT") == "1":
        body["interruptible"] = True
        body["bidPerGpu"] = float(os.environ.get("SC_POD_BID", "1.0"))
    return body


def api(method, path, body=None):
    req = urllib.request.Request(
        REST + path, method=method,
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.4"},
        data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=api_timeout_seconds()) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        print("API ERROR", e.code, e.read().decode()[:500])
        raise


def gql(query):
    req = urllib.request.Request(
        "https://api.runpod.io/graphql", method="POST",
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.4"},
        data=json.dumps({"query": query}).encode())
    with urllib.request.urlopen(req, timeout=api_timeout_seconds()) as r:
        return json.loads(r.read())


def balance():
    d = gql("query { myself { clientBalance spendLimit } }")["data"]["myself"]
    print(f"balance ${d['clientBalance']}, spend limit ${d['spendLimit']}")
    return d["clientBalance"]


def ensure_ssh_key():
    if not SSH_KEY.exists():
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f",
                        str(SSH_KEY), "-C", "runpod-experiment"], check=True)
    pub = (SSH_KEY.with_suffix(".pub")).read_text().strip()
    # add to account (idempotent-ish: RunPod stores a newline list)
    cur = gql('query { myself { pubKey } }')["data"]["myself"].get("pubKey") or ""
    if pub not in cur:
        merged = (cur.strip() + "\n" + pub).strip().replace('"', '\\"')
        gql(f'mutation {{ updateUserSettings(input: {{ pubKey: "{merged}" }}) {{ id }} }}')
        print("registered ssh key on account")
    return pub


def create(gpu=DEFAULT_GPU):
    body = create_body(gpu)
    ensure_ssh_key()
    print("requesting pod", gpu, "cloud", body["cloudType"],
          "allowedCudaVersions", body.get("allowedCudaVersions", "ANY"))
    pod = api("POST", "/pods", body)
    if not isinstance(pod, dict) or not isinstance(pod.get("id"), str) or \
            not pod["id"].strip():
        raise RuntimeError("RunPod create response did not contain a pod ID")
    STATE.write_text(json.dumps(pod, indent=1))
    print("created pod", pod.get("id"))
    return pod


def status():
    pid = json.loads(STATE.read_text())["id"]
    pod = api("GET", f"/pods/{pid}")
    print(json.dumps({k: pod.get(k) for k in
                      ("id", "desiredStatus", "lastStatusChange",
                       "publicIp", "portMappings", "costPerHr")}, indent=1))
    return pod


def ssh_cmd():
    pod = status()
    ip = pod.get("publicIp")
    ports = pod.get("portMappings") or {}
    port = ports.get("22")
    print(f"ssh -i {SSH_KEY} -p {port} -o StrictHostKeyChecking=accept-new root@{ip}")


def terminate():
    pid = json.loads(STATE.read_text())["id"]
    api("DELETE", f"/pods/{pid}")
    print("terminated", pid)
    # Deletion is the lifecycle operation. A subsequent accounting outage must
    # not make callers believe deletion itself failed and allocate another pod.
    try:
        balance()
    except Exception as exc:
        print(f"WARNING: pod deletion succeeded but balance lookup failed: {exc}",
              file=sys.stderr)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "balance":
        balance()
    elif cmd == "create":
        try:
            create(*sys.argv[2:])
        except urllib.error.HTTPError as exc:
            # RunPod documents capacity through the create response poorly; in
            # practice an HTTP 500 is its observed no-allocation response.  It
            # is safe for the bounded wrapper to retry a received HTTP 500.
            # Transport failures remain allocation-ambiguous and are not
            # converted to this status.
            if exc.code == 500:
                raise SystemExit(85) from exc
            raise
    elif cmd == "status":
        status()
    elif cmd == "ssh-cmd":
        ssh_cmd()
    elif cmd == "terminate":
        terminate()
