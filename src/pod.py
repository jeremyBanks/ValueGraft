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
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import urllib.request

KEY = Path(".runpod_key").read_text().strip()
REST = "https://rest.runpod.io/v1"
import os
STATE = Path(os.environ.get("SC_POD_STATE", ".pod_state.json"))
DEFAULT_GPU = "NVIDIA A100 80GB PCIe"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519_runpod"


def api(method, path, body=None):
    req = urllib.request.Request(
        REST + path, method=method,
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.4"},
        data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        print("API ERROR", e.code, e.read().decode()[:500])
        raise


def gql(query):
    req = urllib.request.Request(
        "https://api.runpod.io/graphql", method="POST",
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.4"},
        data=json.dumps({"query": query}).encode())
    with urllib.request.urlopen(req) as r:
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
    ensure_ssh_key()
    body = {
        "name": "semantic-continuity",
        "imageName": IMAGE,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "cloudType": os.environ.get("SC_POD_CLOUD", "SECURE"),
        "containerDiskInGb": 200,
        "volumeInGb": 0,
        "supportPublicIp": True,
        "ports": ["22/tcp"],
    }
    pod = api("POST", "/pods", body)
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
    balance()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "balance":
        balance()
    elif cmd == "create":
        create(*sys.argv[2:])
    elif cmd == "status":
        status()
    elif cmd == "ssh-cmd":
        ssh_cmd()
    elif cmd == "terminate":
        terminate()
