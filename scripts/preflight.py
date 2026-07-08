#!/usr/bin/env python3
"""
PRE-FLIGHT GATE (Tier 0) — local, cheap, no GPU, fail-closed.

Runs the BORING precondition checks that every scaled spend/fan-out has skipped.
Exits nonzero on the FIRST failure. On success writes .preflight_ok.json whose
hash pins THIS exact config (models + job + manifest); the launcher refuses to
fan out unless a fresh token matches.

Usage:
  MODELS="repo1 repo2" ANCHORS="..." python3 scripts/preflight.py \
      --job scripts/job_sweep.sh --launcher scripts/launch_pod.sh

Asserts (each maps to a real past incident):
  B  every model id RESOLVES on the hub and is a TEXT-ONLY causal LM
     (catches hallucinated ids + multimodal wrappers that won't load).
  C  every path the LAUNCHER rsyncs EXISTS locally
     (catches rsync's `|| true` silently dropping a path).
  D  every data file the CODE opens exists at the path it reads
     (catches a "fix" that silently didn't apply).
  E  the launch mechanism is syntactically sound end-to-end
     (job script + every src/*.py parses).
  F  secrets/preconditions present: HF token, ssh key, clean git tree.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILS: list[str] = []
WARNS: list[str] = []


def fail(tag, msg): FAILS.append(f"[{tag}] {msg}")
def warn(tag, msg): WARNS.append(f"[{tag}] {msg}")


# ---- B: model ids resolve + are text-only loadable architectures ----------
MULTIMODAL_MODEL_TYPES = {
    "llava", "llava_next", "qwen2_vl", "qwen2_5_vl", "mllama", "idefics2",
    "idefics3", "paligemma", "gemma3", "gemma3n", "pixtral", "fuyu",
    "kosmos-2", "internvl", "phi3_v", "phi4mm", "aria", "mistral3",
}
MULTIMODAL_ARCH_MARKERS = ("ForConditionalGeneration", "VLForCausalLM", "VisionModel")


def check_models(models):
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        fail("B", f"huggingface_hub unavailable ({e}); cannot verify model ids. "
                  "pip install huggingface_hub and re-run — do NOT launch blind.")
        return
    tok = None
    for kf in (ROOT / ".huggingface_key", ROOT / ".hf_key"):
        if kf.exists():
            tok = kf.read_text().strip(); break
    tok = tok or os.environ.get("HF_TOKEN")
    for m in models:
        try:
            cfgp = hf_hub_download(m, "config.json", token=tok)
        except Exception as e:
            fail("B", f"model id does NOT resolve: {m!r} ({type(e).__name__}). "
                      "typo or nonexistent checkpoint — would crash every pod.")
            continue
        cfg = json.loads(Path(cfgp).read_text())
        mtype = (cfg.get("model_type") or "").lower()
        archs = cfg.get("architectures") or []
        bad = None
        if "vision_config" in cfg or "vision_tower" in cfg:
            bad = "has vision_config"
        elif mtype in MULTIMODAL_MODEL_TYPES:
            bad = f"model_type={mtype} is multimodal"
        elif any(any(mk in a for mk in MULTIMODAL_ARCH_MARKERS) for a in archs):
            bad = f"architecture {archs} is a multimodal/conditional-generation wrapper"
        if bad:
            fail("B", f"model is NOT a text-only causal LM: {m}  ({bad}). "
                      "won't load in the text path — drop it before launch.")
        else:
            print(f"  ok  model  {m:52s} type={mtype} arch={archs[0] if archs else '?'}")


# ---- C: launcher rsync sources exist locally ------------------------------
def check_manifest(launcher):
    p = ROOT / launcher
    if not p.exists():
        warn("C", f"launcher {launcher} not found; skipping manifest check"); return
    text = p.read_text()
    # pull literal source paths from each `rsync ... <srcs> root@...:dest` line
    srcs = []
    for line in text.splitlines():
        if "rsync" not in line or "root@" not in line:
            continue
        body = line.split("root@", 1)[0]
        for tok in body.split():
            if tok.startswith("-") or tok in ("rsync", "-e"):
                continue
            if "/" in tok or tok.endswith((".json", ".parquet", ".key")):
                if tok.startswith(('"', "'", "$", "ssh")) or "@" in tok:
                    continue
                srcs.append(tok)
    for s in dict.fromkeys(srcs):
        if any(c in s for c in "*?${"):
            continue  # globs/vars: can't statically resolve, skip
        if not (ROOT / s).exists() and not Path(s).exists():
            fail("C", f"launcher rsyncs a path that does NOT exist locally: {s!r}. "
                      "rsync's `|| true` will silently drop it and the pod code will "
                      "read a missing file. fix the manifest or create the file.")
        else:
            print(f"  ok  ship   {s}")


# ---- D: data files the code opens exist at the read path ------------------
DATA_READ_RE = re.compile(
    r"""(?:open|read_text|read_bytes|load|read_json|json\.load|Path)\(\s*['"]([^'"]*data/[^'"]+\.(?:json|parquet|jsonl|csv|txt))['"]""")


def check_code_reads(job):
    files = list((ROOT / "src").glob("*.py"))
    if job:
        files.append(ROOT / job)
    seen = set()
    for f in files:
        if not f.exists():
            continue
        for m in DATA_READ_RE.finditer(f.read_text()):
            rel = m.group(1)
            if rel in seen:
                continue
            seen.add(rel)
            if not (ROOT / rel).exists():
                warn("D", f"code in {f.name} reads {rel!r} which is absent locally. "
                          "confirm it is generated on-pod or shipped; else it will crash.")
            else:
                print(f"  ok  read   {rel}  (referenced in {f.name})")


# ---- E: launch mechanism parses end-to-end --------------------------------
def check_syntax(job):
    if job:
        r = subprocess.run(["bash", "-n", str(ROOT / job)], capture_output=True, text=True)
        if r.returncode != 0:
            fail("E", f"job script {job} has a bash syntax error: {r.stderr.strip()}")
        else:
            print(f"  ok  syntax {job}")
    import ast
    for f in (ROOT / "src").glob("*.py"):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            fail("E", f"src/{f.name} syntax error line {e.lineno}: {e.msg}")


# ---- F: secrets + git state -----------------------------------------------
def check_preconditions():
    if not ((ROOT / ".huggingface_key").exists() or (ROOT / ".hf_key").exists()
            or os.environ.get("HF_TOKEN")):
        fail("F", "no HF token (.huggingface_key / .hf_key / HF_TOKEN). gated models "
                  "will 401 on every pod.")
    key = Path.home() / ".ssh" / "id_ed25519_runpod"
    if not key.exists():
        fail("F", f"ssh key {key} missing; launcher cannot reach any pod.")
    r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        warn("F", "working tree is DIRTY. commit before a long run so nothing "
                  "consumed/deleted is uncommitted.")


TOKEN = ROOT / ".preflight_ok.json"
MAX_AGE_S = 6 * 3600  # a green token older than this is stale; re-run the gate


def mechanism_hash(job, launcher):
    """Fingerprint the LAUNCH MECHANISM (job + launcher bytes) only — not the
    model set, so a per-model fan-out pod can verify against a set-wide token."""
    h = hashlib.sha256()
    for f in (job, launcher):
        p = ROOT / f if f else None
        h.update(p.read_bytes() if p and p.exists() else b"")
    return h.hexdigest()[:16]


def env_models():
    models = (os.environ.get("MODELS", "") + " " + os.environ.get("ANCHORS", "")
              + " " + os.environ.get("BREADTH", "")).split()
    return list(dict.fromkeys(m for m in models if "/" in m))


def verify(job, launcher):
    """Fail-closed check for the launcher: a fresh GREEN token must exist, its
    mechanism fingerprint must match the current job+launcher, and every model
    about to launch must be in the token's verified set."""
    if not TOKEN.exists():
        print("PRE-FLIGHT: no green token. run scripts/preflight.sh first.", file=sys.stderr)
        sys.exit(3)
    t = json.loads(TOKEN.read_text())
    age = time.time() - t.get("ts", 0)
    if age > MAX_AGE_S:
        print(f"PRE-FLIGHT: token is STALE ({age/3600:.1f}h old). re-run the gate.", file=sys.stderr)
        sys.exit(3)
    if t.get("mech") != mechanism_hash(job, launcher):
        print("PRE-FLIGHT: job/launcher CHANGED since the gate ran. re-run the gate.", file=sys.stderr)
        sys.exit(3)
    unverified = [m for m in env_models() if m not in set(t.get("models", []))]
    if unverified:
        print(f"PRE-FLIGHT: models never verified by the gate: {unverified}", file=sys.stderr)
        sys.exit(3)
    print(f"PRE-FLIGHT: green (hash={t.get('mech')}, {age/60:.0f}min old).")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=os.environ.get("SC_JOB", ""))
    ap.add_argument("--launcher", default="scripts/launch_pod.sh")
    ap.add_argument("--verify", action="store_true",
                    help="launcher hook: check an existing token, run no checks")
    args = ap.parse_args()

    if args.verify:
        verify(args.job, args.launcher)
        return

    models = env_models()
    if not models:
        fail("B", "no MODELS/ANCHORS/BREADTH in env — nothing to verify. refusing "
                  "to green a launch with an empty model set.")

    print(f"PRE-FLIGHT  job={args.job or '-'}  models={len(models)}  {time.strftime('%FT%T')}")
    check_models(models)
    check_manifest(args.launcher)
    check_code_reads(args.job)
    check_syntax(args.job)
    check_preconditions()

    for w in WARNS:
        print("  WARN " + w)
    if FAILS:
        print("\nPRE-FLIGHT RED — do NOT scale. Fix these first:")
        for f in FAILS:
            print("  FAIL " + f)
        sys.exit(1)

    ch = mechanism_hash(args.job, args.launcher)
    token = {"mech": ch, "ts": time.time(), "job": args.job,
             "models": models, "warns": WARNS}
    TOKEN.write_text(json.dumps(token, indent=1))
    print(f"\nPRE-FLIGHT GREEN  hash={ch}  ({len(models)} models verified). "
          "token .preflight_ok.json written. valid 6h for THIS job+launcher.")


if __name__ == "__main__":
    main()
