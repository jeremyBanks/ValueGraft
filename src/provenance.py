"""Provenance manifest standard + stamping helpers (the anti-"wrong-model" tax).

WHY THIS EXISTS (07-09): agents repeatedly reported "we have data for X" when the
data was the wrong model / dtype / intervention / metric / condition, because our
result files self-documented almost nothing and provenance was INFERRED from
directory names (e.g. dtype read off ``results/swegym_30b_bf16`` rather than from
the loaded weights). This module makes every controlled variable RECORDED FROM THE
RUNTIME and carried inside every result JSON + a per-run ``manifest.json``.

THE STANDARD (every result JSON must carry a ``_manifest`` block; every run must
also drop a run-level ``manifest.json``). Fields, and the RULE that each is READ,
not inferred:

  model.repo_id            model.name_or_path / config._name_or_path (the id the
                           weights were actually loaded from -- runtime, not env).
  model.resolved_revision  config._commit_hash -- the HF hub commit transformers
                           resolved the weights to (None if loaded from a local
                           path / not recorded by transformers).
  load.dtype               str(next(model.parameters()).dtype) -- READ FROM A REAL
                           PARAMETER TENSOR, never a dir name or SC_LOAD_DTYPE env.
  load.dtype_env           the SC_LOAD_DTYPE string that was REQUESTED -- kept only
                           so a mismatch (requested vs realized) is auditable.
  load.quantization        model.config.quantization_config as a dict (None = not
                           quantized), read from the loaded model's config.
  load.kv_geometry         num_key_value_heads / head_dim / n_layers from config.
  intervention.arm         arm name (e.g. "E-tuned", "E-champion", "B", "placebo").
  intervention.graft_type  "value" (keys neutral, alpha_K=0) for every graft arm.
  intervention.alpha       the SCALAR alpha (None when a champion map is used).
  intervention.champion_config_path / .champion_config_sha256 / .champion_label
                           for a tuned per-layer/per-head champion (content hash of
                           the exact config file -> the tuned map is pinned, not
                           just named).
  intervention.alignment   positions/alignment mode string.
  metric.definition        EXACT metric string, and whether it is a PROXY. For
                           SWE-Gym: "teacher-forced mean logprob of the true next
                           assistant action (PROXY; NOT resolve/test-pass rate)".
  metric.is_proxy          bool -- True whenever the metric is a stand-in for the
                           quantity a reader would assume from the corpus name.
  condition.summary_kind   "brief" (handicapped mechanism-isolation) |
                           "prod" (production-faithful) | "fixed_external" | ...
  condition.summary_request_sha256   hash of the EXACT summary-request text used.
  condition.summary_source            self-gen | fixed-external | per-model.
  corpus.name / .split / .n / .instance_ids   what was scored, and which items.
  code.git_commit / .git_dirty        repo commit + dirty flag (subprocess git).
  run.timestamp_utc / .gpu_name / .hostname / .transformers_version / .torch_version

This module is IMPORT-SAFE on the CPU box: torch/transformers are imported lazily
inside the functions that need a live model, so ``import provenance`` never pulls
torch. Pure helpers (hashing, git, manifest assembly) have no heavy deps.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import time
from pathlib import Path

MANIFEST_SCHEMA = 1


# --------------------------------------------------------------------------- #
# Pure helpers (no torch)                                                      #
# --------------------------------------------------------------------------- #
def sha256_text(s: str) -> str:
    """Hex sha256 of a UTF-8 string (empty/None -> None)."""
    if s is None:
        return None
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def sha256_file(path) -> str | None:
    """Hex sha256 of a file's bytes, or None if the path is falsy/missing."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_git(repo: str | None = None) -> dict:
    """Repo commit + dirty flag, read via subprocess git (never inferred).

    Returns {"git_commit": <40-hex or None>, "git_dirty": <bool or None>}. A
    non-git environment / missing git yields Nones rather than raising -- a
    manifest with a null commit is still better than a crashed run."""
    repo = repo or os.getcwd()
    out = {"git_commit": None, "git_dirty": None}
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo,
            stderr=subprocess.DEVNULL).decode()
        out["git_dirty"] = bool(status.strip())
    except Exception:  # noqa: BLE001  (git absent / not a repo -> Nones)
        pass
    return out


# --------------------------------------------------------------------------- #
# Runtime reads from the LOADED model (torch lazy)                            #
# --------------------------------------------------------------------------- #
def capture_model_provenance(model, model_id_hint: str | None = None) -> dict:
    """Read model/dtype/quant provenance FROM THE LOADED MODEL OBJECT.

    Every field here is read from the runtime, NOT from an env var or dir name:
      * repo_id            : model.name_or_path / config._name_or_path
      * resolved_revision  : config._commit_hash (HF resolved weights commit)
      * dtype              : str(next(model.parameters()).dtype)  <-- a real tensor
      * quantization       : model.config.quantization_config (dict or None)
      * kv_geometry        : num_key_value_heads / head_dim / num_hidden_layers
    ``model_id_hint`` is only a fallback for repo_id if the model object doesn't
    carry one; it is recorded separately as ``repo_id_requested`` so a mismatch
    (requested vs realized) stays visible."""
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) or cfg

    # dtype -- from a real parameter tensor, the ground truth of what loaded.
    param_dtype = None
    try:
        param_dtype = str(next(model.parameters()).dtype)
    except Exception:  # noqa: BLE001
        param_dtype = None

    # quantization -- from the loaded config, serialized to a plain dict.
    quant = None
    qc = getattr(cfg, "quantization_config", None)
    if qc is not None:
        try:
            quant = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
        except Exception:  # noqa: BLE001
            quant = {"repr": repr(qc)}

    repo_id = (getattr(model, "name_or_path", None)
               or getattr(cfg, "_name_or_path", None))

    def _g(name):
        return getattr(text_cfg, name, None)

    return {
        "repo_id": repo_id,
        "repo_id_requested": model_id_hint,
        "resolved_revision": getattr(cfg, "_commit_hash", None),
        "config_dtype": str(getattr(cfg, "torch_dtype", None))
        if cfg is not None else None,
        "param_dtype": param_dtype,
        "quantization": quant,
        "architecture": getattr(cfg, "model_type", None),
        "kv_geometry": {
            "num_attention_heads": _g("num_attention_heads"),
            "num_key_value_heads": _g("num_key_value_heads"),
            "head_dim": _g("head_dim"),
            "num_hidden_layers": _g("num_hidden_layers"),
            "rope_theta": _g("rope_theta"),
        },
    }


def capture_mlx_provenance(model, model_id: str) -> dict:
    """Read provenance from a LOADED mlx_lm model (the local 4-bit path).

    dtype is read from a real leaf array in the model's parameter tree; the
    quantization config is read from the model's ``config``/``args`` when present.
    ``model_id`` is the repo the weights were loaded from (the runtime variable
    the caller passed to ``mlx_lm.load`` -- the id that actually loaded, not a dir
    name). The exact local Hugging Face snapshot supplies the resolved revision,
    literal config/tokenizer/template hashes, and content-addressed weight-blob
    identifiers even though the Python model has no ``_commit_hash``."""
    param_dtype = None
    dtype_inventory = {}
    try:
        import mlx.core as mx  # noqa: PLC0415
        leaves = []

        def _walk(obj):
            if isinstance(obj, mx.array):
                leaves.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    _walk(v)
        try:
            _walk(model.parameters())
        except Exception:  # noqa: BLE001
            pass
        if leaves:
            for leaf in leaves:
                dtype = str(leaf.dtype)
                dtype_inventory[dtype] = dtype_inventory.get(dtype, 0) + 1
            # Quantized MLX weights legitimately include uint32 packed arrays;
            # retain the historical field but make the complete inventory the
            # authoritative description rather than pretending one leaf is a
            # model-wide compute dtype.
            param_dtype = str(leaves[0].dtype)
    except Exception:  # noqa: BLE001
        pass

    quant = None
    for attr in ("config", "args"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        q = None
        if isinstance(obj, dict):
            q = obj.get("quantization")
        else:
            q = getattr(obj, "quantization", None)
        if q is not None:
            try:
                quant = dict(q) if not isinstance(q, dict) else q
            except Exception:  # noqa: BLE001
                quant = {"repr": repr(q)}
            break

    snapshot = None
    config = {}
    artifact_files = []
    try:
        candidate = Path(model_id).expanduser()
        if candidate.exists():
            snapshot = candidate.resolve()
        else:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
            snapshot = Path(snapshot_download(
                repo_id=model_id, local_files_only=True)).resolve()
        config_path = snapshot / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
        for path in sorted(snapshot.glob("*.safetensors")):
            resolved = path.resolve()
            blob_name = resolved.name
            artifact_files.append({
                "path": path.name,
                "bytes": resolved.stat().st_size,
                "cache_blob_sha256": (
                    blob_name if re.fullmatch(r"[0-9a-f]{64}", blob_name)
                    else None
                ),
            })
    except Exception:  # noqa: BLE001
        snapshot = None
        config = {}
        artifact_files = []

    # MLX conversion repositories put their applied weight quantization in the
    # literal config JSON even when the loaded Python object omits it.
    if quant is None:
        quant = config.get("quantization_config") or config.get("quantization")

    revision = None
    if snapshot is not None and re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
        revision = snapshot.name

    def _cfg(name):
        return config.get(name)

    tokenizer_config = snapshot / "tokenizer_config.json" if snapshot else None
    tokenizer_json = snapshot / "tokenizer.json" if snapshot else None
    config_path = snapshot / "config.json" if snapshot else None
    chat_template = None
    if tokenizer_config and tokenizer_config.exists():
        try:
            chat_template = json.loads(tokenizer_config.read_text()).get(
                "chat_template")
        except Exception:  # noqa: BLE001
            pass
    if chat_template is None and snapshot:
        template_path = snapshot / "chat_template.jinja"
        if template_path.exists():
            chat_template = template_path.read_text()

    return {
        "repo_id": model_id,
        "repo_id_requested": model_id,
        "resolved_revision": revision,
        "snapshot_path": str(snapshot) if snapshot else None,
        "config_sha256": sha256_file(config_path),
        "tokenizer_config_sha256": sha256_file(tokenizer_config),
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "chat_template_sha256": sha256_text(chat_template),
        "artifact_files": artifact_files,
        "config_dtype": str(_cfg("torch_dtype")) if config else None,
        "param_dtype": param_dtype,
        "parameter_dtype_inventory": dtype_inventory,
        "quantization": quant,
        "architecture": _cfg("model_type"),
        "kv_geometry": {
            "num_attention_heads": _cfg("num_attention_heads"),
            "num_key_value_heads": _cfg("num_key_value_heads"),
            "head_dim": _cfg("head_dim"),
            "num_hidden_layers": _cfg("num_hidden_layers"),
            "rope_theta": _cfg("rope_theta"),
        } if config else None,
    }


def runtime_env() -> dict:
    """Timestamp, GPU name, host, library versions -- read from the runtime.

    torch is imported lazily; if it is absent the torch/gpu fields are None."""
    gpu = None
    torch_v = None
    try:
        import torch  # noqa: PLC0415
        torch_v = torch.__version__
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    tf_v = None
    try:
        import transformers  # noqa: PLC0415
        tf_v = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    package_versions = {}
    for package in ("mlx", "mlx-lm", "huggingface-hub"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_name": gpu,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch_v,
        "transformers_version": tf_v,
        "package_versions": package_versions,
    }


# --------------------------------------------------------------------------- #
# Manifest assembly + stamping                                                #
# --------------------------------------------------------------------------- #
def build_manifest(*, model_provenance: dict, dtype_env: str | None,
                   intervention: dict, metric: dict, condition: dict,
                   corpus: dict, harness: str, extra: dict | None = None,
                   repo: str | None = None) -> dict:
    """Assemble the full provenance manifest from RUNTIME-READ components.

    Every argument block is expected to already hold runtime-read values:
      model_provenance : capture_model_provenance(model) output.
      dtype_env        : the SC_LOAD_DTYPE string REQUESTED (audit vs realized).
      intervention     : {arm, graft_type, alpha, champion_config_path,
                          champion_config_sha256, champion_label, alignment}.
      metric           : {definition, is_proxy, resolve_rate_measured}.
      condition        : {summary_kind, summary_request_sha256, summary_source}.
      corpus           : {name, split, n, instance_ids}.
      harness          : the script that produced the data (e.g. run_swegym_hf.py).
    """
    load = {
        "dtype": model_provenance.get("param_dtype"),
        "parameter_dtype_inventory": model_provenance.get(
            "parameter_dtype_inventory"),
        "dtype_env_requested": dtype_env,
        "config_dtype": model_provenance.get("config_dtype"),
        "quantization": model_provenance.get("quantization"),
        "kv_geometry": model_provenance.get("kv_geometry"),
    }
    # LOUD self-audit: did the realized dtype match what was requested?
    realized = (model_provenance.get("param_dtype") or "")
    req = (dtype_env or "")
    load["dtype_matches_request"] = (
        None if not req else (req.replace("torch.", "") in realized))

    man = {
        "manifest_schema": MANIFEST_SCHEMA,
        "harness": harness,
        "model": {
            "repo_id": model_provenance.get("repo_id"),
            "repo_id_requested": model_provenance.get("repo_id_requested"),
            "resolved_revision": model_provenance.get("resolved_revision"),
            "architecture": model_provenance.get("architecture"),
            "snapshot_path": model_provenance.get("snapshot_path"),
            "config_sha256": model_provenance.get("config_sha256"),
            "tokenizer_config_sha256": model_provenance.get(
                "tokenizer_config_sha256"),
            "tokenizer_json_sha256": model_provenance.get(
                "tokenizer_json_sha256"),
            "chat_template_sha256": model_provenance.get(
                "chat_template_sha256"),
            "artifact_files": model_provenance.get("artifact_files"),
        },
        "load": load,
        "intervention": intervention,
        "metric": metric,
        "condition": condition,
        "corpus": corpus,
        "code": code_git(repo),
        "run": runtime_env(),
    }
    if extra:
        man["extra"] = extra
    return man


def stamp(result: dict, manifest: dict) -> dict:
    """Insert the manifest into a result dict under ``_manifest`` (in place)."""
    result["_manifest"] = manifest
    return result


def write_run_manifest(out_dir, manifest: dict, name: str = "manifest.json"):
    """Write the run-level manifest.json into out_dir (atomic)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# Canonical metric-definition strings (single source of truth)               #
# --------------------------------------------------------------------------- #
METRIC_SWEGYM_TF_LOGPROB = {
    "definition": ("teacher-forced mean per-token logprob of the TRUE next "
                   "assistant action, on SWE-Gym/OpenHands-derived trajectories"),
    "is_proxy": True,
    "resolve_rate_measured": False,
    "proxy_note": ("PROXY for action prediction under compaction -- this is NOT "
                   "SWE-bench/SWE-Gym resolve rate or test-pass rate. No patch is "
                   "applied and no test is run. Report as a logprob proxy."),
}

METRIC_CHAT_RAW_EB = {
    "definition": ("raw_EB = lp_E - lp_B: teacher-forced mean logprob lift of the "
                   "value-graft over the compacted baseline, on the SHARED gold "
                   "continuation"),
    "is_proxy": True,
    "resolve_rate_measured": False,
    "proxy_note": ("logprob-based recovery metric; not a task-success/resolve "
                   "rate."),
}
