"""Fail-closed runtime and repository binding for canary v12."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Mapping, Sequence

import torch


class CanaryPreflightError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryPreflightError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_binding(repo: Path, paths: Sequence[Path]) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    rows = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        absolute = path if path.is_absolute() else repo / path
        _require(absolute.is_file(), f"inventory file absent: {path}")
        relative = absolute.relative_to(repo).as_posix()
        tracked = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"HEAD:{relative}"])
        worktree = absolute.read_bytes()
        _require(tracked == worktree, f"inventory file differs from HEAD: {relative}")
        rows.append({"path": relative, "sha256": hashlib.sha256(worktree).hexdigest()})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"repository_commit": commit, "files": rows,
            "inventory_sha256": hashlib.sha256(canonical).hexdigest()}


def _eos_set(value) -> set[int]:
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return set()


def runtime_fingerprint(model, tokenizer, *, requested_model: str,
                        requested_revision: str, resolved_snapshot: str,
                        expected_geometry: Mapping[str, int | float]) -> dict:
    config = getattr(model.config, "text_config", model.config)
    observed = {
        "layers": int(getattr(config, "num_hidden_layers", -1)),
        "attention_heads": int(getattr(config, "num_attention_heads", -1)),
        "kv_heads": int(getattr(config, "num_key_value_heads", -1)),
        "head_dim": int(getattr(config, "head_dim", -1)),
        "rope_theta": float(getattr(config, "rope_theta", -1)),
    }
    _require(observed == dict(expected_geometry),
             f"model geometry differs: {observed} != {dict(expected_geometry)}")
    floating_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()
                              if parameter.is_floating_point()})
    _require(floating_dtypes == ["torch.bfloat16"],
             f"floating parameter dtypes differ: {floating_dtypes}")
    for scope, value in (("model", model.config), ("text", config)):
        fields = {str(getattr(value, name, None)) for name in (
            "_attn_implementation", "_attn_implementation_internal")}
        _require(fields == {"eager"}, f"{scope} attention backend differs: {fields}")
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and \
                "Attention" in type(module).__name__:
            index = getattr(module, "layer_idx", None)
            _require(isinstance(index, int), f"attention layer index absent: {name}")
            local = getattr(module, "config", None)
            _require(local is not None and
                     str(getattr(local, "_attn_implementation", None)) == "eager" and
                     str(getattr(local, "_attn_implementation_internal", None)) == "eager",
                     f"attention layer backend differs: {name}")
            layers.append({"layer": index, "name": name,
                           "class": type(module).__name__, "backend": "eager"})
    _require([row["layer"] for row in layers] == list(range(observed["layers"])),
             "attention layer coverage/order differs")
    model_eos = _eos_set(getattr(model.config, "eos_token_id", None))
    text_eos = _eos_set(getattr(config, "eos_token_id", None))
    tokenizer_eos = _eos_set(getattr(tokenizer, "eos_token_id", None))
    generation_eos = _eos_set(getattr(
        getattr(model, "generation_config", None), "eos_token_id", None))
    nonempty = [value for value in (model_eos, text_eos, tokenizer_eos,
                                    generation_eos) if value]
    _require(nonempty and all(value == nonempty[0] for value in nonempty),
             f"EOS metadata differs: {nonempty}")
    return {
        "requested_model": requested_model,
        "requested_revision": requested_revision,
        "resolved_snapshot": resolved_snapshot,
        "dtype": "torch.bfloat16", "attention_backend": "eager",
        "geometry": observed, "eos_ids": sorted(nonempty[0]),
        "attention_layers": layers,
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "python": platform.python_version(), "platform": platform.platform(),
    }
