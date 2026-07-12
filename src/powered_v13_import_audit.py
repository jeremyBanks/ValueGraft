"""Static parent-inventory/import-closure audit for powered-v13 Stage T."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


DESIGN_ID = "coherent-state-powered-successor-v13"
CONTRACT_SCHEMA = "powered-v13-technical-canary-inventory-contract-v1"
STAGE = "TECHNICAL_CANARY"
REPORT_SCHEMA = "powered-v13-technical-canary-import-audit-v1"
FORBIDDEN_MODULE_PREFIXES = (
    "powered_v13_cycle",
    "powered_v13_permutation",
    "powered_v13_recipe",
    "powered_v13_stats",
    "powered_v13_stimuli",
)
FORBIDDEN_INVENTORY_PATHS = frozenset({
    "data/coherent_state_powered_v13/recipe-foundation-v1.json",
    "src/powered_v13_cycle.py",
    "src/powered_v13_permutation.py",
    "src/powered_v13_recipe.py",
    "src/powered_v13_stats.py",
    "src/powered_v13_stimuli.py",
    "scripts/write_powered_v13_literal_permutations.py",
    "scripts/write_powered_v13_seed_manifest.py",
    "scripts/recompute_powered_v13_bound_independent.py",
    "scripts/simulate_powered_v13_stats.py",
})
ALLOWED_LITERAL_DYNAMIC_IMPORTS = frozenset({"pod"})


class V13ImportAuditError(RuntimeError):
    """Stage-T inventory or reachable local imports escape the closed graph."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13ImportAuditError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V13ImportAuditError(
            f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_path(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} is empty")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and path.as_posix() == value
             and path.parts and all(part not in {"", ".", ".."}
                                    for part in path.parts)
             and path.parts[0] != ".git",
             f"{label} is not a safe normalized repository path")
    return value


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == {
        "schema", "design_id", "stage", "inventory_paths",
    }, "Stage-T inventory contract fields differ")
    _require(value.get("schema") == CONTRACT_SCHEMA
             and value.get("design_id") == DESIGN_ID
             and value.get("stage") == STAGE,
             "Stage-T inventory contract identity differs")
    paths = value.get("inventory_paths")
    _require(isinstance(paths, list) and bool(paths),
             "Stage-T inventory path list is empty")
    frozen = [_safe_path(path, f"inventory path {index}")
              for index, path in enumerate(paths)]
    _require(frozen == sorted(set(frozen)),
             "Stage-T inventory paths are not unique sorted paths")
    forbidden = sorted(set(frozen) & FORBIDDEN_INVENTORY_PATHS)
    _require(not forbidden,
             f"Stage-T inventory contains production-only paths: {forbidden}")
    result = deepcopy(dict(value))
    result["inventory_paths"] = frozen
    return result


def _load_contract(path: Path) -> dict[str, Any]:
    _require(not path.is_symlink() and path.is_file(),
             "Stage-T inventory contract is absent or symlinked")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V13ImportAuditError(
            f"Stage-T inventory contract is not UTF-8 JSON: {exc}") from exc
    contract = validate_contract(value)
    _require(raw == canonical_json_bytes(contract) + b"\n",
             "Stage-T inventory contract is not canonical JSON plus LF")
    return contract


def _module_map(repo: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    source = repo / "src"
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        module = ".".join(parts)
        result[module] = path.relative_to(repo).as_posix()
    return result


def _imports(path: Path) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise V13ImportAuditError(f"cannot parse inventoried Python {path}: {exc}") from exc
    observed: list[tuple[str, str]] = []
    importlib_aliases = {"importlib"}
    import_module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_aliases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            observed.extend((alias.name, "static") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            _require(node.level == 0,
                     f"relative import is unsupported in Stage T: {path}")
            if node.module:
                observed.append((node.module, "static"))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "__import__", "eval", "exec",
            }:
                raise V13ImportAuditError(
                    f"dynamic code/import call is forbidden in Stage T: {path}")
            direct_alias = (isinstance(node.func, ast.Name)
                            and node.func.id in import_module_aliases)
            module_attribute = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in importlib_aliases
                and node.func.attr == "import_module")
            if direct_alias or module_attribute:
                _require(len(node.args) == 1
                         and isinstance(node.args[0], ast.Constant)
                         and isinstance(node.args[0].value, str),
                         f"nonliteral import_module is forbidden: {path}")
                module = node.args[0].value
                _require(module in ALLOWED_LITERAL_DYNAMIC_IMPORTS,
                         f"dynamic module is not Stage-T allowlisted: {module}")
                observed.append((module, "literal_dynamic"))
    return sorted(set(observed))


def _forbidden_module(module: str) -> bool:
    return any(module == prefix or module.startswith(prefix + ".")
               for prefix in FORBIDDEN_MODULE_PREFIXES)


def audit_stage_t_imports(
    repo: Path,
    *,
    contract_path: Path,
    execution_roots: Sequence[str],
) -> dict[str, Any]:
    repo = Path(repo).resolve(strict=True)
    contract_path = Path(contract_path)
    if not contract_path.is_absolute():
        contract_path = repo / contract_path
    contract = _load_contract(contract_path)
    inventory = set(contract["inventory_paths"])
    roots = [_safe_path(path, f"execution root {index}")
             for index, path in enumerate(execution_roots)]
    _require(roots == sorted(set(roots)) and bool(roots),
             "execution roots are not nonempty unique sorted paths")
    _require(set(roots) <= inventory,
             "execution roots are absent from Stage-T inventory")
    for relative in inventory:
        path = repo / relative
        _require(not path.is_symlink() and path.is_file(),
                 f"inventoried path is absent or symlinked: {relative}")

    modules = _module_map(repo)
    rows: list[dict[str, Any]] = []
    local_edges: list[dict[str, str]] = []
    for relative in sorted(inventory):
        if not relative.endswith(".py"):
            continue
        path = repo / relative
        imports = _imports(path)
        for module, kind in imports:
            _require(not _forbidden_module(module),
                     f"production-only module is reachable: {module} from {relative}")
            candidates = [
                (name, module_path) for name, module_path in modules.items()
                if module == name or module.startswith(name + ".")
            ]
            if candidates:
                _name, local_path = max(candidates, key=lambda row: len(row[0]))
                _require(local_path in inventory,
                         f"reachable local module is absent from inventory: "
                         f"{module} -> {local_path}")
                local_edges.append({
                    "from": relative, "module": module,
                    "to": local_path, "kind": kind,
                })
        raw = path.read_bytes()
        rows.append({
            "path": relative,
            "sha256": sha256_bytes(raw),
            "imports": [{"module": module, "kind": kind}
                        for module, kind in imports],
        })
    result = {
        "schema": REPORT_SCHEMA,
        "design_id": DESIGN_ID,
        "stage": STAGE,
        "status": "PASS",
        "semantic_n": 0,
        "production_entropy_requested": False,
        "contract_path": contract_path.relative_to(repo).as_posix(),
        "contract_sha256": sha256_bytes(contract_path.read_bytes()),
        "inventory_path_count": len(inventory),
        "execution_roots": roots,
        "python_files": rows,
        "local_edges": sorted(local_edges, key=lambda row: (
            row["from"], row["module"], row["to"], row["kind"])),
        "forbidden_module_prefixes": list(FORBIDDEN_MODULE_PREFIXES),
        "forbidden_inventory_paths": sorted(FORBIDDEN_INVENTORY_PATHS),
        "allowed_literal_dynamic_imports": sorted(
            ALLOWED_LITERAL_DYNAMIC_IMPORTS),
    }
    result["report_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


__all__ = [
    "CONTRACT_SCHEMA", "DESIGN_ID", "FORBIDDEN_INVENTORY_PATHS",
    "FORBIDDEN_MODULE_PREFIXES", "REPORT_SCHEMA", "STAGE",
    "V13ImportAuditError", "audit_stage_t_imports", "canonical_json_bytes",
    "validate_contract",
]
