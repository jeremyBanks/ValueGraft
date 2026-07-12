from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import powered_v13_import_audit as audit


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_powered_v13_stage_t_imports.py"
CONTRACT = "data/technical-canary-inventory-contract.json"


def _write(repo: Path, relative: str, content: str | bytes):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def _contract(repo: Path, paths):
    document = {
        "schema": audit.CONTRACT_SCHEMA,
        "design_id": audit.DESIGN_ID,
        "stage": audit.STAGE,
        "inventory_paths": sorted(paths),
    }
    _write(repo, CONTRACT, audit.canonical_json_bytes(document) + b"\n")
    return document


def _repo(tmp_path: Path, *, root_source="import helper\n"):
    _write(tmp_path, "scripts/root.py", root_source)
    _write(tmp_path, "src/helper.py", "VALUE = 1\n")
    paths = [CONTRACT, "scripts/root.py", "src/helper.py"]
    _contract(tmp_path, paths)
    return paths


def test_exact_inventoried_local_import_closure_passes(tmp_path):
    _repo(tmp_path)
    result = audit.audit_stage_t_imports(
        tmp_path, contract_path=CONTRACT,
        execution_roots=["scripts/root.py"])
    assert result["status"] == "PASS"
    assert result["semantic_n"] == 0
    assert result["production_entropy_requested"] is False
    assert result["local_edges"] == [{
        "from": "scripts/root.py", "module": "helper",
        "to": "src/helper.py", "kind": "static",
    }]
    core = {key: value for key, value in result.items()
            if key != "report_sha256"}
    assert result["report_sha256"] == audit.sha256_bytes(
        audit.canonical_json_bytes(core))


def test_reachable_local_module_must_be_in_exact_inventory(tmp_path):
    paths = _repo(tmp_path)
    _contract(tmp_path, [path for path in paths if path != "src/helper.py"])
    with pytest.raises(audit.V13ImportAuditError,
                       match="local module is absent"):
        audit.audit_stage_t_imports(
            tmp_path, contract_path=CONTRACT,
            execution_roots=["scripts/root.py"])


@pytest.mark.parametrize("source,match", [
    ("import powered_v13_recipe\n", "production-only module"),
    ("x = __import__('helper')\n", "dynamic __import__"),
    ("import importlib\nimportlib.import_module(name)\n", "nonliteral"),
    ("import importlib\nimportlib.import_module('helper')\n", "not Stage-T allowlisted"),
])
def test_production_or_open_dynamic_imports_fail(source, match, tmp_path):
    _repo(tmp_path, root_source=source)
    with pytest.raises(audit.V13ImportAuditError, match=match):
        audit.audit_stage_t_imports(
            tmp_path, contract_path=CONTRACT,
            execution_roots=["scripts/root.py"])


def test_only_literal_pod_dynamic_import_is_allowed_and_inventoried(tmp_path):
    _write(tmp_path, "scripts/root.py",
           "import importlib\nimportlib.import_module('pod')\n")
    _write(tmp_path, "src/pod.py", "VALUE = 1\n")
    _contract(tmp_path, [CONTRACT, "scripts/root.py", "src/pod.py"])
    result = audit.audit_stage_t_imports(
        tmp_path, contract_path=CONTRACT,
        execution_roots=["scripts/root.py"])
    assert result["local_edges"][0]["kind"] == "literal_dynamic"
    assert result["local_edges"][0]["to"] == "src/pod.py"


def test_contract_rejects_production_path_duplicates_unsorted_and_extra_fields():
    base = {
        "schema": audit.CONTRACT_SCHEMA,
        "design_id": audit.DESIGN_ID,
        "stage": audit.STAGE,
        "inventory_paths": ["src/safe.py"],
    }
    for mutation in (
        {**base, "inventory_paths": ["src/powered_v13_recipe.py"]},
        {**base, "inventory_paths": ["z.py", "a.py"]},
        {**base, "inventory_paths": ["src/safe.py", "src/safe.py"]},
        {**base, "extra": True},
    ):
        with pytest.raises(audit.V13ImportAuditError):
            audit.validate_contract(mutation)


def test_symlinked_inventoried_source_fails(tmp_path):
    paths = _repo(tmp_path)
    helper = tmp_path / "src/helper.py"
    real = tmp_path / "real-helper.py"
    helper.rename(real)
    helper.symlink_to(real)
    _contract(tmp_path, paths)
    with pytest.raises(audit.V13ImportAuditError, match="symlinked"):
        audit.audit_stage_t_imports(
            tmp_path, contract_path=CONTRACT,
            execution_roots=["scripts/root.py"])


def test_cli_exclusive_writes_report(tmp_path):
    _repo(tmp_path)
    output = tmp_path / "report.json"
    command = [
        sys.executable, str(SCRIPT), "--repo", str(tmp_path),
        "--contract", CONTRACT,
        "--execution-roots-json", '["scripts/root.py"]',
        "--output", str(output),
    ]
    first = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    assert json.loads(output.read_text())["status"] == "PASS"
    second = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert second.returncode == 2
    assert "already exists" in second.stderr
