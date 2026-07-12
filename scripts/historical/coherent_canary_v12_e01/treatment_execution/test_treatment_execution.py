from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.error


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLS_PATH = HERE / "receipt_tools.py"
SPEC = importlib.util.spec_from_file_location("v12_receipt_tools", TOOLS_PATH)
TOOLS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = TOOLS
SPEC.loader.exec_module(TOOLS)
WATCHDOG_PATH = HERE / "provider_deadline_watchdog.py"
WATCHDOG_SPEC = importlib.util.spec_from_file_location(
    "v12_provider_deadline_watchdog", WATCHDOG_PATH)
WATCHDOG = importlib.util.module_from_spec(WATCHDOG_SPEC)
assert WATCHDOG_SPEC.loader is not None
sys.modules[WATCHDOG_SPEC.name] = WATCHDOG
WATCHDOG_SPEC.loader.exec_module(WATCHDOG)


def row(path: Path, relative: str) -> dict:
    data = path.read_bytes()
    return {
        "path": relative,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class ReceiptToolsTest(unittest.TestCase):
    def fixture(self, root: Path, *, terminal: str = "PASS",
                runtime_sha: str | None = TOOLS.EXPECTED_RUNTIME_SHA,
                gpu: str | None =
                "NVIDIA A100 80GB PCIe, GPU-test, 580.159.04, 81920"
                ) -> tuple[Path, str]:
        staging = root / "staging"
        staging.mkdir(parents=True)
        (staging / "raw").write_bytes((b"exact-v12-raw\n" * 350_000) + b"end\n")
        (staging / "job_log").write_text("# exact-v12 job\n")
        harvest_present = terminal == "PASS"
        if harvest_present:
            (staging / "harvest").write_text("{\"harvest\": true}\n")
        receipt_relative = (
            "results/coherent_canary_v12_treatment/"
            "coherent-canary-v12-treatment-e01-exact-subject-"
            "20260712T040000Z_receipt.json"
        )
        artifacts = {
            "raw": row(
                staging / "raw",
                "results/coherent_canary_v12_treatment/"
                "coherent-canary-v12-treatment-e01_exact-subject_"
                "20260712T040001123456Z.json",
            ),
            "harvest": (
                row(
                    staging / "harvest",
                    "results/coherent_canary_v12_harvest/"
                    "coherent-canary-v12-harvest-e01-exact-subject-"
                    "20260712T040000Z.json",
                ) if harvest_present else None
            ),
            "job_log": row(
                staging / "job_log",
                "results/coherent_canary_v12_treatment/"
                "coherent-canary-v12-treatment-e01-exact-subject-"
                "20260712T040000Z_job.md",
            ),
        }
        receipt = {
            "schema": TOOLS.RECEIPT_SCHEMA,
            "design_id": TOOLS.DESIGN_ID,
            "case_id": "e01",
            "completed_at_utc": "2026-07-12T04:01:00+00:00",
            "expected_commit": TOOLS.EXPECTED_COMMIT,
            "observed_commit": TOOLS.EXPECTED_COMMIT,
            "runner_exit": 0 if terminal == "PASS" else 1,
            "runner_status": "PASS" if terminal == "PASS" else "ERROR",
            "harvester_exit": 0 if terminal == "PASS" else 1,
            "harvest_status": "PASS" if terminal == "PASS" else "ERROR",
            "terminal_status": terminal,
            "semantic_evidence_eligible": terminal == "PASS",
            "phase_a_release_status": "PRETREATMENT_PASS",
            "phase_a_release_eligible": True,
            "primary_arm_count": 31 if terminal == "PASS" else None,
            "placebo_control_count": 3 if terminal == "PASS" else None,
            "available_placebo_control_count": 2 if terminal == "PASS" else None,
            "provider_hourly_cost_usd": 1.39,
            "runner_wall_cap_seconds": 1500,
            "gpu": gpu,
            "runtime_fingerprint_sha256": runtime_sha,
            "diagnostic_only": True,
            "formal_v12_decision_eligible": False,
            "aggregate_expansion_authorized": False,
            "artifacts": artifacts,
        }
        receipt_path = root / "receipt"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt_path, receipt_relative

    def test_large_raw_is_verified_and_remapped_losslessly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, relative = self.fixture(root)
            document = TOOLS.validate_receipt(receipt, receipt_relative=relative)
            self.assertEqual(document["terminal_status"], "PASS")
            args = type("Args", (), {
                "receipt": receipt,
                "receipt_relative": relative,
                "staging": root / "staging",
                "repo": root / "repo",
                "intact_raw": root / "intact_raw",
            })()
            TOOLS.command_install(args)
            raw_row = document["artifacts"]["raw"]
            installed_raw = args.intact_raw / Path(raw_row["path"]).name
            self.assertGreater(installed_raw.stat().st_size, 4 * 1024 * 1024)
            self.assertEqual(
                hashlib.sha256(installed_raw.read_bytes()).hexdigest(),
                raw_row["sha256"],
            )
            self.assertFalse((args.repo / raw_row["path"]).exists())
            self.assertTrue((args.repo / relative).is_file())
            self.assertTrue((args.repo / document["artifacts"]["harvest"]["path"]).is_file())
            self.assertTrue((args.repo / document["artifacts"]["job_log"]["path"]).is_file())
            TOOLS.command_install(args)

    def test_corrupt_staged_raw_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, relative = self.fixture(root)
            with (root / "staging/raw").open("ab") as handle:
                handle.write(b"corrupt")
            args = type("Args", (), {
                "receipt": receipt,
                "receipt_relative": relative,
                "staging": root / "staging",
                "repo": root / "repo",
                "intact_raw": root / "intact_raw",
            })()
            with self.assertRaises(TOOLS.ReceiptError):
                TOOLS.command_install(args)

    def test_terminal_error_receipt_preserves_available_artifacts(self):
        for runtime_sha, gpu in (
            (None, None),
            ("0" * 64, "device unavailable after raw error"),
        ):
            with self.subTest(runtime_sha=runtime_sha, gpu=gpu), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt, relative = self.fixture(
                    root, terminal="ERROR", runtime_sha=runtime_sha, gpu=gpu)
                document = TOOLS.validate_receipt(
                    receipt, receipt_relative=relative)
                self.assertEqual(document["terminal_status"], "ERROR")
                self.assertEqual(
                    document["runtime_fingerprint_sha256"], runtime_sha)
                self.assertIsNone(document["artifacts"]["harvest"])
                args = type("Args", (), {
                    "receipt": receipt,
                    "receipt_relative": relative,
                    "staging": root / "staging",
                    "repo": root / "repo",
                    "intact_raw": root / "intact_raw",
                })()
                TOOLS.command_install(args)
                self.assertTrue((args.intact_raw / Path(
                    document["artifacts"]["raw"]["path"]).name).is_file())
                self.assertTrue((args.repo / relative).is_file())
                self.assertTrue((args.repo /
                    document["artifacts"]["job_log"]["path"]).is_file())

    def test_terminal_pass_still_requires_exact_runtime_and_gpu(self):
        for runtime_sha, gpu in (
            (None, "NVIDIA A100 80GB PCIe, GPU-test, 580.159.04, 81920"),
            ("0" * 64, "NVIDIA A100 80GB PCIe, GPU-test, 580.159.04, 81920"),
            (TOOLS.EXPECTED_RUNTIME_SHA,
             "NVIDIA A100 80GB PCIe, GPU-test, 580.159.03, 81920"),
        ):
            with self.subTest(runtime_sha=runtime_sha, gpu=gpu), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt, relative = self.fixture(
                    root, runtime_sha=runtime_sha, gpu=gpu)
                with self.assertRaises(TOOLS.ReceiptError):
                    TOOLS.validate_receipt(receipt, receipt_relative=relative)

    def test_all_receipts_require_diagnostic_only_boundary(self):
        for terminal in ("PASS", "ERROR"):
            with self.subTest(terminal=terminal), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt, relative = self.fixture(root, terminal=terminal)
                document = json.loads(receipt.read_text())
                document["formal_v12_decision_eligible"] = True
                receipt.write_text(json.dumps(document))
                with self.assertRaises(TOOLS.ReceiptError):
                    TOOLS.validate_receipt(receipt, receipt_relative=relative)


class FrozenInputTest(unittest.TestCase):
    def test_pinned_commit_contains_every_bound_input_byte(self):
        paths = {
            "COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md":
                "fc02e86d007369121accbee69471383200fa3f566183e2ed96102e2cf7428b17",
            "data/coherent_canary_v12/revision2/session_d/e01.json":
                "6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090",
            "results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T024859Z.json":
                "a8094b4a3355836cfbe0ec0936342a56e271cd992108d7bda61acd8488ee0b48",
            "results/coherent_canary_validation/coherent_canary_v12_phase_a_e01_exact-subject_20260712T030622Z.json":
                "aabac5f0aa64db8f0db49110583be9a17f8453e1a523f1cbc0e0312ec3c15410",
            "results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_20260712T030639073087Z.json":
                "2cd7f6f190b9f4e9598844e45e5488779b72cd54a1fd51dd5b8a5b24154d672d",
        }
        for relative, expected in paths.items():
            data = subprocess.check_output([
                "git", "-C", str(ROOT), "show",
                f"{TOOLS.EXPECTED_COMMIT}:{relative}",
            ])
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected, relative)


class ProviderDeadlineGuardTest(unittest.TestCase):
    def fixture(self, root: Path, *, rate: float = 1.39):
        state = root / "state.json"
        response = root / "response.json"
        job = root / "job.sh"
        record = root / "guard.json"
        state.write_text(json.dumps({"id": "pod-1", "costPerHr": rate}))
        response.write_text(json.dumps({"id": "pod-1", "costPerHr": rate}))
        job.write_text("#!/bin/bash\nexit 0\n")
        return state, response, job, record

    def test_record_binds_rate_and_provider_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, response, job, record = self.fixture(root)
            value = WATCHDOG.build_record(
                name="v12-treatment", state_path=state,
                response_path=response, job_path=job, record_path=record,
                start_epoch=1_000_000)
            self.assertEqual(value["hard_deadline_epoch"], 1_002_300)
            self.assertEqual(value["delete_trigger_epoch"], 1_002_180)
            self.assertEqual(value["created_cost_per_hr"], 1.39)

    def test_overrate_allocation_is_recorded_then_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, response, job, record = self.fixture(root, rate=1.40)
            with self.assertRaises(WATCHDOG.GuardError):
                WATCHDOG.build_record(
                    name="v12-treatment", state_path=state,
                    response_path=response, job_path=job, record_path=record,
                    start_epoch=1_000_000)
            self.assertTrue(record.is_file())
            self.assertEqual(json.loads(record.read_text())["created_cost_per_hr"], 1.40)

    def test_ambiguity_preserves_before_deadline_but_not_at_deadline(self):
        record = {
            "delete_trigger_epoch": 1_002_180,
            "max_cost_per_hr": 1.39,
        }
        self.assertEqual(WATCHDOG.decision(
            now_epoch=1_002_000, record=record, provider_pod=None,
            provider_error=RuntimeError("transient")), "PRESERVE_AMBIGUOUS")
        self.assertEqual(WATCHDOG.decision(
            now_epoch=1_002_180, record=record, provider_pod=None,
            provider_error=RuntimeError("transient")), "DELETE_DEADLINE")
        self.assertEqual(WATCHDOG.decision(
            now_epoch=1_002_000, record=record,
            provider_pod={"costPerHr": 1.40}), "DELETE_RATE")
        not_found = urllib.error.HTTPError(
            "https://example.invalid", 404, "gone", {}, None)
        self.assertEqual(WATCHDOG.decision(
            now_epoch=1_002_000, record=record, provider_pod=None,
            provider_error=not_found), "GONE")


if __name__ == "__main__":
    unittest.main()
