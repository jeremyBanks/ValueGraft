from __future__ import annotations

import pytest
from pathlib import Path

import pod
from pod_admission import PodAdmissionError, parse_gpu_row, validate_gpu


def test_create_body_applies_provider_cuda_filter(monkeypatch):
    monkeypatch.setenv("SC_POD_ALLOWED_CUDA", "13.0")
    monkeypatch.setenv("SC_POD_CLOUD", "SECURE")
    body = pod.create_body("NVIDIA A100 80GB PCIe")
    assert body["allowedCudaVersions"] == ["13.0"]
    assert body["cloudType"] == "SECURE"
    assert body["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"]
    monkeypatch.setenv("SC_POD_ALLOWED_CUDA", "13.0,13.0,12.9")
    assert pod.create_body("gpu")["allowedCudaVersions"] == ["13.0", "12.9"]


def test_create_body_rejects_invalid_cuda_filter(monkeypatch):
    monkeypatch.setenv("SC_POD_ALLOWED_CUDA", "13")
    with pytest.raises(ValueError, match="invalid SC_POD_ALLOWED_CUDA"):
        pod.create_body("gpu")


def test_gpu_admission_requires_one_exact_compatible_gpu():
    row = "NVIDIA A100 80GB PCIe, 580.159.03, 81920"
    assert parse_gpu_row(row) == {
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "driver_version": "580.159.03",
        "memory_total_mib": 81920,
    }
    assert validate_gpu(
        row, expected_name="NVIDIA A100 80GB PCIe",
        minimum_driver="580.65.06", minimum_memory_mib=80000)


@pytest.mark.parametrize("row, message", [
    ("NVIDIA A100 80GB PCIe, 550.90.12, 81920", "driver"),
    ("NVIDIA A100-SXM4-80GB, 580.159.03, 81920", "GPU name"),
    ("NVIDIA A100 80GB PCIe, 580.159.03, 40960", "GPU memory"),
    ("NVIDIA A100 80GB PCIe, 580.159.03, 81920\nextra, 580.1, 1",
     "exactly one"),
])
def test_gpu_admission_rejects_incompatible_hosts(row, message):
    with pytest.raises(PodAdmissionError, match=message):
        validate_gpu(
            row, expected_name="NVIDIA A100 80GB PCIe",
            minimum_driver="580.65.06", minimum_memory_mib=80000)


def test_exact_launch_wrapper_binds_cuda_driver_and_commit_mechanically():
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts/launch_coherent_canary_v12_technical.sh").read_text()
    launcher = (root / "scripts/launch_pod.sh").read_text()
    assert 'HEAD_COMMIT="$(git rev-parse HEAD)"' in wrapper
    assert "SC_POD_ALLOWED_CUDA=13.0" in wrapper
    assert "SC_MIN_NVIDIA_DRIVER=580.65.06" in wrapper
    assert 'SC_EXPECTED_GPU_NAME="NVIDIA A100 80GB PCIe"' in wrapper
    assert "SC_TERMINATE_ON_ADMISSION_FAILURE=1" in wrapper
    assert "MAX_ADMISSION_ATTEMPTS" in wrapper
    assert launcher.index("nvidia-smi --query-gpu=name,driver_version,memory.total") < \
        launcher.index("apt-get update")
    assert "src/pod_admission.py" in launcher
    assert 'exit 86' in launcher
