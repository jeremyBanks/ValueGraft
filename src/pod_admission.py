#!/usr/bin/env python3
"""Fail-closed validation of one newly allocated RunPod GPU host."""

from __future__ import annotations

import json
import os
import re


class PodAdmissionError(RuntimeError):
    pass


def version_tuple(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value):
        raise PodAdmissionError(f"invalid NVIDIA driver version: {value!r}")
    return tuple(int(part) for part in value.split("."))


def parse_gpu_row(value: str) -> dict[str, object]:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PodAdmissionError(
            f"expected exactly one GPU admission row, observed {len(lines)}")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 3:
        raise PodAdmissionError(f"invalid GPU admission row: {lines[0]!r}")
    name, driver, memory = parts
    version_tuple(driver)
    try:
        memory_mib = int(memory)
    except ValueError as exc:
        raise PodAdmissionError(
            f"invalid GPU memory value: {memory!r}") from exc
    if not name or memory_mib <= 0:
        raise PodAdmissionError("GPU name or memory is empty")
    return {"gpu_name": name, "driver_version": driver,
            "memory_total_mib": memory_mib}


def validate_gpu(value: str, *, expected_name: str = "",
                 minimum_driver: str = "",
                 minimum_memory_mib: int = 0) -> dict[str, object]:
    observed = parse_gpu_row(value)
    if expected_name and observed["gpu_name"] != expected_name:
        raise PodAdmissionError(
            f"GPU name {observed['gpu_name']!r} != {expected_name!r}")
    if minimum_driver and version_tuple(str(observed["driver_version"])) < \
            version_tuple(minimum_driver):
        raise PodAdmissionError(
            f"driver {observed['driver_version']} < required {minimum_driver}")
    if int(observed["memory_total_mib"]) < minimum_memory_mib:
        raise PodAdmissionError(
            f"GPU memory {observed['memory_total_mib']} MiB < required "
            f"{minimum_memory_mib} MiB")
    return observed


def main() -> None:
    try:
        minimum_memory = int(os.environ.get("SC_MIN_GPU_MEMORY_MIB", "0"))
        observed = validate_gpu(
            os.environ.get("SC_OBSERVED_GPU", ""),
            expected_name=os.environ.get("SC_EXPECTED_GPU_NAME", ""),
            minimum_driver=os.environ.get("SC_MIN_NVIDIA_DRIVER", ""),
            minimum_memory_mib=minimum_memory)
    except (ValueError, PodAdmissionError) as exc:
        raise SystemExit(f"POD ADMISSION REJECTED: {exc}") from exc
    print("POD ADMISSION PASS " + json.dumps(observed, sort_keys=True))


if __name__ == "__main__":
    main()
