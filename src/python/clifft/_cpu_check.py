"""Import-time CPU checks for wheel baselines."""

from __future__ import annotations

import platform
from pathlib import Path


def _linux_cpu_flags() -> set[str]:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("flags"):
            _, _, value = line.partition(":")
            return set(value.split())
    return set()


def ensure_supported_cpu(cpu_baseline: str, requires_x86_64_v3_baseline: bool) -> None:
    """Raise ImportError early if this wheel requires x86-64-v3 on Linux."""

    if not requires_x86_64_v3_baseline or cpu_baseline != "x86-64-v3":
        return
    if platform.system() != "Linux":
        return
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return

    flags = _linux_cpu_flags()
    required = {"avx2", "bmi2", "fma"}
    missing = sorted(required - flags)
    if not missing:
        return

    missing_str = ", ".join(missing)
    raise ImportError(
        "This Clifft wheel requires an x86-64-v3 CPU baseline (AVX2, BMI2, FMA). "
        f"Missing CPU flags: {missing_str}. Install from source with "
        "'pip install --no-binary clifft clifft' on older x86_64 machines."
    )
