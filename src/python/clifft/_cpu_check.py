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


def _missing_flag_groups(flags: set[str], required: list[set[str]]) -> list[str]:
    return ["|".join(sorted(group)) for group in required if not (group & flags)]


def ensure_supported_cpu(cpu_baseline: str) -> None:
    """Raise ImportError early if an x86_64 wheel baseline is unsupported."""

    required_by_baseline = {
        "x86-64-v2": [
            {"cx16"},
            {"lahf_lm"},
            {"popcnt"},
            {"pni", "sse3"},
            {"ssse3"},
            {"sse4_1"},
            {"sse4_2"},
        ],
        "x86-64-v3": [
            {"cx16"},
            {"lahf_lm"},
            {"popcnt"},
            {"pni", "sse3"},
            {"ssse3"},
            {"sse4_1"},
            {"sse4_2"},
            {"avx"},
            {"avx2"},
            {"bmi1"},
            {"bmi2"},
            {"f16c"},
            {"fma"},
            {"lzcnt", "abm"},
            {"movbe"},
            {"xsave"},
        ],
    }
    required = required_by_baseline.get(cpu_baseline)
    if required is None:
        return
    if platform.system() != "Linux":
        return
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return

    flags = _linux_cpu_flags()
    missing = _missing_flag_groups(flags, required)
    if not missing:
        return

    missing_str = ", ".join(missing)
    raise ImportError(
        f"This Clifft wheel requires an {cpu_baseline} CPU baseline. "
        f"Missing CPU flags: {missing_str}. Install from source with "
        "'pip install --no-binary clifft clifft' on older x86_64 machines."
    )
