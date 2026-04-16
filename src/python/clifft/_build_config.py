"""Source-tree fallback for build-time configuration.

Installed packages overwrite this module with a generated file that reflects
the actual wheel or source-build CPU baseline.
"""

CPU_BASELINE = "native"
REQUIRES_X86_64_V3_BASELINE = False
