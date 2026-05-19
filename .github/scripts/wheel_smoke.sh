#!/usr/bin/env bash
# QEMU-user wheel smoke for clifft.
#
# Runs a tiny clifft program through `qemu-x86_64 -cpu <model>` to verify
# that the installed wheel/extension dispatches correctly on CPUs below
# the v0.4.x x86-64-v3 path. Catches the "AVX-512 leaks into the AVX-2
# dispatch code path" class of bug deterministically (it would SIGILL
# the emulated CPU) and validates that `CLIFFT_FORCE_ISA` traps fire
# cleanly on incompatible hosts.
#
# Usage:
#   wheel_smoke.sh <cpu_model> <force_isa|auto> <expected:pass|fail>
#
# Examples:
#   wheel_smoke.sh Haswell auto    pass   # auto-detect picks avx2
#   wheel_smoke.sh Haswell avx2    pass   # force avx2 explicitly
#   wheel_smoke.sh Haswell avx512  fail   # host lacks avx512 -> clean trap
#   wheel_smoke.sh Nehalem auto    pass   # auto-detect picks scalar
#   wheel_smoke.sh Nehalem avx2    fail   # host lacks avx2 -> clean trap
#
# Requires:
#   - qemu-x86_64 in PATH
#   - PYTHON env var (or `python` in PATH) pointing at an interpreter
#     with clifft installed.
#
# A "fail" expectation requires:
#   - nonzero exit code (but not 132/134 which indicate SIGILL/SIGABRT)
#   - stderr mentions CLIFFT_FORCE_ISA or RuntimeError (clean trap)
#   - stderr does NOT mention "Illegal instruction" or "SIGILL"

set -uo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <cpu_model> <force_isa|auto> <expected:pass|fail>" >&2
    exit 2
fi

cpu_model="$1"
force_isa="$2"
expected="$3"

case "$expected" in
    pass|fail) ;;
    *)
        echo "error: expected must be 'pass' or 'fail', got '$expected'" >&2
        exit 2
        ;;
esac

PYTHON="${PYTHON:-python3}"
if ! command -v qemu-x86_64 >/dev/null 2>&1; then
    echo "error: qemu-x86_64 not found in PATH" >&2
    exit 2
fi

script='import clifft
print(f"version={clifft.__version__}  baseline={clifft.CPU_BASELINE}  backend={clifft.svm_backend()}", flush=True)
prog = clifft.compile("H 0\nCX 0 1\nM 0 1")
ps = clifft.record_probabilities(prog, ["00", "11"])
assert abs(float(ps[0]) - 0.5) < 1e-12 and abs(float(ps[1]) - 0.5) < 1e-12, ps
print("smoke ok", flush=True)
'

echo "==> wheel_smoke: cpu=$cpu_model force=$force_isa expected=$expected"

# Forward CLIFFT_FORCE_ISA explicitly via `-E` so the smoke does not depend
# on whether qemu-user inherits the caller's environment.
qemu_env=()
if [ "$force_isa" != "auto" ]; then
    qemu_env+=(-E "CLIFFT_FORCE_ISA=$force_isa")
fi

output=$(qemu-x86_64 -cpu "$cpu_model" "${qemu_env[@]}" "$PYTHON" -c "$script" 2>&1)
exit_code=$?
echo "$output"
echo "==> exit_code=$exit_code"

if [ "$expected" = "pass" ]; then
    if [ "$exit_code" -ne 0 ]; then
        echo "FAIL: expected pass but got exit code $exit_code" >&2
        echo "available qemu cpus (truncated):" >&2
        qemu-x86_64 -cpu help 2>&1 | head -30 >&2
        exit 1
    fi
    if ! grep -q "smoke ok" <<<"$output"; then
        echo "FAIL: smoke did not print 'smoke ok'" >&2
        exit 1
    fi
    echo "==> PASS (expected pass, smoke succeeded)"
    exit 0
fi

# expected == "fail": demand a clean trap, not a SIGILL.
if [ "$exit_code" -eq 0 ]; then
    echo "FAIL: expected failure but smoke succeeded" >&2
    exit 1
fi
# Shells report SIGILL as 132 and SIGABRT as 134; both indicate a bad
# instruction or hard abort rather than a Python-level error.
if [ "$exit_code" -eq 132 ] || [ "$exit_code" -eq 134 ]; then
    echo "FAIL: got signal-like exit code $exit_code (likely SIGILL/SIGABRT, not a clean trap)" >&2
    exit 1
fi
if grep -qiE 'illegal instruction|sigill' <<<"$output"; then
    echo "FAIL: output mentions Illegal instruction / SIGILL -- not a clean trap" >&2
    exit 1
fi
if ! grep -qE 'CLIFFT_FORCE_ISA|RuntimeError' <<<"$output"; then
    echo "FAIL: failure output does not mention CLIFFT_FORCE_ISA or RuntimeError" >&2
    exit 1
fi

echo "==> PASS (expected fail, smoke raised a clean trap)"
