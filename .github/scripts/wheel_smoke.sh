#!/usr/bin/env bash
# Emulated-CPU wheel smoke for clifft.
#
# Runs a tiny clifft program under an x86 emulator to verify that the
# installed wheel/extension dispatches correctly across CPU generations.
# Catches the "AVX-512 leaks into the AVX-2 dispatch code path" class of
# bug deterministically (it would SIGILL the emulated CPU) and validates
# that `CLIFFT_FORCE_ISA` traps fire cleanly on incompatible hosts.
#
# Two emulators are supported:
#   qemu - `qemu-x86_64 -cpu <model>` (QEMU user-mode, TCG). Covers CPUs at
#          or below the AVX2 baseline, including the scalar and clean-trap
#          fallback paths.
#   sde  - Intel Software Development Emulator. QEMU's TCG engine cannot
#          execute AVX-512 instructions, so SDE is what makes AVX-512
#          kernel execution deterministic in CI rather than dependent on
#          the runner's host CPU.
#
# Usage:
#   wheel_smoke.sh <emulator:qemu|sde> <cpu_model> <force_isa|auto> <expected:pass|fail> [expected_isa]
#
# Examples:
#   wheel_smoke.sh qemu Haswell auto   pass  avx2    # auto-detect picks avx2
#   wheel_smoke.sh qemu Haswell avx2   pass  avx2    # force avx2 explicitly
#   wheel_smoke.sh qemu Haswell avx512 fail          # host lacks avx512 -> clean trap
#   wheel_smoke.sh qemu Nehalem auto   pass  scalar  # auto-detect picks scalar
#   wheel_smoke.sh qemu Nehalem avx2   fail          # host lacks avx2 -> clean trap
#   wheel_smoke.sh sde  skx     auto   pass  avx512  # SDE Skylake-X auto-detect
#   wheel_smoke.sh sde  skx     avx512 pass  avx512  # SDE force avx512 explicitly
#
# Requires:
#   - qemu mode: qemu-x86_64 in PATH.
#   - sde mode: the SDE64 env var pointing at the sde64 binary, or sde64 in PATH.
#   - PYTHON env var (or `python` in PATH) pointing at an interpreter
#     with clifft installed.
#
# A "fail" expectation requires:
#   - nonzero exit code (but not 132/134 which indicate SIGILL/SIGABRT)
#   - stderr mentions CLIFFT_FORCE_ISA or RuntimeError (clean trap)
#   - stderr does NOT mention "Illegal instruction" or "SIGILL"
#
# When expected_isa is given on a "pass" leg, the smoke output must also
# report that resolved ISA (via clifft.runtime_isa()).

set -uo pipefail

if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    echo "usage: $0 <emulator:qemu|sde> <cpu_model> <force_isa|auto> <expected:pass|fail> [expected_isa]" >&2
    exit 2
fi

emulator="$1"
cpu_model="$2"
force_isa="$3"
expected="$4"
expected_isa="${5:-}"

case "$emulator" in
    qemu|sde) ;;
    *)
        echo "error: emulator must be 'qemu' or 'sde', got '$emulator'" >&2
        exit 2
        ;;
esac

case "$expected" in
    pass|fail) ;;
    *)
        echo "error: expected must be 'pass' or 'fail', got '$expected'" >&2
        exit 2
        ;;
esac

PYTHON="${PYTHON:-python3}"

if [ "$emulator" = "qemu" ] && ! command -v qemu-x86_64 >/dev/null 2>&1; then
    echo "error: qemu-x86_64 not found in PATH" >&2
    exit 2
fi
if [ "$emulator" = "sde" ] && ! command -v "${SDE64:-sde64}" >/dev/null 2>&1; then
    echo "error: ${SDE64:-sde64} not found" >&2
    exit 2
fi

script='from pathlib import Path
import clifft
print(f"version={clifft.__version__}  baseline={clifft.CPU_BASELINE}", flush=True)
print(f"isa={clifft.runtime_isa()}", flush=True)
symbolic = clifft.compile(Path("tests/fixtures/qv10.stim").read_text())
result = clifft.sample(symbolic, shots=1, seed=280)
assert result.measurements.shape[0] == 1, result.measurements.shape
prog = clifft.compile("H 0\nCX 0 1\nM 0 1")
ps = clifft.record_probabilities(prog, ["00", "11"])
assert abs(float(ps[0]) - 0.5) < 1e-12 and abs(float(ps[1]) - 0.5) < 1e-12, ps
print("smoke ok", flush=True)
'

echo "==> wheel_smoke: emulator=$emulator cpu=$cpu_model force=$force_isa expected=$expected"

if [ "$emulator" = "qemu" ]; then
    # Forward CLIFFT_FORCE_ISA explicitly via `-E` so the smoke does not
    # depend on whether qemu-user inherits the caller's environment.
    qemu_env=()
    if [ "$force_isa" != "auto" ]; then
        qemu_env+=(-E "CLIFFT_FORCE_ISA=$force_isa")
    fi
    output=$(qemu-x86_64 -cpu "$cpu_model" "${qemu_env[@]}" "$PYTHON" -c "$script" 2>&1)
    exit_code=$?
else
    # SDE children inherit the environment, so export CLIFFT_FORCE_ISA
    # here instead of passing it through an emulator-specific flag.
    if [ "$force_isa" != "auto" ]; then
        export CLIFFT_FORCE_ISA="$force_isa"
    fi
    output=$("${SDE64:-sde64}" -"$cpu_model" -- "$PYTHON" -c "$script" 2>&1)
    exit_code=$?
fi
echo "$output"
echo "==> exit_code=$exit_code"

if [ "$expected" = "pass" ]; then
    if [ "$exit_code" -ne 0 ]; then
        echo "FAIL: expected pass but got exit code $exit_code" >&2
        if [ "$emulator" = "qemu" ]; then
            echo "available qemu cpus (truncated):" >&2
            qemu-x86_64 -cpu help 2>&1 | head -30 >&2
        fi
        exit 1
    fi
    if ! grep -q "smoke ok" <<<"$output"; then
        echo "FAIL: smoke did not print 'smoke ok'" >&2
        exit 1
    fi
    if [ -n "$expected_isa" ] && ! grep -qx "isa=$expected_isa" <<<"$output"; then
        echo "FAIL: expected isa=$expected_isa but it was not reported" >&2
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
