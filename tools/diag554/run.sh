#!/bin/bash
# Temporary diagnosis for issue 554. Runs the saved canary binaries under
# controlled variations and writes everything under diag-out/.
set -uo pipefail

ROOT=${GITHUB_WORKSPACE:-$PWD}
OUT=$ROOT/diag-out
BIN=$ROOT/saved
SHIM=$ROOT/tools/diag554/libmtrace.so
EXPERIMENTS=${EXPERIMENTS:-env warmup normal envpad libpad normal2}
mkdir -p "$OUT"

CPU=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
export CLIFFT_FORCE_ISA=avx2 OMP_DYNAMIC=FALSE OMP_NUM_THREADS=1
unset GLIBC_TUNABLES
FILTER='^sample_surface_d5_r5_high_noise_10000_shots$'
JSON=/tmp/sampling-canary.json
ARCH=$(uname -m)

# One benchmark process. Extra leading arguments are a command prefix
# (env assignments or setarch); the benchmark command line is identical
# across revisions.
bench() {
    local rev=$1 min_time=$2 reps=$3
    shift 3
    "$@" taskset -c "$CPU" "$BIN/$rev" \
        --benchmark_filter="$FILTER" \
        --benchmark_min_time="${min_time}s" \
        --benchmark_min_warmup_time=0.2 \
        --benchmark_repetitions="$reps" \
        --benchmark_out="$JSON" \
        --benchmark_out_format=json >/dev/null 2>&1
}

abba() {
    local name=$1
    shift
    mkdir -p "$OUT/$name"
    for leg in base-first head-first head-second base-second; do
        local rev=${leg%-*}
        rm -f "$JSON"
        bench "$rev" 0.5 3 "$@"
        mv "$JSON" "$OUT/$name/$leg.json"
    done
    echo "done $name"
}

for experiment in $EXPERIMENTS; do
    case $experiment in
    env)
        {
            lscpu
            uname -a
            ldd --version | head -1
            echo "aslr $(cat /proc/sys/kernel/randomize_va_space)"
            for f in /sys/kernel/mm/transparent_hugepage/enabled \
                /sys/kernel/mm/transparent_hugepage/defrag \
                /sys/kernel/mm/transparent_hugepage/shmem_enabled \
                /sys/kernel/mm/transparent_hugepage/khugepaged/defrag; do
                echo "$f: $(cat "$f" 2>&1)"
            done
            grep -i huge /proc/meminfo
            echo "cpu $CPU"
            cat /proc/self/status | grep -i cpus_allowed_list
        } >"$OUT/env.txt" 2>&1
        ;;
    warmup)
        for rev in base head; do
            taskset -c "$CPU" "$BIN/$rev" --benchmark_filter='.*' \
                --benchmark_min_time=0.1s --benchmark_min_warmup_time=0.1 \
                --benchmark_repetitions=1 >/dev/null 2>&1
        done
        ;;
    normal) abba normal ;;
    normal2) abba normal2 ;;
    preload) abba preload env LD_PRELOAD="$SHIM" MTRACE_DISABLE=1 ;;
    trace)
        # Same environment strings in every leg; the trace prefix is renamed
        # after each process exits.
        mkdir -p "$OUT/trace" "$OUT/traces"
        for leg in base-first head-first head-second base-second; do
            rev=${leg%-*}
            rm -f "$JSON" /tmp/mtrace-run.bin /tmp/mtrace-run.maps
            bench "$rev" 0.5 3 env LD_PRELOAD="$SHIM" MTRACE_OUT=/tmp/mtrace-run
            mv "$JSON" "$OUT/trace/$leg.json"
            mv /tmp/mtrace-run.bin "$OUT/traces/timed-$leg.bin"
            mv /tmp/mtrace-run.maps "$OUT/traces/timed-$leg.maps"
        done
        echo "done trace"
        ;;
    aslr_off) abba aslr_off setarch "$ARCH" -R ;;
    tcache_off) abba tcache_off env GLIBC_TUNABLES=glibc.malloc.tcache_count=0 ;;
    procs)
        # Many short fresh processes: a placement that depends on physical
        # pages or ASLR shows up as spread between processes.
        mkdir -p "$OUT/procs"
        for i in 1 2 3 4 5 6 7 8; do
            for rev in base head; do
                rm -f "$JSON"
                bench "$rev" 0.5 1
                mv "$JSON" "$OUT/procs/$rev-$i.json"
            done
        done
        echo "done procs"
        ;;
    smaps)
        mkdir -p "$OUT/smaps"
        for rev in base head; do
            rm -f "$JSON"
            bench "$rev" 4 1 &
            sleep 2.5
            pid=$(pgrep -n -f "$BIN/$rev --benchmark_filter")
            cp "/proc/$pid/smaps" "$OUT/smaps/$rev.smaps" 2>/dev/null
            cp "/proc/$pid/maps" "$OUT/smaps/$rev.maps" 2>/dev/null
            wait
        done
        echo "done smaps"
        ;;
    bt)
        # Backtrace-labelled traces for identifying allocations. Not timed.
        mkdir -p "$OUT/traces"
        for rev in base head; do
            rm -f "$JSON" /tmp/mtrace-run.bin /tmp/mtrace-run.maps
            bench "$rev" 0.5 3 env LD_PRELOAD="$SHIM" MTRACE_OUT=/tmp/mtrace-run MTRACE_BT=1
            mv /tmp/mtrace-run.bin "$OUT/traces/bt-$rev.bin"
            mv /tmp/mtrace-run.maps "$OUT/traces/bt-$rev.maps"
        done
        echo "done bt"
        ;;
    pad)
        # Leak one block in the shim constructor. A request of N bytes
        # occupies an N + 8 byte chunk, so later top-carved chunks shift by
        # that amount while their relative layout stays the same.
        for pad in ${PADS:-24 56 120 248 504 1016 2040 3064}; do
            abba "pad_$((pad + 8))" env LD_PRELOAD="$SHIM" MTRACE_PAD_SIZE="$pad"
        done
        ;;
    envpad)
        # Shift only the initial stack: one extra environment string, no
        # preload, so heap and mmap layout are unchanged.
        for n in ${ENVPADS:-1 9 25 57 121 249 505 1017 2041 3065}; do
            abba "envpad_$n" env DIAG_ENV_PAD="$(head -c "$n" /dev/zero | tr '\\0' x)"
        done
        ;;
    libpad)
        # Shift only the mmap region (shared libraries, TLS, large chunks) by
        # preloading an empty library with a sized .bss. Paths have equal
        # length so the stack is identical across this sweep.
        mkdir -p /tmp/p "$OUT/libpad-maps"
        for pages in ${LIBPADS:-0 1 2 3 4 8 16 32 57 64 128 256 1024 4096 65536}; do
            name=$(printf '/tmp/p/pad_%06d.so' "$pages")
            echo "char diag_pad_bss[$((pages * 4096 + 1))];" >/tmp/p/pad.c
            gcc -O2 -fPIC -shared -o "$name" /tmp/p/pad.c
            LD_PRELOAD="$name" cat /proc/self/maps >"$OUT/libpad-maps/$pages.maps"
            abba "libpad_$pages" env LD_PRELOAD="$name"
        done
        LD_PRELOAD= cat /proc/self/maps >"$OUT/libpad-maps/none.maps"
        ;;
    esac
done

if compgen -G "$OUT/traces/*.bin" >/dev/null; then
    zstd -q --rm -T0 "$OUT"/traces/*.bin
fi

python3 "$ROOT/tools/diag554/summarize.py" "$OUT" | tee "$OUT/summary.md"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$OUT/summary.md" >>"$GITHUB_STEP_SUMMARY"
fi
