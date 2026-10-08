#!/bin/bash
# Temporary diagnosis for issue 554. Runs the saved canary binaries under
# controlled variations and writes everything under diag-out/.
set -uo pipefail

ROOT=${GITHUB_WORKSPACE:-$PWD}
OUT=$ROOT/diag-out
BIN=$ROOT/saved
SHIM=$ROOT/tools/diag554/libmtrace.so
STEER=$ROOT/tools/diag554/steer.so
EXPERIMENTS=${EXPERIMENTS:-env warmup normal maps rusage steer_ctrl steer objects memhog normal2}
mkdir -p "$OUT"
# The steering sweeps only matter on the CPU that shows the slowdown.
if ! grep -q "EPYC 7763" /proc/cpuinfo && [[ -z "${FORCE_ALL:-}" ]]; then
    EXPERIMENTS="env warmup normal maps steer_ctrl"
fi

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

# One process of one revision; records the per-repetition times.
single() {
    local name=$1 rev=$2
    shift 2
    mkdir -p "$OUT/single"
    rm -f "$JSON"
    bench "$rev" 0.5 3 "$@"
    mv "$JSON" "$OUT/single/$name.json"
}

# Steering environment with fixed-width values: the benchmark copies its
# environment into heap buffers, so every steering run and its control must
# have environment strings of identical length.
steer_env() { # size skip addr
    printf 'LD_PRELOAD=%s STEER_SIZE=%06d STEER_SKIP=%04d STEER_ADDR=%012x' \
        "$STEER" "$1" "$2" "$3"
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
    maps)
        # Library placement in the benchmark process itself (it re-execs
        # with ASLR disabled, so a child of the shell is not representative).
        mkdir -p "$OUT/maps"
        for cfg in none steer; do
            for rev in base head; do
                rm -f "$JSON"
                if [[ $cfg == steer ]]; then
                    bench "$rev" 3 1 env LD_PRELOAD="$STEER" &
                else
                    bench "$rev" 3 1 &
                fi
                sleep 2
                pid=$(pgrep -n -f "$BIN/$rev --benchmark_filter")
                cp "/proc/$pid/maps" "$OUT/maps/$cfg-$rev.maps"
                wait
            done
        done
        cp /usr/lib/x86_64-linux-gnu/libm.so.6 /usr/lib/x86_64-linux-gnu/libc.so.6 "$OUT/maps/"
        echo "done maps"
        ;;
    steer_ctrl)
        # shellcheck disable=SC2046
        abba steer_ctrl env $(steer_env 0 0 0x555560000000)
        ;;
    rusage)
        # User/system time and page faults of whole benchmark processes.
        mkdir -p "$OUT/rusage"
        for i in 1 2; do
            for rev in base head; do
                python3 - "$CPU" "$BIN/$rev" "$FILTER" >>"$OUT/rusage/rusage.txt" <<'PY'
import resource, subprocess, sys
cpu, exe, flt = sys.argv[1:4]
subprocess.run(["taskset", "-c", cpu, exe, f"--benchmark_filter={flt}",
                "--benchmark_min_time=0.5s", "--benchmark_min_warmup_time=0.2",
                "--benchmark_repetitions=3"], stdout=subprocess.DEVNULL, check=True)
r = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"{exe.rsplit('/', 1)[1]} utime {r.ru_utime:.3f} stime {r.ru_stime:.3f} "
      f"minflt {r.ru_minflt} majflt {r.ru_majflt} nivcsw {r.ru_nivcsw}")
PY
            done
        done
        cat "$OUT/rusage/rusage.txt"
        ;;
    memhog)
        # Hold 256 MiB of touched anonymous memory in another process on the
        # same CPU so the benchmark's anonymous pages come from elsewhere.
        python3 -c 'import time; b = bytearray(256 << 20); [b.__setitem__(i, 1) for i in range(0, len(b), 4096)]; time.sleep(600)' &
        hog=$!
        sleep 3
        abba memhog
        kill "$hog"
        wait "$hog" 2>/dev/null
        ;;
    objects)
        FAR=0x555560000000
        # Relocate one plan table at a time (matched by size and occurrence)
        # keeping its low 28 address bits, or flipping bit 12.
        while read -r size skip addr; do
            same=$((FAR | (addr & 0xfffffff)))
            flip=$((FAR | ((addr ^ 0x1000) & 0xfffffff)))
            for kind in same flip; do
                target=$same
                [[ $kind == flip ]] && target=$flip
                # shellcheck disable=SC2046
                single "obj_${size}_$kind" head env $(steer_env "$size" "$skip" "$target")
            done
        done <<'LIST'
162964 13 0x55555589f7a0
131072 6 0x555555814d80
65536 13 0x55555572bdb0
62448 6 0x5555558fe810
56392 20 0x555555804d70
32760 6 0x5555557b88d0
28976 12 0x5555557d1fa0
28972 12 0x5555557cae60
15344 6 0x555555858520
7672 6 0x5555557591c0
2048 137 0x555555797370
1176 35 0x55555573c360
LIST
        # shellcheck disable=SC2046
        single ctrl_head_obj head env $(steer_env 0 0 "$FAR")
        echo "done objects"
        ;;
    steer)
        # Relocate only the 792-byte sampling worker. Head's natural address
        # is 0x5555556a0820 and base's is 0x555555699820; FAR keeps their low
        # 28 bits in an unused region.
        FAR=0x555560000000
        HEAD28=$((FAR | (0x5555556a0820 & 0xfffffff)))
        BASE28=$((FAR | (0x555555699820 & 0xfffffff)))
        put() { # name rev addr
            # shellcheck disable=SC2046
            single "$1" "$2" env $(steer_env 792 0 "$3")
        }
        control() {
            # shellcheck disable=SC2046
            single "ctrl_head_$1" head env $(steer_env 0 0 "$FAR")
            # shellcheck disable=SC2046
            single "ctrl_base_$1" base env $(steer_env 0 0 "$FAR")
        }
        control a
        put head_at_head28 head "$HEAD28"
        put base_at_head28 base "$HEAD28"
        put base_at_base28 base "$BASE28"
        put head_at_base28 head "$BASE28"
        put head_far head $((FAR + 0x1000100))
        control b
        for b in $(seq 6 27); do
            put "head_flip_$b" head $((HEAD28 ^ (1 << b)))
        done
        control c
        for pair in 12:27 13:26 14:25 15:20 16:21 17:22 18:23 19:24 12:13 20:21; do
            x=${pair%:*}
            y=${pair#*:}
            put "head_flip_${x}_$y" head $((HEAD28 ^ (1 << x) ^ (1 << y)))
        done
        control d
        for j in $(seq 0 63); do
            put "head_line_$j" head $(((HEAD28 & ~0xfff) + j * 64 + 0x20))
            if ((j % 16 == 15)); then control "e$j"; fi
        done
        for j in $(seq 0 63); do
            put "base_line_$j" base $(((HEAD28 & ~0xfff) + j * 64 + 0x20))
        done
        control f
        echo "done steer"
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
