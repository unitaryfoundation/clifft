#!/bin/bash
# Compare executor variants at controlled executor addresses and stack
# offsets. Writes everything under diag-out/.
set -uo pipefail

ROOT=${GITHUB_WORKSPACE:-$PWD}
OUT=$ROOT/diag-out
BIN=${BIN:-$ROOT/canary-bin}
STEER=$ROOT/tools/diag554/steer.so
VARIANTS=${VARIANTS:-control local compact}
mkdir -p "$OUT/single"

ZEN3=0
grep -q "EPYC 7763" /proc/cpuinfo && ZEN3=1
EXPERIMENTS=${EXPERIMENTS:-}
if [[ -z $EXPERIMENTS ]]; then
    if ((ZEN3)) || [[ -n ${FORCE_ALL:-} ]]; then
        EXPERIMENTS=${ZEN3_EXPERIMENTS:-env warmup maps natural favunfav sweep stack wide natural2}
    else
        EXPERIMENTS=${OTHER_EXPERIMENTS:-env warmup maps natural favunfav wide natural2}
    fi
fi

CPU=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
export CLIFFT_FORCE_ISA=avx2 OMP_DYNAMIC=FALSE OMP_NUM_THREADS=1
unset GLIBC_TUNABLES
JSON=/tmp/sampling-canary.json
HIGH='^sample_surface_d5_r5_high_noise_10000_shots$'

bench() { # variant filter min_time reps [prefix...]
    local variant=$1 filter=$2 min_time=$3 reps=$4
    shift 4
    rm -f "$JSON"
    "$@" taskset -c "$CPU" "$BIN/$variant" \
        --benchmark_filter="$filter" \
        --benchmark_min_time="${min_time}s" \
        --benchmark_min_warmup_time=0.2 \
        --benchmark_repetitions="$reps" \
        --benchmark_out="$JSON" \
        --benchmark_out_format=json >/dev/null 2>&1
}

# Fixed-width steering environment: the benchmark copies its environment into
# heap buffers, so all steered runs keep identical environment lengths.
steer_env() { # size skip addr
    printf 'LD_PRELOAD=%s STEER_SIZE=%06d STEER_SKIP=%04d STEER_ADDR=%012x' \
        "$STEER" "$1" "$2" "$3"
}

# Same, plus a startup heap shift; used only by the layouts experiment.
pad_env() { # pad bytes
    printf 'LD_PRELOAD=%s STEER_SIZE=%06d STEER_SKIP=%04d STEER_ADDR=%012x STEER_PAD=%06d' \
        "$STEER" 0 0 0x555560000000 "$1"
}

put() { # name variant addr
    # shellcheck disable=SC2046
    bench "$2" "$HIGH" 0.5 3 env $(steer_env 792 0 "$3")
    mv "$JSON" "$OUT/single/$1.json"
}

# Six legs in a balanced order so drift affects every variant equally.
LEGS=""
for v in $VARIANTS; do LEGS="$LEGS $v-first"; done
for v in $(echo "$VARIANTS" | tr ' ' '\n' | tac); do LEGS="$LEGS $v-second"; done
balanced() { # dir filter [prefix...]
    local dir=$1 filter=$2
    shift 2
    mkdir -p "$dir"
    for leg in $LEGS; do
        bench "${leg%-*}" "$filter" 0.5 3 "$@"
        mv "$JSON" "$dir/$leg.json"
    done
}

# libm's log table page determines which executor pages alias it. Read
# libm's base from a steered run and pick the first aliasing page at or above
# 0x5555656a0000 (unused address space above the brk heap); the favorable
# page differs only in address bit 12.
pick_pages() {
    python3 - "$OUT/maps/steer-control.maps" <<'PY'
import re, sys
pairs = [(12, 27), (13, 26), (14, 25), (15, 20), (16, 21), (17, 22), (18, 23), (19, 24)]
tag = lambda va: tuple(((va >> a) ^ (va >> b)) & 1 for a, b in pairs)
libm = min(int(l.split("-")[0], 16) for l in open(sys.argv[1]) if "libm.so" in l)
want = tag(libm + 0xB5000)
page = 0x5555656A0000
while tag(page) != want:
    page += 0x1000
assert tag(page ^ 0x1000) != want
print(f"{libm:#x} {page:#x} {page ^ 0x1000:#x}")
PY
}

for experiment in $EXPERIMENTS; do
    case $experiment in
    env)
        {
            lscpu
            uname -a
            ldd --version | head -1
            cat /sys/kernel/mm/transparent_hugepage/enabled
            echo "cpu $CPU"
            for v in $VARIANTS; do
                echo "$v $(stat -c %s "$BIN/$v") bytes"
            done
        } >"$OUT/env.txt" 2>&1
        ;;
    warmup)
        for v in $VARIANTS; do
            taskset -c "$CPU" "$BIN/$v" --benchmark_filter='.*' --benchmark_min_time=0.1s \
                --benchmark_min_warmup_time=0.1 --benchmark_repetitions=1 >/dev/null 2>&1
        done
        ;;
    maps)
        mkdir -p "$OUT/maps"
        for v in $VARIANTS; do
            # shellcheck disable=SC2046
            env $(steer_env 0 0 0x555560000000) taskset -c "$CPU" "$BIN/$v" \
                --benchmark_filter="$HIGH" --benchmark_min_time=3s >/dev/null 2>&1 &
            sleep 2
            pid=$(pgrep -n -f "$BIN/$v --benchmark_filter")
            cp "/proc/$pid/maps" "$OUT/maps/steer-$v.maps"
            wait
        done
        read -r LIBM BAD GOOD < <(pick_pages)
        echo "libm $LIBM bad_page $BAD good_page $GOOD" | tee "$OUT/pages.txt"
        ;;
    natural) balanced "$OUT/natural" "$HIGH" ;;
    natural2) balanced "$OUT/natural2" "$HIGH" ;;
    favunfav)
        read -r LIBM BAD GOOD < <(pick_pages)
        rounds=4
        ((ZEN3)) || rounds=2
        for i in $(seq 1 $rounds); do
            for v in $VARIANTS; do
                put "fav_${v}_$i" "$v" $((GOOD + 0x820))
                put "unfav_${v}_$i" "$v" $((BAD + 0x820))
            done
        done
        echo "done favunfav"
        ;;
    sweep)
        read -r LIBM BAD GOOD < <(pick_pages)
        for j in $(seq 0 63); do
            for v in $VARIANTS; do
                put "sweep_bad_${v}_$j" "$v" $((BAD + j * 64 + 0x20))
            done
        done
        for j in $(seq 0 8 63); do
            for v in $VARIANTS; do
                put "sweep_good_${v}_$j" "$v" $((GOOD + j * 64 + 0x20))
            done
        done
        echo "done sweep"
        ;;
    stack)
        read -r LIBM BAD GOOD < <(pick_pages)
        for v in $VARIANTS; do
            python3 "$ROOT/tools/diag554/stack_collision.py" "$v" "$BIN/$v" "$STEER" "$CPU" \
                "$OUT" "$(printf '%x' $((GOOD + 0x820)))"
        done
        echo "done stack"
        ;;
    stack9)
        # Hot stack lines on the log constants line (set 9) and on sets 8-9.
        read -r LIBM BAD GOOD < <(pick_pages)
        for off in 240 200; do
            for v in $VARIANTS; do
                python3 "$ROOT/tools/diag554/stack_collision.py" "$v" "$BIN/$v" "$STEER" "$CPU" \
                    "$OUT" "$(printf '%x' $((GOOD + 0x820)))" "$off"
            done
        done
        echo "done stack9"
        ;;
    stacksweep)
        read -r LIBM BAD GOOD < <(pick_pages)
        for v in $VARIANTS; do
            (cd "$ROOT/tools/diag554" && python3 stack_sweep.py "$v" "$BIN/$v" "$STEER" "$CPU" \
                "$OUT" "$(printf '%x' $((GOOD + 0x820)))")
        done
        echo "done stacksweep"
        ;;
    sweep2)
        read -r LIBM BAD GOOD < <(pick_pages)
        for j in $(seq 1 2 63); do
            for v in $VARIANTS; do
                put "sweep_bad_${v}_$j" "$v" $((BAD + j * 64 + 0x20))
            done
        done
        for j in $(seq 4 8 63); do
            for v in $VARIANTS; do
                put "sweep_good_${v}_$j" "$v" $((GOOD + j * 64 + 0x20))
            done
        done
        echo "done sweep2"
        ;;
    layouts)
        # Every sampling benchmark under four heap shifts per variant, so a
        # code change is compared across layouts instead of in one layout.
        for name in $("$BIN/control" --benchmark_list_tests=true | grep '^sample_'); do
            mkdir -p "$OUT/layouts/$name"
            for pad in 0 4152 9000 20000; do
                for v in $VARIANTS; do
                    # shellcheck disable=SC2046
                    bench "$v" "^$name\$" 0.5 3 env $(pad_env "$pad")
                    mv "$JSON" "$OUT/layouts/$name/pad${pad}_$v.json"
                done
            done
        done
        echo "done layouts"
        ;;
    wide)
        for name in $("$BIN/control" --benchmark_list_tests=true | grep '^sample_'); do
            balanced "$OUT/wide/$name" "^$name\$"
        done
        echo "done wide"
        ;;
    esac
done

python3 "$ROOT/tools/diag554/summarize4.py" "$OUT" | tee "$OUT/summary.md"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$OUT/summary.md" >>"$GITHUB_STEP_SUMMARY"
fi
