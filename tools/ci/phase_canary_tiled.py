"""Temporary CI experiment preserving binaries and profiles for comparison."""

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "diagnostic-results"
OUT.mkdir(exist_ok=True)
BASE = "bdf0a9259dfb80b94bca30e5570e56ad3045179c"
HEAD = "4896012857cb97eda09e3d0d40ce135dbfe96dfa"
CORE = str(min(os.sched_getaffinity(0)))
CASE = "sample_exp_val_20q_200_probes_100000_shots"


def run(args, log=None, env=None, check=True):
    print("RUN", " ".join(map(str, args)), flush=True)
    if log:
        with (OUT / log).open("w") as output:
            return subprocess.run(
                args, stdout=output, stderr=subprocess.STDOUT, env=env, check=check
            )
    return subprocess.run(args, env=env, check=check)


def build(label):
    run(
        [
            "cmake",
            "-S",
            "benchmarks",
            "-B",
            "benchmark-build",
            f"-DCLIFFT_SOURCE_DIR={ROOT / 'benchmark-source'}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
            "-DCLIFFT_OPENMP=ON",
            "-DCLIFFT_CPU_BASELINE=x86-64-v2",
        ],
        label + "-configure.log",
    )
    run(
        ["cmake", "--build", "benchmark-build", "--target", "clifft_benchmarks", "--parallel"],
        label + "-build.log",
    )
    shutil.copy2("benchmark-build/clifft_benchmarks", OUT / label)
    shutil.copy2("benchmark-build/compile_commands.json", OUT / (label + "-commands.json"))
    run(["nm", "-S", "-C", str(OUT / label)], label + ".nm")


def measure(label, suffix, case=CASE, env=None, prefix=(), fixed=False):
    name = label + "-" + suffix
    args = [
        *prefix,
        "taskset",
        "-c",
        CORE,
        str(OUT / label),
        "--benchmark_filter=^" + case + "$",
        "--benchmark_min_time=" + ("20x" if fixed else "0.5s"),
        "--benchmark_min_warmup_time=0.2",
        "--benchmark_repetitions=" + ("1" if fixed else "3"),
        "--benchmark_out=" + str(OUT / (name + ".json")),
    ]
    result = run(args, name + ".log", env=env, check=not prefix)
    if result.returncode == 0:
        rows = json.loads((OUT / (name + ".json")).read_text())["benchmarks"]
        for row in rows:
            if row.get("aggregate_name") == "median" or fixed:
                print(name, row["cpu_time"], flush=True)


run(["lscpu"], "cpu.txt")
run([os.environ["CXX"], "-v", "-x", "c++", "/dev/null", "-fsyntax-only"], "compiler.txt")
run(["git", "worktree", "add", "--detach", "benchmark-source", BASE])
for label, revision in [("base", BASE), ("head", HEAD)]:
    run(["git", "-C", "benchmark-source", "checkout", "--detach", "--force", revision])
    shutil.rmtree("benchmark-build", ignore_errors=True)
    build(label)
    run([str(OUT / label), "--benchmark_list_tests=true"], label + "-benchmarks.txt")
    run(
        [
            "taskset",
            "-c",
            CORE,
            str(OUT / label),
            "--benchmark_filter=.*",
            "--benchmark_min_time=0.1s",
            "--benchmark_min_warmup_time=0.1",
        ],
        label + "-warmup.log",
    )

for case in [CASE]:
    for leg, label in enumerate(["base", "head", "head", "base"]):
        measure(label, str(leg) + "-" + case, case=case)

perf = sorted(Path("/usr/lib/linux-tools").glob("*/perf"))[-1]
for label in ["base", "head"]:
    data = OUT / (label + ".perf")
    measure(
        label,
        "profile",
        prefix=[str(perf), "record", "-q", "-e", "cycles:u", "-F", "499", "-o", str(data), "--"],
        fixed=True,
    )
    run(
        [
            str(perf),
            "report",
            "-i",
            str(data),
            "--stdio",
            "--no-children",
            "--percent-limit",
            "1",
            "--field-separator",
            "|",
        ],
        label + "-profile.txt",
        check=False,
    )
    run([str(perf), "annotate", "-i", str(data), "--stdio"], label + "-annotate.txt", check=False)

for label, revision in [("base", BASE), ("head", HEAD)]:
    run(["git", "-C", "benchmark-source", "checkout", "--detach", "--force", revision])
    run(["git", "-C", "benchmark-source", "apply", str(ROOT / "tools/ci/phase_canary_tiled.patch")])
    # Git restores old commit timestamps, so invalidate objects before rebuilding.
    run(["cmake", "--build", "benchmark-build", "--target", "clean"])
    build(label + "-tiled")
    for leg, binary in enumerate([label, label + "-tiled", label + "-tiled", label]):
        measure(binary, "tiled-comparison-" + str(leg))
    for variant in [label, label + "-tiled"]:
        measure(
            variant,
            "counters",
            prefix=[
                str(perf),
                "stat",
                "-e",
                "cycles:u,instructions:u,cache-misses:u,dTLB-load-misses:u",
                "--",
            ],
            fixed=True,
        )
