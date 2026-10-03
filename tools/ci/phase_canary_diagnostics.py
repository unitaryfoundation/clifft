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

for case in [CASE, "sample_early_rejection_1000_shots", "squeeze_parallel_t_8192"]:
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

# A single diagnostic binary selects the old or new pipeline at runtime.
harness = ROOT / "benchmarks/clifft_benchmarks.cc"
text = harness.read_text().replace(
    '#if __has_include("clifft/optimizer/phase_polynomial_pass.h")', "#if 0"
)
text = text.replace("#include <array>", "#include <array>\n#include <cstdlib>")
text = text.replace("default_hir_pass_manager().run(hir);", "diagnostic_pass_manager().run(hir);")
marker = "sampling::ExecutablePlan compile_parsed(Circuit circuit) {"
helper = """HirPassManager diagnostic_pass_manager() {
    if (std::getenv("CLIFFT_DIAG_SKIP_PHASE") == nullptr) {
        return default_hir_pass_manager();
    }
    HirPassManager pm;
    pm.add_pass(std::make_unique<PeepholeFusionPass>());
    pm.add_pass(std::make_unique<StatevectorSqueezePass>());
    return pm;
}

"""
text = text.replace(marker, helper + marker)
text = text.replace(
    "return sampling::ExecutablePlan(sampling::plan_sampling(hir));",
    """auto plan = sampling::ExecutablePlan(sampling::plan_sampling(hir));
    if (const char* path = std::getenv("CLIFFT_DIAG_PLAN")) {
        std::ofstream(path) << plan.inspect();
    }
    return plan;""",
)
text = text.replace(
    "sampling::sample(plan, 100000, 0)",
    "sampling::sample(plan, 100000, 0, 1, std::nullopt, "
    'std::getenv("CLIFFT_DIAG_BATCH") ? std::optional<uint32_t>{'
    'static_cast<uint32_t>(std::atoi(std::getenv("CLIFFT_DIAG_BATCH")))} : std::nullopt)',
)
harness.write_text(text)
build("head-diagnostic")
for leg, skip in enumerate([False, True, True, False]):
    env = dict(os.environ)
    if skip:
        env["CLIFFT_DIAG_SKIP_PHASE"] = "1"
    measure("head-diagnostic", "skip-" + str(skip) + "-" + str(leg), env=env)
for batch in ["1", "1024", "2047", "2048", "2049"]:
    measure("head-diagnostic", "batch-" + batch, env=dict(os.environ, CLIFFT_DIAG_BATCH=batch))

for skip in [False, True]:
    env = dict(os.environ)
    if skip:
        env["CLIFFT_DIAG_SKIP_PHASE"] = "1"
    env["CLIFFT_DIAG_PLAN"] = str(OUT / ("plan-old.txt" if skip else "plan-new.txt"))
    measure("head-diagnostic", "plan-" + str(skip), env=env, fixed=True)
