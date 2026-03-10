"""Standalone benchmark: UCC vs Qiskit-Aer.

Generates circuits in Qiskit, converts to Stim format for UCC,
runs both simulators across two parameter sweeps, and plots results.

Usage:
    python run_benchmark.py
    python run_benchmark.py --output benchmark_comparison.png
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]

TIMEOUT_S = 120
MEM_LIMIT_GB = 6.5
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
CSV_HEADER = ["Sweep", "N", "t", "k", "Tool", "Status", "Exec_s", "PeakMem_MB"]

QUBIT_N_VALUES = [16, 20, 24, 26, 28, 29]
RANK_K_VALUES = [8, 12, 16, 20, 22, 24, 25]


# ---------------------------------------------------------------------------
# Circuit generation (Qiskit) and QASM-to-Stim conversion
# ---------------------------------------------------------------------------


def build_circuit_qiskit(n: int, t: int, k: int) -> str:
    """Build a benchmark circuit in Qiskit, return QASM 2.0 string.

    The circuit places T-gates (non-Clifford) on k active qubits and
    pads the remaining N-k qubits with Cliffords.  A dense-state
    simulator must track 2^N amplitudes, but UCC's factored-state
    representation only needs 2^k.
    """
    from qiskit import QuantumCircuit

    if n < 1 or k < 1:
        raise ValueError(f"n and k must be >= 1, got n={n}, k={k}")

    actual_k = min(k, n)
    qc = QuantumCircuit(n, n)

    # Superposition on the active core
    for i in range(actual_k):
        qc.h(i)

    # T-gates interleaved with Cliffords on the active core
    for i in range(t):
        tgt = i % actual_k
        qc.t(tgt)
        qc.h(tgt)
        if actual_k > 1:
            nxt = (tgt + 1) % actual_k
            qc.cx(tgt, nxt)

    # Clifford padding on spectator qubits
    for i in range(actual_k, n):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    # Measure all
    qc.measure(range(n), range(n))

    from qiskit.qasm2 import dumps

    return str(dumps(qc))


def qasm_to_stim(qasm: str) -> str:
    """Convert an OpenQASM 2.0 string to Stim circuit format.

    Handles the small gate set used by our benchmark circuits:
    h, t, cx, and measure.
    """
    lines: list[str] = []
    for raw in qasm.splitlines():
        line = raw.strip().rstrip(";")
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg", "//")):
            continue

        # h q[3];
        m = re.match(r"^h\s+q\[(\d+)\]$", line)
        if m:
            lines.append(f"H {m.group(1)}")
            continue

        # t q[3];
        m = re.match(r"^t\s+q\[(\d+)\]$", line)
        if m:
            lines.append(f"T {m.group(1)}")
            continue

        # cx q[0],q[1];
        m = re.match(r"^cx\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]$", line)
        if m:
            lines.append(f"CX {m.group(1)} {m.group(2)}")
            continue

        # measure q[0] -> c[0];
        m = re.match(r"^measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]$", line)
        if m:
            lines.append(f"M {m.group(1)}")
            continue

        # Fail fast on unsupported instructions
        raise ValueError(f"Unsupported QASM instruction: {line!r}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Subprocess isolation for clean timing / memory measurement
# ---------------------------------------------------------------------------


def run_worker(tool: str, qasm_str: str, stim_str: str) -> dict[str, Any]:
    """Run a single simulation in an isolated subprocess."""
    import tempfile

    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as qf,
        tempfile.NamedTemporaryFile(mode="w", suffix=".stim", delete=False) as sf,
    ):
        qf.write(qasm_str)
        sf.write(stim_str)
        qasm_path = qf.name
        stim_path = sf.name

    try:
        cmd = [
            sys.executable,
            __file__,
            "--internal-worker",
            tool,
            qasm_path,
            stim_path,
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["UCC_BENCH_MEM_LIMIT_GB"] = str(MEM_LIMIT_GB)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=env)

        if result.returncode in (-9, 137) or any(
            s in result.stderr for s in ("MemoryError", "bad_alloc")
        ):
            return {"status": "OOM", "exec_s": 0.0, "peak_mb": 0.0}

        if result.returncode != 0:
            return {
                "status": "ERROR",
                "exec_s": 0.0,
                "peak_mb": 0.0,
                "msg": result.stderr.strip()[:200],
            }

        out_lines = result.stdout.strip().split("\n")
        return json.loads(out_lines[-1])  # type: ignore[no-any-return]

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "exec_s": float(TIMEOUT_S), "peak_mb": 0.0}
    except (json.JSONDecodeError, IndexError):
        return {"status": "ERROR", "exec_s": 0.0, "peak_mb": 0.0}
    finally:
        os.unlink(qasm_path)
        os.unlink(stim_path)


def execute_internal(tool: str, qasm_path: str, stim_path: str) -> None:
    """Payload executed inside the isolated subprocess."""
    if resource is not None and sys.platform.startswith("linux"):
        mem_str = os.environ.get("UCC_BENCH_MEM_LIMIT_GB")
        if mem_str:
            limit = int(float(mem_str) * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    start = time.perf_counter()

    if tool == "qiskit":
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit.from_qasm_file(qasm_path)
        sim = AerSimulator(method="statevector", max_parallel_threads=1)
        sim.run(transpile(qc, sim), shots=1).result()

    elif tool == "ucc":
        import ucc

        with open(stim_path) as f:
            program = ucc.compile(f.read())
        ucc.sample(program, shots=1)

    else:
        print(f"Unknown tool: {tool}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - start

    if resource is not None:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak_kb / 1024 if sys.platform.startswith("linux") else peak_kb / (1024 * 1024)
    else:
        peak_mb = 0.0

    print(json.dumps({"status": "SUCCESS", "exec_s": elapsed, "peak_mb": peak_mb}))
    sys.exit(0)


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------


def check_theoretical_oom(tool: str, n: int, k: int, max_ram_gb: float = 7.0) -> bool:
    """Skip simulations that mathematically exceed available RAM."""
    headroom = max(max_ram_gb - 2, 0)
    max_elems = int((headroom * 1024**3) // 16)
    if tool == "qiskit":
        return bool(2**n > max_elems)
    return bool(2**k > max_elems)


def run_sweeps() -> str:
    """Run both sweeps and write CSV. Returns path to CSV."""
    rows: list[list[Any]] = []

    # Sweep A: qubit scaling
    n_values = QUBIT_N_VALUES
    t_a, k_a = 20, 12
    for n in n_values:
        for tool in ["ucc", "qiskit"]:
            label = f"[qubit_scaling] {tool:<7} N={n}"
            print(f"{label} -> ", end="", flush=True)

            if check_theoretical_oom(tool, n, k_a):
                print("SKIPPED (OOM)")
                rows.append(["qubit_scaling", n, t_a, k_a, tool, "OOM", 0.0, 0.0])
                continue

            qasm = build_circuit_qiskit(n, t_a, k_a)
            stim = qasm_to_stim(qasm)
            res = run_worker(tool, qasm, stim)
            status = res["status"]
            print(f"{status} ({res['exec_s']:.2f}s | {res['peak_mb']:.1f}MB)")
            rows.append(["qubit_scaling", n, t_a, k_a, tool, status, res["exec_s"], res["peak_mb"]])

    # Sweep B: rank scaling
    n_b, t_b = 24, 40
    for k in RANK_K_VALUES:
        for tool in ["ucc", "qiskit"]:
            label = f"[rank_scaling]  {tool:<7} k={k}"
            print(f"{label} -> ", end="", flush=True)

            if check_theoretical_oom(tool, n_b, k):
                print("SKIPPED (OOM)")
                rows.append(["rank_scaling", n_b, t_b, k, tool, "OOM", 0.0, 0.0])
                continue

            qasm = build_circuit_qiskit(n_b, t_b, k)
            stim = qasm_to_stim(qasm)
            res = run_worker(tool, qasm, stim)
            status = res["status"]
            print(f"{status} ({res['exec_s']:.2f}s | {res['peak_mb']:.1f}MB)")
            rows.append(["rank_scaling", n_b, t_b, k, tool, status, res["exec_s"], res["peak_mb"]])

    with open(RESULTS_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)

    print(f"\nResults written to {RESULTS_FILE}")
    return RESULTS_FILE


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(csv_path: str, output_path: str) -> None:
    """Generate a 2x2 comparison plot from the results CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    TOOL_STYLE = {
        "qiskit": {"color": "#E74C3C", "marker": "s", "label": "Qiskit-Aer (statevector)"},
        "ucc": {"color": "#2ECC71", "marker": "o", "label": "UCC"},
    }

    TIMEOUT_CEIL = 120.0

    with open(csv_path, newline="") as f:
        results = list(csv.DictReader(f))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.15, wspace=0.25, bottom=0.14, top=0.91)

    panels = [
        ("qubit_scaling", "N", "Physical Qubits (N)", "Qubit Scaling (k=12, t=20)"),
        ("rank_scaling", "k", "Active Rank (k)", "Rank Scaling (N=24, t=40)"),
    ]

    # Compute shared Y limits across both panels
    all_times = [
        float(r["Exec_s"]) for r in results if r["Status"] == "SUCCESS" and float(r["Exec_s"]) > 0
    ]
    all_mems = [
        float(r["PeakMem_MB"])
        for r in results
        if r["Status"] == "SUCCESS" and float(r["PeakMem_MB"]) > 0
    ]
    y_floor_t = min(all_times) * 0.4 if all_times else 0.01
    y_ceil_t = max(max(all_times) * 3.0, TIMEOUT_CEIL) if all_times else TIMEOUT_CEIL
    y_floor_m = min(all_mems) * 0.5 if all_mems else 1.0
    y_ceil_m = max(all_mems) * 3.0 if all_mems else 1024.0

    def time_fmt(val: float, _pos: Any) -> str:
        if val >= 1.0:
            return f"{val:.0f}s"
        if val >= 0.01:
            return f"{val * 1000:.0f}ms"
        return f"{val * 1000:.1f}ms"

    def mem_fmt(val: float, _pos: Any) -> str:
        if val >= 1024.0:
            return f"{val / 1024:.1f}GB"
        return f"{val:.0f}MB"

    for col, (sweep, x_col, x_label, title) in enumerate(panels):
        ax_time = axes[0][col]
        ax_mem = axes[1][col]
        panel_rows = [r for r in results if r["Sweep"] == sweep]

        for tool in ["ucc", "qiskit"]:
            style = TOOL_STYLE[tool]
            ok = [r for r in panel_rows if r["Tool"] == tool and r["Status"] == "SUCCESS"]
            fail = [
                r for r in panel_rows if r["Tool"] == tool and r["Status"] in ("OOM", "TIMEOUT")
            ]

            xs = [float(r[x_col]) for r in ok]
            times = [float(r["Exec_s"]) for r in ok]
            mems = [float(r["PeakMem_MB"]) for r in ok]

            if xs:
                ax_time.plot(
                    xs,
                    times,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=7,
                    linewidth=2,
                    label=style["label"],
                )
                ax_mem.plot(
                    xs,
                    mems,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=7,
                    linewidth=2,
                    label=style["label"],
                )

            # Mark failures with dashed vertical lines
            for idx, r in enumerate(fail):
                fx = float(r[x_col])
                status_text = "OOM" if r["Status"] == "OOM" else "Timeout"
                ax_time.axvline(
                    x=fx,
                    color=style["color"],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )
                y_marker = y_ceil_t * (0.5 if idx == 0 else 0.25)
                ax_time.scatter(
                    [fx],
                    [y_marker],
                    marker="X",
                    s=120,
                    color="red",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=10,
                )
                ax_time.annotate(
                    status_text,
                    (fx, y_marker * 0.5),
                    fontsize=8,
                    ha="center",
                    color=style["color"],
                    fontweight="bold",
                )

        for ax in (ax_time, ax_mem):
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3, linewidth=0.5)

        ax_time.set_ylim(y_floor_t, y_ceil_t)
        ax_mem.set_ylim(y_floor_m, y_ceil_m)

        ax_time.set_title(title, fontsize=11, fontweight="bold")
        ax_time.tick_params(axis="x", labelbottom=False)
        ax_mem.set_xlabel(x_label, fontsize=10)

        ax_time.yaxis.set_major_formatter(ticker.FuncFormatter(time_fmt))
        ax_time.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax_mem.yaxis.set_major_formatter(ticker.FuncFormatter(mem_fmt))
        ax_mem.yaxis.set_minor_formatter(ticker.NullFormatter())

    axes[0][0].set_ylabel("Execution Time", fontsize=10)
    axes[1][0].set_ylabel("Peak Memory", fontsize=10)

    # Deduplicated legend
    handles, labels = [], []
    seen: set[str] = set()
    for ax in axes.flat:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in seen:
                seen.add(lbl)
                handles.append(h)
                labels.append(lbl)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        fontsize=10,
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "UCC vs Qiskit-Aer: Simulation Performance",
        fontsize=13,
        fontweight="bold",
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="UCC vs Qiskit-Aer benchmark")
    parser.add_argument(
        "--internal-worker",
        nargs=3,
        metavar=("TOOL", "QASM", "STIM"),
        help="(internal) Run one simulation in subprocess isolation.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip running benchmarks; just re-plot from existing CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "images", "benchmark_comparison.png"),
        help="Output plot path.",
    )
    args = parser.parse_args()

    if args.internal_worker:
        execute_internal(*args.internal_worker)
        return

    if args.plot_only:
        csv_path = RESULTS_FILE
    else:
        csv_path = run_sweeps()

    plot_results(csv_path, args.output)


if __name__ == "__main__":
    main()
