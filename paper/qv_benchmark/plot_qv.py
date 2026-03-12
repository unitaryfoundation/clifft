"""Generate publication-ready QV scaling plots from benchmark results.

Reads ``results.csv`` and produces a log-scale comparison of execution
times across simulators as a function of qubit count.

Usage
-----
    python -m paper.qv_benchmark.plot_qv [--input results.csv] [--output qv_scaling.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_DEFAULT_INPUT: str = "paper/qv_benchmark/results.csv"
_DEFAULT_OUTPUT: str = "paper/qv_benchmark/qv_scaling.pdf"
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

_SIM_STYLE: Dict[str, Dict[str, object]] = {
    "ucc": {"color": "#1f77b4", "marker": "o", "label": "UCC"},
    "qiskit": {"color": "#ff7f0e", "marker": "s", "label": "Qiskit-Aer"},
    "qulacs": {"color": "#2ca02c", "marker": "^", "label": "Qulacs"},
    "qsim": {"color": "#d62728", "marker": "D", "label": "Qsim"},
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot QV benchmark scaling results.")
    p.add_argument("--input", type=str, default=_DEFAULT_INPUT, help="Input CSV.")
    p.add_argument("--output", type=str, default=_DEFAULT_OUTPUT, help="Output plot path.")
    return p


def _load_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate the results CSV."""
    df = pd.read_csv(csv_path)
    for col in ("N", "simulator", "status", "exec_s"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")
    return df


def _compute_stats(
    df: pd.DataFrame,
) -> Dict[str, Tuple[List[int], List[float], List[float], List[float]]]:
    """Compute median, min, max exec times per (simulator, N).

    Returns a dict mapping simulator name to (n_vals, medians, mins, maxs).
    Only includes rows with status == 'SUCCESS'.
    """
    success = df[df["status"] == "SUCCESS"].copy()
    success["exec_s"] = success["exec_s"].astype(float)

    stats: Dict[str, Tuple[List[int], List[float], List[float], List[float]]] = {}

    for sim in success["simulator"].unique():
        sim_df = success[success["simulator"] == sim]
        grouped = sim_df.groupby("N")["exec_s"]
        agg = grouped.agg(["median", "min", "max"]).sort_index()

        n_vals = agg.index.tolist()
        medians = agg["median"].tolist()
        mins = agg["min"].tolist()
        maxs = agg["max"].tolist()
        stats[sim] = (n_vals, medians, mins, maxs)

    return stats


def _find_failures(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Find (simulator, N) combos where all repeats failed (OOM/TIMEOUT/ERROR)."""
    failures: Dict[str, List[int]] = {}
    for sim in df["simulator"].unique():
        sim_df = df[df["simulator"] == sim]
        failed_ns: List[int] = []
        for n, grp in sim_df.groupby("N"):
            if (grp["status"] != "SUCCESS").all():
                failed_ns.append(int(n))
        if failed_ns:
            failures[sim] = failed_ns
    return failures


def plot(csv_path: Path, output_path: Path) -> None:
    """Generate the QV scaling plot."""
    df = _load_data(csv_path)
    stats = _compute_stats(df)
    failures = _find_failures(df)

    fig, ax = plt.subplots(figsize=(8, 5))

    for sim, (n_vals, medians, mins, maxs) in stats.items():
        style = _SIM_STYLE.get(sim, {"color": "gray", "marker": "x", "label": sim})
        n_arr = np.array(n_vals)
        med_arr = np.array(medians)
        min_arr = np.array(mins)
        max_arr = np.array(maxs)

        yerr_lo = med_arr - min_arr
        yerr_hi = max_arr - med_arr

        ax.errorbar(
            n_arr,
            med_arr,
            yerr=[yerr_lo, yerr_hi],
            fmt=f"-{style['marker']}",
            color=style["color"],
            label=str(style["label"]),
            capsize=3,
            markersize=6,
            linewidth=1.5,
        )

    # Mark failure points
    y_ceiling = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for sim, failed_ns in failures.items():
        style = _SIM_STYLE.get(sim, {"color": "gray", "marker": "x", "label": sim})
        for n in failed_ns:
            ax.plot(
                n,
                y_ceiling * 0.9,
                "x",
                color="red",
                markersize=12,
                markeredgewidth=2,
                zorder=10,
            )
            ax.annotate(
                f"{style['label']} OOM",
                (n, y_ceiling * 0.9),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color="red",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Number of Qubits (N)", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title("Quantum Volume Benchmark: Worst-Case Scaling", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    csv_path = Path(args.input)
    if not csv_path.is_absolute():
        csv_path = _PROJECT_ROOT / csv_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path

    plot(csv_path, output_path)


if __name__ == "__main__":
    main()
