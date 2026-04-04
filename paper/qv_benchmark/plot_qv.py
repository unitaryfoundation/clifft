"""Generate publication-ready QV scaling plots from benchmark results.

Reads ``results.csv`` and produces a log-scale comparison of execution
times across simulators as a function of qubit count.

Usage
-----
    python plot_qv.py [--input results.csv] [--output qv_scaling.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_HERE: Path = Path(__file__).resolve().parent

_SIM_STYLE: dict[str, dict[str, object]] = {
    "ucc": {"color": "#1f77b4", "marker": "o", "label": "UCC"},
    "qiskit": {"color": "#ff7f0e", "marker": "s", "label": "Qiskit-Aer"},
    "qulacs": {"color": "#2ca02c", "marker": "^", "label": "Qulacs"},
    "qsim": {"color": "#d62728", "marker": "D", "label": "Qsim"},
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot QV benchmark scaling results.")
    p.add_argument("--input", type=str, default=str(_HERE / "results.csv"), help="Input CSV.")
    p.add_argument("--output", type=str, default=str(_HERE / "qv_scaling.pdf"), help="Output plot.")
    return p


def plot(csv_path: Path, output_path: Path) -> None:
    """Generate the QV scaling plot."""
    df = pd.read_csv(csv_path)
    success = df[df["status"] == "SUCCESS"].copy()
    success["exec_s"] = success["exec_s"].astype(float)

    stats = success.groupby(["simulator", "N"])["exec_s"].agg(["median", "min", "max"])

    fig, ax = plt.subplots(figsize=(8, 5))

    for sim in stats.index.get_level_values("simulator").unique():
        style = _SIM_STYLE.get(sim, {"color": "gray", "marker": "x", "label": sim})
        sim_stats = stats.loc[sim]
        n_vals = sim_stats.index.values
        medians = sim_stats["median"].values
        yerr_lo = medians - sim_stats["min"].values
        yerr_hi = sim_stats["max"].values - medians

        ax.errorbar(
            n_vals,
            medians,
            yerr=[yerr_lo, yerr_hi],
            fmt=f"-{style['marker']}",
            color=style["color"],
            label=str(style["label"]),
            capsize=3,
            markersize=6,
            linewidth=1.5,
        )

    # Mark failure points (all repeats failed for a given N)
    all_failed = df.groupby(["simulator", "N"]).filter(lambda g: (g["status"] != "SUCCESS").all())
    if not all_failed.empty:
        y_ceiling = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
        for (sim, n), _ in all_failed.groupby(["simulator", "N"]):
            style = _SIM_STYLE.get(sim, {"color": "gray", "marker": "x", "label": sim})
            ax.plot(
                n, y_ceiling * 0.9, "x", color="red", markersize=12, markeredgewidth=2, zorder=10
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
    args = _build_parser().parse_args(argv)
    plot(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
