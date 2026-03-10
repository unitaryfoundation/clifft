"""Generate the Three Exponential Walls 2x3-panel figure.

Reads results_walls.csv and produces a 2x3 panel plot showing each
simulator hitting its respective exponential wall.
Top row: execution time.  Bottom row: peak memory.

Usage:
    python plot_walls.py                    # default: results_walls.csv
    python plot_walls.py -i my_results.csv  # custom input
    python plot_walls.py -o figure.pdf      # custom output
"""

import argparse
import csv
import sys
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -- Visual config --
TOOL_STYLE: dict[str, dict[str, Any]] = {
    "qiskit": {"color": "#E74C3C", "marker": "s", "label": "Qiskit-Aer (1 CPU)"},
    "ucc": {"color": "#2ECC71", "marker": "o", "label": "UCC (1 CPU)"},
    "tsim": {"color": "#3498DB", "marker": "^", "label": "tsim (GPU)"},
}

FAIL_MARKER_STYLE: dict[str, Any] = {
    "marker": "X",
    "s": 120,
    "zorder": 10,
    "edgecolors": "black",
    "linewidths": 0.8,
}

# Panel definitions: (panel_key, x_column, x_label, title)
PANEL_DEFS = [
    ("Panel_A", "N", "Physical Qubits (N)", "A: Physical Qubit Wall"),
    ("Panel_B", "k", "Active Rank (k)", "B: Active Rank Wall"),
    ("Panel_C", "t", "Non-Clifford Gates (t)", "C: Stabilizer Rank Wall"),
]

TIMEOUT_CEIL = 120.0


def load_results(path: str) -> list[dict[str, str]]:
    """Load CSV results into a list of row dicts."""
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("Panel")]


def plot_walls(results: list[dict[str, str]], output_path: str) -> None:
    """Render the 2x3 wall comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.subplots_adjust(wspace=0.12, hspace=0.10, bottom=0.12, top=0.90)

    # Share Y axes within each row
    for col in range(1, 3):
        axes[0][col].sharey(axes[0][0])
        axes[1][col].sharey(axes[1][0])

    # Compute global Y range for time (top row)
    all_times = [
        float(r["Exec_s"]) for r in results if r["Status"] == "SUCCESS" and float(r["Exec_s"]) > 0
    ]
    if all_times:
        y_floor_t = min(all_times) * 0.5
        y_ceil_t = max(max(all_times) * 3.0, TIMEOUT_CEIL)
    else:
        y_floor_t, y_ceil_t = 0.01, TIMEOUT_CEIL

    # Compute global Y range for memory (bottom row)
    all_mems = [
        float(r["PeakMem_MB"])
        for r in results
        if r["Status"] == "SUCCESS" and float(r["PeakMem_MB"]) > 0
    ]
    if all_mems:
        y_floor_m = min(all_mems) * 0.5
        y_ceil_m = max(all_mems) * 3.0
    else:
        y_floor_m, y_ceil_m = 1.0, 1024.0

    for col, (panel_key, x_col, x_label, title) in enumerate(PANEL_DEFS):
        ax_time = axes[0][col]
        ax_mem = axes[1][col]

        panel_rows = [r for r in results if r["Panel"] == panel_key]

        if not panel_rows:
            ax_time.set_title(title, fontsize=11, fontweight="bold")
            for ax in (ax_time, ax_mem):
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                )
            ax_mem.set_xlabel(x_label, fontsize=10)
            continue

        # Group by tool
        tool_data: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"x": [], "time": [], "mem": []}
        )
        fail_points: list[tuple[str, float]] = []

        for row in panel_rows:
            tool = row["Tool"]
            x_val = float(row[x_col])
            status = row["Status"]

            if status == "SUCCESS":
                tool_data[tool]["x"].append(x_val)
                tool_data[tool]["time"].append(float(row["Exec_s"]))
                tool_data[tool]["mem"].append(float(row["PeakMem_MB"]))
            elif status in ("OOM", "TIMEOUT"):
                fail_points.append((tool, x_val))

        # Plot successful runs -- time row
        for tool, data in tool_data.items():
            style = TOOL_STYLE.get(tool, {"color": "gray", "marker": "D", "label": tool})
            ax_time.plot(
                data["x"],
                data["time"],
                color=style["color"],
                marker=style["marker"],
                markersize=7,
                linewidth=2,
                label=style["label"],
            )

        # Plot successful runs -- memory row
        for tool, data in tool_data.items():
            style = TOOL_STYLE.get(tool, {"color": "gray", "marker": "D", "label": tool})
            ax_mem.plot(
                data["x"],
                data["mem"],
                color=style["color"],
                marker=style["marker"],
                markersize=7,
                linewidth=2,
                label=style["label"],
            )

        # Failure markers on time row only
        for tool, x_val in fail_points:
            color = TOOL_STYLE.get(tool, {"color": "gray"})["color"]
            ax_time.scatter(
                [x_val],
                [y_ceil_t * 0.6],
                color="red",
                **FAIL_MARKER_STYLE,
            )
            ax_time.axvline(
                x=x_val,
                color=color,
                linestyle="--",
                alpha=0.4,
                linewidth=1,
            )

        # --- Axis formatting ---
        # Time row
        ax_time.set_yscale("log")
        ax_time.set_ylim(y_floor_t, y_ceil_t)
        ax_time.set_title(title, fontsize=11, fontweight="bold")
        ax_time.grid(True, which="both", alpha=0.3, linewidth=0.5)
        ax_time.yaxis.set_major_formatter(ticker.FuncFormatter(_time_fmt))
        ax_time.yaxis.set_minor_formatter(ticker.NullFormatter())
        # Hide x-axis tick labels on top row
        ax_time.tick_params(axis="x", labelbottom=False)

        # Memory row
        ax_mem.set_yscale("log")
        ax_mem.set_ylim(y_floor_m, y_ceil_m)
        ax_mem.set_xlabel(x_label, fontsize=10)
        ax_mem.grid(True, which="both", alpha=0.3, linewidth=0.5)
        ax_mem.yaxis.set_major_formatter(ticker.FuncFormatter(_mem_fmt))
        ax_mem.yaxis.set_minor_formatter(ticker.NullFormatter())

        # Panel C: log scale on x-axis for both rows
        if col == 2:
            ax_time.set_xscale("log")
            ax_mem.set_xscale("log")

    # Y-axis labels on leftmost panels only
    axes[0][0].set_ylabel("Execution Time", fontsize=10)
    axes[1][0].set_ylabel("Peak Memory", fontsize=10)

    # Unified legend below the figure (deduplicated across all 6 axes)
    handles, labels = _collect_legend(axes.flat)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            fontsize=9,
            frameon=True,
            fancybox=True,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(
        "The Three Exponential Walls",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.close(fig)


def _time_fmt(val: float, _pos: Any) -> str:
    """Format log-scale time axis values."""
    if val >= 1.0:
        return f"{val:.0f}s"
    elif val >= 0.01:
        return f"{val * 1000:.0f}ms"
    else:
        return f"{val * 1000:.1f}ms"


def _mem_fmt(val: float, _pos: Any) -> str:
    """Format log-scale memory axis values."""
    if val >= 1024.0:
        return f"{val / 1024.0:.1f}GB"
    else:
        return f"{val:.0f}MB"


def _collect_legend(
    axes: Any,
) -> tuple[list[Any], list[str]]:
    """Deduplicate legend entries across all axes."""
    seen: set[str] = set()
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in seen:
                seen.add(lbl)
                handles.append(h)
                labels.append(lbl)
    return handles, labels


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Plot Three Walls benchmark results")
    parser.add_argument(
        "-i",
        "--input",
        default="results_walls.csv",
        help="Path to results CSV (default: results_walls.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="three_walls.png",
        help="Output figure path (default: three_walls.png)",
    )
    args = parser.parse_args()

    results = load_results(args.input)
    if not results:
        print("No results found in CSV.", file=sys.stderr)
        sys.exit(1)

    plot_walls(results, args.output)


if __name__ == "__main__":
    main()
