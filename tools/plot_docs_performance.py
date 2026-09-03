#!/usr/bin/env python3
"""Generate the light and dark performance figures used by the documentation.

Run from the repository root with:

    uv run --with matplotlib python tools/plot_docs_performance.py
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/assets/performance/data"
OUTPUT = ROOT / "docs/assets/performance"

WORKLOAD_LABELS = {
    "coherent-surface-d3-r1-p1e-3-rz2e-2": "Coherent d3, r1",
    "coherent-surface-d3-r3-p1e-3-rz2e-2": "Coherent d3, r3",
    "coherent-surface-d5-r1-p1e-3-rz2e-2": "Coherent d5, r1",
    "coherent-surface-d5-r5-p1e-3-rz2e-2": "Coherent d5, r5",
    "distillation-color-code-85q-p5e-2": "85q distillation",
    "msc-d3-inject-cultivate-p1e-3": "Cultivation d3",
    "msc-d5-inject-cultivate-p1e-3": "Cultivation d5",
    "surface-code-d7-r7-p1e-3": "Surface code d7, r7",
}
WORKLOAD_ORDER = tuple(WORKLOAD_LABELS)


@dataclass(frozen=True)
class Theme:
    name: str
    foreground: str
    muted: str
    grid: str
    blue: str
    orange: str
    green: str
    red: str


THEMES = (
    Theme(
        "light",
        foreground="#172033",
        muted="#64748B",
        grid="#CBD5E1",
        blue="#3C64B4",
        orange="#C26713",
        green="#147D64",
        red="#B8465F",
    ),
    Theme(
        "dark",
        foreground="#E6EDF7",
        muted="#AAB6C8",
        grid="#526077",
        blue="#83A7F2",
        orange="#F2A65A",
        green="#57C7A5",
        red="#F08AA0",
    ),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _configure(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": theme.muted,
            "axes.labelcolor": theme.foreground,
            "axes.labelsize": 12,
            "axes.titlecolor": theme.foreground,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "savefig.facecolor": "none",
            "text.color": theme.foreground,
            "xtick.color": theme.muted,
            "xtick.labelsize": 10.5,
            "ytick.color": theme.foreground,
            "ytick.labelsize": 11.5,
        }
    )


def _clean_axis(axis: Any, theme: Theme, *, x_grid: bool = True) -> None:
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="both", length=0)
    if x_grid:
        axis.grid(axis="x", color=theme.grid, linewidth=0.8, alpha=0.48)
        axis.set_axisbelow(True)


def _ratio_tick(value: float, _position: float) -> str:
    return f"{value:g}x"


def _ratio_label(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


def _comparison_rows(comparison_id: str) -> list[dict[str, str]]:
    rows = _read_csv(DATA / "release-comparisons.csv")
    selected = [row for row in rows if row["comparison_id"] == comparison_id]
    if {row["workload_id"] for row in selected} != set(WORKLOAD_ORDER):
        raise ValueError(f"comparison {comparison_id!r} does not cover every workload")
    return selected


def _plot_ratios(
    theme: Theme,
    *,
    rows: list[dict[str, str]],
    output_name: str,
    ratios: Iterable[float],
    marker_shapes: Iterable[str],
    marker_filled: Iterable[bool],
    xlabel: str,
    ticks: list[float],
    legend: list[Line2D],
) -> None:
    points = sorted(
        zip(rows, ratios, marker_shapes, marker_filled, strict=True),
        key=lambda point: point[1],
    )
    figure, axis = plt.subplots(figsize=(9.6, 4.5))
    positions = list(range(len(points)))
    maximum = max(point[1] for point in points)
    upper = maximum * 1.55

    axis.axvline(1, color=theme.muted, linewidth=1.2, linestyle=(0, (3, 3)))
    for position, (_row, ratio, shape, filled) in zip(positions, points, strict=True):
        axis.plot(
            [1, ratio],
            [position, position],
            color=theme.blue,
            linewidth=4.5,
            alpha=0.32,
            solid_capstyle="round",
        )
        axis.scatter(
            ratio,
            position,
            marker=shape,
            s=88,
            facecolor=theme.blue if filled else "none",
            edgecolor=theme.blue,
            linewidth=2,
            zorder=3,
        )
        place_left = ratio > upper / 2.4
        axis.annotate(
            _ratio_label(ratio),
            (ratio, position),
            xytext=(-9 if place_left else 9, 0),
            textcoords="offset points",
            ha="right" if place_left else "left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=theme.foreground,
        )

    axis.set_xscale("log")
    axis.set_xlim(0.88, upper)
    visible_ticks = [value for value in ticks if value <= upper]
    axis.xaxis.set_major_locator(FixedLocator(visible_ticks))
    axis.xaxis.set_major_formatter(FuncFormatter(_ratio_tick))
    axis.set_yticks(
        positions,
        labels=[WORKLOAD_LABELS[point[0]["workload_id"]] for point in points],
    )
    axis.set_xlabel(xlabel, labelpad=12)
    if legend:
        axis.legend(
            handles=legend,
            loc="lower right",
            frameon=False,
            ncols=len(legend),
            labelcolor=theme.foreground,
            columnspacing=1.2,
            handletextpad=0.45,
            bbox_to_anchor=(1, 1.005),
        )
    _clean_axis(axis, theme)
    figure.subplots_adjust(left=0.24, right=0.98, top=0.9, bottom=0.17)
    figure.savefig(
        OUTPUT / f"{output_name}-{theme.name}.png",
        dpi=200,
        transparent=True,
    )
    plt.close(figure)


def _plot_release_comparison(theme: Theme) -> None:
    rows = _comparison_rows("current-vs-previous")
    ratios = [float(row["ratio_candidate_over_baseline"]) for row in rows]
    packed = [int(row["candidate_batch_size_effective"]) > 1 for row in rows]
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=theme.blue,
            markeredgecolor=theme.blue,
            markersize=7,
            label="v0.10 packed",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=theme.blue,
            markeredgewidth=1.5,
            markersize=7,
            label="v0.10 scalar",
        ),
    ]
    _plot_ratios(
        theme,
        rows=rows,
        output_name="v010-vs-v009",
        ratios=ratios,
        marker_shapes=["o"] * len(rows),
        marker_filled=packed,
        xlabel="Clifft v0.10 throughput relative to v0.9",
        ticks=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        legend=legend,
    )


def _plot_tool_comparison(theme: Theme) -> None:
    rows = _comparison_rows("alternatives-vs-current")
    ratios = [1 / float(row["ratio_candidate_over_baseline"]) for row in rows]
    clifft_packed = [int(row["baseline_batch_size_effective"]) > 1 for row in rows]
    symft_packed = [int(row["candidate_batch_size_effective"]) > 1 for row in rows]
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=theme.muted,
            markeredgewidth=1.5,
            markersize=7,
            label="SymFT scalar",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=theme.muted,
            markeredgewidth=1.5,
            markersize=6.5,
            label="SymFT packed",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markerfacecolor=theme.blue,
            markeredgecolor=theme.blue,
            markersize=7,
            label="filled: Clifft packed",
        ),
    ]
    _plot_ratios(
        theme,
        rows=rows,
        output_name="clifft-vs-symft",
        ratios=ratios,
        marker_shapes=["D" if value else "o" for value in symft_packed],
        marker_filled=clifft_packed,
        xlabel="Clifft v0.10 throughput relative to SymFT v0.1",
        ticks=[1, 2, 5, 10, 20, 50, 100],
        legend=legend,
    )


def _history_medians() -> tuple[list[str], list[float]]:
    rows = _read_csv(DATA / "history-cases.csv")
    release_rows = _comparison_rows("current-vs-previous")
    anchor_version = release_rows[0]["baseline_simulator_display_version"]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    versions: list[str] = []
    for row in rows:
        if row["status"] != "success":
            continue
        version = row["simulator_display_version"]
        if version not in versions:
            versions.append(version)
        grouped[(row["workload_id"], version)].append(
            float(row["median_attempted_shots_per_second"])
        )
    versions = versions[: versions.index(anchor_version) + 1]

    speedups: dict[str, list[float]] = {}
    for workload in WORKLOAD_ORDER:
        rates = [statistics.median(grouped[(workload, version)]) for version in versions]
        speedups[workload] = [rate / rates[0] for rate in rates]

    versions.append(release_rows[0]["candidate_simulator_display_version"])
    ratios = {
        row["workload_id"]: float(row["ratio_candidate_over_baseline"]) for row in release_rows
    }
    for workload in WORKLOAD_ORDER:
        speedups[workload].append(speedups[workload][-1] * ratios[workload])

    medians = [
        statistics.median(speedups[workload][index] for workload in WORKLOAD_ORDER)
        for index in range(len(versions))
    ]
    return versions, medians


def _plot_history(theme: Theme) -> None:
    versions, medians = _history_medians()
    positions = list(range(len(versions)))
    figure, axis = plt.subplots(figsize=(9.6, 3.8))
    axis.axhline(1, color=theme.muted, linewidth=1.2, linestyle=(0, (3, 3)))
    axis.plot(
        positions,
        medians,
        color=theme.blue,
        linewidth=3.5,
        marker="o",
        markersize=6.5,
        solid_capstyle="round",
        zorder=3,
    )
    axis.fill_between(positions, 1, medians, color=theme.blue, alpha=0.09)
    axis.annotate(
        f"{medians[-1]:.0f}x median",
        (positions[-1], medians[-1]),
        xytext=(-8, -22),
        textcoords="offset points",
        ha="right",
        color=theme.blue,
        fontsize=11.5,
        fontweight="bold",
    )
    axis.annotate(
        "symbolic plans",
        (positions[-3], medians[-3]),
        xytext=(0, 28),
        textcoords="offset points",
        ha="center",
        color=theme.muted,
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": theme.grid, "linewidth": 1},
    )
    axis.annotate(
        "v0.10: packing + compiler",
        (positions[-1], medians[-1]),
        xytext=(-72, 20),
        textcoords="offset points",
        ha="center",
        color=theme.muted,
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": theme.grid, "linewidth": 1},
    )
    axis.set_yscale("log", base=2)
    axis.set_ylim(0.72, 10.5)
    axis.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    axis.yaxis.set_major_formatter(FuncFormatter(_ratio_tick))
    axis.set_xticks(positions, labels=[f"v{version}" for version in versions])
    axis.set_ylabel("Median speedup vs v0.1")
    _clean_axis(axis, theme)
    figure.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.18)
    figure.savefig(
        OUTPUT / f"performance-over-time-{theme.name}.png",
        dpi=200,
        transparent=True,
    )
    plt.close(figure)


def _plot_qv(theme: Theme) -> None:
    rows = _read_csv(DATA / "qv-cases.csv")
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            grouped[(row["simulator"], int(row["qubits"]))].append(float(row["execution_seconds"]))

    styles = {
        "clifft": ("Clifft 0.10", theme.blue, "o"),
        "qiskit": ("Qiskit Aer", theme.orange, "s"),
        "qsim": ("qsim", theme.green, "D"),
        "qulacs": ("Qulacs", theme.red, "^"),
    }
    figure, axis = plt.subplots(figsize=(9.6, 4.7))
    for simulator, (label, color, marker) in styles.items():
        widths = sorted(qubits for tool, qubits in grouped if tool == simulator)
        medians = [statistics.median(grouped[(simulator, width)]) for width in widths]
        axis.plot(
            widths,
            medians,
            label=label,
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=2.3,
        )
    axis.set_yscale("log")
    axis.set_xticks(range(6, 29, 2))
    axis.set_xlabel("Quantum Volume circuit width and depth")
    axis.set_ylabel("Execution time (seconds; lower is better)")
    axis.legend(
        loc="upper left",
        frameon=False,
        ncols=4,
        labelcolor=theme.foreground,
        handletextpad=0.5,
        columnspacing=1.2,
        bbox_to_anchor=(0, 1.03),
    )
    _clean_axis(axis, theme)
    figure.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.17)
    figure.savefig(
        OUTPUT / f"quantum-volume-{theme.name}.png",
        dpi=200,
        transparent=True,
    )
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        _configure(theme)
        _plot_release_comparison(theme)
        _plot_tool_comparison(theme)
        _plot_history(theme)
        _plot_qv(theme)


if __name__ == "__main__":
    main()
