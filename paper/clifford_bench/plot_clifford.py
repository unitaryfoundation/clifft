"""Plot Clifford-limit benchmark results.

Produces a sample-time vs physical error rate plot (log-log),
styled to match the tsim paper's d=7 surface code figure.

Accepts multiple input CSVs with optional label suffixes to rename
the simulator column, enabling separate CPU/GPU runs to be combined:

    python plot_clifford.py --input results.csv tsim_cpu.csv:tsim-cpu tsim_gpu.csv:tsim-gpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

_HERE = Path(__file__).resolve().parent

STYLES: dict[str, dict] = {
    "stim": {
        "color": "#555555",
        "marker": "None",
        "linestyle": "--",
        "linewidth": 2.5,
        "label": "Stim CPU",
    },
    "clifft": {
        "color": "#E74C3C",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2,
        "label": "Clifft CPU",
    },
    "tsim-cpu": {
        "color": "#3498DB",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2,
        "label": "tsim CPU",
    },
    "tsim-gpu": {
        "color": "#2ECC71",
        "marker": "s",
        "linestyle": "-",
        "linewidth": 2,
        "label": "tsim GPU",
    },
}

_STYLE_ORDER = list(STYLES.keys())


def _parse_input_spec(spec: str) -> tuple[str, str | None]:
    """Parse 'path.csv:label' into (path, label).  Label is None if omitted."""
    if ":" in spec:
        path, label = spec.rsplit(":", 1)
        return path, label
    return spec, None


def _load_inputs(specs: list[str]) -> pd.DataFrame:
    """Load and concatenate CSVs, applying label remaps."""
    frames = []
    for spec in specs:
        path, label = _parse_input_spec(spec)
        df = pd.read_csv(path)
        if label is not None:
            df["simulator"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Clifford-limit benchmark results.")
    p.add_argument(
        "--input",
        nargs="+",
        default=[str(_HERE / "results.csv")],
        help=(
            "Input CSV files.  Use path:label to rename the simulator column "
            "(e.g. tsim_cpu.csv:tsim-cpu).  Multiple files are concatenated."
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_HERE / "clifford_bench.png"),
        help="Output plot path.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    df = _load_inputs(args.input)
    df = df[df["status"] == "SUCCESS"].copy()

    group_cols = ["distance", "rounds", "shots"]
    plot_groups = df.groupby(group_cols)

    plot_keys = sorted(plot_groups.groups.keys())

    for d, rounds, shots in plot_keys:
        fig, ax = plt.subplots(figsize=(6, 5))
        group = plot_groups.get_group((d, rounds, shots))

        sims_present = sorted(
            group["simulator"].unique(),
            key=lambda s: _STYLE_ORDER.index(s) if s in _STYLE_ORDER else 99,
        )

        for sim in sims_present:
            style = STYLES.get(
                sim,
                {"color": "gray", "marker": "s", "linestyle": "-", "linewidth": 1.5, "label": sim},
            )
            sim_data = group[group["simulator"] == sim]
            medians = sim_data.groupby("phys_error_rate")["sample_s"].median()

            ax.plot(
                medians.index,
                medians.values,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                color=style["color"],
                label=style["label"],
                markersize=8,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Error rate", fontsize=13)
        ax.set_ylabel("Sample time (s)", fontsize=13)
        ax.set_title(
            f"d={d} rotated surface code\n({rounds} rounds, {shots:,} shots)",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        out_path = Path(args.output)
        if len(plot_keys) > 1:
            out_path = out_path.with_stem(f"{out_path.stem}_d{d}_r{rounds}_s{shots}")
        plt.savefig(str(out_path), dpi=150)
        print(f"Saved {out_path}")
        plt.close()


if __name__ == "__main__":
    main()
