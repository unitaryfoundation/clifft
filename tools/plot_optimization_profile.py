"""Regenerate the optimization introduction's active-width figures.

Run with an installed Clifft build and Matplotlib:
    uv run --with matplotlib python tools/plot_optimization_profile.py

The three-qubit example is adapted from
20261015.riverlane/optimization-profile.tex in unitaryfoundation/clifft-publications.
Traces come from the current compiler, not manually transcribed slide data.
"""

from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import clifft

CIRCUIT = """
R_XX(0.125) 0 1
R_XX(0.25) 1 2
R_XX(0.125) 0 1
MPP Z1*Z2 Z2
"""


def profiles() -> list[tuple[str, list[int], int]]:
    hir = clifft.trace(clifft.parse(CIRCUIT))
    stages = [
        ("Before optimization", None),
        ("After fusion", clifft.PeepholeFusionPass()),
        ("After fusion and squeezing", clifft.StatevectorSqueezePass()),
    ]
    result = []
    for title, pass_ in stages:
        if pass_ is not None:
            manager = clifft.HirPassManager()
            manager.add(pass_)
            manager.run(hir)
        trace = clifft.active_width_trace(hir)
        widths = [trace["initial"], *trace["widths"]]
        # Every operation in this example touches the dense state. Measurements
        # process the pre-collapse width; rotations process the resulting width.
        assert all(
            effect in {"rotation_promote", "rotation_neutral", "measure_active"}
            for effect in trace["effects"]
        )
        work = sum(2 ** max(before, after) for before, after in pairwise(widths))
        assert trace["peak"] == clifft.lower(hir).peak_active_width
        result.append((title, widths, work))
    return result


def main() -> None:
    data = profiles()
    output = Path(__file__).resolve().parents[1] / "docs" / "assets" / "theory"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.hashsalt": "clifft-optimization-profile",
        }
    )
    for theme, foreground, muted, blue in [
        ("light", "#243044", "#596579", "#3c64b4"),
        ("dark", "#e2e8f0", "#a9b5c8", "#7ba3f0"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1), sharey=True)
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.70, wspace=0.23)
        for ax, (title, widths, work) in zip(axes, data, strict=True):
            positions = list(range(len(widths)))
            ax.step(positions, widths, where="post", color=blue, linewidth=2.3)
            ax.fill_between(positions, widths, step="post", color=blue, alpha=0.10)
            ax.scatter(positions, widths, color=blue, s=25, zorder=3, clip_on=False)
            ax.set_title(title, fontsize=12, color=foreground, loc="left", pad=43)
            ax.text(
                0,
                1.12,
                f"Peak width {max(widths)}  |  Work estimate {work}",
                transform=ax.transAxes,
                color=muted,
                fontsize=10,
            )
            ax.set_xlim(0, positions[-1])
            ax.set_ylim(0, 2.25)
            ax.set_xticks(positions)
            ax.set_yticks([0, 1, 2])
            ax.tick_params(colors=muted, length=0, pad=7)
            ax.grid(axis="y", color=muted, alpha=0.18, linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0].set_ylabel("Active width $k$", color=foreground, labelpad=12)
        fig.supxlabel("Operations completed in each circuit", color=muted, fontsize=11, y=0.03)
        target = output / f"optimization-profile-{theme}.svg"
        fig.savefig(target, transparent=True, metadata={"Date": None})
        # Match repository whitespace checks so regeneration leaves a clean diff.
        target.write_text(
            "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
        )
        plt.close(fig)
    for title, widths, work in data:
        print(f"{title}: widths={widths}, dense work={work}")


if __name__ == "__main__":
    main()
