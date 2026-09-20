"""Render the recorded gadget sweep without rerunning benchmarks."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [c for c in json.loads(args.source.read_text())["cases"] if c["probability"] == 0.001]
    distances = [c["distance"] for c in rows]
    measured = [c for c in rows if c["ordinary_seconds_per_attempt"] is not None]
    fig, (time, memory) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    time.plot(
        distances,
        [c["compiled_seconds_per_attempt"] * 1e6 for c in rows],
        "o-",
        label="Compiled gadget",
    )
    time.plot(
        [c["distance"] for c in measured],
        [c["ordinary_seconds_per_attempt"] * 1e6 for c in measured],
        "s-",
        label="Ordinary scalar Clifft",
    )
    time.set(
        title="Measured complete gadget attempts",
        ylabel="Microseconds per attempt (log scale)",
        yscale="log",
    )
    time.annotate(
        "Dense d=9 timing not measured",
        (0.04, 0.94),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
    )
    memory.plot(
        distances,
        [c["contraction_numeric_payload_bytes"] / 2**20 for c in rows],
        "o-",
        label="Contraction numeric payload",
    )
    memory.plot(
        distances,
        [c["ordinary_coefficient_bytes"] / 2**20 for c in rows],
        "s--",
        label="Planned dense coefficients only",
    )
    memory.set(title="Selected storage requirements", ylabel="MiB (log scale)", yscale="log")
    for ax in (time, memory):
        ax.set_xlabel("Code distance")
        ax.set_xticks(distances)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Unflagged geometric gadget family | p=0.001 | one CPU worker")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
