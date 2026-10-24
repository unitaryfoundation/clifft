#!/usr/bin/env python3
"""Delayed-loss surface-code tutorial for the Clifft documentation.

The experiment follows the idea behind Fig. 8 of "Leveraging Qubit Loss
Detection in Fault Tolerant Quantum Algorithms" (arXiv:2502.20558):
losses of the same data qubit at two different times produce different
detector patterns even though both are heralded only at final readout.

Usage:
    uv run --with matplotlib python docs/guide/scripts/delayed_loss_tutorial.py

Generates:
    docs/guide/images/delayed_loss_detectors.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim
from numpy.typing import NDArray

from clifft import noncomp

DISTANCE = 3
ROUNDS = 3
DATA_QUBIT = 10
FORCED_SHOTS = 20_000
STOCHASTIC_SHOTS = 10_000
IMAGE_PATH = Path(__file__).resolve().parent.parent / "images" / "delayed_loss_detectors.png"


def surface_code() -> stim.Circuit:
    """Return the unrolled, noiseless memory experiment used by the tutorial."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=DISTANCE,
        rounds=ROUNDS,
    ).flattened()


def force_loss_after_first_interaction(
    circuit: stim.Circuit,
    *,
    round_number: int,
) -> str:
    """Insert certain loss after qubit 10's first CX in a one-based round."""
    interactions_per_round = 4
    wanted_interaction = interactions_per_round * (round_number - 1) + 1
    seen_interactions = 0
    inserted = False
    lines: list[str] = []

    for line in str(circuit).splitlines():
        lines.append(line)
        words = line.split()
        if words and words[0] == "CX" and str(DATA_QUBIT) in words[1:]:
            seen_interactions += 1
            if seen_interactions == wanted_interaction:
                lines.append(f"LOSS(1) {DATA_QUBIT}")
                inserted = True

    if not inserted:
        raise ValueError(f"round {round_number} is outside the generated circuit")
    return "\n".join(lines)


def state_selective_classifier() -> noncomp.Classifier:
    """Return state-selective readout with a third, heralded symbol."""
    return noncomp.Classifier(
        [
            [1, 0, 0, 0, 0],  # g -> 0
            [0, 1, 0, 0, 0],  # e -> 1
            [0, 0, 1, 1, 1],  # leak_g, leak_e, lost -> herald
        ]
    )


def detector_counts_by_time(
    detector_probabilities: NDArray[np.float64],
    coordinates: dict[int, list[float]],
) -> list[tuple[int, int, int]]:
    """Return time, active count, and total count for each detector slice."""
    times = sorted({int(coord[2]) for coord in coordinates.values()})
    rows: list[tuple[int, int, int]] = []
    for time in times:
        indices = [index for index, coord in coordinates.items() if int(coord[2]) == time]
        active = sum(detector_probabilities[index] > 0.25 for index in indices)
        rows.append((time, active, len(indices)))
    return rows


def plot_detector_activity(
    histories: dict[str, NDArray[np.float64]],
    coordinates: dict[int, list[float]],
) -> None:
    """Plot detector activation probability for each forced loss history."""
    times = sorted({int(coord[2]) for coord in coordinates.values()})
    figure, axes = plt.subplots(
        len(histories),
        len(times),
        figsize=(10, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    points = None

    for row, (label, probabilities) in enumerate(histories.items()):
        for column, time in enumerate(times):
            axis = axes[row, column]
            indices = [index for index, coord in coordinates.items() if int(coord[2]) == time]
            xs = [coordinates[index][0] for index in indices]
            ys = [coordinates[index][1] for index in indices]
            values = [probabilities[index] for index in indices]
            points = axis.scatter(
                xs,
                ys,
                c=values,
                cmap="magma",
                vmin=0,
                vmax=0.5,
                s=150,
                edgecolors="#303030",
                linewidths=0.7,
            )
            axis.set_title(f"t = {time}")
            axis.set_aspect("equal")
            axis.set_xlim(-0.75, 6.75)
            axis.set_ylim(6.75, -0.75)
            axis.set_xticks([0, 2, 4, 6])
            axis.set_yticks([0, 2, 4, 6])
            if column == 0:
                axis.set_ylabel(f"{label}\nspatial y")
            if row == len(histories) - 1:
                axis.set_xlabel("spatial x")

    if points is None:
        raise ValueError("at least one history is required")
    figure.colorbar(points, ax=axes, label="Detector activation probability", shrink=0.85)
    figure.suptitle(f"Delayed loss of data qubit {DATA_QUBIT}", fontsize=14)
    figure.savefig(IMAGE_PATH, dpi=180)
    plt.close(figure)


def stochastic_loss_model(
    *,
    two_qubit_loss_probability: float,
) -> tuple[noncomp.Model, float]:
    """Build the paper's independent per-operand gate-cancellation model."""
    site_probability = 1 - np.sqrt(1 - two_qubit_loss_probability)
    transition = [[0.0] * 5 for _ in range(5)]
    transition[noncomp.Level.LOST][noncomp.Level.G] = site_probability
    transition[noncomp.Level.LOST][noncomp.Level.E] = site_probability
    model = noncomp.Model(
        transitions={"CX": transition},
        classifier=state_selective_classifier(),
        reset_restores_lost=True,
    )
    return model, float(site_probability)


def main() -> None:
    base = surface_code()
    model = noncomp.Model(classifier=state_selective_classifier())
    coordinates = base.get_detector_coordinates()

    early = noncomp.sample(
        force_loss_after_first_interaction(base, round_number=2),
        model,
        shots=FORCED_SHOTS,
        seed=2,
    )
    late = noncomp.sample(
        force_loss_after_first_interaction(base, round_number=3),
        model,
        shots=FORCED_SHOTS,
        seed=3,
    )

    histories = {
        "loss in round 2": early.detectors.mean(axis=0),
        "loss in round 3": late.detectors.mean(axis=0),
    }
    plot_detector_activity(histories, coordinates)

    print("active detectors by time (activation probability > 0.25)")
    print("time  round-2 loss  round-3 loss")
    early_counts = detector_counts_by_time(histories["loss in round 2"], coordinates)
    late_counts = detector_counts_by_time(histories["loss in round 3"], coordinates)
    for (time, early_active, total), (_, late_active, _) in zip(early_counts, late_counts):
        print(f"{time:>4}  {early_active:>2}/{total:<2}          {late_active:>2}/{total:<2}")

    early_slots = np.flatnonzero(early.heralds.any(axis=0))
    late_slots = np.flatnonzero(late.heralds.any(axis=0))
    print(f"heralded measurement slots: {early_slots.tolist()} and {late_slots.tolist()}")

    stochastic_model, site_probability = stochastic_loss_model(two_qubit_loss_probability=0.002)
    stochastic = noncomp.sample(
        str(base),
        stochastic_model,
        shots=STOCHASTIC_SHOTS,
        seed=7,
    )
    heralded = stochastic.heralds.any(axis=1)
    final_loss = (stochastic.final_status == noncomp.QubitStatus.LOST).any(axis=1)
    print(f"per-operand loss probability: {site_probability:.7f}")
    print(f"shots with detected loss: {heralded.mean():.1%}")
    print(f"shots ending with a lost data site: {final_loss.mean():.1%}")
    print(f"detector activation in heralded shots: {stochastic.detectors[heralded].mean():.1%}")


if __name__ == "__main__":
    main()
