#!/usr/bin/env python3
"""Reproduce a neutral-atom logical noise sweep and test no-jump approximations.

The schedules and decoder conventions derive from the public SqaleSim Figure 9
supplementary artifact at Zenodo record 17137995, released under Apache-2.0.

Run from the repository root with:

    uv run python docs/guide/scripts/neutral_atom_leakage_tutorial.py

Generate the tutorial figures with:

    uv run --with matplotlib python docs/guide/scripts/neutral_atom_leakage_tutorial.py \
        --figures
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import asin, sin, sqrt
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from clifft import noncomp

ModelKind = Literal["matched", "exact"]
Counts: TypeAlias = Mapping[str, float | int]
MutableCounts: TypeAlias = dict[str, float | int]
MODEL_KINDS: tuple[ModelKind, ...] = ("matched", "exact")

CIRCUIT_DIR = Path(__file__).parents[1] / "circuits" / "neutral_atom"
CIRCUIT_FILES = {
    "unencoded": "unencoded_alpha1.stim",
    "two_row": "two_row_alpha1.stim",
    "two_row_ldu": "two_row_ldu_alpha1.stim",
    "three_row": "three_row_alpha1.stim",
}
IDEAL_DISTRIBUTION = {"000": 0.25, "010": 0.25, "101": 0.25, "111": 0.25}
FIGURE9_ALPHAS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
ASYMMETRY_ALPHAS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
ASYMMETRIES_PP = (0.5, 2.0, 4.0)
IMAGE_DIR = Path(__file__).parents[1] / "images"
FIGURE9_IMAGE = IMAGE_DIR / "neutral_atom_figure9.png"
ASYMMETRY_IMAGE = IMAGE_DIR / "neutral_atom_rate_asymmetry.png"

INITIAL_LEVELS = (0.007 / 2, 1 - 0.007 - 0.014, 0.007 / 2, 0.014, 0.0)
CLASSIFIER_ERRORS = (0.002, 0.023)
CZ_PHASE_ERROR = 0.02168835419643766
MOVEMENT_PHASE_ERROR = 0.012714743326508494
CZ_LEVEL_TRANSITIONS = (
    (0.00001740, 0.00018500, 0.00000486, 0.00016541, 0.0),
    (0.00001853, 0.00019750, 0.00000461, 0.00017774, 0.0),
    (0.00003113, 0.00042026, 0.00004574, 0.00120999, 0.0),
    (0.00004212, 0.00059021, 0.00005294, 0.00190101, 0.0),
    (0.00000000, 0.00385371, 0.00000000, 0.00000000, 0.0),
)
RZ_LEVEL_TRANSITIONS = (
    (0.0, 2 * 0.00066 * 0.1571993, 0.0, 0.0, 0.0),
    (0.0, 2 * 0.00066 * 0.1692620, 0.0, 0.0, 0.0),
    (0.0, 2 * 0.00066 * 0.3972632, 0.0, 0.0, 0.0),
    (0.0, 2 * 0.00066 * 0.2762855, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
)


def compose_jump_matrices(first: npt.ArrayLike, second: npt.ArrayLike) -> np.ndarray:
    """Compose two jump channels using the public artifact's convention."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    overlap = second_array @ first_array
    second_only = second_array * (1 - first_array.sum(axis=0, keepdims=True))
    first_only = first_array * (1 - second_array.sum(axis=0, keepdims=True).T)
    composed: np.ndarray = overlap + second_only + first_only
    return composed


def equalize_computational_jump_rates(matrix: npt.ArrayLike) -> np.ndarray:
    """Add a self-jump so the computational source columns have equal rates."""
    probabilities = np.array(matrix, dtype=float, copy=True)
    p_g, p_e = probabilities[:, :2].sum(axis=0)
    if p_g > p_e:
        probabilities[noncomp.Level.E, noncomp.Level.E] += p_g - p_e
    else:
        probabilities[noncomp.Level.G, noncomp.Level.G] += p_e - p_g
    return probabilities


def set_computational_jump_rate_difference(
    matrix: npt.ArrayLike,
    difference: float,
) -> np.ndarray:
    """Set p_e - p_g while preserving their mean and jump destinations."""
    probabilities = np.array(matrix, dtype=float, copy=True)
    p_g, p_e = probabilities[:, :2].sum(axis=0)
    mean = (p_g + p_e) / 2
    target_g = mean - difference / 2
    target_e = mean + difference / 2
    if target_g < 0 or target_e > 1:
        raise ValueError(f"rate difference {difference} is incompatible with mean {mean}")
    for column, current, target in (
        (noncomp.Level.G, p_g, target_g),
        (noncomp.Level.E, p_e, target_e),
    ):
        if current <= 0:
            raise ValueError("cannot rescale an empty computational source column")
        probabilities[:, column] *= target / current
    return probabilities


def cz_jump_matrix(alpha: float = 1.0) -> np.ndarray:
    """Combine CZ level changes with the artifact's phase-flip jump channel."""
    phase = np.zeros((5, 5), dtype=float)
    phase[noncomp.Level.G, noncomp.Level.G] = 2 * alpha * CZ_PHASE_ERROR
    phase[noncomp.Level.E, noncomp.Level.E] = 2 * alpha * CZ_PHASE_ERROR
    return compose_jump_matrices(CZ_LEVEL_TRANSITIONS, phase)


def classifier_matrix() -> np.ndarray:
    """Return P[symbol][level] for zero, one, and heralded loss."""
    g_error, e_error = CLASSIFIER_ERRORS
    return np.array(
        [
            [1 - g_error, e_error, 1 - g_error, e_error, 0.0],
            [g_error, 1 - e_error, g_error, 1 - e_error, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )


def make_model(
    kind: ModelKind,
    *,
    alpha: float = 1.0,
    cz_asymmetry: float | None = None,
    isolate_cz_asymmetry: bool = False,
) -> noncomp.Model:
    """Build either the approximation-matched or exact trajectory model."""
    if kind not in ("matched", "exact"):
        raise ValueError(f"unknown model kind: {kind!r}")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    cz_transition = cz_jump_matrix(alpha)
    if cz_asymmetry is not None:
        cz_transition = set_computational_jump_rate_difference(cz_transition, cz_asymmetry)
    rz_transition = np.asarray(RZ_LEVEL_TRANSITIONS, dtype=float)
    if isolate_cz_asymmetry:
        # Equalizing RZ in both arms leaves CZ as the only changed no-jump filter.
        rz_transition = equalize_computational_jump_rates(rz_transition)
    transitions = {
        "CZ": cz_transition,
        "RZ_TRANSITION": rz_transition,
    }
    if kind == "matched":
        transitions = {
            name: equalize_computational_jump_rates(matrix) for name, matrix in transitions.items()
        }

    transition_data: dict[str, list[list[float]]] = {
        name: [[float(value) for value in row] for row in matrix]
        for name, matrix in transitions.items()
    }
    classifier_data = [[float(value) for value in row] for row in classifier_matrix()]

    return noncomp.Model(
        initial_state=INITIAL_LEVELS,
        transitions=transition_data,
        classifier=noncomp.Classifier(classifier_data),
        reset_restores_lost=False,
        damping="neglect" if kind == "matched" else "exact",
    )


def scale_circuit_noise(circuit_text: str, alpha: float) -> str:
    """Apply the public artifact's selected alpha scaling to an alpha=1 circuit."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    def replace_probability(match: re.Match[str]) -> str:
        probability = float(match.group(1))
        if np.isclose(probability, MOVEMENT_PHASE_ERROR, rtol=0, atol=1e-15):
            scaled = alpha * probability
        else:
            # Physical RZ overrotation scales as an angle, not as a probability.
            scaled = sin(alpha * asin(sqrt(probability))) ** 2
        return f"Z_ERROR({scaled:.17g})"

    return re.sub(r"Z_ERROR\(([^)]+)\)", replace_probability, circuit_text)


def process_counts(
    counts: Counts,
    data_qubits: Sequence[int] | None = None,
    flag_qubits: Sequence[int] = (),
    *,
    post_select_on_loss: bool = True,
) -> MutableCounts:
    """Select measurement positions and discard invalid flags or losses."""
    processed: MutableCounts = {}
    for key, value in counts.items():
        if not all(key[index] == "0" for index in flag_qubits):
            continue
        if data_qubits is None:
            output = "".join(bit for index, bit in enumerate(key) if index not in flag_qubits)
        else:
            output = "".join(key[index] for index in data_qubits)
        if post_select_on_loss and not set(output) <= {"0", "1"}:
            continue
        processed[output] = processed.get(output, 0) + value
    return processed


PHYSICAL_TO_LOGICAL = {
    "0000": "00",
    "1111": "00",
    "0011": "01",
    "1100": "01",
    "0101": "10",
    "1010": "10",
    "0110": "11",
    "1001": "11",
}


def decode_blocks(counts: Counts, *, correct_loss: bool = False) -> MutableCounts:
    """Decode concatenated [[4, 2, 2]] measurement blocks."""
    decoded: MutableCounts = {}
    for key, value in counts.items():
        if len(key) % 4:
            raise ValueError("encoded strings must contain complete four-bit blocks")
        blocks = []
        for index in range(0, len(key), 4):
            block = key[index : index + 4]
            if correct_loss and block.count("2") == 1:
                block = block.replace("2", str(block.count("1") % 2))
            blocks.append(PHYSICAL_TO_LOGICAL.get(block))
        if None not in blocks:
            output = "".join(block for block in blocks if block is not None)
            decoded[output] = decoded.get(output, 0) + value
    return decoded


def decode_two_row(counts: Counts) -> MutableCounts:
    """Decode the two-row schedule after flag and loss postselection."""
    data = [index for index in range(10) if index % 5]
    selected = process_counts(counts, data, [0])
    return process_counts(decode_blocks(selected, correct_loss=True), [0, 1, 2])


def decode_two_row_ldu(counts: Counts) -> MutableCounts:
    """Decode the LDU schedule with zero-valued preparation and LDU flags."""
    data = [index for index in range(5, 15) if index % 5]
    flags = [5, 1, 2, 3, 4, 16, 17, 18, 19]
    selected = process_counts(counts, data, flags)
    return process_counts(decode_blocks(selected, correct_loss=True), [0, 1, 2])


def decode_three_row(counts: Counts) -> MutableCounts:
    """Decode two logical samples from each accepted three-row trajectory."""
    data = [index for index in range(15) if index % 5]
    selected = process_counts(counts, data, [0, 5, 10])
    decoded = decode_blocks(selected, correct_loss=True)
    first = Counter(process_counts(decoded, [0, 2, 4]))
    second = Counter(process_counts(decoded, [1, 3, 5]))
    return dict(first + second)


def decode_counts(circuit: str, counts: Counts) -> MutableCounts:
    """Apply the published Figure 9 postprocessing for one schedule."""
    if circuit == "unencoded":
        return {key: value for key, value in counts.items() if set(key) <= {"0", "1"}}
    if circuit == "two_row":
        return decode_two_row(counts)
    if circuit == "two_row_ldu":
        return decode_two_row_ldu(counts)
    if circuit == "three_row":
        return decode_three_row(counts)
    raise ValueError(f"unknown circuit: {circuit!r}")


def total_variation_distance(observed: Counts, expected: Counts) -> float:
    """Return TVD after independently normalizing two count mappings."""
    observed_total = float(sum(observed.values()))
    expected_total = float(sum(expected.values()))
    if observed_total <= 0 or expected_total <= 0:
        raise ValueError("TVD requires nonempty distributions")
    outcomes = set(observed) | set(expected)
    return 0.5 * sum(
        abs(observed.get(key, 0) / observed_total - expected.get(key, 0) / expected_total)
        for key in outcomes
    )


@dataclass(frozen=True)
class ExperimentResult:
    """Summary of one sampled circuit and model."""

    circuit: str
    model: ModelKind
    alpha: float
    shots: int
    accepted: int
    decoded_samples: int
    heralded: int
    tvd: float

    @property
    def acceptance(self) -> float:
        return self.accepted / self.shots


def run_experiment(
    circuit: str,
    model: ModelKind,
    *,
    shots: int,
    seed: int,
    alpha: float = 1.0,
    cz_asymmetry: float | None = None,
    isolate_cz_asymmetry: bool = False,
) -> ExperimentResult:
    """Sample and decode one checked-in schedule at a selected noise multiplier."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    try:
        circuit_file = CIRCUIT_FILES[circuit]
    except KeyError as error:
        raise ValueError(f"unknown circuit: {circuit!r}") from error

    circuit_text = (CIRCUIT_DIR / circuit_file).read_text()
    result = noncomp.sample(
        scale_circuit_noise(circuit_text, alpha),
        make_model(
            model,
            alpha=alpha,
            cz_asymmetry=cz_asymmetry,
            isolate_cz_asymmetry=isolate_cz_asymmetry,
        ),
        shots=shots,
        seed=seed,
    )
    raw_counts = Counter("".join(str(symbol) for symbol in row) for row in result.symbols())
    decoded = decode_counts(circuit, raw_counts)
    decoded_samples = int(sum(decoded.values()))
    accepted = decoded_samples // 2 if circuit == "three_row" else decoded_samples
    return ExperimentResult(
        circuit=circuit,
        model=model,
        alpha=alpha,
        shots=shots,
        accepted=accepted,
        decoded_samples=decoded_samples,
        heralded=int(result.heralds.any(axis=1).sum()),
        tvd=total_variation_distance(decoded, IDEAL_DISTRIBUTION),
    )


@dataclass(frozen=True)
class AcceptanceDifference:
    """Exact-minus-matched acceptance for one controlled asymmetry point."""

    alpha: float
    asymmetry_pp: float
    difference: float
    standard_error: float


def figure9_sweep(*, shots: int, seed: int) -> list[ExperimentResult]:
    """Run both models across the paper's four schedules and alpha grid."""
    results = []
    for circuit_index, circuit in enumerate(CIRCUIT_FILES):
        for alpha_index, alpha in enumerate(FIGURE9_ALPHAS):
            for model_index, model in enumerate(MODEL_KINDS):
                print(
                    f"Figure 9 sweep: {circuit}, {model}, alpha={alpha:g}",
                    flush=True,
                )
                results.append(
                    run_experiment(
                        circuit,
                        model,
                        shots=shots,
                        seed=seed + 10_000 * model_index + 100 * circuit_index + alpha_index,
                        alpha=alpha,
                    )
                )
    return results


def asymmetry_sweep(*, shots: int, seed: int) -> list[AcceptanceDifference]:
    """Measure approximation bias as CZ rate asymmetry and alpha vary."""
    results = []
    for asymmetry_index, asymmetry_pp in enumerate(ASYMMETRIES_PP):
        for alpha_index, alpha in enumerate(ASYMMETRY_ALPHAS):
            point_seed = seed + 1000 + 100 * asymmetry_index + alpha_index
            print(
                f"Asymmetry sweep: delta={asymmetry_pp:g} pp, alpha={alpha:g}",
                flush=True,
            )
            matched = run_experiment(
                "two_row_ldu",
                "matched",
                shots=shots,
                seed=point_seed,
                alpha=alpha,
                cz_asymmetry=asymmetry_pp / 100,
                isolate_cz_asymmetry=True,
            )
            exact = run_experiment(
                "two_row_ldu",
                "exact",
                shots=shots,
                seed=point_seed + 10_000,
                alpha=alpha,
                cz_asymmetry=asymmetry_pp / 100,
                isolate_cz_asymmetry=True,
            )
            standard_error = sqrt(
                matched.acceptance * (1 - matched.acceptance) / shots
                + exact.acceptance * (1 - exact.acceptance) / shots
            )
            results.append(
                AcceptanceDifference(
                    alpha=alpha,
                    asymmetry_pp=asymmetry_pp,
                    difference=exact.acceptance - matched.acceptance,
                    standard_error=standard_error,
                )
            )
    return results


def configure_plot_style() -> None:
    """Use a compact documentation-friendly plotting style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )


def plot_figure9(results: Sequence[ExperimentResult]) -> None:
    """Plot the Clifft reconstruction of the paper's Figure 9 sweep."""
    import matplotlib.pyplot as plt

    colors = {
        "unencoded": "#222222",
        "two_row": "#D97706",
        "two_row_ldu": "#2563A5",
        "three_row": "#16805D",
    }
    labels = {
        "unencoded": "unencoded",
        "two_row": "two-row",
        "two_row_ldu": "two-row + LDU",
        "three_row": "three-row",
    }
    figure, axis = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    for circuit in CIRCUIT_FILES:
        matched = sorted(
            (
                result
                for result in results
                if result.circuit == circuit and result.model == "matched"
            ),
            key=lambda result: result.alpha,
        )
        axis.plot(
            [point.alpha for point in matched],
            [point.tvd for point in matched],
            marker="o",
            linewidth=2,
            markersize=4.5,
            color=colors[circuit],
            label=labels[circuit],
        )
        exact = sorted(
            (result for result in results if result.circuit == circuit and result.model == "exact"),
            key=lambda result: result.alpha,
        )
        axis.plot(
            [point.alpha for point in exact],
            [point.tvd for point in exact],
            linestyle="none",
            marker="x",
            markeredgewidth=1.4,
            markersize=5.5,
            color=colors[circuit],
            label="exact" if circuit == "unencoded" else None,
        )
    axis.axvline(1, color="#777777", linestyle=":", linewidth=1)
    axis.set(xlabel="selected noise multiplier alpha", ylabel="TVD after postselection")
    axis.set_xlim(0.4, 5.1)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3)
    axis.set_title("Clifft reconstruction of the Figure 9 noise sweep")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE9_IMAGE, dpi=180)
    plt.close(figure)


def plot_asymmetry(results: Sequence[AcceptanceDifference]) -> None:
    """Plot exact-minus-matched acceptance under controlled CZ asymmetry."""
    import matplotlib.pyplot as plt

    colors = {0.5: "#D97706", 2.0: "#2563A5", 4.0: "#B42318"}
    figure, axis = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    axis.axhline(0, color="#666666", linewidth=1, linestyle="--")
    for asymmetry_pp in ASYMMETRIES_PP:
        points = sorted(
            (point for point in results if point.asymmetry_pp == asymmetry_pp),
            key=lambda point: point.alpha,
        )
        xs = [point.alpha for point in points]
        ys = [100 * point.difference for point in points]
        errors = [196 * point.standard_error for point in points]
        axis.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            color=colors[asymmetry_pp],
            label=f"{asymmetry_pp:g} pp",
        )
        axis.fill_between(
            xs,
            [value - error for value, error in zip(ys, errors, strict=True)],
            [value + error for value, error in zip(ys, errors, strict=True)],
            color=colors[asymmetry_pp],
            alpha=0.12,
            linewidth=0,
        )
    axis.set(
        xlabel="selected noise multiplier alpha",
        ylabel="accepted-shot change, exact - rebalanced (pp)",
    )
    axis.set_xlim(0.4, 5.1)
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(title="CZ |p_g - p_e|", frameon=False, ncol=3)
    axis.set_title("Rate asymmetry exposes rebalancing bias")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASYMMETRY_IMAGE, dpi=180)
    plt.close(figure)


def generate_figures(*, figure9_shots: int, asymmetry_shots: int, seed: int) -> None:
    """Run both tutorial sweeps and write their figures."""
    configure_plot_style()
    plot_figure9(figure9_sweep(shots=figure9_shots, seed=seed))
    plot_asymmetry(asymmetry_sweep(shots=asymmetry_shots, seed=seed))
    print(f"wrote {FIGURE9_IMAGE}")
    print(f"wrote {ASYMMETRY_IMAGE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuits",
        nargs="+",
        choices=tuple(CIRCUIT_FILES),
        default=tuple(CIRCUIT_FILES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KINDS,
        default=("matched",),
    )
    parser.add_argument("--shots", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--figures",
        action="store_true",
        help="run the noise and asymmetry sweeps and write both tutorial figures",
    )
    parser.add_argument(
        "--figure9-shots",
        type=int,
        default=2_000,
        help="trajectories per model and point in the Figure 9 sweep",
    )
    parser.add_argument(
        "--asymmetry-shots",
        type=int,
        default=1_000,
        help="trajectories per model and point in the CZ-asymmetry sweep",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shots <= 0 or args.figure9_shots <= 0 or args.asymmetry_shots <= 0:
        raise ValueError("shot counts must be positive")
    if args.figures:
        generate_figures(
            figure9_shots=args.figure9_shots,
            asymmetry_shots=args.asymmetry_shots,
            seed=args.seed,
        )
        return 0

    print("circuit       model      acceptance   heralded       TVD")
    for circuit_index, circuit in enumerate(args.circuits):
        for model in args.models:
            result = run_experiment(
                circuit,
                model,
                shots=args.shots,
                seed=args.seed + circuit_index,
            )
            print(
                f"{circuit:<13} {model:<10} {result.acceptance:>9.1%} "
                f"{result.heralded / result.shots:>10.1%} {result.tvd:>9.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
