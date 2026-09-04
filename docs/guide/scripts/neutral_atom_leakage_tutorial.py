#!/usr/bin/env python3
"""Compare matched and exact leakage models on neutral-atom Shor circuits.

The schedules and decoder conventions derive from the public SqaleSim Figure 9
supplementary artifact at Zenodo record 17137995, released under Apache-2.0.

Run from the repository root with:

    uv run python docs/guide/scripts/neutral_atom_leakage_tutorial.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from clifft import noncomp

ModelKind = Literal["matched", "exact"]
Counts: TypeAlias = Mapping[str, float | int]
MutableCounts: TypeAlias = dict[str, float | int]

CIRCUIT_DIR = Path(__file__).parents[1] / "circuits" / "neutral_atom"
CIRCUIT_FILES = {
    "unencoded": "unencoded_alpha1.stim",
    "two_row": "two_row_alpha1.stim",
    "two_row_ldu": "two_row_ldu_alpha1.stim",
    "three_row": "three_row_alpha1.stim",
}
IDEAL_DISTRIBUTION = {"000": 0.25, "010": 0.25, "101": 0.25, "111": 0.25}

INITIAL_LEVELS = (0.007 / 2, 1 - 0.007 - 0.014, 0.007 / 2, 0.014, 0.0)
CLASSIFIER_ERRORS = (0.002, 0.023)
CZ_PHASE_ERROR = 0.02168835419643766
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


def cz_jump_matrix() -> np.ndarray:
    """Combine CZ level changes with the artifact's phase-flip jump channel."""
    phase = np.zeros((5, 5), dtype=float)
    phase[noncomp.Level.G, noncomp.Level.G] = 2 * CZ_PHASE_ERROR
    phase[noncomp.Level.E, noncomp.Level.E] = 2 * CZ_PHASE_ERROR
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


def make_model(kind: ModelKind) -> noncomp.Model:
    """Build either the approximation-matched or exact trajectory model."""
    if kind not in ("matched", "exact"):
        raise ValueError(f"unknown model kind: {kind!r}")

    transitions = {
        "CZ": cz_jump_matrix(),
        "RZ_TRANSITION": np.asarray(RZ_LEVEL_TRANSITIONS, dtype=float),
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
) -> ExperimentResult:
    """Sample and decode one checked-in alpha=1 schedule."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    try:
        circuit_file = CIRCUIT_FILES[circuit]
    except KeyError as error:
        raise ValueError(f"unknown circuit: {circuit!r}") from error

    result = noncomp.sample(
        (CIRCUIT_DIR / circuit_file).read_text(),
        make_model(model),
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
        shots=shots,
        accepted=accepted,
        decoded_samples=decoded_samples,
        heralded=int(result.heralds.any(axis=1).sum()),
        tvd=total_variation_distance(decoded, IDEAL_DISTRIBUTION),
    )


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
        choices=("matched", "exact"),
        default=("matched",),
    )
    parser.add_argument("--shots", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
