"""Noncomputational (leakage/loss) sampling.

Drives a structural leakage/loss trajectory model on top of the ordinary Clifft
sampler:

    import clifft
    from clifft import noncomp

    model = noncomp.Model(
        initial_state=[1, 0, 0, 0, 0],                  # P(level) over the 5-level set
        transitions={"S": T},                           # gate -> T[to][from]
        classifier=noncomp.Classifier(["0", "1"], P),   # optional; P[symbol][level]
        reset_restores_lost=False,
    )
    r = noncomp.sample("H 0\\nCX 0 1\\nS 0\\nM 0\\nM 1\\n", model, shots=1000, seed=7)
    r.measurements   # np.uint8 [shots, num_measurements]
    r.final_status   # np.uint8 [shots, num_qubits], values in QubitStatusKind

This API supports exactly the built-in five-level set, named by ``Level`` and
``LEVELS`` (g, e, leak_g, leak_e, lost); matrix rows and columns are indexed by
``Level``. A classifier has two or three symbols with stochastic columns: the
first two symbols are the record bit, an optional third symbol heralds the
measurement (reported per slot in ``heralds``; the visible record stays binary
with a uniformly drawn bit). Substochastic (reject) columns and richer
alphabets raise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Iterator

import numpy as np
import numpy.typing as npt

from clifft import _clifft_core

__all__ = [
    "LEVELS",
    "Classifier",
    "Level",
    "Model",
    "NonComputationalSample",
    "QubitStatusKind",
    "sample",
]

Matrix = Sequence[Sequence[float]]


class QubitStatusKind(IntEnum):
    """A qubit's final status in the per-shot sidecar."""

    COMPUTATIONAL = 0
    LEAKED = 1
    LOST = 2


class Level(IntEnum):
    """Indices of the built-in five-level model, for naming matrix rows/columns."""

    G = 0
    E = 1
    LEAK_G = 2
    LEAK_E = 3
    LOST = 4


LEVELS = ("g", "e", "leak_g", "leak_e", "lost")


def _as_matrix(matrix: Matrix) -> list[list[float]]:
    """Normalize a nested sequence or 2-D array to list-of-lists of float."""
    return [[float(x) for x in row] for row in matrix]


class Classifier:
    """A measurement classifier: symbol labels and ``P[symbol][level]``.

    Two or three symbols; each level's column must sum to one (substochastic
    reject columns are not supported). The first two symbols map directly to
    the measurement record bit. An optional third symbol heralds the
    measurement -- typically the loss outcome -- reported per record slot in
    :attr:`NonComputationalSample.heralds` while the visible record keeps a
    uniformly drawn bit, so the record layout is unchanged.
    """

    __slots__ = ("symbols", "matrix")

    def __init__(self, symbols: Sequence[str], matrix: Matrix) -> None:
        self.symbols = [str(s) for s in symbols]
        self.matrix = _as_matrix(matrix)


class Model:
    """A noncomputational trajectory model over the built-in five-level set.

    Args:
        initial_state: probability per level, ``P(level)``, summing to one.
        transitions: maps a gate-name string to its ``T[to][from]`` matrix.
        classifier: optional :class:`Classifier` for leaked/lost measurements.
        reset_restores_lost: if true, a reset on a lost qubit restores it.
        unknown_source_policy: how a source-dependent transition on a qubit
            whose computational state is unknown is handled. ``"reject"``
            (the default) raises; ``"equalize_rates"`` opts into an
            approximation that pads every computational column with a
            diagonal pseudo-jump up to the maximum computational jump rate,
            draws the source uniformly, and collapses the carrier on every
            jump. The approximation matches unbiased unknown-source
            marginals; deterministic-but-untracked states remain
            approximate, and destination-collapse correlations are
            discarded.
        lost_leaked_ops: how an operation with no representable effect on a
            leaked or lost operand is handled. ``"reject"`` (the default)
            raises; ``"drop"`` opts into excising the whole operation,
            acting as the identity on the surviving operands. Measurements
            are never dropped; their record slot is kept and the classifier
            supplies the outcome.

    Construction validates shapes, probabilities, gate keys, policy values,
    and level table consistency in C++, raising ``ValueError`` on any
    problem.
    """

    __slots__ = ("_handle",)

    def __init__(
        self,
        initial_state: Sequence[float],
        transitions: Mapping[str, Matrix] | None = None,
        classifier: Classifier | None = None,
        reset_restores_lost: bool = False,
        unknown_source_policy: str = "reject",
        lost_leaked_ops: str = "reject",
    ) -> None:
        transition_matrices = {
            str(gate): _as_matrix(matrix) for gate, matrix in (transitions or {}).items()
        }
        symbols = None if classifier is None else classifier.symbols
        matrix = None if classifier is None else classifier.matrix
        self._handle = _clifft_core._build_noncomp_model(
            [float(p) for p in initial_state],
            transition_matrices,
            symbols,
            matrix,
            bool(reset_restores_lost),
            str(unknown_source_policy),
            str(lost_leaked_ops),
        )


class NonComputationalSample:
    """Result of :func:`sample`: the visible records plus a status sidecar.

    Attributes:
        measurements, detectors, observables: uint8 arrays, shape (shots, width).
        final_status: uint8 array (shots, num_qubits) of :class:`QubitStatusKind`.
            Coarse: it reports computational/leaked/lost, not the specific leaked
            or lost level.
        heralds: uint8 array (shots, num_measurements); 1 where the classifier
            sampled the herald (third) symbol for that slot, else 0.
        shots, num_qubits, num_measurements, num_detectors, num_observables: ints.
    """

    __slots__ = (
        "measurements",
        "detectors",
        "observables",
        "final_status",
        "heralds",
        "shots",
        "num_qubits",
        "num_measurements",
        "num_detectors",
        "num_observables",
    )

    def __init__(
        self,
        measurements: npt.NDArray[np.uint8],
        detectors: npt.NDArray[np.uint8],
        observables: npt.NDArray[np.uint8],
        final_status: npt.NDArray[np.uint8],
        heralds: npt.NDArray[np.uint8],
        num_qubits: int,
        num_measurements: int,
        num_detectors: int,
        num_observables: int,
    ) -> None:
        self.measurements = measurements
        self.detectors = detectors
        self.observables = observables
        self.final_status = final_status
        self.heralds = heralds
        self.shots = int(measurements.shape[0])
        self.num_qubits = int(num_qubits)
        self.num_measurements = int(num_measurements)
        self.num_detectors = int(num_detectors)
        self.num_observables = int(num_observables)

    def symbols(self) -> npt.NDArray[np.uint8]:
        """Per-slot classifier symbols: the record bit, or 2 where heralded.

        A ternary view of the record for comparing against simulators whose
        measurement outcomes carry the herald in-band as a third value.
        """
        out = self.measurements.copy()
        out[self.heralds != 0] = 2
        return out

    def __iter__(self) -> Iterator[npt.NDArray[np.uint8]]:
        """Yield (measurements, detectors, observables) for tuple unpacking."""
        yield self.measurements
        yield self.detectors
        yield self.observables

    def __repr__(self) -> str:
        return (
            f"NonComputationalSample(shots={self.shots}, num_qubits={self.num_qubits}, "
            f"num_measurements={self.num_measurements}, num_detectors={self.num_detectors}, "
            f"num_observables={self.num_observables})"
        )


def sample(
    circuit: object,
    model: Model,
    shots: int,
    seed: int | None = None,
) -> NonComputationalSample:
    """Sample ``circuit`` under ``model`` for ``shots`` shots.

    ``circuit`` is a parsed ``clifft.Circuit`` or a Stim-format string. Returns a
    :class:`NonComputationalSample`. Raises ``ValueError`` when the trajectory
    policy rejects an operation, when a leaked/lost measurement needs a
    classifier the model lacks, or when a classifier column is unsupported.
    """
    if isinstance(circuit, str):
        circuit = _clifft_core.parse(circuit)
    meas, det, obs, status, heralds, num_qubits, num_meas, num_det, num_obs = (
        _clifft_core._sample_noncomputational(circuit, model._handle, shots, seed)
    )
    return NonComputationalSample(
        meas, det, obs, status, heralds, num_qubits, num_meas, num_det, num_obs
    )
