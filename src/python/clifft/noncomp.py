"""Noncomputational (leakage/loss) sampling.

This module is new and actively evolving. Its API and supported models may
change as use cases develop.

Samples five-level leakage/loss trajectories using Clifft's VM:

    import clifft
    from clifft import noncomp

    model = noncomp.Model(
        initial_state=[1, 0, 0, 0, 0],                  # P(level) over the 5-level set
        transitions={"S": T},                           # gate -> T[to][from]
        classifier=noncomp.Classifier(P),               # optional; P[symbol][level]
    )
    r = noncomp.sample("H 0\\nCX 0 1\\nS 0\\nM 0\\nM 1\\n", model, shots=1000, seed=7)
    r.measurements   # np.uint8 [shots, num_measurements]
    r.final_status   # np.uint8 [shots, num_qubits], values in QubitStatus

This API supports exactly the built-in five-level set: ``Level.G``, ``Level.E``,
``Level.LEAK_G``, ``Level.LEAK_E``, and ``Level.LOST``. Matrix rows and columns
are indexed by ``Level``. A classifier has two or three symbols, and each
column must sum to one. The first two symbols are the record bit; an optional
third symbol heralds the measurement (reported per slot in ``heralds`` while
the visible record stays binary with a uniformly drawn bit). Classifiers with
other alphabet sizes are rejected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Iterator

import numpy as np
import numpy.typing as npt

from clifft import _clifft_core
from clifft._clifft_core import Circuit

__all__ = [
    "Classifier",
    "Level",
    "Model",
    "NonComputationalSample",
    "QubitStatus",
    "sample",
]

Matrix = Sequence[Sequence[float]]


class QubitStatus(IntEnum):
    """Per-site status stored in ``NonComputationalSample.final_status``.

    These are per-site *status* codes, not matrix indices. ``Level`` names
    matrix rows and columns (indices 0--4); ``QubitStatus`` names per-qubit
    outcomes (codes 0--3). The two enums share member names (``LEAK_G``,
    ``LEAK_E``, ``LOST``) with *different* integer values -- never substitute
    one for the other.

    ``LEAK_G`` and ``LEAK_E`` are individually distinguishable in
    ``final_status``, unlike the coarse leaked/lost grouping some tools use.
    """

    COMPUTATIONAL = 0
    LEAK_G = 1
    LEAK_E = 2
    LOST = 3


class Level(IntEnum):
    """Indices of the built-in five-level model, for naming matrix rows/columns."""

    G = 0
    E = 1
    LEAK_G = 2
    LEAK_E = 3
    LOST = 4


def _as_matrix(matrix: Matrix) -> list[list[float]]:
    """Normalize a nested sequence or 2-D array to list-of-lists of float."""
    return [[float(x) for x in row] for row in matrix]


class Classifier:
    """A measurement classifier: ``P[symbol][level]`` stochastic matrix.

    The matrix must have two or three rows, and every level column must sum to
    one. The first two rows give the probabilities of recording 0 or 1. An
    optional third row heralds the measurement, typically for loss.
    ``NonComputationalSample.heralds`` reports that symbol separately while
    the binary measurement record receives a uniformly sampled placeholder.

    For ``M`` and ``MR`` on a computational site, the ``g`` and ``e`` columns
    can model Z-basis readout confusion after the quantum measurement.
    Computational ``MX``, ``MY``, ``MRX``, and ``MRY`` measurements do not use
    those columns. On a leaked or lost site, every supported single-site
    measurement uses the corresponding classifier column regardless of basis.
    A computational column may not assign probability to the herald symbol.
    """

    __slots__ = ("matrix",)

    def __init__(self, matrix: Matrix) -> None:
        self.matrix = _as_matrix(matrix)


class Model:
    """A noncomputational trajectory model over the built-in five-level set.

    Args:
        initial_state: probability per level, ``P(level)``, summing to one.
            Defaults to ``[1.0, 0.0, 0.0, 0.0, 0.0]`` (all qubits start in
            the ground state).
        transitions: maps a name to its ``T[to][from]`` matrix. A key that
            names a gate (e.g. ``"CZ"``) is a *hook*: it expands to a
            ``LEVEL_TRANSITION[key]`` annotation after every occurrence of that
            gate. A key naming an instruction that never parses into a node
            of its own is rejected, since its hook could never fire:
            ``MXX``/``MYY``/``MZZ`` (desugared to ``MPP``),
            ``CH``/``CCX``/``CCZ`` (decomposed by the parser), and identity
            no-ops. Annotate those positions explicitly instead. Any key --
            gate-named or not -- can be referenced directly from the circuit
            with ``LEVEL_TRANSITION[key] q``, and ``LOSS(p) q`` applies a
            uniform loss inline. A transition fires at its circuit position,
            with the source taken from the qubit's state there.
        classifier: Optional [Classifier][clifft.noncomp.Classifier] supplying
            leaked/lost measurement outcomes and computational readout
            confusion.
        reset_restores_lost: if true, a reset on a lost qubit restores it to
            a computational state; if false (default), the reset acts on the
            vacated site and is dropped.
        damping: handling of the no-transition update when the total
            transition probability differs between ``g`` and ``e`` for a
            coherent qubit that is not yet represented in the state vector.
            ``"exact"`` (the default) adds the qubit to the state vector at
            that site, increasing peak rank by one. ``"neglect"`` avoids the
            expansion but omits the state update caused by observing that no
            transition occurred. It is exact when ``g`` and ``e`` have the
            same total transition probability; otherwise the bias is of order
            ``|p_g - p_e|``.

    An operation with no representable effect on a leaked or lost operand --
    e.g. a two-qubit gate onto a vacated site -- is dropped, acting as the
    identity on the surviving operands. Single-qubit measurements (``M``,
    ``MX``, ``MY``) keep their record slot; once the qubit has left the
    computational subspace the readout basis is incidental and the
    classifier supplies the bit. A
    measure-and-reset (``MR``/``MRX``/``MRY``) keeps its record the same
    way; its reset half re-prepares the site only when the reset restores
    it (a leaked qubit always; a lost qubit only with
    ``reset_restores_lost``). Parity measurements
    (``MPP``) are not supported when the model can leak or lose qubits -- they
    have no faithful single-bit classifier substitution -- and raise before
    sampling begins. A model that can leak or lose qubits also requires a
    classifier when the circuit measures a qubit.

    Construction validates shapes, probabilities, gate keys, policy values,
    and level table consistency, raising ``ValueError`` on any problem.
    """

    __slots__ = (
        "_handle",
        "_transition_keys",
        "_classifier_rows",
        "_reset_restores_lost",
        "_damping",
    )

    def __init__(
        self,
        initial_state: Sequence[float] | None = None,
        transitions: Mapping[str, Matrix] | None = None,
        classifier: Classifier | None = None,
        reset_restores_lost: bool = False,
        damping: str = "exact",
    ) -> None:
        if initial_state is None:
            initial_state = [1.0, 0.0, 0.0, 0.0, 0.0]
        transition_matrices = {
            str(gate): _as_matrix(matrix) for gate, matrix in (transitions or {}).items()
        }
        matrix = None if classifier is None else classifier.matrix
        self._handle = _clifft_core._build_noncomp_model(
            [float(p) for p in initial_state],
            transition_matrices,
            matrix,
            bool(reset_restores_lost),
            str(damping),
        )
        self._transition_keys: list[str] = sorted(transition_matrices.keys())
        self._classifier_rows: int | None = None if classifier is None else len(classifier.matrix)
        self._reset_restores_lost: bool = bool(reset_restores_lost)
        self._damping: str = str(damping)

    def __repr__(self) -> str:
        parts = [f"transitions={self._transition_keys!r}"]
        if self._classifier_rows is not None:
            parts.append(f"classifier={self._classifier_rows}-symbol")
        parts.append(f"reset_restores_lost={self._reset_restores_lost!r}")
        parts.append(f"damping={self._damping!r}")
        return f"Model({', '.join(parts)})"


class NonComputationalSample:
    """Measurement results and final site statuses returned by ``sample()``.

    Attributes:
        measurements, detectors, observables: uint8 arrays, shape (shots, width).
        final_status: uint8 array (shots, num_qubits) of
            [QubitStatus][clifft.noncomp.QubitStatus] values.
            Reports the definite noncomputational level per site and shot:
            ``LEAK_G`` and ``LEAK_E`` are individually distinguishable.
            Computational sites report as ``QubitStatus.COMPUTATIONAL`` rather
            than ``G`` or ``E`` because their state remains quantum in the VM
            and may not be a definite level.
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
        """Return measurement symbols, using 2 for heralded slots.

        This returns a copy of ``measurements`` with each heralded placeholder
        replaced by 2.
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
    circuit: Circuit | str,
    model: Model,
    shots: int,
    seed: int | None = None,
    max_rank: int | None = None,
) -> NonComputationalSample:
    """Sample ``circuit`` under ``model`` for ``shots`` shots.

    On a leaked or lost site, ``M``, ``MX``, ``MY``, ``MR``, ``MRX``, and
    ``MRY`` sample the classifier without regard to measurement basis. A model
    that can leak or lose sites requires a classifier when the circuit
    measures a physical site. Parity measurements (``MPP``) and ``EXP_VAL``
    probes are not supported with such models.

    Continuations are compiled with the default optimization passes that
    preserve measurement-record order, omitting
    [StatevectorSqueezePass][clifft.StatevectorSqueezePass]. Reordering can
    change the placement of internal collapse outcomes relative to later
    records. This API does not currently accept custom pass managers.

    Args:
        circuit: Parsed ``clifft.Circuit`` or Stim-format circuit string.
        model: Leakage and loss model.
        shots: Number of trajectories to sample.
        seed: Seed for reproducible sampling. The same seed and arguments
            produce identical results. When ``None``, each call uses fresh OS
            entropy.
        max_rank: Optional cap on the peak rank of every compiled
            continuation. The check is conservative because a continuation
            may contain branches that the current shot will not take.

    Returns:
        [NonComputationalSample][clifft.noncomp.NonComputationalSample]
        containing measurement, detector, observable, herald, and final-status
        arrays.

    Raises:
        ValueError: If a model or circuit contract is violated, an annotation
            is malformed, or a continuation exceeds ``max_rank``.
    """
    if isinstance(circuit, str):
        circuit = _clifft_core.parse(circuit)
    meas, det, obs, status, heralds, num_qubits, num_meas, num_det, num_obs = (
        _clifft_core._sample_noncomputational(circuit, model._handle, shots, seed, max_rank)
    )
    return NonComputationalSample(
        meas, det, obs, status, heralds, num_qubits, num_meas, num_det, num_obs
    )
