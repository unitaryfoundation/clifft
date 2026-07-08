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

    The computational columns define readout confusion for Z-basis
    measurements of computational qubits: the true outcome is misreported
    with the column's off-diagonal probability, an asymmetric in-record
    flip (the qubit still collapses to its true state). Identity
    computational columns add nothing, and a computational column may not
    place probability beyond the two record symbols.
    """

    __slots__ = ("symbols", "matrix")

    def __init__(self, symbols: Sequence[str], matrix: Matrix) -> None:
        self.symbols = [str(s) for s in symbols]
        if len(self.symbols) != len(set(self.symbols)):
            duplicates = sorted({s for s in self.symbols if self.symbols.count(s) > 1})
            raise ValueError(f"noncomp classifier: duplicate symbol label(s): {duplicates}")
        self.matrix = _as_matrix(matrix)


class Model:
    """A noncomputational trajectory model over the built-in five-level set.

    Args:
        initial_state: probability per level, ``P(level)``, summing to one.
        transitions: maps a name to its ``T[to][from]`` matrix. A key that
            names a gate (e.g. ``"CZ"``) is a *hook*: it expands to a
            ``LEVEL_TRANSITION[key]`` annotation after every occurrence of that
            gate. Any key -- gate-named or not -- can be referenced
            directly from the circuit with ``LEVEL_TRANSITION[key] q``, and
            ``LOSS(p) q`` applies a uniform loss inline. A transition
            fires at its circuit position, with the source taken from the
            qubit's state there.
        classifier: optional :class:`Classifier` supplying leaked/lost
            measurement outcomes and computational readout confusion.
        reset_restores_lost: if true, a reset on a lost qubit restores it to
            a computational state; if false (default), the reset acts on the
            vacated site and is dropped.
        damping: exact-mode handling of sites whose no-fire back-action is
            genuinely non-Clifford (a source-dependent transition on a
            coherent qubit outside the amplitude array). ``"exact"`` (the
            default) expands the qubit into the array, adding one to the
            circuit's rank at that site; ``"neglect"`` keeps the rank and
            omits the no-fire back-action, a survivorship tilt of order
            ``|p_g - p_e|`` with no effect on source-independent rates.
            Only meaningful at coherent dormant sites (see the design note).

    An operation with no representable effect on a leaked or lost operand --
    e.g. a two-qubit gate onto a vacated site -- is dropped, acting as the
    identity on the surviving operands. Single-qubit measurements (``M``,
    ``MX``, ``MY``) keep their record slot; on a vacated carrier the readout
    basis is incidental and the classifier supplies the bit. A
    measure-and-reset (``MR``/``MRX``/``MRY``) is kept the same way, with
    the reset additionally re-preparing the site. Parity measurements
    (``MPP``) are not supported when the model can leak or lose qubits — they
    have no faithful single-bit classifier substitution — and raise before
    sampling begins. A model that can leak or lose qubits also requires a
    classifier when the circuit measures.

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
        damping: str = "exact",
    ) -> None:
        transition_matrices = {
            str(gate): _as_matrix(matrix) for gate, matrix in (transitions or {}).items()
        }
        num_symbols = None if classifier is None else len(classifier.symbols)
        matrix = None if classifier is None else classifier.matrix
        self._handle = _clifft_core._build_noncomp_model(
            [float(p) for p in initial_state],
            transition_matrices,
            num_symbols,
            matrix,
            bool(reset_restores_lost),
            str(damping),
        )


class NonComputationalSample:
    """Result of :func:`sample`: the visible records plus a status sidecar.

    Attributes:
        measurements, detectors, observables: uint8 arrays, shape (shots, width).
        final_status: uint8 array (shots, num_qubits) of :class:`QubitStatusKind`.
            Leaked/lost statuses are per-shot truth. Computational qubits
            report as a single category: transitions with computational
            destinations resolve entirely inside the simulator, so no
            final level is claimed.
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
    max_rank: int | None = None,
) -> NonComputationalSample:
    """Sample ``circuit`` under ``model`` for ``shots`` shots.

    ``circuit`` is a parsed ``clifft.Circuit`` or a Stim-format string. Returns a
    :class:`NonComputationalSample`. Single-qubit measurements (``M``, ``MX``,
    ``MY``) of a leaked or lost qubit read the classifier; the readout basis is
    incidental on a vacated carrier. A model that can leak or lose qubits
    requires a classifier when the circuit measures, and parity measurements
    (``MPP``) are not supported with such models — both are rejected before
    sampling begins. Raises ``ValueError`` when one of these contracts is
    violated, when a classifier column is unsupported, or when an operation
    has no representable effect on a noncomputational operand.

    ``max_rank`` caps the compiled peak rank under exact-mode compilation;
    the cap is enforced at each continuation compile, failing with the
    offending circuit line named instead of attempting a ``2**k``
    allocation. Unlimited when ``None``.
    """
    if isinstance(circuit, str):
        circuit = _clifft_core.parse(circuit)
    meas, det, obs, status, heralds, num_qubits, num_meas, num_det, num_obs = (
        _clifft_core._sample_noncomputational(circuit, model._handle, shots, seed, max_rank)
    )
    return NonComputationalSample(
        meas, det, obs, status, heralds, num_qubits, num_meas, num_det, num_obs
    )
