"""Noncomputational (leakage/loss) sampling.

Samples five-level leakage/loss trajectories using Clifft's executor:

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
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import Callable, Iterator, Literal

import numpy as np
import numpy.typing as npt

import clifft._clifft_core as _clifft_core
from clifft._clifft_core import Circuit

__all__ = [
    "Classifier",
    "InteractionRule",
    "PartnerEffect",
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


@dataclass(frozen=True, slots=True, init=False)
class PartnerEffect:
    """Noise on a computational partner of a leaked or lost operand.

    ``pauli=(px, py, pz)`` specifies mutually exclusive X, Y, and Z errors;
    identity has probability ``1 - px - py - pz``. Full depolarization is
    ``(0.25, 0.25, 0.25)``. The optional leakage attempt happens after the
    Pauli channel, with source-preserving ``g -> leak_g`` and ``e -> leak_e``
    semantics on the partner. Spreading is supported only for leaked sources.
    Both effects default to zero. Values are validated and copied on creation.
    """

    pauli: tuple[float, float, float]
    spread_probability: float

    def __init__(
        self, *, pauli: Sequence[float] = (0.0, 0.0, 0.0), spread_probability: float = 0.0
    ) -> None:
        values = tuple(float(p) for p in pauli)
        if len(values) != 3:
            raise ValueError("pauli requires three probabilities in X, Y, Z order")
        if any(not isfinite(p) or p < 0 or p > 1 for p in values) or sum(values) > 1:
            raise ValueError(
                "Pauli probabilities must be finite, nonnegative, and sum to at most 1"
            )
        spread = float(spread_probability)
        if not isfinite(spread) or not 0 <= spread <= 1:
            raise ValueError("spread_probability must be finite and in [0, 1]")
        object.__setattr__(self, "pauli", values)
        object.__setattr__(self, "spread_probability", spread)

    def _values(self) -> list[float]:
        return [*self.pauli, self.spread_probability]


@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionRule:
    """Replace a gate's default effect for one source status and direction.

    ``source_operand`` is 0 for the first operand in each ordered gate pair,
    or 1 for the second. ``source_status`` is ``"leaked"`` (either leaked
    level) or ``"lost"``. The other operand must be computational. The entire
    ``effect`` replaces the matching default; ``PartnerEffect()`` disables it.
    ``Model`` validates gate support and rejects duplicate canonical rules,
    including aliases such as CX and CNOT. Rules do not affect virtual feedback.
    """

    gate: str
    source_operand: int
    source_status: Literal["leaked", "lost"]
    effect: PartnerEffect

    def __post_init__(self) -> None:
        if not isinstance(self.gate, str):
            raise TypeError("gate must be a gate name")
        if not isinstance(self.source_operand, int) or self.source_operand not in (0, 1):
            raise ValueError("source_operand must be 0 or 1")
        if self.source_status not in ("leaked", "lost"):
            raise ValueError("source_status must be 'leaked' or 'lost'")
        if not isinstance(self.effect, PartnerEffect):
            raise TypeError("effect must be a PartnerEffect")
        if self.source_status == "lost" and self.effect.spread_probability != 0:
            raise ValueError("spreading is only supported for a leaked source")


class Model:
    """A noncomputational trajectory model over the built-in five-level set.

    Args:
        initial_state: probability per level, ``P(level)``, summing to one.
            Defaults to ``[1.0, 0.0, 0.0, 0.0, 0.0]`` (all qubits start in
            the ground state).
        transitions: maps a name to its ``T[to][from]`` matrix. A key that
            names a gate (e.g. ``"CZ"``) is a *hook*: it expands to a
            ``LEVEL_TRANSITION[key]`` annotation after every occurrence of that
            gate. A key naming a recognized instruction that cannot be hooked
            is rejected, since it would otherwise look like a hook that never
            fires. These include noise channels and annotations, as well as
            instructions that never parse into a node of their own:
            ``MXX``/``MYY``/``MZZ`` (desugared to ``MPP``),
            ``CH``/``CCX``/``CCZ`` (decomposed by the parser), and identity
            no-ops. Annotate those positions explicitly instead. Any key --
            whether an arbitrary name or a hookable gate name -- can be
            referenced directly from the circuit with
            ``LEVEL_TRANSITION[key] q``. ``LEAKAGE(p) q`` applies
            source-preserving leakage inline, while ``LOSS(p) q`` applies
            uniform loss. A transition fires at its circuit position, with
            the source taken from the qubit's state there.
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
            that site, increasing peak active width by one. ``"neglect"`` avoids the
            expansion but omits the state update caused by observing that no
            transition occurred. It is exact when ``g`` and ``e`` have the
            same total transition probability; otherwise the bias is of order
            ``|p_g - p_e|``.

        gate_partner_effects: Defaults keyed by ``"leaked"`` or ``"lost"``,
            each a [PartnerEffect][clifft.noncomp.PartnerEffect]. Apply in both
            directions of supported native physical two-qubit unitary gates
            when the other operand is computational. Omitted statuses have no
            partner effect. These defaults exclude noise and virtual feedback.
        interactions: Specific [InteractionRule][clifft.noncomp.InteractionRule]
            overrides. Each replaces the whole matching default. Generated
            interactions run after the gate and before its level-transition
            hooks. Explicit circuit annotations run at their own expanded
            position and compose with generated effects without overriding them.

    By default, an operation with no representable effect on a leaked or lost operand --
    e.g. a two-qubit gate onto a vacated site -- is dropped, acting as the
    identity on the surviving operands. Configured partner interactions replace
    that identity with Pauli noise and optional leakage spreading. Single-qubit measurements (``M``,
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
        "_gate_partner_effects",
        "_interactions",
    )

    def __init__(
        self,
        initial_state: Sequence[float] | None = None,
        transitions: Mapping[str, Matrix] | None = None,
        classifier: Classifier | None = None,
        reset_restores_lost: bool = False,
        damping: str = "exact",
        *,
        gate_partner_effects: Mapping[str, PartnerEffect] | None = None,
        interactions: Sequence[InteractionRule] | None = None,
    ) -> None:
        if initial_state is None:
            initial_state = [1.0, 0.0, 0.0, 0.0, 0.0]
        transition_matrices = {
            str(gate): _as_matrix(matrix) for gate, matrix in (transitions or {}).items()
        }
        matrix = None if classifier is None else classifier.matrix
        self._gate_partner_effects = dict(gate_partner_effects or {})
        self._interactions = tuple(interactions or ())
        if any(
            not isinstance(effect, PartnerEffect) for effect in self._gate_partner_effects.values()
        ):
            raise TypeError("gate_partner_effects values must be PartnerEffect objects")
        if any(not isinstance(rule, InteractionRule) for rule in self._interactions):
            raise TypeError("interactions must contain InteractionRule objects")
        self._handle = _clifft_core._build_noncomp_model(
            [float(p) for p in initial_state],
            transition_matrices,
            matrix,
            bool(reset_restores_lost),
            str(damping),
            {status: effect._values() for status, effect in self._gate_partner_effects.items()},
            [
                (rule.gate, rule.source_operand, rule.source_status, rule.effect._values())
                for rule in self._interactions
            ],
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
        if self._gate_partner_effects:
            parts.append(f"gate_partner_effects={self._gate_partner_effects!r}")
        if self._interactions:
            parts.append(f"interactions={self._interactions!r}")
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
            than ``G`` or ``E`` because their state remains quantum in the executor
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
    max_active_width: int | None = None,
    threads: int | Literal["auto"] = 1,
) -> NonComputationalSample:
    """Sample ``circuit`` under ``model`` for ``shots`` shots.

    On a leaked or lost site, ``M``, ``MX``, ``MY``, ``MR``, ``MRX``, and
    ``MRY`` sample the classifier without regard to measurement basis. A model
    that can leak or lose sites requires a classifier when the circuit
    measures a physical site. Parity measurements (``MPP``) and ``EXP_VAL``
    probes are not supported with such models.

    ``HERALD_LEAKAGE_EVENT`` and ``HERALD_LOSS_EVENT`` append nondestructive
    status checks to the ordinary measurement record. An optional probability
    suppresses positive results only. They require no classifier and leave
    their entries in the classifier ``heralds`` sidecar zero.

    Continuations are compiled with the default optimization passes that
    preserve measurement-record order and instrument-prefix stability, omitting
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
        max_active_width: Optional cap on the peak active width of every compiled
            continuation. The check is conservative because a continuation
            may contain branches that the current shot will not take.
        threads: Number of cross-shot workers. Defaults to 1. Pass ``"auto"``
            to use the implementation-reported hardware concurrency. Seeded
            results are identical for every worker count.

    Returns:
        [NonComputationalSample][clifft.noncomp.NonComputationalSample]
        containing measurement, detector, observable, herald, and final-status
        arrays.

    Raises:
        ValueError: If a model or circuit contract is violated, an annotation
            is malformed, or a continuation exceeds ``max_active_width``.
    """
    return _sample_with(
        circuit,
        model,
        shots,
        seed,
        max_active_width,
        threads,
        _clifft_core._sample_noncomputational,
    )


def _sample_with(
    circuit: Circuit | str,
    model: Model,
    shots: int,
    seed: int | None,
    max_active_width: int | None,
    threads: int | Literal["auto"],
    sampler: Callable,
) -> NonComputationalSample:
    if isinstance(circuit, str):
        circuit = _clifft_core.parse(circuit)
    meas, det, obs, status, heralds, num_qubits, num_meas, num_det, num_obs = sampler(
        circuit, model._handle, shots, seed, max_active_width, threads
    )
    return NonComputationalSample(
        meas, det, obs, status, heralds, num_qubits, num_meas, num_det, num_obs
    )
