"""Postselected phase rewrites checked against exact two-qubit branch probabilities."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np
import pytest
from utils_conformance import SamplingMode, assert_joint_distribution, expectation_atol

import clifft

PROBES = ["".join(axes) for axes in product("IXYZ", repeat=2) if axes != ("I", "I")]
Outcome = tuple[tuple[int, ...], float] | None


def _source(probe: str, position: str, readout: float, difference: bool) -> str:
    pauli = "*".join(f"{axis}{q}" for q, axis in enumerate(probe) if axis != "I")
    lines = ["H 0 1", "DEPOLARIZE1(0.05) 1", "T 0"]
    if position == "between":
        lines.append(f"EXP_VAL {pauli}")
    lines += ["CX 1 0", "T_DAG 0", "CX 1 0"]
    if position == "before":
        lines.append(f"EXP_VAL {pauli}")
    lines.append(f"M({readout}) 1")
    if position == "after":
        lines.append(f"EXP_VAL {pauli}")
    if difference:
        lines += ["X_ERROR(0.13) 1", f"M({readout}) 1"]
    lines += ["CX rec[-1] 0", "H 0", "M 0"]
    if difference:
        lines += ["DETECTOR rec[-3]", "DETECTOR rec[-3] rec[-2]"]
    else:
        lines.append("DETECTOR rec[-2]")
    return "\n".join(lines)


@dataclass(frozen=True)
class Branch:
    probability: float
    state: Any
    records: tuple[int, ...] = ()
    expectation: float = 0.0


@lru_cache(maxsize=360)
def _reference(
    probe: str, position: str, selected_parity: int, readout: float, difference: bool
) -> dict[Outcome, float]:
    from qiskit.circuit.library import CXGate, HGate, TdgGate, TGate, XGate
    from qiskit.quantum_info import Pauli, Statevector

    branches = [Branch(1.0, Statevector.from_int(0, 4))]

    def evolve(gate: Any, qubits: list[int]) -> None:
        nonlocal branches
        branches = [replace(b, state=b.state.evolve(gate, qubits)) for b in branches]

    def observe() -> None:
        nonlocal branches
        # Qiskit writes the highest-index qubit first in Pauli labels.
        branches = [
            replace(b, expectation=float(b.state.expectation_value(Pauli(probe[::-1])).real))
            for b in branches
        ]

    def fault(choices: list[tuple[str, float]]) -> None:
        nonlocal branches
        branches = [
            replace(b, probability=b.probability * p, state=b.state.evolve(Pauli(axis + "I")))
            for b in branches
            for axis, p in choices
        ]

    def measure(qubit: int, readout_probability: float) -> None:
        nonlocal branches
        collapsed = []
        for b in branches:
            for bit in (0, 1):
                data = np.array(b.state.data)
                data[((np.arange(4) >> qubit) & 1) != bit] = 0
                probability = float(np.vdot(data, data).real)
                if probability < 1e-14:
                    continue
                state = Statevector(data / np.sqrt(probability))
                # Readout changes the reported bit, not the physical collapsed state.
                for flip, p in [(0, 1 - readout_probability), (1, readout_probability)]:
                    if p:
                        collapsed.append(
                            replace(
                                b,
                                probability=b.probability * probability * p,
                                state=state,
                                records=b.records + (bit ^ flip,),
                            )
                        )
        branches = collapsed

    evolve(HGate(), [0])
    evolve(HGate(), [1])
    fault([("I", 0.95), ("X", 0.05 / 3), ("Y", 0.05 / 3), ("Z", 0.05 / 3)])
    evolve(TGate(), [0])
    if position == "between":
        observe()
    evolve(CXGate(), [1, 0])
    evolve(TdgGate(), [0])
    evolve(CXGate(), [1, 0])
    if position == "before":
        observe()
    measure(1, readout)
    if position == "after":
        observe()
    if difference:
        fault([("I", 0.87), ("X", 0.13)])
        measure(1, readout)
    branches = [
        replace(b, state=b.state.evolve(XGate(), [0])) if b.records[-1] else b for b in branches
    ]
    evolve(HGate(), [0])
    measure(0, 0.0)

    probabilities: dict[Outcome, float] = {}
    for b in branches:
        survives = b.records[0] == selected_parity and (
            not difference or b.records[0] == b.records[1]
        )
        # Merge numerical roundoff in equivalent oracle branches. Sampled probes
        # are matched separately with the backend's precision tolerance.
        outcome = (b.records, round(b.expectation, 12)) if survives else None
        probabilities[outcome] = probabilities.get(outcome, 0.0) + b.probability
    np.testing.assert_allclose(sum(probabilities.values()), 1, atol=1e-12, rtol=0)
    return probabilities


def _assert_joint_outcomes(
    result: Any, expected: dict[Outcome, float], sampling_mode: SamplingMode
) -> None:
    outcomes = list(expected)
    width = (len(outcomes) - 1).bit_length()
    probabilities = np.zeros(1 << width)
    bins = np.full(result.total_shots, outcomes.index(None), dtype=np.uint64)
    matched = np.zeros(result.passed_shots, dtype=np.uint8)
    assert result.exp_vals.shape == (result.passed_shots, 1)
    for index, outcome in enumerate(outcomes):
        probabilities[index] = expected[outcome]
        if outcome is None:
            continue
        records, expectation = outcome
        matches = np.all(result.measurements == records, axis=1) & np.isclose(
            result.exp_vals[:, 0],
            expectation,
            atol=expectation_atol(sampling_mode, fp64=1e-10),
            rtol=0,
        )
        matched += matches
        bins[np.flatnonzero(matches)] = index
    np.testing.assert_array_equal(matched, 1, err_msg="Unexpected or ambiguous survivor outcome")
    # Include discards in the same joint histogram so conditional distributions
    # cannot hide an incorrect survival probability.
    encoded = ((bins[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8)
    assert_joint_distribution(encoded, probabilities)


@pytest.mark.parametrize("profile", ["unoptimized", "phase", "default"])
@pytest.mark.parametrize("difference", [False, True], ids=["single-record", "difference"])
@pytest.mark.parametrize("readout", [0.0, 0.17], ids=["ideal-readout", "noisy-readout"])
@pytest.mark.parametrize("selected_parity", [0, 1], ids=["select-zero", "select-one"])
@pytest.mark.parametrize("position", ["between", "before", "after"])
@pytest.mark.parametrize("probe", PROBES)
def test_postselected_phase_probes_match_joint_branch_oracle(
    sampling_mode: SamplingMode,
    probe: str,
    position: str,
    selected_parity: int,
    readout: float,
    difference: bool,
    profile: str,
) -> None:
    source = _source(probe, position, readout, difference)
    expected = _reference(probe, position, selected_parity, readout, difference)
    phase = clifft.PhasePolynomialPass()
    if profile == "phase":
        manager = clifft.HirPassManager()
        manager.add(phase)
    else:
        manager = clifft.default_hir_pass_manager() if profile == "default" else None
    program = sampling_mode.compile(
        source,
        hir_passes=manager,
        postselection_mask=[1, 1] if difference else [1],
        expected_detectors=[selected_parity, 0] if difference else [selected_parity],
    )
    if profile == "phase" and readout == 0 and position == "before":
        if probe == "ZI":
            assert phase.applied and phase.output_t_count == 0
        elif probe == "XI":
            assert not phase.applied and phase.output_t_count == 2
    result = sampling_mode.sample_survivors(program, 2048, seed=974, keep_records=True)
    _assert_joint_outcomes(result, expected, sampling_mode)
