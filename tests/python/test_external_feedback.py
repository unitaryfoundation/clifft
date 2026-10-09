"""Pinned external preparation checked against Clifford faults and a logical Aer oracle."""

from functools import lru_cache
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
from utils_conformance import SamplingMode, assert_joint_distribution

import clifft

FIXTURE = Path(__file__).parents[1] / "fixtures/merlin_bt27.stim"
LOGICAL_QUBITS = [0, 1, 2, 27, 28, 29, 54, 55, 56]


def _source(noise: str = "") -> str:
    lines = FIXTURE.read_text().splitlines()
    if noise:
        first_t = next(i for i, line in enumerate(lines) if line.startswith("T "))
        lines.insert(first_t, noise + " " + " ".join(map(str, range(81))))
    return "\n".join(lines) + "\n"


def _optimize(
    source: str, limit: int | None = None
) -> tuple[clifft.HirModule, clifft.PhasePolynomialPass]:
    hir = clifft.trace(clifft.parse(source))
    phase = (
        clifft.PhasePolynomialPass()
        if limit is None
        else clifft.PhasePolynomialPass(max_variables=limit)
    )
    manager = clifft.HirPassManager()
    for pass_ in (
        clifft.PeepholeFusionPass(),
        phase,
        clifft.RotationSimplificationPass(),
        clifft.StatevectorSqueezePass(),
    ):
        manager.add(pass_)
    manager.run(hir)
    return hir, phase


@pytest.mark.parametrize("noise", ["", "Z_ERROR(0.001)"])
def test_external_preparation_needs_a_complete_phase_region(noise: str) -> None:
    # Z-only noise is a modified diagnostic, not Merlin's depolarizing model.
    source = _source(noise)
    limited, phase = _optimize(source, 32)
    limited_width = clifft.active_width_trace(limited)["peak"]
    assert phase.blocks_capped > 0
    assert phase.expansion_attempts == 0
    for limit in (None, 33, 64):
        reduced, phase = _optimize(source, limit)
        assert reduced.num_t_gates < limited.num_t_gates // 2
        assert clifft.active_width_trace(reduced)["peak"] < limited_width // 2
        assert phase.applied
        assert phase.blocks_expanded > 0
        assert phase.blocks_capped == 0


def test_external_depolarizing_faults_prevent_fixed_constraint_reduction() -> None:
    source = _source("DEPOLARIZE1(0.001)")
    limited, _ = _optimize(source, 32)
    default, _ = _optimize(source)
    assert default.num_t_gates == limited.num_t_gates
    assert clifft.active_width_trace(default)["peak"] == clifft.active_width_trace(limited)["peak"]


@lru_cache(maxsize=None)
def _joint_reference(probability: float) -> np.ndarray:
    import stim
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    # At the pinned revision the decoded logical CCZ tensor consists of these
    # six permutations on three blocks of three logical qubits.
    logical = QuantumCircuit(9)
    logical.h(range(9))
    for a, b, c in permutations(range(3)):
        logical.ccz(a, 3 + b, 6 + c)
    logical.h(range(9))
    logical.save_probabilities()
    result = AerSimulator(method="statevector", max_parallel_threads=1).run(logical).result()
    assert result.success
    expected = np.zeros(1024)
    expected[::2] = np.asarray(result.data()["probabilities"])
    expected[np.abs(expected) < 1e-14] = 0
    if not probability:
        return expected

    # Z faults commute through physical CCZ. The remaining Clifford decoder
    # maps each fault to a detector bit and logical X-outcome flips. Enumerating
    # each single fault then convolving includes every multi-fault branch.
    lines = _source().splitlines()
    first_t = next(i for i, line in enumerate(lines) if line.startswith("T "))
    prefix = "\n".join(lines[:first_t])
    suffix = "\n".join(line for line in lines[first_t:] if not line.startswith(("T ", "T_DAG ")))
    suffix += "\nMX " + " ".join(map(str, LOGICAL_QUBITS))
    for q in range(81):
        circuit = stim.Circuit(prefix + f"\nZ {q}\n" + suffix)
        rows = circuit.compile_sampler(seed=93).sample(shots=4)
        # The first 54 records belong to the switch. Record 54 is detector 0.
        bits = np.column_stack((rows[:, 54], rows[:, -9:])).astype(np.uint8)
        np.testing.assert_array_equal(bits, np.broadcast_to(bits[0], bits.shape))
        mask = int(bits[0].astype(np.int64) @ (1 << np.arange(10)))
        expected = (1 - probability) * expected + probability * expected[np.arange(1024) ^ mask]
    return expected


@pytest.mark.parametrize("probability", [0, 0.001])
def test_external_logical_output_and_syndrome_joint_distribution(
    sampling_mode: SamplingMode, probability: float
) -> None:
    source = _source(f"Z_ERROR({probability})" if probability else "")
    source += "MX " + " ".join(map(str, LOGICAL_QUBITS)) + "\n"
    program = sampling_mode.compile(source)
    assert program.peak_active_width <= 12
    result = sampling_mode.sample(program, shots=8192, seed=579)
    joint = np.column_stack((result.detectors[:, 0], result.measurements[:, -9:]))
    assert_joint_distribution(joint, _joint_reference(probability))
    if not probability:
        np.testing.assert_array_equal(result.detectors, 0)
