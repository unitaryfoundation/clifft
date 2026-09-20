"""Independent checks of reconstructed physical fold-transversal cultivation."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import stim
from folded_msc import FOLD, X_CHECKS, Z_CHECKS, Builder, Circuit, Operation, make_circuit
from validate_folded_msc import aer_trajectory, bind_faults, native_replay, validate_trajectories

import clifft


def test_injection_growth_and_logical_clifford_against_aer():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, Statevector
    from qiskit_aer import AerSimulator

    builder = Builder(0)
    builder.injection()
    builder.growth()
    circuit = bind_faults(builder.circuit, {})
    qc = QuantumCircuit(13)
    for op in circuit.operations:
        getattr(qc, {"R": "reset", "T_DAG": "tdg"}.get(op.name, op.name.lower()))(*op.qubits)
    qc.save_statevector()
    state = Statevector(
        AerSimulator(method="statevector", max_parallel_threads=1)
        .run(qc)
        .result()
        .get_statevector()
    )
    for axis, checks in (("X", X_CHECKS), ("Z", Z_CHECKS)):
        for support in checks:
            label = "".join(axis if q in support else "I" for q in reversed(range(13)))
            assert state.expectation_value(Pauli(label)).real == pytest.approx(1, abs=1e-12)
    # Evaluate the published Clifford expression directly, without controlled
    # gates, cat encoding or the generator's factor construction.
    folded = QuantumCircuit(13)
    folded.x(list(FOLD))
    folded.s([0, 6, 12])
    folded.sdg([3, 9])
    for a, b in ((1, 5), (4, 8), (7, 11), (2, 10)):
        folded.cz(a, b)
    folded.global_phase = -np.pi / 4
    # Preserve the reference because evolve can share the input array while
    # applying a circuit's global phase in the installed Qiskit version.
    expected = state.data.copy()
    np.testing.assert_allclose(state.evolve(folded).data, expected, atol=1e-12)
    logical_x = Pauli("".join("X" if q in FOLD else "I" for q in reversed(range(13))))
    assert state.expectation_value(logical_x).real == pytest.approx(2**-0.5, abs=1e-12)


def test_noiseless_complete_protocol_and_physical_noise_sites():
    circuit = make_circuit(0)
    samples = clifft.sample(clifft.compile(circuit.text()), 256, seed=17, threads=1)
    assert samples.measurements.shape == (256, 43)
    assert samples.detectors.shape == (256, 42)
    assert not samples.detectors.any()
    assert not samples.observables.any()
    protocol = make_circuit(0.001, evaluation="none")
    counts = Counter(op.name for op in protocol.operations if op.probability is None)
    assert sum(counts[name] for name in ("T", "T_DAG", "CCZ")) == 29
    for i, op in enumerate(protocol.operations):
        if op.probability is not None:
            continue
        if op.name == "M":
            noise = protocol.operations[i - 1]
            assert noise.name == "X_ERROR" and noise.qubits == op.qubits
        else:
            noise = protocol.operations[i + 1]
            expected = "X_ERROR" if op.name == "R" else f"DEPOLARIZE{len(op.qubits)}"
            assert noise.name == expected and noise.qubits == op.qubits
        assert noise.probability == 0.001
    assert all(op.probability is None for op in circuit.operations if op.stage.startswith("final_"))


def test_css_checks_match_stim_and_logical_distance():
    checks = [
        stim.PauliString("".join(axis if q in row else "I" for q in range(13)))
        for axis, rows in (("X", X_CHECKS), ("Z", Z_CHECKS))
        for row in rows
    ]
    logical_z = stim.PauliString("ZZZ" + "I" * 10)
    stim.Tableau.from_stabilizers(checks + [logical_z])
    # Enumerate both logical cosets, not just the displayed representatives.
    for rows, initial in ((X_CHECKS, FOLD), (Z_CHECKS, (0, 1, 2))):
        masks = [sum(1 << q for q in row) for row in rows]
        words = [sum(1 << q for q in initial)]
        for mask in masks:
            words += [word ^ mask for word in words]
        assert min(word.bit_count() for word in words) == 3


def test_fault_conditioned_physical_histories_against_aer():
    native = Path("build-study/replay_cultivation")
    if not native.is_file():
        pytest.skip("build replay_cultivation")
    results = validate_trajectories(make_circuit(0.02), native, count=6)
    assert any(result["faults"] for result in results)
    assert any(result["aer_log_probability"] < -0.5 for result in results)


def test_final_logical_readout_detects_a_known_logical_error():
    native = Path("build-study/replay_cultivation")
    if not native.is_file():
        pytest.skip("build replay_cultivation")
    original = bind_faults(make_circuit(0), {})
    boundary = next(i for i, op in enumerate(original.operations) if op.stage == "final_syndrome")
    faults = [Operation("Z", (q,), "test_fault", "logical Z witness") for q in (0, 1, 2)]
    physical = Circuit(
        original.operations[:boundary] + faults + original.operations[boundary:],
        original.measurements,
    )
    records, expected = aer_trajectory(physical, 5)
    assert records[:43] == "0" * 42 + "1"
    result = native_replay(physical, records, native)
    assert result["reachable"]
    assert result["log_probability"] == pytest.approx(expected, abs=1e-12)
