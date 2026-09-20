"""Check the study transformations and independent state diagnostics."""

import numpy as np
import pytest
import stim
from cultivation_pauli_oracle import record_parity
from soft_cultivation_study import clifford_proxy, source_text, stabilizer_diagnostic, zero_noise


def test_zero_noise_preserves_nested_physical_instructions():
    text = "RX 0\nREPEAT 2 {\n    T 0\n    DEPOLARIZE1(0.125) 0\n    MX(0.25) 0\n}\n"
    zero = zero_noise(text)
    assert "T 0" in zero
    assert "DEPOLARIZE1(0)" in zero
    assert "MX(0)" in zero
    proxy = stim.Circuit(clifford_proxy(zero))
    for op in proxy.flattened():
        if op.name in {"DEPOLARIZE1", "MX"}:
            assert op.gate_args_copy() == [0]


def test_pinned_source_has_deterministic_nonzero_clifford_reference():
    source, _ = source_text()
    circuit = stim.Circuit(clifford_proxy(zero_noise(source)))
    circuit.detector_error_model()
    converter = circuit.compile_m2d_converter(skip_reference_sample=True)
    d, o = converter.convert(
        measurements=circuit.reference_sample()[None, :], separate_observables=True
    )
    assert np.flatnonzero(d[0]).tolist() == [270]
    assert not o.any()


@pytest.mark.parametrize("magic_count", [0, 1, 3])
def test_numerical_nullity_for_known_product_states(magic_count):
    state = np.array([1.0 + 0j])
    for q in range(4):
        theta = np.pi / 8 if q < magic_count else 0
        state = np.kron(state, [np.cos(theta), np.sin(theta)])
    result = stabilizer_diagnostic(state)
    assert result["numerical_nullity"] == magic_count


@pytest.mark.parametrize(
    "circuit,selected,expected",
    [
        ("RX 0\nS 0\nMX 0", {0}, 2**-0.5),
        ("RX 0\nS_DAG 0\nMX 0", {0}, 2**-0.5),
        ("RX 0\nS 0\nMX 0\nS 0\nMX 0", {1}, 0.5),
        ("RX 0\nS 0\nMX 0\nMX 0", {0, 1}, 1),
        ("RX 0\nS 0\nR 0\nM 0", {0}, 1),
        ("RX 0\nS 0\nMPP Y0", {0}, 2**-0.5),
        ("RX 0\nS_DAG 0\nMPP Y0", {0}, -(2**-0.5)),
    ],
)
def test_exact_reference_measurement_and_reset_duals(circuit, selected, expected):
    assert record_parity(circuit, selected)["expectation"] == pytest.approx(expected, abs=1e-14)


def test_exact_reference_against_qiskit_unitary_expectations():
    qiskit = pytest.importorskip("qiskit")
    aer = pytest.importorskip("qiskit_aer")
    from qiskit.quantum_info import Pauli, Statevector

    rng = np.random.default_rng(718)
    simulator = aer.AerSimulator(method="statevector", max_parallel_threads=1)
    for _ in range(20):
        qc = qiskit.QuantumCircuit(3)
        qc.h(range(3))
        lines = ["RX 0 1 2"]
        for _ in range(6):
            q = int(rng.integers(3))
            dagger = bool(rng.integers(2))
            (qc.tdg if dagger else qc.t)(q)
            lines.append(f"S{'_DAG' if dagger else ''} {q}")
            c, t = rng.choice(3, 2, replace=False).tolist()
            qc.cx(c, t)
            lines.append(f"CX {c} {t}")
        axes = rng.choice(list("XYZ"), 3).tolist()
        lines.append("MPP " + "*".join(f"{axis}{q}" for q, axis in enumerate(axes)))
        expected = (
            Statevector.from_instruction(qc).expectation_value(Pauli("".join(reversed(axes)))).real
        )
        actual = record_parity("\n".join(lines), {0})["expectation"]
        assert actual == pytest.approx(expected, abs=1e-12)
        qc.save_expectation_value(Pauli("".join(reversed(axes))), [0, 1, 2], label="parity")
        aer_result = simulator.run(qc).result()
        assert aer_result.success
        assert actual == pytest.approx(aer_result.data()["parity"], abs=1e-12)
