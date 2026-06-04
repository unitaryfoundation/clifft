"""Tests for the Clifft Qiskit backend provider."""

import pytest

try:
    from qiskit import QuantumCircuit
    from qiskit.exceptions import QiskitError
    from qiskit_aer import AerSimulator

    from clifft.qiskit import ClifftProvider

    qiskit_missing = False
except ImportError:
    qiskit_missing = True

pytestmark = pytest.mark.skipif(qiskit_missing, reason="qiskit/qiskit-aer not installed")

SHOTS = 4000
TOL = 0.05


def _aer_counts(circuit):
    result = AerSimulator().run(circuit, shots=SHOTS, seed_simulator=1).result()
    return result.get_counts()


def _assert_close(a, b, shots=SHOTS, tol=TOL):
    keys = set(a) | set(b)
    for k in keys:
        pa = a.get(k, 0) / shots
        pb = b.get(k, 0) / shots
        assert abs(pa - pb) < tol, f"{k}: {pa} vs {pb}\n{a}\n{b}"


@pytest.fixture
def backend():
    return ClifftProvider().get_backend("clifft")


def test_bell(backend):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    counts = backend.run(qc, shots=SHOTS).result().get_counts()
    _assert_close(counts, _aer_counts(qc))


def test_non_clifford_t(backend):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.t(0)
    qc.h(0)
    qc.measure(0, 0)
    counts = backend.run(qc, shots=SHOTS).result().get_counts()
    _assert_close(counts, _aer_counts(qc))


def test_higher_level_ccx(backend):
    # ccx is not in the basis; must be transpiled/decomposed into Clifford+T.
    qc = QuantumCircuit(3, 3)
    qc.h([0, 1])
    qc.ccx(0, 1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    counts = backend.run(qc, shots=SHOTS).result().get_counts()
    _assert_close(counts, _aer_counts(qc))


def test_clbit_ordering(backend):
    # X on qubit 0 only -> clbit 0 = 1, others 0 -> Qiskit prints '...01'.
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.measure([0, 1, 2], [0, 1, 2])
    counts = backend.run(qc, shots=SHOTS).result().get_counts()
    assert set(counts) == {"001"}


def test_list_of_circuits(backend):
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    res = backend.run([qc, qc], shots=SHOTS).result()
    assert res.get_counts(0) == {"1": SHOTS}
    assert res.get_counts(1) == {"1": SHOTS}


def test_unsupported_operation_raises(backend):
    qc = QuantumCircuit(1, 1)
    qc.reset(0)
    qc.h(0)
    qc.measure(0, 0)
    with pytest.raises(QiskitError):
        backend.run(qc, shots=SHOTS).result()
