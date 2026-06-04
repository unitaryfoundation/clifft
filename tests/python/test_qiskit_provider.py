"""Tests for the Clifft Qiskit backend provider.

Validates:
- Bell circuit counts against reference
- Statistical equivalence with Qiskit Aer
- Multi-circuit batch runs
- Transpilation of CCX (Toffoli) circuits
- Unsupported gate error handling
- Provider / backend API surface
"""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.providers.jobstatus import JobStatus
from qiskit_aer import AerSimulator

from clifft.qiskit_provider import (
    ClifftBackend,
    ClifftProvider,
    UnsupportedGateError,
    circuit_to_stim,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _counts_close(a: dict[str, int], b: dict[str, int], shots: int, sigma: int = 5) -> bool:
    """Return True if two count dicts agree within 5-sigma for each key."""
    all_keys = set(a) | set(b)
    for key in all_keys:
        ca, cb = a.get(key, 0), b.get(key, 0)
        # Binomial std dev for the larger of the two
        p = max(ca, cb) / shots
        std = math.sqrt(shots * p * (1 - p)) if 0 < p < 1 else 1.0
        if abs(ca - cb) > sigma * std:
            return False
    return True


# ---------------------------------------------------------------------------
# Provider / backend API
# ---------------------------------------------------------------------------


class TestProviderAPI:
    def test_get_backend_returns_clifft_backend(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        assert isinstance(backend, ClifftBackend)

    def test_provider_backends_list(self) -> None:
        p = ClifftProvider()
        assert len(p.backends()) == 1
        assert p.backends()[0].name == "clifft"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            ClifftProvider().get_backend("fake")

    def test_backend_name(self) -> None:
        assert ClifftBackend().name == "clifft"

    def test_backend_max_circuits(self) -> None:
        assert ClifftBackend().max_circuits >= 1


# ---------------------------------------------------------------------------
# Converter unit tests
# ---------------------------------------------------------------------------


class TestConverter:
    def test_bell_stim_text(self) -> None:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        stim = circuit_to_stim(qc)
        assert "H 0" in stim
        assert "CX 0 1" in stim
        assert "M 0" in stim
        assert "M 1" in stim

    def test_unsupported_gate_raises(self) -> None:
        from qiskit.circuit.library import CCXGate

        qc = QuantumCircuit(3, 1)
        qc.append(CCXGate(), [0, 1, 2])
        with pytest.raises(UnsupportedGateError):
            circuit_to_stim(qc)

    def test_rotation_gates_convert(self) -> None:
        qc = QuantumCircuit(1)
        qc.rx(math.pi / 4, 0)
        qc.ry(math.pi / 2, 0)
        qc.rz(math.pi, 0)
        stim = circuit_to_stim(qc)
        assert "R_X" in stim
        assert "R_Y" in stim
        assert "R_Z" in stim

    def test_clifford_gates_convert(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.s(0)
        qc.sdg(0)
        qc.t(0)
        qc.tdg(0)
        qc.cx(0, 1)
        qc.cy(0, 1)
        qc.cz(0, 1)
        stim = circuit_to_stim(qc)
        for gate in ("H", "S", "S_DAG", "T", "T_DAG", "CX", "CY", "CZ"):
            assert gate in stim

    def test_barrier_silently_skipped(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        stim = circuit_to_stim(qc)
        assert "barrier" not in stim.lower()


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestBellCircuit:
    SHOTS = 4096

    def test_bell_only_00_and_11(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        job = backend.run(_bell(), shots=self.SHOTS)
        counts = job.result().get_counts()
        assert set(counts.keys()) <= {"00", "11"}, f"Unexpected keys: {counts.keys()}"

    def test_bell_roughly_half_half(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        job = backend.run(_bell(), shots=self.SHOTS)
        counts = job.result().get_counts()
        total = sum(counts.values())
        assert total == self.SHOTS
        assert abs(counts.get("00", 0) - counts.get("11", 0)) < 0.15 * self.SHOTS

    def test_job_status_done(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        job = backend.run(_bell(), shots=100)
        assert job.status() == JobStatus.DONE

    def test_result_success(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        job = backend.run(_bell(), shots=100)
        assert job.result().success


class TestStatisticalEquivalence:
    """Compare Clifft counts against Qiskit Aer on several circuits."""

    SHOTS = 8192
    SIGMA = 5

    def _aer_counts(self, qc: QuantumCircuit) -> dict[str, int]:
        sim = AerSimulator()
        result = sim.run(qc, shots=self.SHOTS).result()
        return dict(result.get_counts())

    def _clifft_counts(self, qc: QuantumCircuit) -> dict[str, int]:
        backend = ClifftProvider().get_backend("clifft")
        return dict(backend.run(qc, shots=self.SHOTS).result().get_counts())

    def test_bell_matches_aer(self) -> None:
        qc = _bell()
        assert _counts_close(self._clifft_counts(qc), self._aer_counts(qc), self.SHOTS, self.SIGMA)

    def test_ghz_3q_matches_aer(self) -> None:
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])
        assert _counts_close(self._clifft_counts(qc), self._aer_counts(qc), self.SHOTS, self.SIGMA)

    def test_t_gate_circuit_matches_aer(self) -> None:
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.t(0)
        qc.h(0)
        qc.measure(0, 0)
        assert _counts_close(self._clifft_counts(qc), self._aer_counts(qc), self.SHOTS, self.SIGMA)

    def test_4qubit_clifford_matches_aer(self) -> None:
        qc = QuantumCircuit(4, 4)
        qc.h([0, 1, 2, 3])
        qc.cx(0, 1)
        qc.cz(2, 3)
        qc.s(0)
        qc.t(2)
        qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
        assert _counts_close(self._clifft_counts(qc), self._aer_counts(qc), self.SHOTS, self.SIGMA)


class TestToffoliTranspiled:
    """CCX (Toffoli) should work after transpilation to Clifford+T basis."""

    SHOTS = 4096

    def test_ccx_transpiled_runs(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        qc = QuantumCircuit(3, 3)
        qc.x([0, 1])
        qc.ccx(0, 1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])
        transpiled = transpile(qc, backend=backend)
        job = backend.run(transpiled, shots=self.SHOTS)
        counts = job.result().get_counts()
        assert counts.get("111", 0) == self.SHOTS, f"Expected all '111', got {counts}"

    def test_ccz_transpiled_runs(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        qc = QuantumCircuit(3, 3)
        qc.x([0, 1])
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
        qc.measure([0, 1, 2], [0, 1, 2])
        transpiled = transpile(qc, backend=backend)
        job = backend.run(transpiled, shots=self.SHOTS)
        counts = job.result().get_counts()
        # After CCZ with x|0>, x|1>, h|2>: qubit 2 should flip → "110" + measure
        assert sum(counts.values()) == self.SHOTS


class TestMultiCircuit:
    def test_batch_two_circuits(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        qc1 = _bell()
        qc2 = QuantumCircuit(1, 1)
        qc2.x(0)
        qc2.measure(0, 0)
        job = backend.run([qc1, qc2], shots=100)
        result = job.result()
        assert result.success
        counts0 = result.get_counts(0)
        counts1 = result.get_counts(1)
        assert set(counts0.keys()) <= {"00", "11"}
        assert counts1 == {"1": 100}

    def test_deterministic_x_gate(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        counts = backend.run(qc, shots=200).result().get_counts()
        assert counts == {"1": 200}

    def test_deterministic_zero(self) -> None:
        backend = ClifftProvider().get_backend("clifft")
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        counts = backend.run(qc, shots=200).result().get_counts()
        assert counts == {"0": 200}
