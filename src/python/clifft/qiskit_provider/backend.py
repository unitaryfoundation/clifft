"""ClifftBackend: Qiskit BackendV2 that runs circuits through Clifft."""

from __future__ import annotations

from typing import Union

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.transpiler import Target

import clifft

from .converter import UnsupportedGateError, circuit_to_stim, counts_from_measurements
from .job import ClifftJob

# Basis gates the backend natively accepts without transpilation
_BASIS_GATES = [
    "h", "s", "sdg", "t", "tdg",
    "x", "y", "z",
    "cx", "cy", "cz",
    "rx", "ry", "rz",
    "measure", "reset",
]


class ClifftBackend(BackendV2):
    """Qiskit backend that simulates circuits using the Clifft simulator.

    Clifft is a fast exact simulator for near-Clifford circuits.  It accepts
    Stim-format circuits; this backend converts Qiskit QuantumCircuits
    automatically.

    Usage::

        from clifft.qiskit_provider import ClifftProvider

        backend = ClifftProvider().get_backend("clifft")
        job = backend.run(circuit, shots=1000)
        counts = job.result().get_counts()

    Gates that are not in the Clifft basis raise ``UnsupportedGateError``.
    Transpile first with ``qiskit.transpile(circuit, backend=backend)``.
    """

    def __init__(self) -> None:
        super().__init__(name="clifft")
        self._target = self._build_target()

    # ------------------------------------------------------------------
    # BackendV2 interface
    # ------------------------------------------------------------------

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> int:
        return 300

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024)

    def run(
        self,
        circuits: Union[QuantumCircuit, list[QuantumCircuit]],
        *,
        shots: int | None = None,
        **kwargs: object,
    ) -> ClifftJob:
        """Run one or more circuits through Clifft and return a job.

        Args:
            circuits: A single QuantumCircuit or a list of them.
            shots: Number of shots (default: 1024).
            **kwargs: Ignored (for forward compatibility).

        Returns:
            A completed :class:`ClifftJob` whose ``.result()`` is
            immediately available.

        Raises:
            UnsupportedGateError: If a circuit contains gates not in
                the Clifft basis.  Transpile first.
        """
        if shots is None:
            shots = self.options.shots

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        experiment_results = [
            self._run_one(qc, shots=shots) for qc in circuits
        ]

        result = Result(
            backend_name=self.name,
            backend_version="0.1.0",
            qobj_id="",
            job_id="",
            success=all(r.success for r in experiment_results),
            results=experiment_results,
        )
        return ClifftJob(self, result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_one(self, qc: QuantumCircuit, *, shots: int) -> ExperimentResult:
        """Compile and sample a single circuit; return an ExperimentResult."""
        stim_text = circuit_to_stim(qc)
        prog = clifft.compile(stim_text)
        result = clifft.sample(prog, shots)

        num_clbits = qc.num_clbits
        counts = counts_from_measurements(result.measurements, num_clbits)

        data = ExperimentResultData(counts={hex(int(k, 2)): v for k, v in counts.items()})
        return ExperimentResult(
            shots=shots,
            success=True,
            data=data,
            header=_CircuitHeader(qc.name, num_clbits),
            status="DONE",
        )

    @staticmethod
    def _build_target() -> Target:
        """Build a minimal all-to-all Target for transpilation hints."""
        from qiskit.circuit.library import (
            CXGate, CYGate, CZGate,
            HGate, IGate, RXGate, RYGate, RZGate,
            Reset, SGate, SdgGate, TGate, TdgGate,
            XGate, YGate, ZGate,
        )
        from qiskit.circuit.measure import Measure
        from qiskit.circuit import Parameter

        theta = Parameter("θ")
        target = Target(num_qubits=30)
        for gate in [
            HGate(), SGate(), SdgGate(), TGate(), TdgGate(),
            XGate(), YGate(), ZGate(), IGate(),
            RXGate(theta), RYGate(theta), RZGate(theta),
            CXGate(), CYGate(), CZGate(),
            Measure(), Reset(),
        ]:
            target.add_instruction(gate)
        return target


class _CircuitHeader:
    """Minimal header-like object carrying clbit count for Result."""

    def __init__(self, name: str, n_clbits: int) -> None:
        self.name = name
        self.creg_sizes = [["c", n_clbits]]
        self.memory_slots = n_clbits

    def items(self):
        return self.__dict__.items()
