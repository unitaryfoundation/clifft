"""A minimal Qiskit BackendV2 provider that runs circuits on Clifft."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import (
    CXGate,
    CYGate,
    CZGate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV2, JobStatus, JobV1, Options
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.transpiler import Target

import clifft
from clifft.qiskit._translate import (
    CLIFFT_BASIS,
    counts_from_measurements,
    qiskit_to_stim,
)

# clifft is a simulator with all-to-all connectivity and no fixed qubit count;
# the Target needs a concrete width for qiskit.transpile(qc, backend=backend) to
# work, so we advertise a generous default that callers can raise if needed.
DEFAULT_NUM_QUBITS = 64


class ClifftJob(JobV1):
    """Synchronous job holding an already-computed Result."""

    def __init__(self, backend: BackendV2, job_id: str, result: Result):
        super().__init__(backend, job_id)
        self._result = result

    def submit(self):  # work already done synchronously in backend.run
        pass

    def result(self) -> Result:
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.DONE


class ClifftBackend(BackendV2):
    """Run Qiskit circuits on the Clifft near-Clifford simulator."""

    def __init__(
        self,
        provider: ClifftProvider | None = None,
        name: str = "clifft",
        num_qubits: int = DEFAULT_NUM_QUBITS,
    ):
        super().__init__(provider=provider, name=name, backend_version="0.1.0")
        self._target = self._build_target(num_qubits)

    @staticmethod
    def _build_target(num_qubits: int) -> Target:
        # all-to-all ideal target advertising the full supported basis, so
        # qiskit.transpile(qc, backend=backend) decomposes into clifft's basis.
        target = Target(num_qubits=num_qubits)
        theta = Parameter("theta")
        for gate in (
            HGate(),
            XGate(),
            YGate(),
            ZGate(),
            SGate(),
            SdgGate(),
            TGate(),
            TdgGate(),
            CXGate(),
            CYGate(),
            CZGate(),
            RXGate(theta),
            RYGate(theta),
            RZGate(theta),
            Measure(),
        ):
            target.add_instruction(gate, name=gate.name)
        return target

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024, seed=None)

    def run(self, run_input: QuantumCircuit | list[QuantumCircuit], **options: Any) -> ClifftJob:
        shots = options.get("shots", self.options.shots)
        seed = options.get("seed", self.options.seed)

        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)

        experiment_results = []
        for circ in circuits:
            decomposed = transpile(circ, basis_gates=CLIFFT_BASIS, optimization_level=1)
            stim_text, measured_clbits = qiskit_to_stim(decomposed)
            if not measured_clbits:
                raise QiskitError("clifft backend requires at least one measurement.")

            program = clifft.compile(stim_text)
            if seed is None:
                sample = clifft.sample(program, shots=shots)
            else:
                sample = clifft.sample(program, shots=shots, seed=seed)

            counts = counts_from_measurements(
                sample.measurements, measured_clbits, decomposed.num_clbits
            )
            experiment_results.append(
                ExperimentResult(
                    shots=shots,
                    success=True,
                    data=ExperimentResultData(counts=counts),
                    header={"memory_slots": decomposed.num_clbits, "name": circ.name},
                )
            )

        result = Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            qobj_id="clifft",
            job_id="clifft-job",
            success=True,
            results=experiment_results,
        )
        return ClifftJob(self, "clifft-job", result)


class ClifftProvider:
    """Entry point: ``ClifftProvider().get_backend("clifft")``."""

    def get_backend(self, name: str = "clifft") -> ClifftBackend:
        if name != "clifft":
            raise QiskitBackendNotFoundError(f"No backend matches the name '{name}'.")
        return ClifftBackend(provider=self, name=name)

    def backends(self, name: str | None = None) -> list[ClifftBackend]:
        backend = ClifftBackend(provider=self)
        if name in (None, "clifft"):
            return [backend]
        return []
