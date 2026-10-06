"""Regenerate the fixed-input Draper benchmark using MQT Bench 2.3.0."""

import importlib.metadata
import math
from importlib import import_module
from pathlib import Path

from qiskit import QuantumCircuit, transpile


def main() -> None:
    draper_qft_adder = import_module("mqt.bench.benchmarks.draper_qft_adder")

    bits = 16
    circuit = draper_qft_adder.create_circuit(2 * bits).remove_final_measurements(inplace=False)
    prep = QuantumCircuit(2 * bits)
    for q in range(2 * bits):
        if q % 3 == 0:
            prep.x(q)
    circuit = circuit.compose(prep, front=True)
    circuit.measure_all()
    gates = {
        "h": "H",
        "s": "S",
        "sdg": "S_DAG",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "cx": "CX",
        "cy": "CY",
        "cz": "CZ",
        "t": "T",
        "tdg": "T_DAG",
    }
    rotations = {"rx": "R_X", "ry": "R_Y", "rz": "R_Z"}
    circuit = transpile(circuit, basis_gates=[*gates, *rotations], optimization_level=0)
    lines = [
        "# Fixed-size 16-bit Draper adder on 32 qubits.",
        "# Input a=37449 b=18724; output a=37449 b=56173; no noise.",
        "# Records are qubits 0..31 in order; observable 0 is the parity of the two sum MSBs.",
        "# MQT Bench "
        + importlib.metadata.version("mqt-bench")
        + "; Qiskit "
        + importlib.metadata.version("qiskit")
        + "; optimization_level=0; rotations in half-turns.",
    ]
    for instruction in circuit.data:
        name = instruction.operation.name
        targets = " ".join(str(circuit.find_bit(q).index) for q in instruction.qubits)
        if name in ("barrier", "id"):
            continue
        if name == "measure":
            lines.append("M " + targets)
        elif name in gates:
            lines.append(gates[name] + " " + targets)
        elif name in rotations:
            alpha = float(instruction.operation.params[0]) / math.pi
            lines.append(f"{rotations[name]}({alpha}) {targets}")
        else:
            raise ValueError(name)
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]")
    path = Path(__file__).parent / "fixtures/draper_adder_16_basis.stim"
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
