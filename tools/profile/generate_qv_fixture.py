#!/usr/bin/env python3
"""Generate a deterministic Quantum Volume fixture in Clifft circuit syntax."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import qiskit
import qiskit.qasm2
from qiskit import transpile
from qiskit.circuit.library import quantum_volume

EXPECTED_QISKIT_VERSION = "2.3.1"


def generate_qv(width: int, seed: int) -> tuple[str, str]:
    if qiskit.__version__ != EXPECTED_QISKIT_VERSION:
        raise RuntimeError(
            f"Qiskit {EXPECTED_QISKIT_VERSION} is required for a stable fixture; "
            f"found {qiskit.__version__}"
        )

    source = quantum_volume(width, seed=seed)
    circuit = transpile(
        source,
        basis_gates=["u3", "cx"],
        optimization_level=0,
    )
    circuit.measure_all()
    qasm = str(qiskit.qasm2.dumps(circuit))
    round_tripped = qiskit.qasm2.loads(qasm)

    lines: list[str] = []
    for instruction in round_tripped.data:
        operation = instruction.operation
        qubits = [round_tripped.find_bit(qubit).index for qubit in instruction.qubits]
        if operation.name == "u3":
            theta, phi, lam = (float(value) / math.pi for value in operation.params)
            lines.append(f"U3({theta!r},{phi!r},{lam!r}) {qubits[0]}")
        elif operation.name == "cx":
            lines.append(f"CX {qubits[0]} {qubits[1]}")
        elif operation.name == "measure":
            lines.append(f"M {qubits[0]}")
        elif operation.name == "barrier":
            continue
        else:
            raise RuntimeError(f"unexpected transpiled operation: {operation.name}")

    return qasm, "\n".join(lines) + "\n"


def atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="ascii")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=26)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qasm-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.width <= 0:
        parser.error("--width must be positive")
    qasm, contents = generate_qv(args.width, args.seed)
    atomic_write(args.qasm_output, qasm)
    atomic_write(args.output, contents)
    print(
        f"generated {args.qasm_output} and {args.output} with Qiskit "
        f"{qiskit.__version__}: "
        f"width={args.width} depth={args.width} seed={args.seed} "
        f"lines={contents.count(chr(10))}"
    )


if __name__ == "__main__":
    main()
