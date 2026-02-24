"""Standalone numpy-based Clifford+T statevector oracle for testing.

This is a simple dense-matrix simulator used to validate UCC's output.
It uses explicit 2^n x 2^n matrices, so it's limited to small circuits.
"""

import numpy as np


class CliffordTOracle:
    """Dense statevector simulator for Clifford+T circuits."""

    def __init__(self, num_qubits: int):
        if num_qubits > 10:
            raise ValueError("Oracle limited to 10 qubits (dense matrix)")
        self.n = num_qubits
        self.dim = 1 << num_qubits
        # Start in |0...0>
        self.sv = np.zeros(self.dim, dtype=complex)
        self.sv[0] = 1.0

    # Single-qubit gates as 2x2 matrices
    _I = np.eye(2, dtype=complex)
    _X = np.array([[0, 1], [1, 0]], dtype=complex)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    _Z = np.array([[1, 0], [0, -1]], dtype=complex)
    _H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    _S = np.array([[1, 0], [0, 1j]], dtype=complex)
    _S_DAG = np.array([[1, 0], [0, -1j]], dtype=complex)
    _T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    _T_DAG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

    def _single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate via tensor product expansion."""
        # Build full 2^n x 2^n matrix.
        # For little-endian indexing (qubit 0 = LSB), the tensor product order
        # is reversed: qubit n-1 is leftmost, qubit 0 is rightmost.
        full = np.array([[1.0]], dtype=complex)
        for q in range(self.n - 1, -1, -1):  # Reverse order
            if q == qubit:
                full = np.kron(full, gate)
            else:
                full = np.kron(full, self._I)
        self.sv = full @ self.sv

    def _two_qubit_gate(self, gate: np.ndarray, q1: int, q2: int) -> None:
        """Apply a two-qubit gate (4x4 matrix on qubits q1, q2).

        The gate matrix is defined in the standard basis |b1, b2> where b1 is
        the qubit q1 value and b2 is the qubit q2 value. Matrix indices use
        row/col = 2*b1 + b2.
        """
        new_sv = np.zeros(self.dim, dtype=complex)
        for i in range(self.dim):
            if abs(self.sv[i]) < 1e-15:
                continue
            # Extract input bits at q1 and q2
            b1_in = (i >> q1) & 1
            b2_in = (i >> q2) & 1
            # Gate matrix column index
            in_idx = b1_in * 2 + b2_in

            # Apply gate: for this input, look at what outputs it produces
            for b1_out in range(2):
                for b2_out in range(2):
                    out_idx = b1_out * 2 + b2_out
                    coeff = gate[out_idx, in_idx]
                    if abs(coeff) < 1e-15:
                        continue
                    # Construct output computational basis index
                    j = i
                    j = (j & ~(1 << q1)) | (b1_out << q1)
                    j = (j & ~(1 << q2)) | (b2_out << q2)
                    new_sv[j] += coeff * self.sv[i]
        self.sv = new_sv

    # Gate implementations
    def h(self, qubit: int) -> None:
        self._single_qubit_gate(self._H, qubit)

    def x(self, qubit: int) -> None:
        self._single_qubit_gate(self._X, qubit)

    def y(self, qubit: int) -> None:
        self._single_qubit_gate(self._Y, qubit)

    def z(self, qubit: int) -> None:
        self._single_qubit_gate(self._Z, qubit)

    def s(self, qubit: int) -> None:
        self._single_qubit_gate(self._S, qubit)

    def s_dag(self, qubit: int) -> None:
        self._single_qubit_gate(self._S_DAG, qubit)

    def t(self, qubit: int) -> None:
        self._single_qubit_gate(self._T, qubit)

    def t_dag(self, qubit: int) -> None:
        self._single_qubit_gate(self._T_DAG, qubit)

    def cx(self, control: int, target: int) -> None:
        """CNOT gate."""
        # CNOT matrix in computational basis (control=MSB in 2-qubit space)
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        self._two_qubit_gate(cnot, control, target)

    def cy(self, control: int, target: int) -> None:
        """CY gate."""
        cy = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex)
        self._two_qubit_gate(cy, control, target)

    def cz(self, control: int, target: int) -> None:
        """CZ gate."""
        cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
        self._two_qubit_gate(cz, control, target)

    def statevector(self) -> np.ndarray:
        """Return the current statevector."""
        return self.sv.copy()


def simulate_circuit(circuit_str: str) -> np.ndarray:
    """Parse and simulate a circuit string, returning the statevector.

    Supports gates: H, X, Y, Z, S, S_DAG, T, T_DAG, CX, CY, CZ
    """
    # First pass: find max qubit
    max_qubit = 0
    ops: list[tuple[str, list[int]]] = []
    for line in circuit_str.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        gate = parts[0].upper()
        # Skip measurement/reset gates - oracle validates pure unitary evolution only.
        # Circuits with measurements are validated via sampling distribution tests.
        if gate in ("M", "MR", "R"):
            continue
        qubits = [int(p) for p in parts[1:]]
        ops.append((gate, qubits))
        if qubits:
            max_qubit = max(max_qubit, max(qubits))

    # Create simulator
    sim = CliffordTOracle(max_qubit + 1)

    # Apply gates
    for gate, qubits in ops:
        if gate == "H":
            sim.h(qubits[0])
        elif gate == "X":
            sim.x(qubits[0])
        elif gate == "Y":
            sim.y(qubits[0])
        elif gate == "Z":
            sim.z(qubits[0])
        elif gate == "S":
            sim.s(qubits[0])
        elif gate in ("S_DAG", "SDAG", "SDG"):
            sim.s_dag(qubits[0])
        elif gate == "T":
            sim.t(qubits[0])
        elif gate in ("T_DAG", "TDAG", "TDG"):
            sim.t_dag(qubits[0])
        elif gate in ("CX", "CNOT"):
            sim.cx(qubits[0], qubits[1])
        elif gate == "CY":
            sim.cy(qubits[0], qubits[1])
        elif gate == "CZ":
            sim.cz(qubits[0], qubits[1])
        else:
            raise ValueError(f"Unknown gate: {gate}")

    return sim.statevector()
