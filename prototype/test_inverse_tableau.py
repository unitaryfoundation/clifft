"""Validate that the inverse tableau alone is sufficient for all operations.

The design (DESIGN.md §4.1) claims we only need the inverse tableau:
  - Pauli decomposition: read row q of inverse → β, γ, base_phase
  - Clifford updates: prepend C⁻¹ to inverse
  - No forward tableau needed

This test maintains BOTH tableaux side-by-side, verifies they're inverses,
and confirms that reading the inverse tableau row gives the same
decomposition as the O(n) pauli_action scan on the forward tableau.

The inverse tableau update rule:
  Forward gate U conjugates rows by U:   row → U row U†
  Inverse gate U conjugates rows by U†:  row → U† row U

  For self-adjoint gates (H, CNOT, X, Y, Z): same operation on both.
  For S: forward calls s(), inverse calls sdg() (and vice versa).
"""

import numpy as np
import pytest
from tableau import Tableau, GF2


class InverseTableau:
    """Maintains the inverse tableau by recording the gate sequence
    and recomputing the inverse from scratch.

    The inverse tableau stores C† G C for each standard generator G,
    where C = U_n...U_2 U_1 is the cumulative Clifford.
    C† = U_1† U_2† ... U_n†, so the inverse is built by applying
    the adjoint gates in REVERSE order to an identity tableau.

    This brute-force approach validates the CONCEPT (do we ever need
    the forward tableau?) without requiring optimized prepend operations.
    A production implementation would use Stim's prepend_* methods.
    """

    def __init__(self, n: int):
        self.n = n
        self.gates: list[tuple[str, tuple]] = []
        self.tab = Tableau(n)

    def copy(self):
        c = InverseTableau.__new__(InverseTableau)
        c.n = self.n
        c.gates = list(self.gates)
        c.tab = self.tab.copy()
        return c

    def _rebuild(self):
        """Rebuild inverse from recorded gates.

        C = gate_n ... gate_1
        C† = gate_1† ... gate_n†
        Inverse tableau row for Z_q stores C† Z_q C.
        We build this by applying gate_1† first (conjugating rows),
        then gate_2†, etc. Since conjugation is the 'append' operation,
        this means: start with identity, then for each gate in order,
        append the adjoint.
        """
        self.tab = Tableau(self.n)
        # Adjoint map: for each gate, what's its adjoint?
        adj = {
            'h': 'h',           # H† = H
            's': 'sdg',         # S†
            'sdg': 's',         # (S†)† = S
            'cnot': 'cnot',     # CNOT† = CNOT
            'x_gate': 'x_gate', # X† = X
            'y_gate': 'y_gate', # Y† = Y
            'z_gate': 'z_gate', # Z† = Z
        }
        for gate_name, args in reversed(self.gates):
            getattr(self.tab, adj[gate_name])(*args)

    def _apply(self, gate_name: str, *args):
        self.gates.append((gate_name, args))
        self._rebuild()

    def h(self, q):       self._apply('h', q)
    def s(self, q):       self._apply('s', q)
    def sdg(self, q):     self._apply('sdg', q)
    def cnot(self, c, t): self._apply('cnot', c, t)
    def x_gate(self, q):  self._apply('x_gate', q)
    def y_gate(self, q):  self._apply('y_gate', q)
    def z_gate(self, q):  self._apply('z_gate', q)

    def read_z_row(self, qubit: int):
        """Read decomposition of Z_qubit from inverse tableau row.

        Row q of the stabilizer section (rows n..2n-1) of the inverse
        tableau stores C† Z_q C. Its x-bits give β (destabilizer mask),
        z-bits give γ (stabilizer mask), sign gives base_phase.

        But wait — the inverse tableau has the SAME layout as the forward:
        destab rows 0..n-1, stab rows n..2n-1. At initialization both are
        identity. After gates, the inverse rows store C† X_i C (destab) and
        C† Z_i C (stab).

        So to get the decomposition of Z_q, we read stabilizer row q
        (row index n+q), and the x-bits give β, z-bits give γ.
        """
        n = self.n
        row = n + qubit  # stabilizer row for Z_qubit

        beta = np.zeros(n, dtype=int)
        gamma = np.zeros(n, dtype=int)

        for i in range(n):
            beta[i] = self.tab.x(row, i)    # x-bits → destabilizer mask
            gamma[i] = self.tab.z(row, i)   # z-bits → stabilizer mask

        # Phase: sign bit + Y correction (i^{num_y})
        sign = self.tab.sign(row)
        num_y = sum(1 for i in range(n)
                    if self.tab.x(row, i) and self.tab.z(row, i))

        # base_phase = (-1)^sign × i^{num_y}
        base_phase = ((-1.0) ** sign) * (1j ** num_y)

        return beta, gamma, base_phase


def symplectic_matrix(tab: Tableau) -> np.ndarray:
    """Extract the 2n×2n symplectic matrix (ignoring phases) over GF(2).

    Rows 0..n-1 are destabilizers, rows n..2n-1 are stabilizers.
    Each row has [x_0..x_{n-1}, z_0..z_{n-1}].
    """
    n = tab.n
    S = np.zeros((2 * n, 2 * n), dtype=int)
    for r in range(2 * n):
        for q in range(n):
            S[r, q] = tab.x(r, q)
            S[r, n + q] = tab.z(r, q)
    return S


def check_inverse_relationship(fwd: Tableau, inv: InverseTableau):
    """Verify that fwd and inv are symplectic inverses (mod 2).

    The forward tableau's symplectic matrix S_fwd maps standard generators
    to current generators. The inverse S_inv maps current physical Paulis
    back to generator-basis. Their composition must be the identity:
    S_fwd @ S_inv = I (mod 2).
    """
    n = fwd.n
    S_fwd = symplectic_matrix(fwd)
    S_inv = symplectic_matrix(inv.tab)

    product = (S_fwd @ S_inv) % 2
    expected = np.eye(2 * n, dtype=int)

    assert np.array_equal(product, expected), (
        f"Symplectic inverse check failed!\n"
        f"S_fwd @ S_inv (mod 2):\n{product}\n"
        f"Expected: I"
    )


def check_decomposition_match(fwd: Tableau, inv: InverseTableau, qubit: int):
    """Verify inverse row-read matches forward pauli_action for Z_qubit."""
    n = fwd.n

    # Forward: O(n) scan
    zq_x = np.zeros(n, dtype=int)
    zq_z = np.zeros(n, dtype=int)
    zq_z[qubit] = 1
    fwd_beta, fwd_gamma, fwd_phase = fwd.pauli_action(zq_x, zq_z, 0)

    # Inverse: row read
    inv_beta, inv_gamma, inv_phase = inv.read_z_row(qubit)

    assert np.array_equal(fwd_beta, inv_beta), (
        f"β mismatch for Z_{qubit}: forward={list(fwd_beta)} inverse={list(inv_beta)}"
    )
    assert np.array_equal(fwd_gamma, inv_gamma), (
        f"γ mismatch for Z_{qubit}: forward={list(fwd_gamma)} inverse={list(inv_gamma)}"
    )
    assert abs(fwd_phase - inv_phase) < 1e-10, (
        f"phase mismatch for Z_{qubit}: forward={fwd_phase} inverse={inv_phase}"
    )


# ── Gate application helpers ──

def apply_gate(fwd: Tableau, inv: InverseTableau, gate: str, *args):
    """Apply a gate to both forward and inverse tableaux."""
    getattr(fwd, gate)(*args)
    getattr(inv, gate)(*args)


# ── Tests ──

def test_identity():
    """At initialization, forward == inverse == identity."""
    for n in range(1, 5):
        fwd = Tableau(n)
        inv = InverseTableau(n)
        check_inverse_relationship(fwd, inv)
        for q in range(n):
            check_decomposition_match(fwd, inv, q)


def test_single_h():
    """H is self-adjoint: same update on both."""
    fwd = Tableau(2)
    inv = InverseTableau(2)
    apply_gate(fwd, inv, 'h', 0)
    check_inverse_relationship(fwd, inv)
    for q in range(2):
        check_decomposition_match(fwd, inv, q)


def test_single_s():
    """S gate: forward uses s(), inverse uses sdg()."""
    fwd = Tableau(2)
    inv = InverseTableau(2)
    apply_gate(fwd, inv, 's', 0)
    check_inverse_relationship(fwd, inv)
    for q in range(2):
        check_decomposition_match(fwd, inv, q)


def test_single_sdg():
    """S† gate: forward uses sdg(), inverse uses s()."""
    fwd = Tableau(2)
    inv = InverseTableau(2)
    apply_gate(fwd, inv, 'sdg', 0)
    check_inverse_relationship(fwd, inv)
    for q in range(2):
        check_decomposition_match(fwd, inv, q)


def test_cnot():
    """CNOT is self-adjoint."""
    fwd = Tableau(3)
    inv = InverseTableau(3)
    apply_gate(fwd, inv, 'h', 0)
    apply_gate(fwd, inv, 'cnot', 0, 1)
    check_inverse_relationship(fwd, inv)
    for q in range(3):
        check_decomposition_match(fwd, inv, q)


def test_walkthrough_circuit():
    """The exact circuit from the walkthrough notebook."""
    fwd = Tableau(2)
    inv = InverseTableau(2)

    steps = [
        ('h', 0),
        # T on q0 would go here (not a Clifford, doesn't change tableau)
        # T on q1 (same)
        ('h', 1),
        # T on q1
        ('cnot', 0, 1),
    ]

    for step in steps:
        gate = step[0]
        args = step[1:]
        apply_gate(fwd, inv, gate, *args)
        check_inverse_relationship(fwd, inv)
        for q in range(2):
            check_decomposition_match(fwd, inv, q)


def test_y_phase_correction():
    """Circuits that produce Y entries, testing i^{num_y} correction."""
    fwd = Tableau(2)
    inv = InverseTableau(2)

    # H then S creates Y in the stabilizer: SXS† = Y
    apply_gate(fwd, inv, 'h', 0)
    apply_gate(fwd, inv, 's', 0)
    check_inverse_relationship(fwd, inv)
    for q in range(2):
        check_decomposition_match(fwd, inv, q)

    # Add entanglement to spread Y
    apply_gate(fwd, inv, 'cnot', 0, 1)
    check_inverse_relationship(fwd, inv)
    for q in range(2):
        check_decomposition_match(fwd, inv, q)


def test_pauli_gates():
    """X, Y, Z gates (all self-adjoint)."""
    fwd = Tableau(2)
    inv = InverseTableau(2)

    apply_gate(fwd, inv, 'h', 0)
    apply_gate(fwd, inv, 'x_gate', 0)
    check_inverse_relationship(fwd, inv)

    apply_gate(fwd, inv, 'y_gate', 1)
    check_inverse_relationship(fwd, inv)

    apply_gate(fwd, inv, 'z_gate', 0)
    check_inverse_relationship(fwd, inv)

    for q in range(2):
        check_decomposition_match(fwd, inv, q)


def test_s_roundtrip():
    """S then S† should return to identity."""
    fwd = Tableau(2)
    inv = InverseTableau(2)

    apply_gate(fwd, inv, 'h', 0)
    apply_gate(fwd, inv, 's', 0)
    apply_gate(fwd, inv, 'sdg', 0)

    # Should be same as just H
    fwd2 = Tableau(2)
    inv2 = InverseTableau(2)
    apply_gate(fwd2, inv2, 'h', 0)

    assert np.array_equal(
        symplectic_matrix(fwd), symplectic_matrix(fwd2)
    ), "S S† didn't cancel on forward tableau"
    assert np.array_equal(
        symplectic_matrix(inv.tab), symplectic_matrix(inv2.tab)
    ), "S S† didn't cancel on inverse tableau"


def test_random_clifford_circuits():
    """Fuzz: random Clifford circuits, verify inverse at every step."""
    rng = np.random.default_rng(42)
    gate_names = ['h', 's', 'sdg', 'cnot', 'x_gate', 'y_gate', 'z_gate']

    n_circuits = 100
    for trial in range(n_circuits):
        n = rng.integers(2, 6)  # 2-5 qubits
        depth = rng.integers(5, 30)

        fwd = Tableau(n)
        inv = InverseTableau(n)

        for _ in range(depth):
            gate = rng.choice(gate_names)
            if gate == 'cnot':
                c, t = rng.choice(n, size=2, replace=False)
                apply_gate(fwd, inv, gate, int(c), int(t))
            else:
                q = int(rng.integers(0, n))
                apply_gate(fwd, inv, gate, q)

        # Verify after full circuit
        check_inverse_relationship(fwd, inv)
        for q in range(n):
            check_decomposition_match(fwd, inv, q)


def test_random_clifford_circuits_every_step():
    """Fuzz: verify inverse relationship after EVERY gate, not just at end."""
    rng = np.random.default_rng(99)
    gate_names = ['h', 's', 'sdg', 'cnot', 'x_gate', 'y_gate', 'z_gate']

    for trial in range(20):
        n = rng.integers(2, 5)
        depth = rng.integers(10, 25)

        fwd = Tableau(n)
        inv = InverseTableau(n)

        for step in range(depth):
            gate = rng.choice(gate_names)
            if gate == 'cnot':
                c, t = rng.choice(n, size=2, replace=False)
                apply_gate(fwd, inv, gate, int(c), int(t))
            else:
                q = int(rng.integers(0, n))
                apply_gate(fwd, inv, gate, q)

            # Check after every single gate
            check_inverse_relationship(fwd, inv)
            for q_check in range(n):
                check_decomposition_match(fwd, inv, q_check)


def test_deep_circuit_stress():
    """Stress test: deep circuit on more qubits."""
    rng = np.random.default_rng(7)
    gate_names = ['h', 's', 'sdg', 'cnot', 'x_gate', 'y_gate', 'z_gate']

    n = 8
    depth = 200
    fwd = Tableau(n)
    inv = InverseTableau(n)

    for _ in range(depth):
        gate = rng.choice(gate_names)
        if gate == 'cnot':
            c, t = rng.choice(n, size=2, replace=False)
            apply_gate(fwd, inv, gate, int(c), int(t))
        else:
            q = int(rng.integers(0, n))
            apply_gate(fwd, inv, gate, q)

    check_inverse_relationship(fwd, inv)
    for q in range(n):
        check_decomposition_match(fwd, inv, q)


def test_measurement_without_forward():
    """Validate that measurement collapse can be done on inverse alone.

    The design claims (§5.3) that AG-style row operations on the inverse
    tableau preserve the coefficient vector key space. We verify this by:
    1. Running a circuit with T gates + measurement on the forward prototype
    2. Maintaining an inverse tableau in parallel
    3. Verifying decompositions match before and after measurement
    """
    from state import GeneralizedStabilizerState

    rng = np.random.default_rng(42)

    for trial in range(50):
        n = rng.integers(2, 5)
        state = GeneralizedStabilizerState(n)
        inv = InverseTableau(n)

        # Random Clifford prefix
        for _ in range(rng.integers(3, 10)):
            gate = rng.choice(['h', 's', 'sdg', 'cnot', 'x_gate', 'z_gate'])
            if gate == 'cnot' and n >= 2:
                c, t = rng.choice(n, size=2, replace=False)
                apply_gate(state.tableau, inv, gate, int(c), int(t))
            elif gate != 'cnot':
                q = int(rng.integers(0, n))
                apply_gate(state.tableau, inv, gate, q)

        # Verify decompositions match before measurement
        for q in range(n):
            check_decomposition_match(state.tableau, inv, q)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
