"""Stabilizer/destabilizer tableau in binary symplectic form.

The tableau is a 2n × (2n+1) matrix over GF(2). Rows 0..n-1 are destabilizers,
rows n..2n-1 are stabilizers. Each row has the format:

    [x_0, x_1, ..., x_{n-1}, z_0, z_1, ..., z_{n-1}, sign]

where x_i and z_i encode the Pauli on qubit i:
    (x=0, z=0) → I
    (x=1, z=0) → X
    (x=1, z=1) → Y
    (x=0, z=1) → Z

and sign=0 means +1, sign=1 means -1.

This follows the Aaronson-Gottesman convention (arXiv:quant-ph/0406196).
"""

import numpy as np
import galois

GF2 = galois.GF(2)

# Pauli labels for display
_PAULI_LABELS = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


class Tableau:
    """Binary symplectic stabilizer/destabilizer tableau for n qubits.

    The initial state is |00...0⟩, with stabilizers Z_i and destabilizers X_i.
    """

    def __init__(self, n: int):
        self.n = n
        # 2n rows × (2n+1) cols over GF(2)
        self.table = GF2(np.zeros((2 * n, 2 * n + 1), dtype=int))
        # Destabilizers (rows 0..n-1): X_i on qubit i
        for i in range(n):
            self.table[i, i] = 1  # x_i = 1 → X
        # Stabilizers (rows n..2n-1): Z_i on qubit i
        for i in range(n):
            self.table[n + i, n + i] = 1  # z_i = 1 → Z

    def copy(self) -> "Tableau":
        t = Tableau.__new__(Tableau)
        t.n = self.n
        t.table = self.table.copy()
        return t

    # ── Accessors ──────────────────────────────────────────────────────

    def x(self, row: int, qubit: int) -> int:
        return int(self.table[row, qubit])

    def z(self, row: int, qubit: int) -> int:
        return int(self.table[row, self.n + qubit])

    def sign(self, row: int) -> int:
        return int(self.table[row, 2 * self.n])

    def set_sign(self, row: int, val: int):
        self.table[row, 2 * self.n] = val % 2

    # ── Display ────────────────────────────────────────────────────────

    def row_to_pauli_string(self, row: int) -> str:
        """Convert a tableau row to a human-readable Pauli string like +XZI."""
        sign_char = "-" if self.sign(row) else "+"
        paulis = []
        for q in range(self.n):
            xi, zi = self.x(row, q), self.z(row, q)
            paulis.append(_PAULI_LABELS[(xi, zi)])
        return sign_char + "".join(paulis)

    def print_tableau(self):
        """Pretty-print all destabilizers and stabilizers."""
        n = self.n
        print(f"Tableau ({n} qubits):")
        print("  Destabilizers:")
        for i in range(n):
            print(f"    d[{i}] = {self.row_to_pauli_string(i)}")
        print("  Stabilizers:")
        for i in range(n):
            print(f"    s[{i}] = {self.row_to_pauli_string(n + i)}")

    # ── Clifford gate updates ─────────────────────────────────────────
    # These update the tableau by conjugation: row → U·row·U†
    # All operate on the full 2n rows (both stabilizers and destabilizers).

    def h(self, qubit: int):
        """Hadamard on qubit. Conjugation: X↔Z, Y→-Y."""
        q = qubit
        n = self.n
        for row in range(2 * n):
            xi, zi = self.x(row, q), self.z(row, q)
            # phase update: sign ^= (x & z) because HYH† = -Y
            self.table[row, 2 * n] ^= GF2(xi & zi)
            # swap x and z
            self.table[row, q], self.table[row, n + q] = (
                self.table[row, n + q],
                self.table[row, q],
            )

    def s(self, qubit: int):
        """S (phase) gate on qubit. Conjugation: X→Y, Z→Z."""
        q = qubit
        n = self.n
        for row in range(2 * n):
            xi, zi = self.x(row, q), self.z(row, q)
            # phase: sign ^= (x & z)  because SYS† = -Y and SXS† = Y
            self.table[row, 2 * n] ^= GF2(xi & zi)
            # z ^= x
            self.table[row, n + q] ^= self.table[row, q]

    def sdg(self, qubit: int):
        """S† gate on qubit. Conjugation: X→-Y, Z→Z."""
        q = qubit
        n = self.n
        for row in range(2 * n):
            xi, zi = self.x(row, q), self.z(row, q)
            # sign ^= x & ~z
            self.table[row, 2 * n] ^= GF2(xi & (1 - zi))
            # z ^= x
            self.table[row, n + q] ^= self.table[row, q]

    def cnot(self, control: int, target: int):
        """CNOT gate. Conjugation: X_c→X_c X_t, Z_t→Z_c Z_t."""
        c, t = control, target
        n = self.n
        for row in range(2 * n):
            xc, zc = self.x(row, c), self.z(row, c)
            xt, zt = self.x(row, t), self.z(row, t)
            # phase: sign ^= x_c * z_t * (x_t ^ z_c ^ 1)
            self.table[row, 2 * n] ^= GF2(xc & zt & (xt ^ zc ^ 1))
            # x_t ^= x_c
            self.table[row, t] ^= self.table[row, c]
            # z_c ^= z_t
            self.table[row, n + c] ^= self.table[row, n + t]

    def x_gate(self, qubit: int):
        """Pauli X gate. Conjugation: Z→-Z, X→X."""
        q = qubit
        n = self.n
        for row in range(2 * n):
            # sign ^= z
            self.table[row, 2 * n] ^= self.table[row, n + q]

    def y_gate(self, qubit: int):
        """Pauli Y gate. Conjugation: X→-X, Z→-Z."""
        q = qubit
        n = self.n
        for row in range(2 * n):
            # sign ^= x ^ z
            self.table[row, 2 * n] ^= self.table[row, q] ^ self.table[row, n + q]

    def z_gate(self, qubit: int):
        """Pauli Z gate. Conjugation: X→-X, Z→Z."""
        q = qubit
        n = self.n
        for row in range(2 * n):
            # sign ^= x
            self.table[row, 2 * n] ^= self.table[row, q]

    # ── Row operations ────────────────────────────────────────────────

    @staticmethod
    def _g(x1: int, z1: int, x2: int, z2: int) -> int:
        """Phase factor for multiplying two single-qubit Paulis.

        Returns g such that the product picks up a factor i^g.
        Uses the Aaronson-Gottesman formula.
        """
        if x1 == 0 and z1 == 0:
            return 0
        elif x1 == 1 and z1 == 1:  # Y
            return z2 - x2
        elif x1 == 1 and z1 == 0:  # X
            return z2 * (2 * x2 - 1)
        else:  # Z
            return x2 * (1 - 2 * z2)

    def rowsum(self, target: int, source: int):
        """Replace row[target] with the product row[source] * row[target].

        Updates the phase correctly using the g-function.
        """
        n = self.n
        g_sum = 0
        for q in range(n):
            g_sum += self._g(
                self.x(source, q),
                self.z(source, q),
                self.x(target, q),
                self.z(target, q),
            )
        # phase: (2*r_target + 2*r_source + g_sum) mod 4 == 0 → new sign is 0
        new_sign = 0 if (2 * self.sign(target) + 2 * self.sign(source) + g_sum) % 4 == 0 else 1
        # XOR the Pauli parts
        for q in range(n):
            self.table[target, q] ^= self.table[source, q]
            self.table[target, n + q] ^= self.table[source, n + q]
        self.set_sign(target, new_sign)

    # ── Pauli action on the basis ────────────────────────────────────

    def pauli_action(self, pauli_x: np.ndarray, pauli_z: np.ndarray, pauli_sign: int):
        """Compute how a Pauli P acts on the basis B(S,D).

        From Eq. (4) of the paper: Q|b_α⟩ = ξ_α(Q) |b_{α⊕β(Q)}⟩

        where:
          β_i(Q) = 1 iff Q anticommutes with stabilizer s_i
          ξ_α(Q) depends on which stabilizers are 'hit' by α

        Returns (beta, stab_bits, base_phase) where:
          beta[i] = 1 iff P anticommutes with stabilizer s_i (the index shift)
          stab_bits[i] = 1 iff P anticommutes with destabilizer d_i (for phase calc)
          base_phase = complex phase from reconstructing P out of generators

        The full phase for a given α is:
          ξ_α = base_phase * (-1)^(Σ stab_bits[i] * α_i)
        """
        n = self.n
        beta = np.zeros(n, dtype=int)
        stab_bits = np.zeros(n, dtype=int)

        # β_i(P) = 1 iff P anticommutes with stabilizer s_i
        for i in range(n):
            sip = 0
            for q in range(n):
                sip += pauli_x[q] * int(self.table[n + i, n + q])  # x_P * z_s
                sip += pauli_z[q] * int(self.table[n + i, q])      # z_P * x_s
            if sip % 2:
                beta[i] = 1

        # stab_bits[i] = 1 iff P anticommutes with destabilizer d_i
        for i in range(n):
            sip = 0
            for q in range(n):
                sip += pauli_x[q] * int(self.table[i, n + q])
                sip += pauli_z[q] * int(self.table[i, q])
            if sip % 2:
                stab_bits[i] = 1

        # Compute base_phase by reconstructing P from generators and
        # comparing to the original. P = base_phase * (∏ d_i^{beta[i]}) * (∏ s_i^{stab_bits[i]})
        #
        # Why this ordering: anticommutation with s_i means d_i appears in the
        # decomposition (beta[i] indexes destabilizer rows 0..n-1), and
        # anticommutation with d_i means s_i appears (stab_bits[i] indexes
        # stabilizer rows n..2n-1). This is the standard symplectic dual.
        accum_x = np.zeros(n, dtype=int)
        accum_z = np.zeros(n, dtype=int)
        accum_phase = 1.0 + 0j  # complex phase of the accumulator

        for i in range(n):
            if beta[i]:
                accum_phase = self._multiply_row_into(accum_x, accum_z, accum_phase, i)
        for i in range(n):
            if stab_bits[i]:
                accum_phase = self._multiply_row_into(accum_x, accum_z, accum_phase, n + i)

        # The original Pauli has phase (-1)^pauli_sign
        orig_phase = (-1.0) ** pauli_sign
        # base_phase = orig_phase / accum_phase
        base_phase = orig_phase / accum_phase

        return beta, stab_bits, base_phase

    def _multiply_row_into(self, accum_x, accum_z, accum_phase, row):
        """Multiply tableau row onto the RIGHT of accumulator: accum ← accum * row.

        Tracks full complex phase through the Pauli group multiplication.
        Uses the _g function: for each qubit, _g(a_x, a_z, r_x, r_z) gives
        the i-exponent from (accum_qubit)(row_qubit).
        """
        n = self.n
        row_phase = (-1.0 + 0j) ** self.sign(row)

        # Compute phase from multiplying accum * row (accum on LEFT, row on RIGHT)
        g_sum = 0
        for q in range(n):
            rx = int(self.table[row, q])
            rz = int(self.table[row, n + q])
            g_sum += self._g(int(accum_x[q]), int(accum_z[q]), rx, rz)

        # Total phase: accum_phase * row_phase * i^g_sum
        new_phase = accum_phase * row_phase * (1j ** g_sum)

        # Update Pauli part by XOR
        for q in range(n):
            accum_x[q] ^= int(self.table[row, q])
            accum_z[q] ^= int(self.table[row, n + q])

        return new_phase

    # ── Measurement support ───────────────────────────────────────────

    def find_anticommuting_stabilizer(self, qubit: int) -> int | None:
        """Find the first stabilizer that anticommutes with Z on qubit.

        Z_q anticommutes with a Pauli P iff P has an X or Y on qubit q,
        i.e., the x-bit of P on qubit q is 1.

        Returns the stabilizer index (0-based), or None if Z_qubit commutes
        with all stabilizers (meaning Z_qubit is in the stabilizer group,
        and measurement is deterministic).
        """
        n = self.n
        for i in range(n):
            # Check stabilizer row (n+i), x-bit at qubit
            if self.x(n + i, qubit):
                return i
        return None

    def measurement_pivot(self, qubit: int, outcome: int, pivot_idx: int):
        """Perform the tableau pivot for a non-deterministic Z_qubit measurement.

        After this:
        - The old destabilizer d[pivot_idx] is replaced by the old stabilizer s[pivot_idx]
        - The old stabilizer s[pivot_idx] is replaced by Z_qubit with the given outcome sign
        - All other rows that anticommute with the new stabilizer are fixed up via rowsum
        """
        n = self.n
        p = pivot_idx

        # Fix up all other stabilizers that anticommute with Z_qubit
        for i in range(n):
            if i != p and self.x(n + i, qubit):  # stab[i] anticommutes
                self.rowsum(n + i, n + p)  # stab[i] *= stab[p]
        # Fix up all destabilizers that anticommute with Z_qubit
        for i in range(n):
            if self.x(i, qubit):  # destab[i] anticommutes
                self.rowsum(i, n + p)  # destab[i] *= stab[p]

        # Replace destab[p] with old stab[p]
        self.table[p, :] = self.table[n + p, :].copy()

        # Replace stab[p] with Z_qubit, sign = outcome
        self.table[n + p, :] = GF2(np.zeros(2 * n + 1, dtype=int))
        self.table[n + p, n + qubit] = 1  # Z on this qubit
        self.set_sign(n + p, outcome)

    def get_deterministic_outcome(self, qubit: int) -> tuple[int, np.ndarray]:
        """For a deterministic Z_qubit measurement, compute the outcome.

        When Z_qubit commutes with all stabilizers that have x-bit set on qubit,
        we use the Aaronson-Gottesman trick: scratch-simulate using destabilizer
        rows to figure out which stabilizers multiply to give Z_qubit.

        Returns (outcome, stab_bits) where stab_bits[i]=1 means s_i contributes.
        """
        n = self.n

        # Find which destabilizers anticommute with Z_qubit
        # (these have x-bit set on qubit)
        # Use rowsum on a scratch row to find the stabilizer product
        stab_bits = np.zeros(n, dtype=int)

        # Create a scratch row representing Z_qubit
        # We'll track which stabilizers we need by finding destabilizers
        # that anticommute with Z_qubit, then using their paired stabilizers
        scratch_x = np.zeros(n, dtype=int)
        scratch_z = np.zeros(n, dtype=int)
        scratch_z[qubit] = 1
        scratch_sign = 0

        for i in range(n):
            # Does destabilizer d_i anticommute with our scratch Pauli?
            sip = 0
            for q in range(n):
                sip += int(self.table[i, q]) * scratch_z[q]       # x_d * z_scratch
                sip += int(self.table[i, n + q]) * scratch_x[q]   # z_d * x_scratch
            if sip % 2 == 1:  # anticommutes
                stab_bits[i] = 1
                # Multiply stabilizer s_i into scratch
                g_sum = 0
                for q in range(n):
                    sx = int(self.table[n + i, q])
                    sz = int(self.table[n + i, n + q])
                    g_sum += self._g(sx, sz, scratch_x[q], scratch_z[q])
                s_sign = self.sign(n + i)
                new_sign = 0 if (2 * scratch_sign + 2 * s_sign + g_sum) % 4 == 0 else 1
                for q in range(n):
                    scratch_x[q] ^= int(self.table[n + i, q])
                    scratch_z[q] ^= int(self.table[n + i, n + q])
                scratch_sign = new_sign

        # After multiplying all relevant stabilizers, scratch should be ±I
        # (since Z_q = ± product of stabilizers)
        # The sign tells us the outcome
        outcome = scratch_sign
        return outcome, stab_bits
