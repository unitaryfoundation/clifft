"""Generalized stabilizer state: (v, B(S,D)) representation.

The quantum state is |ϕ⟩ = Σ_α v_α |b_α⟩ where:
  - B(S,D) is the orthonormal basis induced by the stabilizer-destabilizer tableau
  - |b_α⟩ = d^α |ψ_S⟩ (product of destabilizers applied to the stabilizer state)
  - v is a sparse complex coefficient vector

Key properties:
  - Clifford gates: update tableau only, v unchanged, |v'| = |v|
  - T/T† gates: each entry can split into at most 2, |v'| ≤ 2|v|
  - Measurements: entries are filtered, |v'| ≤ |v|
"""

import math
import numpy as np
from tableau import Tableau


class GeneralizedStabilizerState:
    """A quantum state in the generalized stabilizer representation."""

    def __init__(self, n: int):
        self.n = n
        self.tableau = Tableau(n)
        # Sparse coefficient vector: maps int (bit-string α) → complex amplitude
        # Initial state |00...0⟩ = |b_0⟩ with coefficient 1
        self.v: dict[int, complex] = {0: 1.0 + 0j}

    def copy(self) -> "GeneralizedStabilizerState":
        s = GeneralizedStabilizerState.__new__(GeneralizedStabilizerState)
        s.n = self.n
        s.tableau = self.tableau.copy()
        s.v = dict(self.v)
        return s

    @property
    def num_entries(self) -> int:
        """Number of nonzero entries |v|."""
        return sum(1 for c in self.v.values() if abs(c) > 1e-10)

    def _cleanup(self):
        """Remove near-zero entries."""
        self.v = {k: c for k, c in self.v.items() if abs(c) > 1e-10}

    def _renormalize(self):
        """Renormalize the coefficient vector."""
        self._cleanup()
        total = sum(abs(c) ** 2 for c in self.v.values())
        if total > 1e-15:
            factor = math.sqrt(total)
            self.v = {k: c / factor for k, c in self.v.items()}

    # ── Clifford gates (update tableau only) ──────────────────────

    def apply_h(self, qubit: int):
        """Hadamard gate. Updates tableau; v unchanged."""
        self.tableau.h(qubit)

    def apply_s(self, qubit: int):
        """S (phase) gate. Updates tableau; v unchanged."""
        self.tableau.s(qubit)

    def apply_sdg(self, qubit: int):
        """S† gate. Updates tableau; v unchanged."""
        self.tableau.sdg(qubit)

    def apply_cnot(self, control: int, target: int):
        """CNOT gate. Updates tableau; v unchanged."""
        self.tableau.cnot(control, target)

    def apply_x(self, qubit: int):
        """Pauli X gate. Updates tableau; v unchanged."""
        self.tableau.x_gate(qubit)

    def apply_y(self, qubit: int):
        """Pauli Y gate. Updates tableau; v unchanged."""
        self.tableau.y_gate(qubit)

    def apply_z(self, qubit: int):
        """Pauli Z gate. Updates tableau; v unchanged."""
        self.tableau.z_gate(qubit)

    # ── T/T† gates ────────────────────────────────────────────────

    def apply_t(self, qubit: int):
        """T gate on qubit. May double |v|.

        T = e^{iπ/8} [cos(π/8) I - i sin(π/8) Z]

        Using Eq.(4): Z|b_α⟩ = ξ_α(Z) |b_{α⊕β(Z)}⟩
        So T|b_α⟩ = cos(π/8)|b_α⟩ - i·sin(π/8)·ξ_α(Z_q)·|b_{α⊕β(Z_q)}⟩
        """
        self._apply_t_internal(qubit, dagger=False)

    def apply_tdg(self, qubit: int):
        """T† gate on qubit. May double |v|.

        T† = e^{-iπ/8} [cos(π/8) I + i sin(π/8) Z]
        """
        self._apply_t_internal(qubit, dagger=True)

    def _apply_t_internal(self, qubit: int, dagger: bool):
        """Internal T/T† implementation."""
        n = self.n
        cos_val = math.cos(math.pi / 8)
        sin_val = math.sin(math.pi / 8)
        sign = 1.0 if dagger else -1.0  # T uses -i·sin, T† uses +i·sin

        # Decompose Z_qubit in the current basis
        z_x = np.zeros(n, dtype=int)
        z_z = np.zeros(n, dtype=int)
        z_z[qubit] = 1
        beta_bits, stab_bits, base_phase = self.tableau.pauli_action(z_x, z_z, 0)

        # β(Z_q): the index shift from applying Z_q
        beta = 0
        for i in range(n):
            if beta_bits[i]:
                beta |= 1 << (n - 1 - i)

        new_v: dict[int, complex] = {}

        for alpha, v_alpha in self.v.items():
            # Term 1: cos(π/8) · v_α · |b_α⟩
            new_v[alpha] = new_v.get(alpha, 0j) + cos_val * v_alpha

            # Term 2: (sign * i) · sin(π/8) · phase · ξ_α(Z_q) · v_α · |b_{α⊕β}⟩
            # Compute ξ_α(Z_q): the phase from applying Z_q to |b_α⟩
            # ξ_α(Z_q) = base_phase · (-1)^(Σ stab_bits[i] * α_i)
            alpha_bits = [(alpha >> (n - 1 - i)) & 1 for i in range(n)]
            parity = sum(stab_bits[i] * alpha_bits[i] for i in range(n)) % 2
            xi = base_phase * ((-1) ** parity)

            target = alpha ^ beta
            coeff = (sign * 1j) * sin_val * xi * v_alpha
            new_v[target] = new_v.get(target, 0j) + coeff

        self.v = new_v
        self._cleanup()

    # ── Measurement ───────────────────────────────────────────────

    def measure(self, qubit: int, rng=None) -> int:
        """Measure qubit in Z basis. Returns outcome (0 or 1).

        Updates both the tableau and v.
        """
        n = self.n
        if rng is None:
            rng = np.random.default_rng()

        pivot = self.tableau.find_anticommuting_stabilizer(qubit)

        if pivot is None:
            # Z_qubit is in the stabilizer group (β = 0)
            # May still be probabilistic if branches have different eigenvalues
            return self._measure_deterministic(qubit, rng)
        else:
            # Non-deterministic: need to sample
            return self._measure_random(qubit, pivot, rng)

    def _measure_deterministic(self, qubit: int, rng=None) -> int:
        """Handle measurement when Z_q is in the stabilizer group (β = 0).

        Z_q is a product of stabilizers: Z_q = base_phase × Π S_k^{z_k}.
        Since β = 0, there is no index shift: Z_q|b_α⟩ = ξ_α|b_α⟩.

        The eigenvalue ξ_α = base_phase × (-1)^(z · α) can VARY across
        branches when z has nonzero inner product with active α indices.
        When |v| = 1 (pure stabilizer state), the outcome is deterministic.
        When |v| > 1, different branches may have different eigenvalues,
        making the measurement probabilistic — we must filter branches.
        """
        n = self.n
        outcome, stab_bits = self.tableau.get_deterministic_outcome(qubit)

        if self.num_entries <= 1:
            # Pure stabilizer state: truly deterministic, no filtering needed
            return outcome

        # Check if all branches have the same eigenvalue
        # ξ_α = base_phase × (-1)^(stab_bits · α)
        # If stab_bits · α is the same for all active α, outcome is deterministic
        stab_parities = set()
        for alpha in self.v:
            if abs(self.v[alpha]) < 1e-10:
                continue
            alpha_bits = [(alpha >> (n - 1 - i)) & 1 for i in range(n)]
            parity = sum(stab_bits[i] * alpha_bits[i] for i in range(n)) % 2
            stab_parities.add(parity)

        if len(stab_parities) <= 1:
            # All branches have same eigenvalue: truly deterministic
            return outcome

        # Branches have different eigenvalues: measurement is probabilistic.
        # base_phase determines which parity → which outcome:
        #   parity=0: eigenvalue = base_phase  → outcome 0 if base_phase=+1, 1 if -1
        #   parity=1: eigenvalue = -base_phase → outcome 1 if base_phase=+1, 0 if -1
        # The 'outcome' from get_deterministic_outcome corresponds to the
        # stabilizer state (α=0, parity=0).

        # Split branches by parity
        v_same: dict[int, complex] = {}   # parity matches α=0 → gets 'outcome'
        v_diff: dict[int, complex] = {}   # parity differs → gets '1-outcome'

        for alpha, v_alpha in self.v.items():
            if abs(v_alpha) < 1e-10:
                continue
            alpha_bits = [(alpha >> (n - 1 - i)) & 1 for i in range(n)]
            parity = sum(stab_bits[i] * alpha_bits[i] for i in range(n)) % 2
            if parity == 0:
                v_same[alpha] = v_alpha
            else:
                v_diff[alpha] = v_alpha

        prob_same = sum(abs(c) ** 2 for c in v_same.values())
        prob_diff = sum(abs(c) ** 2 for c in v_diff.values())
        total = prob_same + prob_diff

        if rng is None:
            rng = np.random.default_rng()

        if total > 0:
            prob_same /= total

        if rng.random() < prob_same:
            self.v = v_same
            actual_outcome = outcome
        else:
            self.v = v_diff
            actual_outcome = 1 - outcome

        self._renormalize()
        return actual_outcome

    def _measure_random(self, qubit: int, pivot: int, rng) -> int:
        """Handle non-deterministic measurement."""
        n = self.n

        # Decompose Z_qubit
        z_x = np.zeros(n, dtype=int)
        z_z = np.zeros(n, dtype=int)
        z_z[qubit] = 1
        beta_bits, stab_bits, base_phase = self.tableau.pauli_action(z_x, z_z, 0)

        beta = 0
        for i in range(n):
            if beta_bits[i]:
                beta |= 1 << (n - 1 - i)

        # k-bit mask for the pivot destabilizer
        k_mask = 1 << (n - 1 - pivot)

        # Split entries into outcome-0 and outcome-1 bins
        new_v0: dict[int, complex] = {}
        new_v1: dict[int, complex] = {}

        for alpha, v_alpha in self.v.items():
            alpha_bits = [(alpha >> (n - 1 - i)) & 1 for i in range(n)]
            parity = sum(stab_bits[i] * alpha_bits[i] for i in range(n)) % 2
            xi = base_phase * ((-1) ** parity)

            coef = 1.0 / math.sqrt(2)

            if alpha & k_mask:  # α has the pivot bit set
                target = alpha ^ beta  # XOR with full destab pattern
                c0 = coef * xi
                c1 = -c0
                new_v0[target] = new_v0.get(target, 0j) + v_alpha * c0
                new_v1[target] = new_v1.get(target, 0j) + v_alpha * c1
            else:
                new_v0[alpha] = new_v0.get(alpha, 0j) + v_alpha * coef
                new_v1[alpha] = new_v1.get(alpha, 0j) + v_alpha * coef

        # Compute probabilities
        prob0 = sum(abs(c) ** 2 for c in new_v0.values())
        prob1 = sum(abs(c) ** 2 for c in new_v1.values())
        total = prob0 + prob1
        if total > 0:
            prob0 /= total

        outcome = 0 if rng.random() < prob0 else 1

        # Update tableau
        self.tableau.measurement_pivot(qubit, outcome, pivot)

        # Update v
        self.v = new_v0 if outcome == 0 else new_v1
        self._renormalize()
        return outcome

    # ── Reset ─────────────────────────────────────────────────────

    def reset(self, qubit: int):
        """Reset qubit to |0⟩ by measuring and conditionally flipping."""
        outcome = self.measure(qubit)
        if outcome == 1:
            self.apply_x(qubit)

    # ── Statevector expansion (for verification) ──────────────────

    def to_statevector(self) -> np.ndarray:
        """Expand to a full 2^n statevector for verification.

        Constructs each |b_α⟩ = d^α |ψ_S⟩ explicitly and sums v_α |b_α⟩.
        """
        n = self.n
        dim = 1 << n
        sv = np.zeros(dim, dtype=complex)

        # First, find the stabilizer state |ψ_S⟩
        psi_s = self._stabilizer_state_vector()

        for alpha, v_alpha in self.v.items():
            if abs(v_alpha) < 1e-15:
                continue
            # |b_α⟩ = d^α |ψ_S⟩
            b_alpha = self._apply_destabilizer_product(psi_s, alpha)
            sv += v_alpha * b_alpha

        return sv

    def _stabilizer_state_vector(self) -> np.ndarray:
        """Compute the stabilizer state |ψ_S⟩ as a dense vector.

        |ψ_S⟩ is the unique +1 eigenstate of all stabilizers.
        We project each computational basis state and take the first nonzero result.
        """
        n = self.n
        dim = 1 << n

        # Build the full projector P = Π_i (I + s_i)/2
        proj = np.eye(dim, dtype=complex)
        for i in range(n):
            stab_mat = self._pauli_row_to_matrix(n + i)
            proj = 0.5 * (np.eye(dim) + stab_mat) @ proj

        # Apply to computational basis states until we find a nonzero result
        for seed_idx in range(dim):
            seed = np.zeros(dim, dtype=complex)
            seed[seed_idx] = 1.0
            state = proj @ seed
            norm = np.linalg.norm(state)
            if norm > 1e-10:
                return state / norm

        raise RuntimeError("Failed to find stabilizer state")

    def _apply_destabilizer_product(self, state: np.ndarray, alpha: int) -> np.ndarray:
        """Apply d^α = Π_{i where α_i=1} d_i to state."""
        n = self.n
        result = state.copy()
        for i in range(n):
            if (alpha >> (n - 1 - i)) & 1:
                mat = self._pauli_row_to_matrix(i)  # destabilizer i
                result = mat @ result
        return result

    def _pauli_row_to_matrix(self, row: int) -> np.ndarray:
        """Convert a tableau row to a 2^n × 2^n matrix."""
        n = self.n
        # Build single-qubit Pauli matrices and tensor them
        I2 = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

        pauli_map = {(0, 0): I2, (1, 0): X, (1, 1): Y, (0, 1): Z}

        mat = np.array([[1.0 + 0j]])
        for q in range(n):
            xi = self.tableau.x(row, q)
            zi = self.tableau.z(row, q)
            mat = np.kron(mat, pauli_map[(xi, zi)])

        # Apply sign
        if self.tableau.sign(row):
            mat = -mat

        return mat

    # ── Display ───────────────────────────────────────────────────

    def print_state(self, label: str = ""):
        """Pretty-print the full generalized stabilizer state."""
        if label:
            print(f"\n{'═' * 60}")
            print(f"  {label}")
            print(f"{'═' * 60}")

        self.tableau.print_tableau()

        print(f"\n  Coefficient vector v ({self.num_entries} nonzero entries):")
        self._cleanup()
        if not self.v:
            print("    (empty - zero state)")
            return

        n = self.n
        for alpha in sorted(self.v.keys()):
            c = self.v[alpha]
            if abs(c) < 1e-10:
                continue
            bits = format(alpha, f"0{n}b")
            # Show which destabilizers are active
            active = [f"d{i}" for i in range(n) if (alpha >> (n - 1 - i)) & 1]
            destab_str = "·".join(active) if active else "I"
            print(f"    α={bits}  v={c: .6f}  ({destab_str})|ψ_S⟩")

        # Also show the statevector if small enough
        if n <= 4:
            sv = self.to_statevector()
            print(f"\n  Full statevector (2^{n}={1 << n} entries):")
            for i in range(1 << n):
                if abs(sv[i]) > 1e-10:
                    bits = format(i, f"0{n}b")
                    print(f"    |{bits}⟩: {sv[i]: .6f}")
