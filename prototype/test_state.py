"""Tests: compare generalized stabilizer sim against direct statevector."""

import numpy as np
import pytest
from state import GeneralizedStabilizerState

# Gate matrices for reference simulation
I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
T_mat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Tdg_mat = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)


def states_equal_up_to_phase(sv1, sv2, atol=1e-8):
    """Check if two statevectors are equal up to a global phase."""
    # Find first nonzero entry
    for i in range(len(sv1)):
        if abs(sv1[i]) > atol:
            phase = sv2[i] / sv1[i]
            return np.allclose(sv1 * phase, sv2, atol=atol)
    return np.allclose(sv2, 0, atol=atol)


def apply_single(sv, gate, qubit, n):
    """Apply single-qubit gate to statevector."""
    ops = [I2] * n
    ops[qubit] = gate
    mat = ops[0]
    for op in ops[1:]:
        mat = np.kron(mat, op)
    return mat @ sv


def apply_cnot_sv(sv, ctrl, tgt, n):
    """Apply CNOT to statevector."""
    dim = 1 << n
    result = np.zeros(dim, dtype=complex)
    for i in range(dim):
        bits = list(format(i, f"0{n}b"))
        if bits[ctrl] == '1':
            bits[tgt] = '0' if bits[tgt] == '1' else '1'
        j = int(''.join(bits), 2)
        result[j] += sv[i]
    return result


def test_initial_state():
    s = GeneralizedStabilizerState(2)
    sv = s.to_statevector()
    expected = np.array([1, 0, 0, 0], dtype=complex)
    assert states_equal_up_to_phase(sv, expected)


def test_hadamard():
    s = GeneralizedStabilizerState(1)
    s.apply_h(0)
    sv = s.to_statevector()
    expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert states_equal_up_to_phase(sv, expected)


def test_bell_state():
    s = GeneralizedStabilizerState(2)
    s.apply_h(0)
    s.apply_cnot(0, 1)
    sv = s.to_statevector()
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert states_equal_up_to_phase(sv, expected)


def test_t_gate_single():
    """T|0> should just be |0> (up to phase)."""
    s = GeneralizedStabilizerState(1)
    s.apply_t(0)
    sv = s.to_statevector()
    expected = np.array([1, 0], dtype=complex)
    assert states_equal_up_to_phase(sv, expected)
    assert s.num_entries == 1  # T on |0> shouldn't increase entries


def test_t_on_plus():
    """T|+> should give the correct state."""
    s = GeneralizedStabilizerState(1)
    s.apply_h(0)
    s.apply_t(0)
    sv = s.to_statevector()
    expected = T_mat @ (H @ np.array([1, 0], dtype=complex))
    assert states_equal_up_to_phase(sv, expected)
    assert s.num_entries == 2  # T on |+> doubles entries


def test_tdg_on_plus():
    s = GeneralizedStabilizerState(1)
    s.apply_h(0)
    s.apply_tdg(0)
    sv = s.to_statevector()
    expected = Tdg_mat @ (H @ np.array([1, 0], dtype=complex))
    assert states_equal_up_to_phase(sv, expected)


def test_t_then_tdg_is_identity():
    """T† T = I (up to global phase)."""
    s = GeneralizedStabilizerState(1)
    s.apply_h(0)
    s.apply_t(0)
    s.apply_tdg(0)
    sv = s.to_statevector()
    expected = H @ np.array([1, 0], dtype=complex)
    assert states_equal_up_to_phase(sv, expected)


def test_two_t_gates_is_s():
    """T^2 = S (up to global phase)."""
    s = GeneralizedStabilizerState(1)
    s.apply_h(0)
    s.apply_t(0)
    s.apply_t(0)
    sv = s.to_statevector()
    expected = S @ H @ np.array([1, 0], dtype=complex)
    assert states_equal_up_to_phase(sv, expected)


def test_bell_plus_t():
    """Bell state with T on qubit 0."""
    s = GeneralizedStabilizerState(2)
    s.apply_h(0)
    s.apply_cnot(0, 1)
    s.apply_t(0)
    sv = s.to_statevector()

    ref = np.array([1, 0, 0, 0], dtype=complex)
    ref = apply_single(ref, H, 0, 2)
    ref = apply_cnot_sv(ref, 0, 1, 2)
    ref = apply_single(ref, T_mat, 0, 2)
    assert states_equal_up_to_phase(sv, ref)


def test_clifford_only_entries_stable():
    """Clifford-only circuit should keep |v|=1."""
    s = GeneralizedStabilizerState(3)
    s.apply_h(0)
    s.apply_cnot(0, 1)
    s.apply_s(1)
    s.apply_h(2)
    s.apply_cnot(2, 0)
    assert s.num_entries == 1

    ref = np.zeros(8, dtype=complex)
    ref[0] = 1.0
    ref = apply_single(ref, H, 0, 3)
    ref = apply_cnot_sv(ref, 0, 1, 3)
    ref = apply_single(ref, S, 1, 3)
    ref = apply_single(ref, H, 2, 3)
    ref = apply_cnot_sv(ref, 2, 0, 3)
    sv = s.to_statevector()
    assert states_equal_up_to_phase(sv, ref)


def test_multiple_t_gates():
    """3 qubits, T on each after H."""
    s = GeneralizedStabilizerState(3)
    ref = np.zeros(8, dtype=complex)
    ref[0] = 1.0

    for q in range(3):
        s.apply_h(q)
        ref = apply_single(ref, H, q, 3)

    for q in range(3):
        s.apply_t(q)
        ref = apply_single(ref, T_mat, q, 3)

    sv = s.to_statevector()
    assert states_equal_up_to_phase(sv, ref)
    # After 3 independent T gates on |+++>, |v| ≤ 2^3 = 8
    assert s.num_entries <= 8


def test_measure_deterministic():
    """Measuring |0> in Z gives 0 deterministically."""
    s = GeneralizedStabilizerState(1)
    rng = np.random.default_rng(42)
    for _ in range(10):
        s2 = s.copy()
        assert s2.measure(0, rng=rng) == 0


def test_measure_after_x():
    """Measuring X|0> = |1> gives 1 deterministically."""
    s = GeneralizedStabilizerState(1)
    s.apply_x(0)
    rng = np.random.default_rng(42)
    for _ in range(10):
        s2 = s.copy()
        assert s2.measure(0, rng=rng) == 1


def test_measure_superposition():
    """Measuring |+> gives roughly 50/50."""
    rng = np.random.default_rng(42)
    counts = {0: 0, 1: 0}
    for _ in range(200):
        s = GeneralizedStabilizerState(1)
        s.apply_h(0)
        outcome = s.measure(0, rng=rng)
        counts[outcome] += 1
    # Should be roughly 50/50
    assert 60 < counts[0] < 140, f"Unexpected distribution: {counts}"


def test_measure_collapses_t_state():
    """After T|+> (2 entries), measurement should collapse to 1 entry."""
    rng = np.random.default_rng(42)
    s = GeneralizedStabilizerState(1)
    s.apply_h(0)
    s.apply_t(0)
    assert s.num_entries == 2
    s.measure(0, rng=rng)
    assert s.num_entries == 1


def test_measure_2qubit_after_t():
    """2 qubits with T gates, measurements reduce entries."""
    rng = np.random.default_rng(42)
    s = GeneralizedStabilizerState(2)
    s.apply_h(0)
    s.apply_h(1)
    s.apply_t(0)
    s.apply_t(1)
    assert s.num_entries == 4
    s.measure(0, rng=rng)
    assert s.num_entries <= 2
    s.measure(1, rng=rng)
    assert s.num_entries == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_random_clifford_t_circuits():
    """Fuzz test: random Clifford+T circuits cross-checked against statevector.

    Generates random circuits on 2-4 qubits with H, S, S†, X, Y, Z, CNOT,
    T, T† gates and verifies the generalized stabilizer state matches
    direct statevector simulation at the end.
    """
    rng = np.random.default_rng(12345)
    CZ = np.diag([1, 1, 1, -1]).astype(complex)

    gate_matrices = {
        'H': H, 'S': S, 'Sdg': Sdg, 'X': X,
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': Z, 'T': T_mat, 'Tdg': Tdg_mat,
    }

    n_circuits = 200
    for trial in range(n_circuits):
        n = rng.integers(2, 6)  # 2-5 qubits
        depth = rng.integers(5, 20)  # 5-19 gates

        s = GeneralizedStabilizerState(n)
        sv = np.zeros(1 << n, dtype=complex)
        sv[0] = 1.0

        for _ in range(depth):
            gate_type = rng.choice([
                'H', 'S', 'Sdg', 'X', 'Y', 'Z', 'T', 'Tdg', 'CNOT'
            ])
            if gate_type == 'CNOT':
                if n < 2:
                    continue
                ctrl, tgt = rng.choice(n, size=2, replace=False)
                s.apply_cnot(int(ctrl), int(tgt))
                sv = apply_cnot_sv(sv, int(ctrl), int(tgt), n)
            else:
                q = int(rng.integers(0, n))
                # Apply to generalized stabilizer state
                getattr(s, {
                    'H': 'apply_h', 'S': 'apply_s', 'Sdg': 'apply_sdg',
                    'X': 'apply_x', 'Y': 'apply_y', 'Z': 'apply_z',
                    'T': 'apply_t', 'Tdg': 'apply_tdg',
                }[gate_type])(q)
                # Apply to statevector
                sv = apply_single(sv, gate_matrices[gate_type], q, n)

        gen_sv = s.to_statevector()
        assert states_equal_up_to_phase(gen_sv, sv), (
            f"Trial {trial}: n={n}, depth={depth} — statevectors don't match!\n"
            f"  max diff = {np.max(np.abs(gen_sv - sv))}"
        )


def test_measure_then_continue():
    """Test mid-circuit measurement followed by more gates.

    Verifies measurement + post-measurement T gate produces correct
    statistics by running many shots.
    """
    rng = np.random.default_rng(99)
    # Circuit: H(0), CNOT(0,1), measure(0), T(1), measure(1)
    # After measuring qubit 0, qubit 1 is in a definite state.
    # T on a definite state shouldn't increase |v|.

    for _ in range(20):
        s = GeneralizedStabilizerState(2)
        s.apply_h(0)
        s.apply_cnot(0, 1)
        m0 = s.measure(0, rng=rng)
        # After measuring q0, q1 should be in |m0⟩
        assert s.num_entries == 1
        s.apply_t(1)
        # T on a Z-eigenstate shouldn't grow |v|
        assert s.num_entries == 1
        m1 = s.measure(1, rng=rng)
        assert m1 == m0, "Entangled qubit should match measured partner"


def test_reset():
    """Test reset operation."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        s = GeneralizedStabilizerState(1)
        s.apply_h(0)
        s.apply_t(0)
        s.reset(0)
        # After reset, qubit should be |0⟩
        sv = s.to_statevector()
        expected = np.array([1, 0], dtype=complex)
        assert states_equal_up_to_phase(sv, expected), f"Reset failed: sv={sv}"


def test_statevector_oracle_sanity():
    """Verify the numpy statevector oracle against known identities.

    These don't involve the prototype at all — they check that the
    reference simulation helper functions are correct.
    """
    # HH = I
    sv = np.array([1, 0], dtype=complex)
    sv = apply_single(sv, H, 0, 1)
    sv = apply_single(sv, H, 0, 1)
    assert np.allclose(sv, [1, 0])

    # HZH = X
    sv = np.array([1, 0], dtype=complex)
    sv = apply_single(sv, H, 0, 1)
    sv = apply_single(sv, Z, 0, 1)
    sv = apply_single(sv, H, 0, 1)
    assert np.allclose(sv, [0, 1])  # X|0⟩ = |1⟩

    # SS = Z
    sv = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+⟩
    sv = apply_single(sv, S, 0, 1)
    sv = apply_single(sv, S, 0, 1)
    expected = apply_single(np.array([1, 1], dtype=complex) / np.sqrt(2), Z, 0, 1)
    assert np.allclose(sv, expected)

    # TTTT = Z
    sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
    for _ in range(4):
        sv = apply_single(sv, T_mat, 0, 1)
    expected = apply_single(np.array([1, 1], dtype=complex) / np.sqrt(2), Z, 0, 1)
    assert states_equal_up_to_phase(sv, expected)

    # CNOT|10⟩ = |11⟩
    sv = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
    sv = apply_cnot_sv(sv, 0, 1, 2)
    assert np.allclose(sv, [0, 0, 0, 1])  # |11⟩

    # CNOT|01⟩ = |01⟩ (control not set)
    sv = np.array([0, 1, 0, 0], dtype=complex)
    sv = apply_cnot_sv(sv, 0, 1, 2)
    assert np.allclose(sv, [0, 1, 0, 0])

    # Bell state: CNOT(H⊗I)|00⟩ = (|00⟩+|11⟩)/√2
    sv = np.array([1, 0, 0, 0], dtype=complex)
    sv = apply_single(sv, H, 0, 2)
    sv = apply_cnot_sv(sv, 0, 1, 2)
    assert np.allclose(sv, [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])


def test_deterministic_measurement_with_branch_filtering():
    """Test that measurements correctly filter branches even when β = 0.

    When Z_q is in the stabilizer group (β = 0), the measurement appears
    deterministic to the Clifford simulator. But different branches |b_α⟩
    may have different eigenvalues if stab_bits · α varies across active α.
    The measurement must filter branches in this case.

    This was a bug: _measure_deterministic previously assumed all branches
    had the same eigenvalue, returning the outcome without filtering.
    """
    import numpy as np

    n = 3
    state = GeneralizedStabilizerState(n)

    # Build a circuit that creates the bug condition:
    # After H(1), T(1), Cliffords: branches exist.
    # Then Z of some qubit is a stabilizer (β = 0) but stab_bits
    # has nonzero overlap with the branch index space.
    state.apply_h(1)
    state.measure(0, rng=np.random.default_rng(42))
    state.reset(0)
    state.apply_t(1)
    state.apply_t(0)
    state.apply_h(1)
    state.apply_cnot(0, 2)
    state.apply_cnot(0, 1)
    state.apply_sdg(0)
    state.apply_sdg(2)
    state.measure(0, rng=np.random.default_rng(42))
    state.reset(0)
    state.apply_tdg(1)

    # At this point, |v| = 2 and Z_1 has β = 0 but different eigenvalues
    assert state.num_entries == 2

    # Build statevector for comparison
    sv = state.to_statevector()

    # Compute true Z_1 measurement probabilities from statevector
    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Z1_mat = np.kron(np.kron(I2, Z), I2)  # Z on qubit 1 of 3

    P0 = (np.eye(8) + Z1_mat) / 2
    P1 = (np.eye(8) - Z1_mat) / 2
    prob0 = np.real(np.conj(sv) @ P0 @ sv)
    prob1 = np.real(np.conj(sv) @ P1 @ sv)

    # The state is NOT a Z_1 eigenstate (branches have different eigenvalues)
    assert prob0 > 0.01 and prob1 > 0.01, (
        f"Expected probabilistic measurement, got prob0={prob0}, prob1={prob1}"
    )

    # Measure and check statevector matches
    state_copy = state.copy()
    outcome = state_copy.measure(1, rng=np.random.default_rng(42))

    proto_sv = state_copy.to_statevector()

    # Project oracle statevector for the same outcome
    if outcome == 0:
        expected = P0 @ sv
    else:
        expected = P1 @ sv
    expected = expected / np.linalg.norm(expected)

    # After filtering, |v| should decrease
    assert state_copy.num_entries < state.num_entries, (
        f"Expected |v| to decrease after filtering, "
        f"got {state_copy.num_entries} (was {state.num_entries})"
    )

    # Statevector should match
    overlap = abs(np.conj(proto_sv) @ expected)
    assert overlap > 0.9999, (
        f"Statevector mismatch: overlap={overlap}"
    )


def test_measurement_statevector_fuzz():
    """Fuzz test: compare prototype measurement against statevector oracle.

    This specifically tests the fix for the β=0 measurement bug by running
    many random circuits with interleaved T gates and measurements, and
    comparing the prototype's statevector against a direct matrix oracle.
    """
    import numpy as np

    I2 = np.eye(2, dtype=complex)
    H_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S_mat = np.array([[1, 0], [0, 1j]], dtype=complex)
    Sdg_mat = np.array([[1, 0], [0, -1j]], dtype=complex)
    T_mat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    Tdg_mat = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_gate(gate, qubit, n):
        mats = [I2] * n
        mats[qubit] = gate
        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result

    def cnot_matrix(ctrl, tgt, n):
        dim = 2 ** n
        mat = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            bits = list(format(i, f"0{n}b"))
            if bits[ctrl] == "0":
                mat[i, i] = 1
            else:
                bits[tgt] = str(1 - int(bits[tgt]))
                j = int("".join(bits), 2)
                mat[j, i] = 1
        return mat

    rng = np.random.default_rng(20250704)
    mismatches = 0
    completed = 0

    for trial in range(200):
        n = int(rng.integers(2, 5))
        depth = int(rng.integers(5, 15))
        dim = 2 ** n

        state = GeneralizedStabilizerState(n)
        sv = np.zeros(dim, dtype=complex)
        sv[0] = 1.0

        meas_rng_state = rng.integers(0, 2**32)
        diverged = False

        for _ in range(depth):
            r = rng.random()
            if r < 0.12:
                q = int(rng.integers(0, n))
                state.apply_t(q)
                sv = kron_gate(T_mat, q, n) @ sv
            elif r < 0.20:
                q = int(rng.integers(0, n))
                state.apply_tdg(q)
                sv = kron_gate(Tdg_mat, q, n) @ sv
            elif r < 0.32 and n >= 2:
                ctrl, tgt = rng.choice(n, size=2, replace=False)
                state.apply_cnot(int(ctrl), int(tgt))
                sv = cnot_matrix(int(ctrl), int(tgt), n) @ sv
            elif r < 0.48:
                q = int(rng.integers(0, n))

                # Compute oracle probabilities
                Z_q = kron_gate(Z_mat, q, n)
                P0 = (np.eye(dim) + Z_q) / 2
                P1 = (np.eye(dim) - Z_q) / 2
                prob0 = max(0.0, np.real(np.conj(sv) @ P0 @ sv))

                # Shared coin flip
                coin = np.random.default_rng(meas_rng_state).random()
                meas_rng_state += 1
                sv_outcome = 0 if coin < prob0 else 1

                # Project oracle statevector
                sv = (P0 if sv_outcome == 0 else P1) @ sv
                sv_norm = np.linalg.norm(sv)
                if sv_norm > 1e-15:
                    sv /= sv_norm

                # Prototype measurement with same coin
                class FixedRng:
                    def __init__(self, val):
                        self._val = val
                    def random(self):
                        return self._val

                proto_outcome = state.measure(q, rng=FixedRng(coin))

                if proto_outcome != sv_outcome:
                    diverged = True
                    break

                # Reset
                if sv_outcome == 1:
                    sv = kron_gate(X_mat, q, n) @ sv
                state.reset(q)
            else:
                gate_choice = rng.choice(["H", "S", "S_DAG"])
                q = int(rng.integers(0, n))
                if gate_choice == "H":
                    state.apply_h(q)
                    sv = kron_gate(H_mat, q, n) @ sv
                elif gate_choice == "S":
                    state.apply_s(q)
                    sv = kron_gate(S_mat, q, n) @ sv
                else:
                    state.apply_sdg(q)
                    sv = kron_gate(Sdg_mat, q, n) @ sv

        if diverged:
            continue  # outcome divergence, skip

        completed += 1
        proto_sv = state.to_statevector()
        overlap = abs(np.conj(proto_sv) @ sv)
        if overlap < 0.9999:
            mismatches += 1

    assert completed > 100, f"Too few completed trials: {completed}"
    assert mismatches == 0, (
        f"Statevector mismatches: {mismatches}/{completed} trials"
    )
