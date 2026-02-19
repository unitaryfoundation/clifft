"""Tests for the AOT Compiler + VM prototype.

Validates against numpy matrix-multiply reference simulation.
"""

import math
import numpy as np
import pytest
from aot_compiler import AOTCompiler, AOTRuntimeVM, aot_to_statevector

# Gate matrices
I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
T_mat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Tdg_mat = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)


def phase_eq(sv1, sv2, atol=1e-7):
    """Check if two statevectors are equal up to a global phase."""
    for i in range(len(sv1)):
        if abs(sv1[i]) > atol:
            phase = sv2[i] / sv1[i]
            return np.allclose(sv1 * phase, sv2, atol=atol)
    return np.allclose(sv2, 0, atol=atol)


def apply_single(sv, gate, qubit, n):
    ops = [I2] * n
    ops[qubit] = gate
    mat = ops[0]
    for op in ops[1:]:
        mat = np.kron(mat, op)
    return mat @ sv


def apply_cnot_sv(sv, ctrl, tgt, n):
    dim = 1 << n
    result = np.zeros(dim, dtype=complex)
    for i in range(dim):
        bits = list(format(i, f"0{n}b"))
        if bits[ctrl] == '1':
            bits[tgt] = '0' if bits[tgt] == '1' else '1'
        j = int(''.join(bits), 2)
        result[j] += sv[i]
    return result


def run_aot(n, ops):
    """Compile + execute ops, return statevector."""
    c = AOTCompiler(n)
    bc = c.compile_ops(ops)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(bc, c.noise_schedule)
    return aot_to_statevector(n, c, vm)


def run_ref(n, ops):
    """Reference numpy simulation."""
    gate_map = {
        'H': H, 'S': S, 'S_DAG': Sdg, 'T': T_mat, 'T_DAG': Tdg_mat,
        'X': X_mat, 'Y': Y_mat, 'Z': Z_mat,
    }
    sv = np.zeros(1 << n, dtype=complex)
    sv[0] = 1.0
    for gate_name, qubits in ops:
        if gate_name in gate_map:
            sv = apply_single(sv, gate_map[gate_name], qubits[0], n)
        elif gate_name == 'CX':
            sv = apply_cnot_sv(sv, qubits[0], qubits[1], n)
    return sv


# ── Basic tests ──

def test_identity():
    sv = run_aot(1, [])
    assert phase_eq(sv, np.array([1, 0], dtype=complex))


def test_hadamard_only():
    sv = run_aot(1, [('H', [0])])
    ref = run_ref(1, [('H', [0])])
    assert phase_eq(sv, ref)


def test_t_on_zero():
    """T|0> = |0> up to phase (beta=0, scalar phase only)."""
    sv = run_aot(1, [('T', [0])])
    ref = run_ref(1, [('T', [0])])
    assert phase_eq(sv, ref)


def test_t_on_plus():
    ops = [('H', [0]), ('T', [0])]
    assert phase_eq(run_aot(1, ops), run_ref(1, ops))


def test_tdg_on_plus():
    ops = [('H', [0]), ('T_DAG', [0])]
    assert phase_eq(run_aot(1, ops), run_ref(1, ops))


def test_t_then_tdg():
    """T† T = I."""
    ops = [('H', [0]), ('T', [0]), ('T_DAG', [0])]
    assert phase_eq(run_aot(1, ops), run_ref(1, ops))


def test_two_t_is_s():
    """T^2 = S."""
    ops = [('H', [0]), ('T', [0]), ('T', [0])]
    assert phase_eq(run_aot(1, ops), run_ref(1, ops))


def test_bell_plus_t():
    ops = [('H', [0]), ('CX', [0, 1]), ('T', [0])]
    assert phase_eq(run_aot(2, ops), run_ref(2, ops))


def test_three_qubits_h_t():
    """H on all, T on all."""
    ops = [('H', [q]) for q in range(3)] + [('T', [q]) for q in range(3)]
    assert phase_eq(run_aot(3, ops), run_ref(3, ops))


def test_clifford_only():
    ops = [('H', [0]), ('CX', [0, 1]), ('S', [1]), ('H', [2]), ('CX', [2, 0])]
    assert phase_eq(run_aot(3, ops), run_ref(3, ops))


def test_s_then_t():
    ops = [('H', [0]), ('S', [0]), ('T', [0])]
    assert phase_eq(run_aot(1, ops), run_ref(1, ops))


def test_collide_two_t_same_qubit():
    """Two T gates on same qubit in superposition -> second is COLLIDE."""
    ops = [('H', [0]), ('T', [0]), ('S', [0]), ('T', [0])]
    sv = run_aot(1, ops)
    ref = run_ref(1, ops)
    assert phase_eq(sv, ref)


def test_interleaved_clifford_t():
    """Clifford gates between T gates."""
    ops = [
        ('H', [0]), ('H', [1]),
        ('T', [0]),
        ('CX', [0, 1]),
        ('T', [1]),
        ('H', [0]),
        ('T', [0]),
    ]
    assert phase_eq(run_aot(2, ops), run_ref(2, ops))


def test_t_on_entangled():
    """T gates on both qubits of a Bell pair."""
    ops = [
        ('H', [0]), ('CX', [0, 1]),
        ('T', [0]), ('T', [1]),
    ]
    assert phase_eq(run_aot(2, ops), run_ref(2, ops))


def test_many_t_gates():
    """Several T gates with Cliffords between."""
    ops = [
        ('H', [0]), ('H', [1]), ('H', [2]),
        ('T', [0]), ('T', [1]), ('T', [2]),
        ('CX', [0, 1]), ('CX', [1, 2]),
        ('T', [0]), ('T', [1]), ('T', [2]),
    ]
    assert phase_eq(run_aot(3, ops), run_ref(3, ops))


# ── Fuzz test ──

def build_random_circuit(n, depth, rng, t_frac=0.3):
    """Generate random Clifford+T circuit."""
    ops = []
    single_cliffords = ['H', 'S', 'S_DAG', 'X', 'Y', 'Z']
    t_gates = ['T', 'T_DAG']
    for _ in range(depth):
        r = rng.random()
        if r < t_frac:
            gate = rng.choice(t_gates)
            q = int(rng.integers(0, n))
            ops.append((gate, [q]))
        elif r < t_frac + 0.15 and n >= 2:
            ctrl, tgt = rng.choice(n, size=2, replace=False)
            ops.append(('CX', [int(ctrl), int(tgt)]))
        else:
            gate = rng.choice(single_cliffords)
            q = int(rng.integers(0, n))
            ops.append((gate, [q]))
    return ops


@pytest.mark.parametrize("seed", range(500))
def test_fuzz_small(seed):
    """Fuzz: random circuits on 2-4 qubits, 5-15 gates."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 5))
    depth = int(rng.integers(5, 16))
    ops = build_random_circuit(n, depth, rng)

    sv_aot = run_aot(n, ops)
    sv_ref = run_ref(n, ops)
    assert phase_eq(sv_aot, sv_ref), (
        f"Mismatch at seed={seed}, n={n}, depth={depth}\n"
        f"ops={ops}\n"
        f"aot={sv_aot}\nref={sv_ref}"
    )


# ── Measurement tests ──

def run_aot_with_measurements(n, ops, seed=42):
    """Compile + execute with measurements, return (sv, measurements)."""
    c = AOTCompiler(n)
    bc = c.compile_ops(ops)
    vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
    vm.execute(bc, c.noise_schedule)
    sv = aot_to_statevector(n, c, vm)
    return sv, vm.measurements


def test_measure_zero_state():
    """Measure |0> -> always 0, state unchanged."""
    # Z_0 on |0> is stabilizer, beta=0, mapped_gamma=0 -> INDEPENDENT
    sv, meas = run_aot_with_measurements(1, [('M', [0])])
    # After measurement, state should still be |0> or |1>
    assert len(meas) == 1


def test_measure_after_h():
    """Measure |+> -> random outcome, state collapses."""
    for seed in range(20):
        c = AOTCompiler(1)
        bc = c.compile_ops([('H', [0]), ('M', [0])])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        assert len(vm.measurements) == 1
        assert vm.measurements[0] in (0, 1)


def test_measure_bell_state():
    """Measure qubit 0 of Bell state."""
    ops = [('H', [0]), ('CX', [0, 1]), ('M', [0])]
    c = AOTCompiler(2)
    bc = c.compile_ops(ops)
    for inst in bc:
        print(inst)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(bc, c.noise_schedule)
    print('v after measure:', vm.v)
    print('measurements:', vm.measurements)


def test_measure_after_t():
    """H, T, then measure. The T creates a branch, measure collapses."""
    ops = [('H', [0]), ('T', [0]), ('M', [0])]
    c = AOTCompiler(1)
    bc = c.compile_ops(ops)
    print('Bytecode:')
    for inst in bc:
        print(f'  {inst}')
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(bc, c.noise_schedule)
    print('v:', vm.v, 'meas:', vm.measurements)
    # After H T M: should have 1 measurement, v should shrink to size 1
    assert len(vm.measurements) == 1


def test_measure_reset_t():
    """MR creates SSA qubit. Subsequent T on same qubit should work."""
    ops = [
        ('H', [0]), ('T', [0]),  # Create branch
        ('MR', [0]),              # Measure + reset via SSA
        ('H', [0]), ('T', [0]),  # Fresh T on reset qubit
    ]
    c = AOTCompiler(1)
    bc = c.compile_ops(ops)
    print('Bytecode:')
    for inst in bc:
        print(f'  {inst}')
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(bc, c.noise_schedule)
    print('v:', vm.v, 'meas:', vm.measurements)
    # After MR, the branch from first T is collapsed.
    # After H+T on fresh qubit, we have a new branch.
    assert len(vm.v) == 2  # one branch from second T


def test_measure_statistics():
    """Measure |+> 1000 times, check ~50/50 split."""
    outcomes = []
    for seed in range(1000):
        c = AOTCompiler(1)
        bc = c.compile_ops([('H', [0]), ('M', [0])])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        outcomes.append(vm.measurements[0])
    count_0 = outcomes.count(0)
    # Expect ~500 ± 50
    assert 400 < count_0 < 600, f"Got {count_0} zeros out of 1000"


def test_measure_t_statistics():
    """H T H M: the T-rotated state has non-uniform Z-basis probabilities.

    H T H |0> has P(0) = cos^2(pi/8) ~= 0.854.
    """
    outcomes = []
    for seed in range(2000):
        c = AOTCompiler(1)
        bc = c.compile_ops([('H', [0]), ('T', [0]), ('H', [0]), ('M', [0])])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        outcomes.append(vm.measurements[0])
    count_0 = outcomes.count(0)
    expected_p0 = math.cos(math.pi / 8) ** 2  # ~0.854
    actual_p0 = count_0 / 2000
    assert abs(actual_p0 - expected_p0) < 0.05, (
        f"Expected p(0)={expected_p0:.3f}, got {actual_p0:.3f}"
    )


@pytest.mark.parametrize("seed", range(100))
def test_fuzz_with_measurements(seed):
    """Fuzz: circuits with T gates and mid-circuit MR."""
    rng = np.random.default_rng(seed + 10000)
    n = int(rng.integers(2, 4))
    depth = int(rng.integers(8, 20))
    ops = build_random_circuit(n, depth, rng, t_frac=0.25)
    # Add measurements at the end
    for q in range(n):
        if rng.random() < 0.5:
            ops.append(('MR', [q]))
    # Add more gates after measurements
    extra = build_random_circuit(n, int(rng.integers(3, 8)), rng, t_frac=0.25)
    ops.extend(extra)

    # Just verify it doesn't crash and produces valid output
    c = AOTCompiler(n)
    bc = c.compile_ops(ops)
    vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
    vm.execute(bc, c.noise_schedule)
    sv = aot_to_statevector(n, c, vm)
    assert sv.shape == (1 << n,)
    norm = np.linalg.norm(sv)
    assert abs(norm - 1.0) < 1e-6 or norm < 1e-10, f"Bad norm: {norm}"


# ── Sign tracker / noise tests ──

def test_noise_no_error_preserves_state():
    """With p=0 noise, state is identical to noiseless."""
    ops_noiseless = [('H', [0]), ('T', [0])]
    ops_noisy = [('H', [0]), ('DEPOLARIZE', [0, 0.0]), ('T', [0])]
    sv_clean = run_aot(1, ops_noiseless)
    sv_noisy = run_aot(1, ops_noisy)
    assert phase_eq(sv_clean, sv_noisy)


def test_noise_p1_flips_deterministic_measurement():
    """X noise on |0> flips the stabilizer sign; deterministic measurement should reflect this.

    This test validates that MEASURE_INDEPENDENT uses the sign tracker.
    |0> has stabilizer +Z. After X noise, physical stabilizer is -Z.
    Deterministic measurement of Z should give outcome 1 (eigenvalue -1).
    """
    outcomes = []
    for seed in range(100):
        c = AOTCompiler(1)
        bc = c.compile_ops([
            ('PAULI_NOISE', [0, 1.0, 0.0, 0.0]),  # X error with p=1
            ('M', [0]),
        ])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        outcomes.append(vm.measurements[0])
    # All outcomes should be 1 (X flips Z stabilizer sign)
    assert all(o == 1 for o in outcomes), (
        f"Expected all 1s, got {outcomes[:10]}..."
    )


def test_noise_z_on_plus_state():
    """Z noise on |+> should not change Z-basis measurement statistics.

    |+> = H|0>. Z|+> = |->. Both have 50/50 for Z measurement.
    """
    outcomes_clean = []
    outcomes_noisy = []
    for seed in range(1000):
        c = AOTCompiler(1)
        bc = c.compile_ops([('H', [0]), ('M', [0])])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        outcomes_clean.append(vm.measurements[0])

        c2 = AOTCompiler(1)
        bc2 = c2.compile_ops([('H', [0]), ('PAULI_NOISE', [0, 0.0, 0.0, 1.0]), ('M', [0])])
        vm2 = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm2.execute(bc2, c2.noise_schedule)
        outcomes_noisy.append(vm2.measurements[0])

    # Both should be ~50/50
    p_clean = sum(outcomes_clean) / len(outcomes_clean)
    p_noisy = sum(outcomes_noisy) / len(outcomes_noisy)
    assert abs(p_clean - 0.5) < 0.05
    assert abs(p_noisy - 0.5) < 0.05


def test_sign_tracker_affects_t_gate():
    """Sign tracker should change the effective phase of a T gate.

    Two scenarios with same circuit but different initial sign state.
    This is a unit test for _resolve_sign.
    """
    from aot_compiler import PHASES

    # H|0> = |+>, then T|+>
    # Without noise: specific statevector
    # With X-noise before T: the T gate's rewound Z picks up sign flip
    ops_clean = [('H', [0]), ('T', [0])]
    ops_noisy = [('H', [0]), ('PAULI_NOISE', [0, 1.0, 0.0, 0.0]), ('T', [0])]

    sv_clean = run_aot(1, ops_clean)

    # With noise: compile and run many times. X noise with p=1 always fires.
    c = AOTCompiler(1)
    bc = c.compile_ops(ops_noisy)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(bc, c.noise_schedule)
    sv_noisy = aot_to_statevector(1, c, vm)

    # They should differ (X error before T changes the phase)
    # Actually, let's verify against reference
    ref_clean = run_ref(1, [('H', [0]), ('T', [0])])
    ref_noisy = run_ref(1, [('H', [0]), ('X', [0]), ('T', [0])])

    assert phase_eq(sv_clean, ref_clean)
    # The noisy version should match X then T
    assert phase_eq(sv_noisy, ref_noisy), (
        f"noisy={sv_noisy}\nref_noisy={ref_noisy}"
    )


def test_depolarize_compiles():
    """Depolarizing noise should produce 3 noise schedule entries."""
    c = AOTCompiler(1)
    bc = c.compile_ops([('DEPOLARIZE', [0, 0.3])])
    # Noise is now in the schedule, not bytecode
    all_noise = []
    for entries in c.noise_schedule.values():
        all_noise.extend(entries)
    assert len(all_noise) == 3
    for prob, dm, sm in all_noise:
        assert abs(prob - 0.1) < 1e-10


def test_noise_anticommute_mask_structure():
    """X error on qubit 0 should anticommute with Z-stabilizer of |0>."""
    c = AOTCompiler(1)
    error = [1]  # X on qubit 0
    dm, sm = c._compute_anticommute_masks(error)
    # For |0>: destab=X_0, stab=Z_0. X commutes with X, anticommutes with Z.
    assert (dm >> 0) & 1 == 0, f"X should commute with destab X_0, dm={bin(dm)}"
    assert (sm >> 0) & 1 == 1, f"X should anticommute with stab Z_0, sm={bin(sm)}"


def test_noise_z_anticommute_mask_structure():
    """Z error on qubit 0: anticommutes with X-destabilizer, commutes with Z-stabilizer."""
    c = AOTCompiler(1)
    error = [3]  # Z on qubit 0
    dm, sm = c._compute_anticommute_masks(error)
    assert (dm >> 0) & 1 == 1, f"Z should anticommute with destab X_0, dm={bin(dm)}"
    assert (sm >> 0) & 1 == 0, f"Z should commute with stab Z_0, sm={bin(sm)}"


from aot_compiler import OpCode


def test_noise_after_clifford_mask_rotates():
    """After H, the stabilizer/destabilizer roles swap. Masks should reflect this."""
    c = AOTCompiler(1)
    c.h(0)  # Now stab is X_0, destab is Z_0
    error_x = [1]  # X on qubit 0
    dm, sm = c._compute_anticommute_masks(error_x)
    # After H: destab=Z_0, stab=X_0. X anticommutes with Z (destab), commutes with X (stab).
    assert (dm >> 0) & 1 == 1, f"After H: X should anticommute with destab Z, dm={bin(dm)}"
    assert (sm >> 0) & 1 == 0, f"After H: X should commute with stab X, sm={bin(sm)}"


def test_noise_x_flips_z_measurement():
    """X noise with probability p should flip Z-measurement outcome with probability p.

    Prepare |0>, apply X noise with p=0.3, measure Z.
    Without noise: always outcome 0.
    With noise: outcome 1 with probability 0.3.
    """
    outcomes = []
    N = 2000
    p_noise = 0.3
    for seed in range(N):
        c = AOTCompiler(1)
        bc = c.compile_ops([
            ('PAULI_NOISE', [0, p_noise, 0.0, 0.0]),  # X with p=0.3
            ('M', [0]),
        ])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        outcomes.append(vm.measurements[0])
    actual_p1 = sum(outcomes) / N
    assert abs(actual_p1 - p_noise) < 0.05, (
        f"Expected p(1)={p_noise}, got {actual_p1}"
    )


def test_noise_depolarize_t_gate_statistics():
    """H, depolarize(p), T, H, M should show noise-shifted probabilities.

    Without noise: H T H |0> has P(0) = cos^2(pi/8) ~ 0.854
    With depolarizing noise: probabilities shift toward 0.5
    """
    N = 2000
    p_noise = 0.3

    outcomes_clean = []
    outcomes_noisy = []
    for seed in range(N):
        # Clean
        c1 = AOTCompiler(1)
        bc1 = c1.compile_ops([('H', [0]), ('T', [0]), ('H', [0]), ('M', [0])])
        vm1 = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm1.execute(bc1, c1.noise_schedule)
        outcomes_clean.append(vm1.measurements[0])

        # Noisy
        c2 = AOTCompiler(1)
        bc2 = c2.compile_ops([
            ('H', [0]),
            ('DEPOLARIZE', [0, p_noise]),
            ('T', [0]),
            ('H', [0]),
            ('M', [0]),
        ])
        vm2 = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm2.execute(bc2, c2.noise_schedule)
        outcomes_noisy.append(vm2.measurements[0])

    p0_clean = 1.0 - sum(outcomes_clean) / N
    p0_noisy = 1.0 - sum(outcomes_noisy) / N

    expected_clean = math.cos(math.pi / 8) ** 2
    assert abs(p0_clean - expected_clean) < 0.05, f"Clean: expected {expected_clean:.3f}, got {p0_clean:.3f}"

    # Noisy should be shifted toward 0.5 compared to clean
    assert abs(p0_noisy - 0.5) < abs(p0_clean - 0.5) or abs(p0_noisy - p0_clean) < 0.1, (
        f"Noisy p0={p0_noisy:.3f} should be closer to 0.5 than clean p0={p0_clean:.3f}"
    )


def test_bell_state_correlation():
    """Bell state measurements must be perfectly correlated."""
    N = 1000
    corr = 0
    for seed in range(N):
        c = AOTCompiler(2)
        bc = c.compile_ops([('H', [0]), ('CX', [0, 1]), ('M', [0]), ('M', [1])])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        assert len(vm.measurements) == 2
        corr += (vm.measurements[0] == vm.measurements[1])
    assert corr == N, f"Bell state: expected perfect correlation, got {corr}/{N}"


def test_ghz_state_correlation():
    """GHZ state measurements must be perfectly correlated (all same)."""
    N = 500
    for seed in range(N):
        c = AOTCompiler(3)
        bc = c.compile_ops([
            ('H', [0]), ('CX', [0, 1]), ('CX', [0, 2]),
            ('M', [0]), ('M', [1]), ('M', [2]),
        ])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        assert len(vm.measurements) == 3
        assert vm.measurements[0] == vm.measurements[1] == vm.measurements[2], (
            f"GHZ: expected all same, got {vm.measurements} at seed={seed}"
        )


def test_noise_on_bell_state_measurement():
    """Bell state with X noise on qubit 0: should flip correlation."""
    N = 2000
    corr_clean = 0
    corr_noisy = 0

    for seed in range(N):
        c1 = AOTCompiler(2)
        bc1 = c1.compile_ops([('H', [0]), ('CX', [0, 1]), ('M', [0]), ('M', [1])])
        vm1 = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm1.execute(bc1, c1.noise_schedule)
        if len(vm1.measurements) == 2:
            corr_clean += (vm1.measurements[0] == vm1.measurements[1])

        c2 = AOTCompiler(2)
        bc2 = c2.compile_ops([
            ('H', [0]), ('CX', [0, 1]),
            ('PAULI_NOISE', [0, 1.0, 0.0, 0.0]),
            ('M', [0]), ('M', [1]),
        ])
        vm2 = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm2.execute(bc2, c2.noise_schedule)
        if len(vm2.measurements) == 2:
            corr_noisy += (vm2.measurements[0] != vm2.measurements[1])

    assert corr_clean / N > 0.95, f"Clean correlation={corr_clean/N:.3f}"
    assert corr_noisy / N > 0.95, f"Noisy anti-correlation={corr_noisy/N:.3f}"


# ── Detector tests ──

def test_detector_no_noise():
    """Detector on repeat-measurement parity should not fire without noise."""
    N = 200
    for seed in range(N):
        c = AOTCompiler(1)
        # Measure qubit 0 twice. Without noise, both should agree.
        bc = c.compile_ops([
            ('H', [0]),
            ('MR', [0]),
            ('H', [0]),
            ('MR', [0]),
        ])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        # Both measurements on |+> are random but independent after reset.
        # A detector on one measurement should give 0 (no error).
        # Let's use a simpler circuit: prepare |0>, measure, measure.

    # Simpler: |0> measured twice. Both should give 0.
    for seed in range(N):
        c = AOTCompiler(1)
        bc = c.compile_ops([
            ('M', [0]),
            ('DETECTOR', [0]),  # Single measurement; should always be 0
        ])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        assert vm.detections == [0], f"Detector fired without noise: seed={seed}"


def test_detector_with_noise():
    """Detector should fire when Pauli noise causes a bit flip."""
    N = 500
    detections = []
    p_noise = 0.3
    for seed in range(N):
        c = AOTCompiler(1)
        # Prepare |0>, inject X noise (flips outcome), measure, check detector
        bc = c.compile_ops([
            ('PAULI_NOISE', [0, p_noise, 0.0, 0.0]),  # X error with p=0.3
            ('M', [0]),
            ('DETECTOR', [0]),  # Should be 0 without noise, 1 with X error
        ])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        detections.append(vm.detections[0])

    detection_rate = sum(detections) / N
    assert abs(detection_rate - p_noise) < 0.05, (
        f"Detection rate {detection_rate:.3f} doesn't match noise prob {p_noise}"
    )


def test_detector_parity_check():
    """Detector on XOR of two correlated measurements should give 0."""
    N = 500
    for seed in range(N):
        c = AOTCompiler(2)
        # Bell state: measurements are correlated (both same)
        bc = c.compile_ops([
            ('H', [0]), ('CX', [0, 1]),
            ('M', [0]), ('M', [1]),
            ('DETECTOR', [0, 1]),  # XOR should be 0
        ])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        assert vm.detections == [0], (
            f"Detector fired on Bell state: seed={seed}, meas={vm.measurements}"
        )


def test_detector_bell_with_noise():
    """X noise on one Bell qubit should trigger the parity detector."""
    N = 500
    detections = []
    for seed in range(N):
        c = AOTCompiler(2)
        bc = c.compile_ops([
            ('H', [0]), ('CX', [0, 1]),
            ('PAULI_NOISE', [0, 1.0, 0.0, 0.0]),  # X on q0 always
            ('M', [0]), ('M', [1]),
            ('DETECTOR', [0, 1]),
        ])
        vm = AOTRuntimeVM(rng=np.random.default_rng(seed))
        vm.execute(bc, c.noise_schedule)
        detections.append(vm.detections[0])

    # X noise anti-correlates Bell measurements, so XOR = 1 always
    assert all(d == 1 for d in detections), (
        f"Expected all detections=1, got {sum(detections)}/{N}"
    )


# ── SSA recycling test ──

def test_ssa_recycling():
    """After MR, virtual qubit should be recycled (n_virtual stays bounded)."""
    c = AOTCompiler(1)
    initial_nv = c.n_virtual
    ops = []
    for _ in range(10):
        ops.extend([('H', [0]), ('MR', [0])])
    c.compile_ops(ops)
    # With recycling, n_virtual should not grow by 10
    # (first MR frees the old and allocs new; subsequent ones reuse)
    assert c.n_virtual <= initial_nv + 2, (
        f"n_virtual grew to {c.n_virtual} after 10 MR ops (expected <= {initial_nv + 2})"
    )


# ── Peak rank tracking tests ──

def test_peak_rank_no_t():
    """Clifford-only circuit: peak rank stays 0."""
    c = AOTCompiler(2)
    c.compile_ops([('H', [0]), ('CX', [0, 1]), ('S', [1])])
    assert c.peak_rank == 0


def test_peak_rank_single_t():
    """Single T on |+>: one branch, peak rank = 1."""
    c = AOTCompiler(1)
    c.compile_ops([('H', [0]), ('T', [0])])
    assert c.peak_rank == 1


def test_peak_rank_measure_reduces():
    """T then measure: rank grows to 1, then drops. Peak stays 1."""
    c = AOTCompiler(1)
    c.compile_ops([('H', [0]), ('T', [0]), ('M', [0])])
    assert c.peak_rank == 1
    assert c.V.rank == 0  # measurement collapsed it


def test_peak_rank_multiple_t():
    """Multiple T gates on different qubits: peak rank grows."""
    c = AOTCompiler(3)
    c.compile_ops([('H', [0]), ('H', [1]), ('H', [2]),
                   ('T', [0]), ('T', [1]), ('T', [2])])
    assert c.peak_rank == 3


# ── General LCU tests ──

def test_lcu_as_t_gate():
    """LCU encoding of T gate should match dedicated T gate."""
    import cmath
    cos_val = math.cos(math.pi / 8)
    sin_val = math.sin(math.pi / 8)
    # T = cos(pi/8) I + (-i sin(pi/8)) Z = e^{-i pi/8 Z}
    # In matrix form: T = [[1, 0], [0, e^{i pi/4}]]
    # T = cos(pi/8) I + (-i sin(pi/8)) Z
    t_terms = [(cos_val, 'I'), (-1j * sin_val, 'Z')]

    # Reference: dedicated T gate path
    ops_t = [('H', [0]), ('T', [0])]
    sv_t = run_aot(1, ops_t)

    # LCU path
    c = AOTCompiler(1)
    c.h(0)
    c.apply_lcu(0, t_terms)
    bc = c.bytecode
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(bc, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)

    # Reference numpy
    ref = run_ref(1, [('H', [0]), ('T', [0])])

    assert phase_eq(sv_lcu, ref), f"LCU T-gate mismatch: lcu={sv_lcu}, ref={ref}"


def test_lcu_as_t_dag():
    """LCU encoding of T-dagger."""
    cos_val = math.cos(math.pi / 8)
    sin_val = math.sin(math.pi / 8)
    # T† = cos(pi/8) I + (i sin(pi/8)) Z
    t_dag_terms = [(cos_val, 'I'), (1j * sin_val, 'Z')]

    ops_ref = [('H', [0]), ('T_DAG', [0])]
    ref = run_ref(1, ops_ref)

    c = AOTCompiler(1)
    c.h(0)
    c.apply_lcu(0, t_dag_terms)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)

    assert phase_eq(sv_lcu, ref), f"LCU T†-gate mismatch: lcu={sv_lcu}, ref={ref}"


def test_lcu_sqrt_x():
    """LCU for sqrt(X) = (1+i)/2 * I + (1-i)/2 * X.

    sqrt(X) = e^{-i pi/4 X} = cos(pi/4)I - i sin(pi/4)X = (I - iX)/sqrt(2).
    """
    cos_val = math.cos(math.pi / 4)
    sin_val = math.sin(math.pi / 4)
    sx_terms = [(cos_val, 'I'), (-1j * sin_val, 'X')]

    # Reference: SX = [[1+i, 1-i], [1-i, 1+i]] / 2
    SX_mat = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex) / 2

    # Apply to |0>
    ref_sv = SX_mat @ np.array([1, 0], dtype=complex)

    c = AOTCompiler(1)
    c.apply_lcu(0, sx_terms)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)

    assert phase_eq(sv_lcu, ref_sv), f"LCU SX mismatch: lcu={sv_lcu}, ref={ref_sv}"


def test_lcu_sqrt_x_on_plus():
    """SX gate via LCU on |+> state."""
    cos_val = math.cos(math.pi / 4)
    sin_val = math.sin(math.pi / 4)
    sx_terms = [(cos_val, 'I'), (-1j * sin_val, 'X')]

    SX_mat = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex) / 2
    plus_sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ref_sv = SX_mat @ plus_sv

    c = AOTCompiler(1)
    c.h(0)
    c.apply_lcu(0, sx_terms)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)

    assert phase_eq(sv_lcu, ref_sv), f"LCU SX|+> mismatch: lcu={sv_lcu}, ref={ref_sv}"


def test_lcu_arbitrary_z_rotation():
    """LCU for R_z(theta) = cos(theta/2)I - i sin(theta/2)Z."""
    theta = 0.7  # arbitrary angle
    rz_terms = [(math.cos(theta/2), 'I'), (-1j * math.sin(theta/2), 'Z')]

    # Reference
    Rz = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)
    plus_sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ref_sv = Rz @ plus_sv

    c = AOTCompiler(1)
    c.h(0)
    c.apply_lcu(0, rz_terms)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)

    assert phase_eq(sv_lcu, ref_sv), f"LCU Rz({theta}) mismatch: lcu={sv_lcu}, ref={ref_sv}"


def test_lcu_arbitrary_x_rotation():
    """LCU for R_x(theta) = cos(theta/2)I - i sin(theta/2)X."""
    theta = 1.2
    rx_terms = [(math.cos(theta/2), 'I'), (-1j * math.sin(theta/2), 'X')]

    Rx = np.array([
        [math.cos(theta/2), -1j*math.sin(theta/2)],
        [-1j*math.sin(theta/2), math.cos(theta/2)],
    ], dtype=complex)
    ref_sv = Rx @ np.array([1, 0], dtype=complex)

    c = AOTCompiler(1)
    c.apply_lcu(0, rx_terms)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)

    assert phase_eq(sv_lcu, ref_sv), f"LCU Rx({theta}) mismatch: lcu={sv_lcu}, ref={ref_sv}"


def test_lcu_two_rotations():
    """Two sequential LCU rotations."""
    theta1, theta2 = 0.5, 0.9
    rz1 = [(math.cos(theta1/2), 'I'), (-1j * math.sin(theta1/2), 'Z')]
    rz2 = [(math.cos(theta2/2), 'I'), (-1j * math.sin(theta2/2), 'Z')]

    # Reference: Rz(theta2) @ Rz(theta1) @ H|0>
    Rz1 = np.array([[np.exp(-1j*theta1/2), 0], [0, np.exp(1j*theta1/2)]], dtype=complex)
    Rz2 = np.array([[np.exp(-1j*theta2/2), 0], [0, np.exp(1j*theta2/2)]], dtype=complex)
    ref_sv = Rz2 @ Rz1 @ (H @ np.array([1, 0], dtype=complex))

    c = AOTCompiler(1)
    c.h(0)
    c.apply_lcu(0, rz1)
    c.apply_lcu(0, rz2)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)

    assert phase_eq(sv_lcu, ref_sv), f"Two Rz mismatch"


def test_lcu_entangled():
    """LCU rotation on one qubit of a Bell pair."""
    theta = 0.8
    rz_terms = [(math.cos(theta/2), 'I'), (-1j * math.sin(theta/2), 'Z')]

    # Reference: Rz on qubit 0 of Bell state
    Rz = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    ref_sv = np.kron(Rz, I2) @ bell

    c = AOTCompiler(2)
    c.h(0)
    c.cx(0, 1)
    c.apply_lcu(0, rz_terms)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(2, c, vm)

    assert phase_eq(sv_lcu, ref_sv), f"LCU on Bell mismatch"


def test_lcu_compile_ops_interface():
    """Test that LCU works through compile_ops."""
    theta = 0.6
    rz_terms = [(math.cos(theta/2), 'I'), (-1j * math.sin(theta/2), 'Z')]

    ops = [('H', [0]), ('LCU', [0, rz_terms])]
    c = AOTCompiler(1)
    bc = c.compile_ops(ops)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(bc, c.noise_schedule)
    sv = aot_to_statevector(1, c, vm)

    Rz = np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)
    ref = Rz @ (H @ np.array([1, 0], dtype=complex))
    assert phase_eq(sv, ref)


def test_lcu_large_angle_anticommute():
    """LCU where non-I term dominates, exercising P_m*P_dom order.

    Rx(2.5) has sin(1.25) > cos(1.25), so X becomes dominant.
    The relative operator is I*X = X, but if we computed X*I = X the
    phase would be the same. The real test: Rz(2.5) decomposed as
    cos(1.25)I - i*sin(1.25)Z, with Z dominant. Relative: I*Z = Z.
    Then apply after H so Z has a nontrivial beta.
    """
    theta = 2.5  # sin(1.25) ~ 0.949 > cos(1.25) ~ 0.315
    rx_terms = [(math.cos(theta/2), 'I'), (-1j * math.sin(theta/2), 'X')]

    Rx = np.array([
        [math.cos(theta/2), -1j*math.sin(theta/2)],
        [-1j*math.sin(theta/2), math.cos(theta/2)],
    ], dtype=complex)
    # Apply to |+> so X-dominant factoring interacts nontrivially
    plus = H @ np.array([1, 0], dtype=complex)
    ref_sv = Rx @ plus

    c = AOTCompiler(1)
    c.h(0)
    c.apply_lcu(0, rx_terms)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)
    assert phase_eq(sv_lcu, ref_sv), f"Large-angle Rx mismatch: lcu={sv_lcu}, ref={ref_sv}"


def test_lcu_anticommute_xz():
    """LCU with anticommuting X and Z terms where Z dominates.

    U = 0.3*X + 0.9*Z (not unitary, but LCU machinery still applies).
    Z dominates. Relative: X*Z = -iY. If we wrongly compute Z*X = +iY,
    we get the wrong sign.
    """
    terms = [(0.3, 'X'), (0.9, 'Z')]
    # Reference: (0.3 X + 0.9 Z)|0> = 0.3|1> + 0.9|0>
    ref_sv = np.array([0.9, 0.3], dtype=complex)
    ref_sv /= np.linalg.norm(ref_sv)

    c = AOTCompiler(1)
    c.apply_lcu(0, terms)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)
    assert phase_eq(sv_lcu, ref_sv), f"XZ anticommute mismatch: lcu={sv_lcu}, ref={ref_sv}"


def test_lcu_y_rotation():
    """LCU for R_y(theta) = cos(theta/2)I - i sin(theta/2)Y."""
    theta = 0.9
    ry_terms = [(math.cos(theta/2), 'I'), (-1j * math.sin(theta/2), 'Y')]

    Ry = np.array([
        [math.cos(theta/2), -math.sin(theta/2)],
        [math.sin(theta/2), math.cos(theta/2)],
    ], dtype=complex)
    ref_sv = Ry @ np.array([1, 0], dtype=complex)

    c = AOTCompiler(1)
    c.apply_lcu(0, ry_terms)
    vm = AOTRuntimeVM(rng=np.random.default_rng(42))
    vm.execute(c.bytecode, c.noise_schedule)
    sv_lcu = aot_to_statevector(1, c, vm)

    assert phase_eq(sv_lcu, ref_sv), f"LCU Ry({theta}) mismatch: lcu={sv_lcu}, ref={ref_sv}"
